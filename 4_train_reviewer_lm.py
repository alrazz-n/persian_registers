import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/scratch/project_462001491/nima"
)

EXPERIMENT_DIR = (
    BASE_DIR / "corpus_experiment"
)


# ------------------------------------------------------------
# IMPORTANT:
# Qwen3.6 has its own tokenized-data directory.
#
# This is deliberately different from the old Qwen2.5
# tokenized directory.
# ------------------------------------------------------------

TOKEN_DIR = (
    EXPERIMENT_DIR / "tokenized_Qwen3_6"
)


# ------------------------------------------------------------
# Qwen3.6 validation set
#
# This matches the validation-generation script:
#
# validation/validation_Qwen3_6_5m.bin
# ------------------------------------------------------------

VALIDATION_FILE = (
    EXPERIMENT_DIR
    / "validation"
    / "validation_Qwen3_6_5m.bin"
)


# ------------------------------------------------------------
# Qwen3.6-specific results directory
#
# Deliberately separate from any previous experiment.
# ------------------------------------------------------------

RESULT_DIR = (
    EXPERIMENT_DIR
    / "lm_results_Qwen3_6"
)


# ============================================================
# TOKENIZER
# ============================================================

TOKENIZER_NAME = (
    "Qwen/Qwen3.6-35B-A3B"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 1234

CONTEXT_LENGTH = 2048

TARGET_TRAINING_TOKENS = 100_000_000


# ============================================================
# MODEL
# ============================================================

# Small randomly initialized GPT-2-style model.
#
# This is NOT the Qwen3.6 pretrained model.
# It uses the Qwen3.6 tokenizer/vocabulary.
#
# Suitable as a controlled reviewer sanity-check model.

N_LAYER = 6
N_HEAD = 12
N_EMBD = 768


# ============================================================
# TRAINING
# ============================================================

PER_DEVICE_BATCH_SIZE = 4

GRADIENT_ACCUMULATION = 8

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.1

WARMUP_RATIO = 0.02


# ============================================================
# DATASET
# ============================================================

class MemmapDataset(Dataset):

    def __init__(
        self,
        path,
        context_length
    ):

        self.path = Path(path)

        self.tokens = np.memmap(
            self.path,
            dtype=np.uint32,
            mode="r"
        )

        self.context_length = (
            context_length
        )

        # Need context_length + 1 tokens:
        #
        # input:
        #   tokens[0:2048]
        #
        # labels:
        #   tokens[1:2049]
        #
        self.block_size = (
            context_length + 1
        )

        self.num_sequences = (
            len(self.tokens)
            // self.block_size
        )

    def __len__(self):

        return self.num_sequences

    def __getitem__(self, index):

        start = (
            index * self.block_size
        )

        end = (
            start + self.block_size
        )

        x = torch.tensor(
            self.tokens[start:end],
            dtype=torch.long
        )

        return {
            "input_ids": x[:-1],
            "labels": x[1:],
        }


# ============================================================
# MODEL
# ============================================================

def create_model(
    vocab_size,
    eos_token_id
):

    config = GPT2Config(

        vocab_size=vocab_size,

        n_positions=CONTEXT_LENGTH,

        n_ctx=CONTEXT_LENGTH,

        n_embd=N_EMBD,

        n_layer=N_LAYER,

        n_head=N_HEAD,

        resid_pdrop=0.0,

        embd_pdrop=0.0,

        attn_pdrop=0.0,

        bos_token_id=eos_token_id,

        eos_token_id=eos_token_id,

    )

    model = GPT2LMHeadModel(
        config
    )

    return model


# ============================================================
# CORPUS PATH
# ============================================================

def get_corpus_path(name):

    # IMPORTANT:
    #
    # These filenames exactly match the Qwen3.6
    # tokenization script.
    #
    paths = {

        "hplt3":
            TOKEN_DIR
            / "hplt3_Qwen3_6_100000000.bin",

        "random20":
            TOKEN_DIR
            / "hplt3_random20_Qwen3_6_100000000.bin",

        "perref":
            TOKEN_DIR
            / "perref_Qwen3_6_100000000.bin",
    }

    return paths[name]


# ============================================================
# SAFETY CHECKS
# ============================================================

def check_file(path, description):

    if not path.exists():

        raise FileNotFoundError(
            f"{description} does not exist:\n"
            f"{path}"
        )

    if path.stat().st_size == 0:

        raise RuntimeError(
            f"{description} is empty:\n"
            f"{path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        required=True,
        choices=[
            "hplt3",
            "random20",
            "perref",
        ]
    )

    args = parser.parse_args()


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SYSTEM")
    print("=" * 70)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print(
        "Loading Qwen3.6 tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True
    )

    tokenizer.model_max_length = (
        10**9
    )

    vocab_size = len(tokenizer)

    eos_token_id = (
        tokenizer.eos_token_id
    )

    if eos_token_id is None:

        raise ValueError(
            "Qwen3.6 tokenizer has no EOS token."
        )


    print(
        f"Tokenizer: "
        f"{TOKENIZER_NAME}"
    )

    print(
        f"Vocabulary size: "
        f"{vocab_size:,}"
    )

    print(
        f"EOS token ID: "
        f"{eos_token_id}"
    )


    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    train_file = get_corpus_path(
        args.corpus
    )

    check_file(
        train_file,
        "Training file"
    )

    check_file(
        VALIDATION_FILE,
        "Validation file"
    )


    print()
    print("=" * 70)
    print("DATA")
    print("=" * 70)

    print(
        f"Training file:\n"
        f"{train_file}"
    )

    print(
        f"Validation file:\n"
        f"{VALIDATION_FILE}"
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = MemmapDataset(
        train_file,
        CONTEXT_LENGTH
    )

    validation_dataset = MemmapDataset(
        VALIDATION_FILE,
        CONTEXT_LENGTH
    )


    # --------------------------------------------------------
    # Report actual usable tokens
    # --------------------------------------------------------

    actual_training_tokens = (
        len(train_dataset)
        * CONTEXT_LENGTH
    )

    actual_validation_tokens = (
        len(validation_dataset)
        * CONTEXT_LENGTH
    )


    print(
        f"Training tokens in file: "
        f"{len(train_dataset.tokens):,}"
    )

    print(
        f"Usable training tokens: "
        f"{actual_training_tokens:,}"
    )

    print(
        f"Training sequences: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation tokens in file: "
        f"{len(validation_dataset.tokens):,}"
    )

    print(
        f"Usable validation tokens: "
        f"{actual_validation_tokens:,}"
    )

    print(
        f"Validation sequences: "
        f"{len(validation_dataset):,}"
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Creating randomly initialized model..."
    )

    model = create_model(
        vocab_size=vocab_size,
        eos_token_id=eos_token_id
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )


    # --------------------------------------------------------
    # Calculate optimizer steps
    # --------------------------------------------------------

    tokens_per_optimizer_step = (
        PER_DEVICE_BATCH_SIZE
        * GRADIENT_ACCUMULATION
        * CONTEXT_LENGTH
    )

    max_steps = math.ceil(
        TARGET_TRAINING_TOKENS
        / tokens_per_optimizer_step
    )

    print()
    print(
        f"Tokens per optimizer step: "
        f"{tokens_per_optimizer_step:,}"
    )

    print(
        f"Target training tokens: "
        f"{TARGET_TRAINING_TOKENS:,}"
    )

    print(
        f"Training steps: "
        f"{max_steps:,}"
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        RESULT_DIR / args.corpus
    )

    # --------------------------------------------------------
    # IMPORTANT SAFETY CHECK
    #
    # Do NOT silently overwrite an existing Qwen3.6 run.
    # --------------------------------------------------------

    if output_dir.exists():

        existing_files = list(
            output_dir.iterdir()
        )

        if existing_files:

            raise FileExistsError(
                "\nRefusing to overwrite existing "
                "Qwen3.6 training results.\n\n"
                f"Output directory:\n"
                f"{output_dir}\n\n"
                "Move/delete the old directory "
                "manually if you intentionally want "
                "to rerun this experiment."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    bf16_available = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    fp16_available = (
        torch.cuda.is_available()
        and not bf16_available
    )


    training_args = TrainingArguments(

        output_dir=str(
            output_dir
        ),

        overwrite_output_dir=False,

        max_steps=max_steps,

        per_device_train_batch_size=(
            PER_DEVICE_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        warmup_ratio=WARMUP_RATIO,

        lr_scheduler_type="cosine",

        logging_steps=50,

        eval_strategy="steps",

        eval_steps=500,

        save_strategy="steps",

        save_steps=500,

        save_total_limit=2,

        bf16=bf16_available,

        fp16=fp16_available,

        tf32=False,

        report_to="none",

        seed=SEED,

        data_seed=SEED,

        dataloader_num_workers=2,

        remove_unused_columns=False,
    )


    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"STARTING QWEN3.6 TRAINING: "
        f"{args.corpus}"
    )
    print("=" * 70)

    trainer.train()


    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    metrics = trainer.evaluate()

    eval_loss = metrics[
        "eval_loss"
    ]

    perplexity = math.exp(
        eval_loss
    )


    print(
        f"Corpus: "
        f"{args.corpus}"
    )

    print(
        f"Validation loss: "
        f"{eval_loss:.6f}"
    )

    print(
        f"Validation perplexity: "
        f"{perplexity:.4f}"
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "corpus":
            args.corpus,

        "tokenizer":
            TOKENIZER_NAME,

        "tokenizer_vocab_size":
            int(vocab_size),

        "eos_token_id":
            int(eos_token_id),

        "seed":
            SEED,

        "training_tokens_in_file":
            int(len(train_dataset.tokens)),

        "usable_training_tokens":
            int(actual_training_tokens),

        "validation_tokens_in_file":
            int(len(validation_dataset.tokens)),

        "usable_validation_tokens":
            int(actual_validation_tokens),

        "context_length":
            CONTEXT_LENGTH,

        "parameters":
            int(parameter_count),

        "training_steps":
            int(max_steps),

        "per_device_batch_size":
            PER_DEVICE_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION,

        "effective_batch_tokens":
            int(tokens_per_optimizer_step),

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_ratio":
            WARMUP_RATIO,

        "eval_loss":
            float(eval_loss),

        "perplexity":
            float(perplexity),
    }


    result_file = (
        output_dir
        / "final_results.json"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


    print()
    print(
        "Results saved to:"
    )

    print(
        result_file
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":

    main()
