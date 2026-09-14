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

TOKEN_DIR = (
    EXPERIMENT_DIR / "tokenized"
)

VALIDATION_FILE = (
    EXPERIMENT_DIR
    / "validation"
    / "validation_5m.bin"
)

RESULT_DIR = (
    EXPERIMENT_DIR / "lm_results"
)


# ============================================================
# TOKENIZER
# ============================================================

TOKENIZER_NAME = (
    "Qwen/Qwen2.5-0.5B"
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

# Small model suitable for a reviewer sanity check.

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

    paths = {

        "hplt3":
            TOKEN_DIR
            / "hplt3_100000000.bin",

        "random20":
            TOKEN_DIR
            / "hplt3_random20_100000000.bin",

        "perref":
            TOKEN_DIR
            / "perref_100000000.bin",
    }

    return paths[name]


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
        "Loading tokenizer..."
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
            "Tokenizer has no EOS token."
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

    if not train_file.exists():

        raise FileNotFoundError(
            train_file
        )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            VALIDATION_FILE
        )


    print()
    print("=" * 70)
    print("DATA")
    print("=" * 70)

    print(
        f"Training file:\n{train_file}"
    )

    print(
        f"Validation file:\n{VALIDATION_FILE}"
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


    print(
        f"Training tokens: "
        f"{len(train_dataset.tokens):,}"
    )

    print(
        f"Training sequences: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation tokens: "
        f"{len(validation_dataset.tokens):,}"
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
        f"Training steps: "
        f"{max_steps:,}"
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        RESULT_DIR / args.corpus
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

        overwrite_output_dir=True,

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
        f"STARTING TRAINING: {args.corpus}"
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
        f"Corpus: {args.corpus}"
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

        "corpus": args.corpus,

        "seed": SEED,

        "training_tokens":
            int(len(train_dataset.tokens)),

        "validation_tokens":
            int(len(validation_dataset.tokens)),

        "context_length":
            CONTEXT_LENGTH,

        "parameters":
            int(parameter_count),

        "training_steps":
            int(max_steps),

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
        f"Results saved to:"
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