import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Evaluate a Qwen3.6-tokenizer LM on the held-out "
        "human-annotated classifier test set."
    )
)

parser.add_argument(
    "--corpus",
    required=True,
    choices=[
        "hplt3",
        "random20",
        "perref",
    ],
    help="Which trained LM to evaluate.",
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/scratch/project_462001491/nima"
)

EXPERIMENT_DIR = (
    BASE_DIR / "corpus_experiment"
)


# ------------------------------------------------------------
# Classifier dataset
#
# This is the held-out human-annotated dataset.
# ------------------------------------------------------------

DATASET_DIR = (
    BASE_DIR / "binary_dataset"
)


# ------------------------------------------------------------
# Qwen3.6 trained LM checkpoints
#
# IMPORTANT:
# This is deliberately different from the previous:
#
#     corpus_experiment/lm_results
#
# The Qwen3.6 training script writes to:
#
#     corpus_experiment/lm_results_Qwen3_6
# ------------------------------------------------------------

MODEL_DIR = (
    EXPERIMENT_DIR
    / "lm_results_Qwen3_6"
)


# ------------------------------------------------------------
# Qwen3.6 evaluation results
#
# Separate from any previous Qwen2.5 evaluation results.
# ------------------------------------------------------------

RESULT_DIR = (
    EXPERIMENT_DIR
    / "lm_test_results_Qwen3_6"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Tokenizer
#
# MUST match the tokenizer used to create:
#
#   hplt3_Qwen3_6_100000000.bin
#   hplt3_random20_Qwen3_6_100000000.bin
#   perref_Qwen3_6_100000000.bin
#
# and:
#
#   validation_Qwen3_6_5m.bin
# ------------------------------------------------------------

TOKENIZER_NAME = (
    "Qwen/Qwen3.6-35B-A3B"
)


# ------------------------------------------------------------
# Context length
# ------------------------------------------------------------

CONTEXT_LENGTH = 2048


# ------------------------------------------------------------
# Dataset label column
# ------------------------------------------------------------

LABEL_COLUMN = "Binary"


# ------------------------------------------------------------
# Classifier labels
#
# 1 = high quality
# 0 = low quality
# ------------------------------------------------------------

HIGH_QUALITY_LABEL = 1
LOW_QUALITY_LABEL = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 70)
print("SYSTEM")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )


# ============================================================
# LOAD CLASSIFIER DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING CLASSIFIER DATASET")
print("=" * 70)

print(
    f"Dataset directory:\n"
    f"{DATASET_DIR}"
)

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset directory does not exist:\n"
        f"{DATASET_DIR}"
    )


dataset = load_from_disk(
    str(DATASET_DIR)
)


print()
print("Dataset:")
print(dataset)


print()
print(
    f"Available splits: "
    f"{list(dataset.keys())}"
)


# ------------------------------------------------------------
# Use the held-out test split
# ------------------------------------------------------------

if "test" not in dataset:

    raise KeyError(
        "The dataset does not contain a 'test' split."
    )


test_dataset = dataset["test"]


print()
print(
    f"Test documents: "
    f"{len(test_dataset):,}"
)

print(
    f"Test columns: "
    f"{test_dataset.column_names}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "text" not in test_dataset.column_names:

    raise KeyError(
        "Test dataset does not contain a 'text' column."
    )


if LABEL_COLUMN not in test_dataset.column_names:

    raise KeyError(
        f"Test dataset does not contain the "
        f"'{LABEL_COLUMN}' column."
    )


# ============================================================
# LOAD QWEN3.6 TOKENIZER
# ============================================================

print()
print("=" * 70)
print("LOADING QWEN3.6 TOKENIZER")
print("=" * 70)

print(
    f"Tokenizer: "
    f"{TOKENIZER_NAME}"
)


tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True
)


tokenizer.model_max_length = 10**9


vocab_size = len(
    tokenizer
)


eos_token_id = (
    tokenizer.eos_token_id
)


if eos_token_id is None:

    raise ValueError(
        "Qwen3.6 tokenizer does not have an EOS token."
    )


print(
    f"Vocabulary size: "
    f"{vocab_size:,}"
)


print(
    f"EOS token ID: "
    f"{eos_token_id}"
)


# ============================================================
# TOKENIZE TEST DOCUMENTS
# ============================================================

print()
print("=" * 70)
print("TOKENIZING TEST DOCUMENTS")
print("=" * 70)

print(
    f"Tokenizer: "
    f"{TOKENIZER_NAME}"
)

print(
    f"Label column: "
    f"{LABEL_COLUMN}"
)

print(
    "High-quality label: 1"
)

print(
    "Low-quality label:  0"
)

print(
    "No EOS token is added."
)

print(
    "Documents remain separate during evaluation."
)


documents = []

total_tokens = 0

skipped_empty = 0

skipped_short = 0

unknown_labels = 0


for index, example in enumerate(
    test_dataset
):

    text = example["text"]


    # --------------------------------------------------------
    # Empty text
    # --------------------------------------------------------

    if not text:

        skipped_empty += 1

        continue


    # --------------------------------------------------------
    # Read classifier label
    # --------------------------------------------------------

    label = int(
        example[LABEL_COLUMN]
    )


    # --------------------------------------------------------
    # Check label
    # --------------------------------------------------------

    if label not in [
        HIGH_QUALITY_LABEL,
        LOW_QUALITY_LABEL,
    ]:

        unknown_labels += 1

        print(
            f"WARNING: unknown label "
            f"{label} at test index {index}"
        )

        continue


    # --------------------------------------------------------
    # Tokenize
    #
    # IMPORTANT:
    # This matches the Qwen3.6 tokenization scripts:
    #
    #     add_special_tokens=False
    #
    # No EOS is manually appended.
    # --------------------------------------------------------

    token_ids = tokenizer(
        text,
        add_special_tokens=False
    )["input_ids"]


    # --------------------------------------------------------
    # Need at least two tokens
    # --------------------------------------------------------

    if len(token_ids) < 2:

        skipped_short += 1

        continue


    documents.append({

        "test_index":
            index,

        "label":
            label,

        "tokens":
            token_ids,

    })


    total_tokens += len(
        token_ids
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        (index + 1) % 100 == 0
    ):

        print(
            f"Processed "
            f"{index + 1:,} / "
            f"{len(test_dataset):,} documents"
        )


# ============================================================
# TOKENIZATION SUMMARY
# ============================================================

print()
print("=" * 70)
print("TOKENIZATION COMPLETE")
print("=" * 70)


print(
    f"Original test documents: "
    f"{len(test_dataset):,}"
)


print(
    f"Usable test documents:   "
    f"{len(documents):,}"
)


print(
    f"Skipped empty:           "
    f"{skipped_empty:,}"
)


print(
    f"Skipped <2 tokens:       "
    f"{skipped_short:,}"
)


print(
    f"Unknown labels:          "
    f"{unknown_labels:,}"
)


print(
    f"Total document tokens:   "
    f"{total_tokens:,}"
)


if not documents:

    raise RuntimeError(
        "No usable test documents were found."
    )


# ============================================================
# LABEL STATISTICS
# ============================================================

print()
print("=" * 70)
print("TEST SET LABEL STATISTICS")
print("=" * 70)


label_counts = {

    HIGH_QUALITY_LABEL: {
        "documents": 0,
        "tokens": 0,
    },

    LOW_QUALITY_LABEL: {
        "documents": 0,
        "tokens": 0,
    },

}


for document in documents:

    label = document["label"]

    n_tokens = len(
        document["tokens"]
    )


    label_counts[label][
        "documents"
    ] += 1


    label_counts[label][
        "tokens"
    ] += n_tokens


print()

print(
    "HIGH QUALITY (Binary=1):"
)

print(
    f"  documents: "
    f"{label_counts[HIGH_QUALITY_LABEL]['documents']:,}"
)

print(
    f"  tokens:    "
    f"{label_counts[HIGH_QUALITY_LABEL]['tokens']:,}"
)


print()

print(
    "LOW QUALITY (Binary=0):"
)

print(
    f"  documents: "
    f"{label_counts[LOW_QUALITY_LABEL]['documents']:,}"
)

print(
    f"  tokens:    "
    f"{label_counts[LOW_QUALITY_LABEL]['tokens']:,}"
)


# ============================================================
# FIND FINAL CHECKPOINT
# ============================================================

def find_final_checkpoint(
    corpus_name
):
    """
    Find the checkpoint with the largest training step.

    Only searches inside the Qwen3.6-specific model directory.
    """

    corpus_dir = (
        MODEL_DIR / corpus_name
    )


    if not corpus_dir.exists():

        raise FileNotFoundError(
            f"Qwen3.6 model directory does not exist:\n"
            f"{corpus_dir}"
        )


    checkpoints = []


    for path in corpus_dir.glob(
        "checkpoint-*"
    ):

        if not path.is_dir():

            continue


        try:

            step = int(
                path.name.split("-")[1]
            )

        except (
            ValueError,
            IndexError,
        ):

            continue


        checkpoints.append(
            (step, path)
        )


    if not checkpoints:

        raise RuntimeError(
            f"No checkpoints found in:\n"
            f"{corpus_dir}"
        )


    checkpoints.sort(
        key=lambda x: x[0]
    )


    final_step, final_path = (
        checkpoints[-1]
    )


    print()
    print(
        f"Found {len(checkpoints):,} "
        f"Qwen3.6 checkpoints."
    )


    print(
        f"Final checkpoint step: "
        f"{final_step:,}"
    )


    print(
        f"Final checkpoint:\n"
        f"{final_path}"
    )


    return final_path


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_path
):
    """
    Evaluate one trained Qwen3.6-tokenizer LM on the
    held-out human-annotated test documents.

    Loss is weighted by the number of predicted tokens.

    Documents are evaluated independently.

    No predictions cross document boundaries.

    Long documents are evaluated in independent chunks of
    CONTEXT_LENGTH tokens. Within each chunk, every token
    after the first is predicted from the preceding tokens.
    """

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)


    print(
        f"Model path:\n"
        f"{model_path}"
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = GPT2LMHeadModel.from_pretrained(
        model_path
    )


    # --------------------------------------------------------
    # Model/tokenizer vocabulary consistency check
    # --------------------------------------------------------

    model_vocab_size = (
        model.config.vocab_size
    )


    print(
        f"Model vocabulary size: "
        f"{model_vocab_size:,}"
    )


    print(
        f"Tokenizer vocabulary size: "
        f"{vocab_size:,}"
    )


    if model_vocab_size != vocab_size:

        raise ValueError(
            "\nVocabulary mismatch!\n\n"
            f"Model:     {model_vocab_size:,}\n"
            f"Tokenizer: {vocab_size:,}\n\n"
            "This checkpoint does not appear to "
            "match the Qwen3.6 tokenizer."
        )


    # --------------------------------------------------------
    # Context-length consistency check
    # --------------------------------------------------------

    model_max_positions = getattr(
        model.config,
        "n_positions",
        None
    )


    if (
        model_max_positions is not None
        and model_max_positions
        < CONTEXT_LENGTH
    ):

        raise ValueError(
            "\nContext-length mismatch!\n\n"
            f"Model n_positions: "
            f"{model_max_positions}\n"
            f"Required: "
            f"{CONTEXT_LENGTH}"
        )


    # --------------------------------------------------------
    # Move model to device
    # --------------------------------------------------------

    model.to(DEVICE)

    model.eval()


    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )


    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    total_loss = 0.0

    total_predicted_tokens = 0


    # --------------------------------------------------------
    # Per-label statistics
    # --------------------------------------------------------

    label_stats = {

        HIGH_QUALITY_LABEL: {
            "loss": 0.0,
            "tokens": 0,
            "documents": 0,
        },

        LOW_QUALITY_LABEL: {
            "loss": 0.0,
            "tokens": 0,
            "documents": 0,
        },

    }


    # --------------------------------------------------------
    # Per-document statistics
    # --------------------------------------------------------

    document_results = []


    # ========================================================
    # EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)


    with torch.no_grad():

        for doc_index, document in enumerate(
            documents
        ):

            token_ids = document["tokens"]

            label = document["label"]

            original_test_index = (
                document["test_index"]
            )


            # ------------------------------------------------
            # Document statistics
            # ------------------------------------------------

            document_loss = 0.0

            document_predicted_tokens = 0


            # ------------------------------------------------
            # Split document into chunks
            #
            # IMPORTANT:
            #
            # Documents remain independent.
            #
            # We do NOT concatenate separate documents.
            #
            # Each chunk contains at most:
            #
            #     CONTEXT_LENGTH + 1
            #
            # tokens.
            #
            # This gives at most CONTEXT_LENGTH
            # next-token predictions.
            # ------------------------------------------------

            for start in range(
                0,
                len(token_ids) - 1,
                CONTEXT_LENGTH
            ):

                chunk = token_ids[
                    start:
                    start + CONTEXT_LENGTH + 1
                ]


                if len(chunk) < 2:

                    continue


                # ------------------------------------------------
                # Input tokens
                # ------------------------------------------------

                input_ids = torch.tensor(
                    chunk[:-1],
                    dtype=torch.long,
                    device=DEVICE
                ).unsqueeze(0)


                # ------------------------------------------------
                # Target tokens
                # ------------------------------------------------

                labels = torch.tensor(
                    chunk[1:],
                    dtype=torch.long,
                    device=DEVICE
                ).unsqueeze(0)


                # ------------------------------------------------
                # Forward pass
                # ------------------------------------------------

                outputs = model(
                    input_ids=input_ids,
                    labels=labels
                )


                loss = outputs.loss


                n_tokens = labels.numel()


                # ------------------------------------------------
                # Overall statistics
                # ------------------------------------------------

                total_loss += (
                    loss.item()
                    * n_tokens
                )


                total_predicted_tokens += (
                    n_tokens
                )


                # ------------------------------------------------
                # Document statistics
                # ------------------------------------------------

                document_loss += (
                    loss.item()
                    * n_tokens
                )


                document_predicted_tokens += (
                    n_tokens
                )


                # ------------------------------------------------
                # Label statistics
                # ------------------------------------------------

                label_stats[label][
                    "loss"
                ] += (
                    loss.item()
                    * n_tokens
                )


                label_stats[label][
                    "tokens"
                ] += n_tokens


            # ------------------------------------------------
            # Store document result
            # ------------------------------------------------

            if document_predicted_tokens > 0:

                doc_loss = (
                    document_loss
                    / document_predicted_tokens
                )


                document_results.append({

                    "test_index":
                        int(
                            original_test_index
                        ),

                    "label":
                        int(label),

                    "loss":
                        float(
                            doc_loss
                        ),

                    "perplexity":
                        float(
                            math.exp(
                                doc_loss
                            )
                        ),

                    "tokens":
                        int(
                            document_predicted_tokens
                        ),

                })


                label_stats[label][
                    "documents"
                ] += 1


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (doc_index + 1) % 25 == 0
                or
                (doc_index + 1) == len(documents)
            ):

                print(
                    f"Evaluated "
                    f"{doc_index + 1:,} / "
                    f"{len(documents):,} documents"
                )


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    if total_predicted_tokens == 0:

        raise RuntimeError(
            "No tokens were evaluated."
        )


    overall_loss = (
        total_loss
        / total_predicted_tokens
    )


    overall_ppl = math.exp(
        overall_loss
    )


    # ========================================================
    # PER-LABEL RESULTS
    # ========================================================

    label_results = {}


    for label in [
        HIGH_QUALITY_LABEL,
        LOW_QUALITY_LABEL,
    ]:

        stats = label_stats[label]


        if stats["tokens"] == 0:

            label_results[label] = {

                "documents":
                    stats["documents"],

                "tokens":
                    stats["tokens"],

                "loss":
                    None,

                "perplexity":
                    None,

            }

            continue


        loss = (
            stats["loss"]
            / stats["tokens"]
        )


        ppl = math.exp(
            loss
        )


        label_results[label] = {

            "documents":
                stats["documents"],

            "tokens":
                stats["tokens"],

            "loss":
                float(loss),

            "perplexity":
                float(ppl),

        }


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)


    print()

    print(
        f"Corpus: "
        f"{args.corpus}"
    )


    print(
        f"Tokenizer: "
        f"{TOKENIZER_NAME}"
    )


    print(
        f"Checkpoint:\n"
        f"{model_path}"
    )


    print()

    print(
        f"Overall loss: "
        f"{overall_loss:.6f}"
    )


    print(
        f"Overall PPL:  "
        f"{overall_ppl:.4f}"
    )


    # --------------------------------------------------------
    # High quality
    # --------------------------------------------------------

    high_result = label_results[
        HIGH_QUALITY_LABEL
    ]


    print()

    print(
        "HIGH QUALITY (Binary=1):"
    )


    print(
        f"  documents: "
        f"{high_result['documents']:,}"
    )


    print(
        f"  tokens:    "
        f"{high_result['tokens']:,}"
    )


    if high_result["loss"] is not None:

        print(
            f"  loss:      "
            f"{high_result['loss']:.6f}"
        )


        print(
            f"  PPL:       "
            f"{high_result['perplexity']:.4f}"
        )


    # --------------------------------------------------------
    # Low quality
    # --------------------------------------------------------

    low_result = label_results[
        LOW_QUALITY_LABEL
    ]


    print()

    print(
        "LOW QUALITY (Binary=0):"
    )


    print(
        f"  documents: "
        f"{low_result['documents']:,}"
    )


    print(
        f"  tokens:    "
        f"{low_result['tokens']:,}"
    )


    if low_result["loss"] is not None:

        print(
            f"  loss:      "
            f"{low_result['loss']:.6f}"
        )


        print(
            f"  PPL:       "
            f"{low_result['perplexity']:.4f}"
        )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "corpus":
            args.corpus,

        "tokenizer":
            TOKENIZER_NAME,

        "tokenizer_vocab_size":
            int(vocab_size),

        "eos_token_id":
            int(eos_token_id),

        "context_length":
            CONTEXT_LENGTH,

        "checkpoint":
            str(model_path),

        "parameters":
            int(parameter_count),

        "test_documents_original":
            int(len(test_dataset)),

        "test_documents_evaluated":
            int(len(documents)),

        "test_tokens":
            int(total_tokens),

        "predicted_tokens":
            int(total_predicted_tokens),

        "overall_loss":
            float(overall_loss),

        "overall_ppl":
            float(overall_ppl),

        "high_quality":
            label_results[
                HIGH_QUALITY_LABEL
            ],

        "low_quality":
            label_results[
                LOW_QUALITY_LABEL
            ],

        "document_results":
            document_results,

    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("QWEN3.6 REVIEWER LM EVALUATION")
    print("=" * 70)


    print(
        f"Corpus: "
        f"{args.corpus}"
    )


    print(
        f"Tokenizer: "
        f"{TOKENIZER_NAME}"
    )


    print(
        f"Context length: "
        f"{CONTEXT_LENGTH}"
    )


    print(
        f"Dataset label column: "
        f"{LABEL_COLUMN}"
    )


    print(
        f"High-quality label: "
        f"{HIGH_QUALITY_LABEL}"
    )


    print(
        f"Low-quality label: "
        f"{LOW_QUALITY_LABEL}"
    )


    # --------------------------------------------------------
    # Find final checkpoint
    # --------------------------------------------------------

    checkpoint = find_final_checkpoint(
        args.corpus
    )


    # --------------------------------------------------------
    # Evaluate model
    # --------------------------------------------------------

    result = evaluate_model(
        checkpoint
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        RESULT_DIR
        / f"{args.corpus}_Qwen3_6.json"
    )


    # --------------------------------------------------------
    # Safety check:
    # Do not overwrite an existing evaluation.
    # --------------------------------------------------------

    if output_file.exists():

        raise FileExistsError(
            "\nRefusing to overwrite existing "
            "Qwen3.6 evaluation result:\n"
            f"{output_file}\n\n"
            "Delete or rename the existing result "
            "only if you intentionally want to rerun it."
        )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)


    print()

    print(
        f"{'Corpus':<15}"
        f"{'Overall PPL':>15}"
        f"{'High PPL':>15}"
        f"{'Low PPL':>15}"
    )


    print(
        "-" * 60
    )


    high_ppl = (
        result[
            "high_quality"
        ]["perplexity"]
    )


    low_ppl = (
        result[
            "low_quality"
        ]["perplexity"]
    )


    high_ppl_string = (
        f"{high_ppl:.4f}"
        if high_ppl is not None
        else "N/A"
    )


    low_ppl_string = (
        f"{low_ppl:.4f}"
        if low_ppl is not None
        else "N/A"
    )


    print(
        f"{args.corpus:<15}"
        f"{result['overall_ppl']:>15.4f}"
        f"{high_ppl_string:>15}"
        f"{low_ppl_string:>15}"
    )


    print()

    print(
        f"Results saved to:"
    )


    print(
        output_file
    )


    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)
