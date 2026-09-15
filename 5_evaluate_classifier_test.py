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
        "Evaluate a trained LM on the held-out "
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
    help=(
        "Which trained LM to evaluate."
    ),
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/scratch/project_462001491/nima"
)

# ------------------------------------------------------------
# Classifier dataset
# ------------------------------------------------------------

DATASET_DIR = (
    BASE_DIR / "binary_dataset"
)

# ------------------------------------------------------------
# Trained LM checkpoints
# ------------------------------------------------------------

MODEL_DIR = (
    BASE_DIR
    / "corpus_experiment"
    / "lm_results"
)

# ------------------------------------------------------------
# Output results
# ------------------------------------------------------------

RESULT_DIR = (
    BASE_DIR
    / "corpus_experiment"
    / "lm_test_results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

TOKENIZER_NAME = (
    "Qwen/Qwen2.5-0.5B"
)

# ------------------------------------------------------------
# Evaluation context
# ------------------------------------------------------------

CONTEXT_LENGTH = 2048

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
    f"Dataset directory:\n{DATASET_DIR}"
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

if "labels" not in test_dataset.column_names:

    raise KeyError(
        "Test dataset does not contain a 'labels' column."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

print(
    f"Tokenizer: {TOKENIZER_NAME}"
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True
)

tokenizer.model_max_length = 10**9

print(
    f"Vocabulary size: "
    f"{len(tokenizer):,}"
)

print(
    f"EOS token ID: "
    f"{tokenizer.eos_token_id}"
)


# ============================================================
# TOKENIZE TEST DOCUMENTS
# ============================================================

print()
print("=" * 70)
print("TOKENIZING TEST DOCUMENTS")
print("=" * 70)

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

    if not text:

        skipped_empty += 1
        continue


    label = int(
        example["labels"]
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
    # --------------------------------------------------------

    token_ids = tokenizer(
        text,
        add_special_tokens=False
    )["input_ids"]


    # --------------------------------------------------------
    # Skip documents with fewer than 2 tokens
    #
    # At least two tokens are needed to create one
    # next-token prediction.
    # --------------------------------------------------------

    if len(token_ids) < 2:

        skipped_short += 1
        continue


    documents.append({

        "label":
            label,

        "tokens":
            token_ids,

    })


    total_tokens += len(token_ids)


    if (
        (index + 1) % 100 == 0
    ):

        print(
            f"Processed "
            f"{index + 1:,} / "
            f"{len(test_dataset):,} documents"
        )


print()
print(
    "=" * 70
)
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
    f"Skipped empty:            "
    f"{skipped_empty:,}"
)

print(
    f"Skipped <2 tokens:        "
    f"{skipped_short:,}"
)

print(
    f"Unknown labels:           "
    f"{unknown_labels:,}"
)

print(
    f"Total document tokens:    "
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
    "HIGH QUALITY (label=1):"
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
    "LOW QUALITY (label=0):"
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
    """

    corpus_dir = (
        MODEL_DIR / corpus_name
    )

    if not corpus_dir.exists():

        raise FileNotFoundError(
            f"Model directory does not exist:\n"
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
        f"checkpoints."
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
    Evaluate one trained LM on all test documents.

    Loss is weighted by the number of predicted tokens,
    which gives a standard corpus-level cross-entropy.

    Each document is evaluated independently. Therefore,
    predictions do not cross document boundaries.
    """

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print(
        f"Model path:\n{model_path}"
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = GPT2LMHeadModel.from_pretrained(
        model_path
    )

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


            # ------------------------------------------------
            # Document statistics
            # ------------------------------------------------

            document_loss = 0.0

            document_predicted_tokens = 0


            # ------------------------------------------------
            # Split document into chunks
            #
            # Each chunk contains up to:
            #
            #   CONTEXT_LENGTH + 1
            #
            # tokens.
            #
            # The first token is the input context and
            # subsequent tokens are predicted.
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
                # Input
                # ------------------------------------------------

                input_ids = torch.tensor(
                    chunk[:-1],
                    dtype=torch.long,
                    device=DEVICE
                ).unsqueeze(0)


                # ------------------------------------------------
                # Next-token labels
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
                # Overall
                # ------------------------------------------------

                total_loss += (
                    loss.item()
                    * n_tokens
                )

                total_predicted_tokens += (
                    n_tokens
                )


                # ------------------------------------------------
                # Document
                # ------------------------------------------------

                document_loss += (
                    loss.item()
                    * n_tokens
                )

                document_predicted_tokens += (
                    n_tokens
                )


                # ------------------------------------------------
                # Label
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
                        doc_index,

                    "label":
                        label,

                    "loss":
                        float(doc_loss),

                    "perplexity":
                        float(
                            math.exp(doc_loss)
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
    # CALCULATE OVERALL RESULTS
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
    # CALCULATE LABEL RESULTS
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
        "HIGH QUALITY (label=1):"
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
        "LOW QUALITY (label=0):"
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

        "checkpoint":
            str(model_path),

        "parameters":
            int(parameter_count),

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
    print("REVIEWER LM EVALUATION")
    print("=" * 70)

    print(
        f"Corpus: "
        f"{args.corpus}"
    )

    print(
        f"Context length: "
        f"{CONTEXT_LENGTH}"
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
        / f"{args.corpus}.json"
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
        result["high_quality"]["perplexity"]
    )

    low_ppl = (
        result["low_quality"]["perplexity"]
    )


    print(
        f"{args.corpus:<15}"
        f"{result['overall_ppl']:>15.4f}"
        f"{high_ppl:>15.4f}"
        f"{low_ppl:>15.4f}"
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
