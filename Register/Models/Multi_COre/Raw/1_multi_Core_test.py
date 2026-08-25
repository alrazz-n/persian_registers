import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import load_from_disk
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path(
    r"/scratch/project_462001491/nima/Hybrid_SP_ID_effect/Dataset_Hugging_face/without_NA"
)

RESULTS_ROOT = DATASET_ROOT / "_evaluation"

MODEL_ID = "TurkuNLP/web-register-classification-multilingual"

BATCH_SIZE = 8
MAX_LENGTH = 512

# Threshold reported for Persian in the Hugging Face model card
THRESHOLD = 0.35


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Load model and tokenizer
# ============================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-large"
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID
)

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# Model labels
# ============================================================

id2label = model.config.id2label

model_labels = [
    id2label[i]
    for i in range(len(id2label))
]

print("\nModel labels:")
print(model_labels)

print(
    "\nNumber of model labels:",
    len(model_labels)
)


# ============================================================
# Main labels
# ============================================================

main_labels = [
    label
    for label in model_labels
    if label.isupper()
]

print("\nMain labels:")
print(main_labels)


# ============================================================
# Select dataset for this SLURM array task
# ============================================================

dataset_dirs = sorted(
    [
        path
        for path in DATASET_ROOT.iterdir()
        if path.is_dir()
        and (path / "dataset_dict.json").exists()
    ]
)

if not dataset_dirs:
    raise RuntimeError(
        f"No Hugging Face datasets found in:\n{DATASET_ROOT}"
    )

# Get array task ID
if "SLURM_ARRAY_TASK_ID" not in os.environ:
    raise RuntimeError(
        "SLURM_ARRAY_TASK_ID is not set. "
        "This script is intended to run as a Slurm array job."
    )

task_id = int(
    os.environ["SLURM_ARRAY_TASK_ID"]
)

if task_id >= len(dataset_dirs):
    raise IndexError(
        f"SLURM_ARRAY_TASK_ID={task_id}, "
        f"but only {len(dataset_dirs)} datasets were found."
    )

dataset_path = dataset_dirs[task_id]
dataset_name = dataset_path.name

print("\n" + "=" * 80)
print("SLURM ARRAY INFORMATION")
print("=" * 80)

print("Task ID:", task_id)
print("Dataset:", dataset_name)
print("Dataset path:", dataset_path)
print("=" * 80)


# ============================================================
# Load selected dataset
# ============================================================

print(
    f"\nLoading dataset: {dataset_name}"
)

dataset = load_from_disk(
    str(dataset_path)
)

# ============================================================
# Evaluation function
# ============================================================

def evaluate_dataset(
    dataset_name,
    dataset,
    threshold=THRESHOLD,
):

    print("\n" + "=" * 80)
    print(f"Evaluating: {dataset_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------

    test_dataset = dataset["test"]

    texts = test_dataset["text"]

    y_true = np.array(
        test_dataset["labels"],
        dtype=np.int32,
    )

    print(
        f"Test examples: {len(texts)}"
    )

    # --------------------------------------------------------
    # Load dataset metadata
    # --------------------------------------------------------

    metadata_file = (
        DATASET_ROOT
        / dataset_name
        / "split_metadata.json"
    )

    if not metadata_file.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{metadata_file}"
        )

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    dataset_labels = metadata["labels"]

    print(
        f"Dataset labels ({len(dataset_labels)}):"
    )
    print(dataset_labels)

    # --------------------------------------------------------
    # Verify label counts
    # --------------------------------------------------------

    if y_true.shape[1] != len(dataset_labels):

        raise ValueError(
            f"Number of label columns does not match metadata "
            f"for {dataset_name}.\n"
            f"y_true columns: {y_true.shape[1]}\n"
            f"metadata labels: {len(dataset_labels)}"
        )

    # --------------------------------------------------------
    # Align dataset gold-label columns to model-label order
    #
    # Dataset labels may have a different order from the model.
    #
    # After this:
    #
    # y_true_aligned[:, 0] -> model_labels[0]
    # y_true_aligned[:, 1] -> model_labels[1]
    # etc.
    # --------------------------------------------------------

    label_to_dataset_index = {
        label: i
        for i, label in enumerate(dataset_labels)
    }

    missing_labels = [
        label
        for label in model_labels
        if label not in label_to_dataset_index
    ]

    if missing_labels:

        raise ValueError(
            f"\nThe following model labels are missing "
            f"from dataset '{dataset_name}':\n"
            f"{missing_labels}\n\n"
            f"This means the dataset cannot be directly "
            f"evaluated against this model."
        )

    dataset_indices_for_model = [
        label_to_dataset_index[label]
        for label in model_labels
    ]

    y_true_aligned = y_true[
        :,
        dataset_indices_for_model
    ]

    # --------------------------------------------------------
    # Model predictions
    # --------------------------------------------------------

    print("\nRunning model predictions...")

    all_probabilities = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(texts),
        )

        batch_texts = texts[start:end]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

            probabilities = torch.sigmoid(
                outputs.logits
            )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        # Progress
        if end == len(texts) or end % (BATCH_SIZE * 10) == 0:

            print(
                f"  Processed {end}/{len(texts)}"
            )

    y_prob = np.concatenate(
        all_probabilities,
        axis=0,
    )

    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    y_pred = (
        y_prob >= threshold
    ).astype(np.int32)

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if y_true_aligned.shape != y_pred.shape:

        raise ValueError(
            f"Shape mismatch for {dataset_name}.\n"
            f"Gold: {y_true_aligned.shape}\n"
            f"Pred: {y_pred.shape}"
        )

    # --------------------------------------------------------
    # Metrics: all labels
    # --------------------------------------------------------

    metrics_all = {

        "dataset": dataset_name,

        "threshold": threshold,

        "num_test_examples": len(texts),

        "micro_f1_all":
            f1_score(
                y_true_aligned,
                y_pred,
                average="micro",
                zero_division=0,
            ),

        "macro_f1_all":
            f1_score(
                y_true_aligned,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "weighted_f1_all":
            f1_score(
                y_true_aligned,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

        "micro_precision_all":
            precision_score(
                y_true_aligned,
                y_pred,
                average="micro",
                zero_division=0,
            ),

        "micro_recall_all":
            recall_score(
                y_true_aligned,
                y_pred,
                average="micro",
                zero_division=0,
            ),
    }

    # ========================================================
    # Classification report
    # ========================================================

    report = classification_report(
        y_true_aligned,
        y_pred,
        target_names=model_labels,
        zero_division=0,
        output_dict=True,
    )

    # --------------------------------------------------------
    # Print classification report
    # --------------------------------------------------------

    print("\nClassification report:")

    print(
        classification_report(
            y_true_aligned,
            y_pred,
            target_names=model_labels,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save classification report as JSON
    # --------------------------------------------------------

    report_file = (
        RESULTS_ROOT
        / f"{dataset_name}_classification_report.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save classification report as CSV
    # --------------------------------------------------------

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_csv = (
        RESULTS_ROOT
        / f"{dataset_name}_classification_report.csv"
    )

    report_df.to_csv(
        report_csv,
        encoding="utf-8-sig",
    )

    print(
        "\nClassification report saved to:"
        f"\n  {report_file}"
        f"\n  {report_csv}"
    )

    # ========================================================
    # Main labels only
    # ========================================================

    main_model_indices = [
        i
        for i, label in enumerate(model_labels)
        if label in main_labels
    ]

    y_true_main = y_true_aligned[
        :,
        main_model_indices
    ]

    y_pred_main = y_pred[
        :,
        main_model_indices
    ]

    metrics_all[
        "micro_f1_main"
    ] = f1_score(
        y_true_main,
        y_pred_main,
        average="micro",
        zero_division=0,
    )

    metrics_all[
        "macro_f1_main"
    ] = f1_score(
        y_true_main,
        y_pred_main,
        average="macro",
        zero_division=0,
    )

    metrics_all[
        "weighted_f1_main"
    ] = f1_score(
        y_true_main,
        y_pred_main,
        average="weighted",
        zero_division=0,
    )

    # ========================================================
    # Per-label metrics
    # ========================================================

    per_label_f1 = f1_score(
        y_true_aligned,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label_precision = precision_score(
        y_true_aligned,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label_recall = recall_score(
        y_true_aligned,
        y_pred,
        average=None,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Store per-label metrics
    # --------------------------------------------------------

    for i, label in enumerate(model_labels):

        metrics_all[
            f"{label}_f1"
        ] = per_label_f1[i]

        metrics_all[
            f"{label}_precision"
        ] = per_label_precision[i]

        metrics_all[
            f"{label}_recall"
        ] = per_label_recall[i]

    # ========================================================
    # Predictions dataframe
    # ========================================================

    prediction_rows = []

    for i, text in enumerate(texts):

        row = {
            "index": i,
            "text": text,
        }

        # ----------------------------------------------------
        # Model probability + prediction for each label
        # ----------------------------------------------------

        for j, label in enumerate(model_labels):

            row[
                f"prob_{label}"
            ] = float(
                y_prob[i, j]
            )

            row[
                f"pred_{label}"
            ] = int(
                y_pred[i, j]
            )

        # ----------------------------------------------------
        # Gold labels
        #
        # Use ORIGINAL dataset ordering here.
        # ----------------------------------------------------

        gold_indices = np.where(
            y_true[i] == 1
        )[0]

        row["gold_labels"] = " ".join(
            dataset_labels[j]
            for j in gold_indices
        )

        # ----------------------------------------------------
        # Predicted labels
        # ----------------------------------------------------

        predicted_indices = np.where(
            y_pred[i] == 1
        )[0]

        row["predicted_labels"] = " ".join(
            model_labels[j]
            for j in predicted_indices
        )

        prediction_rows.append(row)

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    return (
        metrics_all,
        predictions_df,
    )


# ============================================================
# Create results directory
# ============================================================

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Evaluate selected dataset
# ============================================================

metrics, predictions = evaluate_dataset(
    dataset_name=dataset_name,
    dataset=dataset,
    threshold=THRESHOLD,
)


# ============================================================
# Save predictions
# ============================================================

prediction_file = (
    RESULTS_ROOT
    / f"{dataset_name}_predictions.csv"
)

predictions.to_csv(
    prediction_file,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"\nPredictions saved to:"
    f"\n  {prediction_file}"
)


# ============================================================
# Save metrics
# ============================================================

metrics_file = (
    RESULTS_ROOT
    / f"{dataset_name}_metrics.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )

print(
    f"\nMetrics saved to:"
    f"\n  {metrics_file}"
)


# ============================================================
# Create comparison DataFrame for current dataset
# ============================================================

results_df = pd.DataFrame([metrics])


# ------------------------------------------------------------
# Put important columns first
# ------------------------------------------------------------

preferred_columns = [
    "dataset",
    "threshold",
    "num_test_examples",

    "micro_f1_all",
    "macro_f1_all",
    "weighted_f1_all",

    "micro_f1_main",
    "macro_f1_main",
    "weighted_f1_main",

    "micro_precision_all",
    "micro_recall_all",
]

remaining_columns = [
    col
    for col in results_df.columns
    if col not in preferred_columns
]

results_df = results_df[
    preferred_columns
    + remaining_columns
]


# ============================================================
# Print final comparison
# ============================================================

print("\n" + "=" * 100)
print("FINAL RESULTS")
print("=" * 100)

print(
    results_df[
        [
            "dataset",
            "micro_f1_all",
            "macro_f1_all",
            "weighted_f1_all",
            "micro_f1_main",
            "macro_f1_main",
            "weighted_f1_main",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Save comparison results
# ============================================================

results_csv = (
    RESULTS_ROOT
    / "model_comparison.csv"
)

results_json = (
    RESULTS_ROOT
    / "model_comparison.json"
)

results_df.to_csv(
    results_csv,
    index=False,
    encoding="utf-8-sig",
)

results_df.to_json(
    results_json,
    orient="records",
    indent=2,
)

print("\nComparison results saved:")
print(f"  {results_csv}")
print(f"  {results_json}")


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 100)
print("EVALUATION COMPLETE")
print("=" * 100)

print(
    f"\nAll results are in:\n"
    f"  {RESULTS_ROOT}"
)