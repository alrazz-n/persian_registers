import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from datasets import load_from_disk

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"/scratch/project_462001491/nima/Hybrid_SP_ID_effect/Dataset_Hugging_face/without_NA"
)

MODEL_ROOT = (
    DATASET_ROOT
    / "_MultiCore_XLMR_finetuned_evaluation"
)

THRESHOLD_ROOT = Path(
    r"/projappl/project_462001491/nima/Hybrid_SP_ID_effect /Models/Multi_COre/Fine_tuned/XLMR_Threshold_tuning"
)

RESULT_ROOT = (
    THRESHOLD_ROOT
    / "_test_evaluation"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)



# ============================================================
# SLURM ARRAY
# ============================================================

DATASET_NAMES = [
    "Combined_hybrid_no_NA",
    "Combined_ID_hybrid_no_NA",
    "Combined_single_no_NA",
    "Combined_SP_hybrid_no_NA",
]


if "SLURM_ARRAY_TASK_ID" not in os.environ:
    raise RuntimeError(
        "SLURM_ARRAY_TASK_ID is not set."
    )


task_id = int(
    os.environ["SLURM_ARRAY_TASK_ID"]
)


if task_id >= len(DATASET_NAMES):
    raise IndexError(
        f"SLURM_ARRAY_TASK_ID={task_id}, "
        f"but only {len(DATASET_NAMES)} datasets exist."
    )


dataset_name = DATASET_NAMES[task_id]


print("\n" + "=" * 100)
print("SLURM ARRAY INFORMATION")
print("=" * 100)

print("Task ID:", task_id)
print("Dataset:", dataset_name)

print("=" * 100)


# ============================================================
# PATHS FOR THIS DATASET
# ============================================================

DATASET_PATH = (
    DATASET_ROOT
    / dataset_name
)

MODEL_PATH = (
    MODEL_ROOT
    / dataset_name
    / "final_model"
)

THRESHOLD_DIR = (
    THRESHOLD_ROOT
    / dataset_name
)

OUTPUT_DIR = (
    RESULT_ROOT
    / dataset_name
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("\nPaths:")
print("Dataset:", DATASET_PATH)
print("Model:", MODEL_PATH)
print("Thresholds:", THRESHOLD_DIR)
print("Output:", OUTPUT_DIR)


# ============================================================
# CHECK PATHS
# ============================================================

required_paths = [
    DATASET_PATH,
    MODEL_PATH,
    THRESHOLD_DIR,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path does not exist:\n{path}"
        )


global_threshold_file = (
    THRESHOLD_DIR
    / "best_global_threshold.json"
)

per_class_threshold_file = (
    THRESHOLD_DIR
    / "per_class_thresholds.json"
)

if not global_threshold_file.exists():

    raise FileNotFoundError(
        f"Global threshold file not found:\n"
        f"{global_threshold_file}"
    )

if not per_class_threshold_file.exists():

    raise FileNotFoundError(
        f"Per-class threshold file not found:\n"
        f"{per_class_threshold_file}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = load_from_disk(
    str(DATASET_PATH)
)

print(dataset)


# ============================================================
# LOAD METADATA
# ============================================================

metadata_file = (
    DATASET_PATH
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


print("\nDataset labels:")
print(dataset_labels)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-large"
)

print("Tokenizer loaded.")


# ============================================================
# MODEL
# ============================================================

print("\nLoading fine-tuned model...")

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_PATH
    )
)

print("Model loaded.")


# ============================================================
# MODEL LABELS
# ============================================================

id2label = model.config.id2label

model_labels = [
    id2label[i]
    for i in range(
        len(id2label)
    )
]


print("\nModel labels:")
print(model_labels)


# ============================================================
# VERIFY LABEL SPACE
# ============================================================

if len(dataset_labels) != len(model_labels):

    raise ValueError(
        "Dataset/model label count mismatch:\n"
        f"Dataset: {len(dataset_labels)}\n"
        f"Model: {len(model_labels)}"
    )


missing_labels = [
    label
    for label in model_labels
    if label not in dataset_labels
]

extra_labels = [
    label
    for label in dataset_labels
    if label not in model_labels
]


if missing_labels:

    raise ValueError(
        f"Labels missing from dataset:\n"
        f"{missing_labels}"
    )


if extra_labels:

    raise ValueError(
        f"Extra dataset labels:\n"
        f"{extra_labels}"
    )


# ============================================================
# LABEL ALIGNMENT
# ============================================================

label_to_dataset_index = {
    label: i
    for i, label in enumerate(
        dataset_labels
    )
}


dataset_indices_for_model = [
    label_to_dataset_index[label]
    for label in model_labels
]


print("\nLabel mapping:")

for model_index, label in enumerate(
    model_labels
):

    dataset_index = (
        dataset_indices_for_model[
            model_index
        ]
    )

    print(
        f"{model_index:2d}: "
        f"{label} <- "
        f"dataset column "
        f"{dataset_index}"
    )


# ============================================================
# PREPARE TEST DATASET
# ============================================================

def prepare_test_dataset(ds):

    def align_labels(example):

        original_labels = np.asarray(
            example["labels"],
            dtype=np.float32,
        )

        aligned_labels = (
            original_labels[
                dataset_indices_for_model
            ]
        )

        return {
            "labels":
                aligned_labels.tolist()
        }


    ds = ds.map(
        align_labels
    )


    def tokenize(example):

        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
        )


    ds = ds.map(
        tokenize,
        batched=True,
    )


    columns_to_keep = [
        "input_ids",
        "attention_mask",
        "labels",
    ]


    if "token_type_ids" in ds.column_names:

        columns_to_keep.append(
            "token_type_ids"
        )


    ds = ds.remove_columns(
        [
            column
            for column in ds.column_names
            if column not in columns_to_keep
        ]
    )


    return ds


print("\nPreparing test dataset...")

test_dataset = prepare_test_dataset(
    dataset["test"]
)

print(
    "Test examples:",
    len(test_dataset)
)


# ============================================================
# TRAINER FOR INFERENCE ONLY
# ============================================================

args = TrainingArguments(

    output_dir=str(
        OUTPUT_DIR / "tmp_eval"
    ),

    per_device_eval_batch_size=16,

    report_to="none",

    bf16=True,

)


trainer = Trainer(

    model=model,

    args=args,

    tokenizer=tokenizer,

    data_collator=DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    ),

)


# ============================================================
# TEST PREDICTION
# ============================================================

print("\nRunning prediction on TEST set...")

test_output = trainer.predict(
    test_dataset
)


test_logits = test_output.predictions

test_labels = test_output.label_ids


print(
    "Test logits shape:",
    test_logits.shape
)

print(
    "Test labels shape:",
    test_labels.shape
)


# ============================================================
# SAVE TEST LOGITS
# ============================================================

np.save(
    OUTPUT_DIR / "test_logits.npy",
    test_logits,
)

np.save(
    OUTPUT_DIR / "test_labels.npy",
    test_labels,
)


# ============================================================
# LOGITS -> PROBABILITIES
# ============================================================

test_probabilities = (
    1.0
    /
    (
        1.0
        + np.exp(
            -test_logits
        )
    )
)


# ============================================================
# LOAD GLOBAL THRESHOLD
# ============================================================

with open(
    global_threshold_file,
    "r",
    encoding="utf-8",
) as f:

    global_data = json.load(f)


global_threshold = float(
    global_data["threshold"]
)


print(
    "\nGlobal threshold:",
    global_threshold
)


# ============================================================
# LOAD PER-CLASS THRESHOLDS
# ============================================================

with open(
    per_class_threshold_file,
    "r",
    encoding="utf-8",
) as f:

    per_class_thresholds_dict = json.load(f)


per_class_thresholds = np.asarray(
    [
        per_class_thresholds_dict[label]
        for label in model_labels
    ],
    dtype=np.float64,
)


print("\nPer-class thresholds:")

for label, threshold in zip(
    model_labels,
    per_class_thresholds,
):

    print(
        f"{label:10s}: "
        f"{threshold:.4f}"
    )


# ============================================================
# PREDICTIONS
# ============================================================

# ------------------------------------------------------------
# 0.50
# ------------------------------------------------------------

pred_05 = (
    test_probabilities >= 0.50
).astype(np.int32)


# ------------------------------------------------------------
# GLOBAL
# ------------------------------------------------------------

pred_global = (
    test_probabilities
    >= global_threshold
).astype(np.int32)


# ------------------------------------------------------------
# PER CLASS
# ------------------------------------------------------------

pred_per_class = (
    test_probabilities
    >= per_class_thresholds
).astype(np.int32)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    return {

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "micro_precision": precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),

        "micro_recall": recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),

    }


# ============================================================
# OVERALL RESULTS
# ============================================================

results = pd.DataFrame([

    {
        "dataset": dataset_name,

        "method": "0.50",

        "threshold": 0.50,

        **calculate_metrics(
            test_labels,
            pred_05,
        ),
    },

    {
        "dataset": dataset_name,

        "method": "global",

        "threshold": global_threshold,

        **calculate_metrics(
            test_labels,
            pred_global,
        ),
    },

    {
        "dataset": dataset_name,

        "method": "per_class",

        "threshold": "individual",

        **calculate_metrics(
            test_labels,
            pred_per_class,
        ),
    },

])


print("\n" + "=" * 100)
print("TEST RESULTS")
print("=" * 100)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# SAVE OVERALL RESULTS
# ============================================================

results_file = (
    OUTPUT_DIR
    / "test_threshold_comparison.csv"
)

results.to_csv(
    results_file,
    index=False,
)


# ============================================================
# PER-CLASS TEST RESULTS
# ============================================================

per_class_results = []


for class_idx, label in enumerate(
    model_labels
):

    y_true = test_labels[
        :, class_idx
    ]


    row = {

        "class_index":
            class_idx,

        "label":
            label,

        "threshold_per_class":
            per_class_thresholds[
                class_idx
            ],

        "actual_positives":
            int(y_true.sum()),

    }


    # --------------------------------------------------------
    # 0.50
    # --------------------------------------------------------

    row["f1_0.50"] = f1_score(
        y_true,
        pred_05[:, class_idx],
        zero_division=0,
    )

    row["precision_0.50"] = precision_score(
        y_true,
        pred_05[:, class_idx],
        zero_division=0,
    )

    row["recall_0.50"] = recall_score(
        y_true,
        pred_05[:, class_idx],
        zero_division=0,
    )


    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    row["f1_global"] = f1_score(
        y_true,
        pred_global[:, class_idx],
        zero_division=0,
    )

    row["precision_global"] = precision_score(
        y_true,
        pred_global[:, class_idx],
        zero_division=0,
    )

    row["recall_global"] = recall_score(
        y_true,
        pred_global[:, class_idx],
        zero_division=0,
    )


    # --------------------------------------------------------
    # Per class
    # --------------------------------------------------------

    row["f1_per_class"] = f1_score(
        y_true,
        pred_per_class[:, class_idx],
        zero_division=0,
    )

    row["precision_per_class"] = precision_score(
        y_true,
        pred_per_class[:, class_idx],
        zero_division=0,
    )

    row["recall_per_class"] = recall_score(
        y_true,
        pred_per_class[:, class_idx],
        zero_division=0,
    )


    row["predicted_positives_0.50"] = int(
        pred_05[:, class_idx].sum()
    )

    row["predicted_positives_global"] = int(
        pred_global[:, class_idx].sum()
    )

    row["predicted_positives_per_class"] = int(
        pred_per_class[:, class_idx].sum()
    )


    per_class_results.append(
        row
    )


per_class_df = pd.DataFrame(
    per_class_results
)


# ============================================================
# SAVE PER-CLASS RESULTS
# ============================================================

per_class_file = (
    OUTPUT_DIR
    / "test_per_class_comparison.csv"
)

per_class_df.to_csv(
    per_class_file,
    index=False,
)


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

for method, predictions in [

    ("0.50", pred_05),

    ("global", pred_global),

    ("per_class", pred_per_class),

]:

    report = classification_report(

        test_labels,

        predictions,

        target_names=model_labels,

        zero_division=0,

        output_dict=True,

    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    report_file = (
        OUTPUT_DIR
        / f"test_classification_report_{method}.csv"
    )


    report_df.to_csv(
        report_file,
        encoding="utf-8-sig",
    )


# ============================================================
# SAVE THRESHOLDS USED
# ============================================================

threshold_summary = {

    "dataset": dataset_name,

    "global_threshold":
        global_threshold,

    "per_class_thresholds":
        {
            label: float(threshold)
            for label, threshold
            in zip(
                model_labels,
                per_class_thresholds,
            )
        },

}


with open(
    OUTPUT_DIR
    / "thresholds_used.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        threshold_summary,
        f,
        indent=2,
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 100)
print("EVALUATION COMPLETE")
print("=" * 100)

print(
    "Dataset:",
    dataset_name
)

print(
    "Results:",
    OUTPUT_DIR
)

print(
    "\nBest Test Macro-F1:"
)

best_row = results.loc[
    results["macro_f1"].idxmax()
]

print(
    best_row.to_string()
)
