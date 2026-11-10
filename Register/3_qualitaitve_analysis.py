
# ============================================================
# QUALITATIVE COMPARATIVE ERROR ANALYSIS
# PRETRAINED VS FINE-TUNED BGE-M3
# ============================================================
#
# Objectives:
# 1. Identify documents incorrectly classified by the base model
#    but correctly classified by the fine-tuned model.
# 2. Identify partially recovered documents.
# 3. Identify corrected false negatives and false positives.
# 4. Save document text, ground truth, predictions and probabilities.
# 5. Generate class-level improvement statistics.
# 6. Save a qualitative analysis dataset for manual inspection.
#
# NO TRAINING REQUIRED
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from datasets import load_from_disk

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"/scratch/project_462001491/nima/Hybrid_SP_ID_effect/Dataset_Hugging_face/without_NA"
)

# Base model
BASE_MODEL_ID = (
    "TurkuNLP/web-register-classification-multilingual-bge"
)

# Fine-tuned model directory
MODEL_RESULTS_ROOT = (
    DATASET_ROOT
    / "_MultiCore_BGE_m3_finetuned_evaluation"
)

# Output directory
RESULTS_ROOT = (
    DATASET_ROOT
    / "_MultiCore_BGE_m3_qualitative_comparative_analysis"
)

HF_CACHE = Path(
    "/scratch/project_462001491/nima/huggingface"
)

TOKENIZER_ID = (
    "BAAI/bge-m3-retromae"
)

MAX_LENGTH = 1024

BATCH_SIZE = 8

# Classification threshold
THRESHOLD = 0.35

# Minimum number of recovered examples to save per class
# (This does not limit the complete results CSV.)
MIN_CLASS_EXAMPLES = 1

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# SELECT DATASET USING SLURM ARRAY
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
    raise RuntimeError("No datasets found.")

if "SLURM_ARRAY_TASK_ID" not in os.environ:
    raise RuntimeError("SLURM_ARRAY_TASK_ID is not set.")

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

if task_id < 0 or task_id >= len(dataset_dirs):
    raise IndexError("Invalid SLURM_ARRAY_TASK_ID.")

dataset_path = dataset_dirs[task_id]

dataset_name = dataset_path.name

print("\nDataset:", dataset_name)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_MODEL_PATH = BASE_MODEL_ID

FINETUNED_MODEL_PATH = (
    MODEL_RESULTS_ROOT
    / dataset_name
    / "final_model"
)

if not Path(FINETUNED_MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Fine-tuned model not found: {FINETUNED_MODEL_PATH}"
    )


OUTPUT_DIR = (
    RESULTS_ROOT
    / dataset_name
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Base model:", BASE_MODEL_PATH)

print("Fine-tuned model:", FINETUNED_MODEL_PATH)

print("Output:", OUTPUT_DIR)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading test dataset...")

dataset = load_from_disk(
    str(dataset_path)
)

test_dataset = dataset["test"]

print("Number of test documents:", len(test_dataset))


# ============================================================
# LOAD METADATA
# ============================================================

metadata_file = (
    dataset_path
    / "split_metadata.json"
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
# PRESERVE ORIGINAL TEXT AND LABELS
# ============================================================

texts = list(
    test_dataset["text"]
)

original_labels = np.asarray(
    test_dataset["labels"],
    dtype=np.float32,
)

if original_labels.ndim != 2:
    raise ValueError("Expected multilabel ground-truth arrays.")

if len(texts) != len(original_labels):
    raise ValueError("Text and label counts do not match.")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    cache_dir=str(HF_CACHE),
)


# ============================================================
# LOAD BOTH MODELS
# ============================================================

print("\nLoading base model...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    cache_dir=str(HF_CACHE),
)

base_model.to(device)

base_model.eval()


print("\nLoading fine-tuned model...")

finetuned_model = AutoModelForSequenceClassification.from_pretrained(
    str(FINETUNED_MODEL_PATH),
    cache_dir=str(HF_CACHE),
)

finetuned_model.to(device)

finetuned_model.eval()


# ============================================================
# VERIFY LABEL ORDER
# ============================================================

def get_model_labels(model):

    id2label = model.config.id2label

    return [
        id2label.get(i, id2label.get(str(i)))
        for i in range(model.config.num_labels)
    ]


base_labels = get_model_labels(base_model)

finetuned_labels = get_model_labels(finetuned_model)


if base_labels != finetuned_labels:

    raise ValueError(
        "Base and fine-tuned model label orders differ.\n"
        f"Base: {base_labels}\n"
        f"Fine-tuned: {finetuned_labels}"
    )


model_labels = base_labels


if len(model_labels) != len(dataset_labels):

    raise ValueError("Dataset and model label counts differ.")


if set(model_labels) != set(dataset_labels):

    raise ValueError("Dataset and model label names differ.")


# Align ground-truth labels to model label order
label_to_dataset_index = {
    label: i
    for i, label in enumerate(dataset_labels)
}

dataset_indices_for_model = [
    label_to_dataset_index[label]
    for label in model_labels
]

ground_truth = original_labels[
    :, dataset_indices_for_model
]

print("\nVerified label order:")
print(model_labels)


# ============================================================
# TOKENIZE TEST DATA ONCE
# ============================================================

print("\nTokenizing test documents...")

tokenized = tokenizer(
    texts,
    truncation=True,
    max_length=MAX_LENGTH,
)

from datasets import Dataset

inference_dataset = Dataset.from_dict(
    tokenized
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)

test_loader = DataLoader(
    inference_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
)


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def get_predictions(model, loader):

    all_probabilities = []

    with torch.inference_mode():

        for batch_idx, batch in enumerate(loader):

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(
                **inputs,
                return_dict=True,
            )

            probabilities = torch.sigmoid(
                outputs.logits.float()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            if (batch_idx + 1) % 50 == 0:

                print(
                    f"Processed batches: {batch_idx + 1}"
                )

    return np.concatenate(
        all_probabilities,
        axis=0,
    )


# ============================================================
# BASE MODEL INFERENCE
# ============================================================

print("\nRunning base model inference...")

base_probabilities = get_predictions(
    base_model,
    test_loader,
)


# ============================================================
# FINE-TUNED MODEL INFERENCE
# ============================================================

print("\nRunning fine-tuned model inference...")

finetuned_probabilities = get_predictions(
    finetuned_model,
    test_loader,
)


# ============================================================
# VERIFY INFERENCE RESULTS
# ============================================================

assert base_probabilities.shape == ground_truth.shape

assert finetuned_probabilities.shape == ground_truth.shape

assert len(base_probabilities) == len(texts)


# ============================================================
# GENERATE MULTILABEL PREDICTIONS
# ============================================================

base_predictions = (
    base_probabilities >= THRESHOLD
).astype(int)

finetuned_predictions = (
    finetuned_probabilities >= THRESHOLD
).astype(int)

ground_truth_binary = (
    ground_truth >= 0.5
).astype(int)


# ============================================================
# DOCUMENT-LEVEL CORRECTNESS
# ============================================================

base_exact_match = np.all(
    base_predictions == ground_truth_binary,
    axis=1,
)

finetuned_exact_match = np.all(
    finetuned_predictions == ground_truth_binary,
    axis=1,
)


# Base incorrect, fine-tuned correct
fully_recovered = (
    ~base_exact_match
    & finetuned_exact_match
)


# Base correct, fine-tuned incorrect
regressed = (
    base_exact_match
    & ~finetuned_exact_match
)


# Both incorrect
both_incorrect = (
    ~base_exact_match
    & ~finetuned_exact_match
)


# Both correct
both_correct = (
    base_exact_match
    & finetuned_exact_match
)


# ============================================================
# DOCUMENT-LEVEL LABEL RECOVERY
# ============================================================

base_false_negatives = (
    (ground_truth_binary == 1)
    & (base_predictions == 0)
)

finetuned_false_negatives = (
    (ground_truth_binary == 1)
    & (finetuned_predictions == 0)
)

base_false_positives = (
    (ground_truth_binary == 0)
    & (base_predictions == 1)
)

finetuned_false_positives = (
    (ground_truth_binary == 0)
    & (finetuned_predictions == 1)
)


# Corrected false negatives
recovered_false_negatives = (
    base_false_negatives
    & ~finetuned_false_negatives
)


# Corrected false positives
corrected_false_positives = (
    base_false_positives
    & ~finetuned_false_positives
)


# Newly introduced false positives
new_false_positives = (
    ~base_false_positives
    & finetuned_false_positives
)


# Remaining false negatives
remaining_false_negatives = (
    base_false_negatives
    & finetuned_false_negatives
)


# Partially recovered:
# At least one missing true label was recovered,
# but the complete document prediction is still incorrect.
recovered_label_counts = (
    recovered_false_negatives.sum(axis=1)
)

partially_recovered = (
    (recovered_label_counts > 0)
    & ~finetuned_exact_match
)


# Any document with at least one recovered false negative
any_recovered = (
    recovered_label_counts > 0
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_label_names(binary_array):

    return [
        model_labels[i]
        for i in np.where(binary_array == 1)[0]
    ]


def get_probability_dict(probabilities):

    return {
        f"Probability_{label}": float(probabilities[i])
        for i, label in enumerate(model_labels)
    }


def get_error_details(
    ground_truth_row,
    prediction_row,
):

    false_negatives = (
        (ground_truth_row == 1)
        & (prediction_row == 0)
    )

    false_positives = (
        (ground_truth_row == 0)
        & (prediction_row == 1)
    )

    return (
        get_label_names(false_negatives),
        get_label_names(false_positives),
    )


# ============================================================
# BUILD DOCUMENT-LEVEL COMPARISON DATAFRAME
# ============================================================

print("\nBuilding comparative analysis dataframe...")

records = []

for i in range(len(texts)):

    gt = ground_truth_binary[i]

    base_pred = base_predictions[i]

    ft_pred = finetuned_predictions[i]

    base_fn, base_fp = get_error_details(
        gt,
        base_pred,
    )

    ft_fn, ft_fp = get_error_details(
        gt,
        ft_pred,
    )

    recovered_fn = get_label_names(
        recovered_false_negatives[i].astype(int)
    )

    corrected_fp = get_label_names(
        corrected_false_positives[i].astype(int)
    )

    introduced_fp = get_label_names(
        new_false_positives[i].astype(int)
    )

    remaining_fn = get_label_names(
        remaining_false_negatives[i].astype(int)
    )

    if fully_recovered[i]:

        status = "FULLY_RECOVERED"

    elif partially_recovered[i]:

        status = "PARTIALLY_RECOVERED"

    elif both_correct[i]:

        status = "BOTH_CORRECT"

    elif regressed[i]:

        status = "REGRESSED"

    else:

        status = "BOTH_INCORRECT"

    record = {

        "Document_ID": i,

        "Text": texts[i],

        "Status": status,

        "GroundTruth": get_label_names(gt),

        "Base_Predictions": get_label_names(base_pred),

        "Finetuned_Predictions": get_label_names(ft_pred),

        "Base_Exact_Match": bool(base_exact_match[i]),

        "Finetuned_Exact_Match": bool(finetuned_exact_match[i]),

        "Base_False_Negatives": base_fn,

        "Finetuned_False_Negatives": ft_fn,

        "Base_False_Positives": base_fp,

        "Finetuned_False_Positives": ft_fp,

        "Recovered_Labels": recovered_fn,

        "Corrected_False_Positives": corrected_fp,

        "New_False_Positives": introduced_fp,

        "Remaining_Missed_Labels": remaining_fn,

        "Number_Recovered_Labels": len(recovered_fn),

        "Number_New_False_Positives": len(introduced_fp),

    }

    record.update({

        f"Base_{key}": value

        for key, value in get_probability_dict(
            base_probabilities[i]
        ).items()

    })

    record.update({

        f"Finetuned_{key}": value

        for key, value in get_probability_dict(
            finetuned_probabilities[i]
        ).items()

    })

    records.append(record)


comparison_df = pd.DataFrame(records)


# ============================================================
# SAVE COMPLETE COMPARISON
# ============================================================

comparison_df.to_csv(
    OUTPUT_DIR / "all_documents_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE FULLY RECOVERED DOCUMENTS
# ============================================================

fully_recovered_df = comparison_df[
    comparison_df["Status"] == "FULLY_RECOVERED"
].copy()

fully_recovered_df.to_csv(
    OUTPUT_DIR / "fully_recovered_documents.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE PARTIALLY RECOVERED DOCUMENTS
# ============================================================

partially_recovered_df = comparison_df[
    comparison_df["Status"] == "PARTIALLY_RECOVERED"
].copy()

partially_recovered_df.to_csv(
    OUTPUT_DIR / "partially_recovered_documents.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE ALL DOCUMENTS WITH RECOVERED LABELS
# ============================================================

any_recovered_df = comparison_df[
    comparison_df["Number_Recovered_Labels"] > 0
].copy()

any_recovered_df.to_csv(
    OUTPUT_DIR / "documents_with_recovered_labels.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE REGRESSED DOCUMENTS
# ============================================================

regressed_df = comparison_df[
    comparison_df["Status"] == "REGRESSED"
].copy()

regressed_df.to_csv(
    OUTPUT_DIR / "regressed_documents.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# CLASS-LEVEL IMPROVEMENT ANALYSIS
# ============================================================

class_records = []

for class_idx, class_name in enumerate(model_labels):

    gt = ground_truth_binary[:, class_idx]

    base_pred = base_predictions[:, class_idx]

    ft_pred = finetuned_predictions[:, class_idx]

    base_fn = (
        (gt == 1) & (base_pred == 0)
    ).sum()

    ft_fn = (
        (gt == 1) & (ft_pred == 0)
    ).sum()

    base_fp = (
        (gt == 0) & (base_pred == 1)
    ).sum()

    ft_fp = (
        (gt == 0) & (ft_pred == 1)
    ).sum()

    recovered_fn = recovered_false_negatives[
        :, class_idx
    ].sum()

    corrected_fp = corrected_false_positives[
        :, class_idx
    ].sum()

    introduced_fp = new_false_positives[
        :, class_idx
    ].sum()

    support = gt.sum()

    class_records.append({

        "Class": class_name,

        "GroundTruth_Support": int(support),

        "Base_False_Negatives": int(base_fn),

        "Finetuned_False_Negatives": int(ft_fn),

        "Recovered_False_Negatives": int(recovered_fn),

        "Base_False_Positives": int(base_fp),

        "Finetuned_False_Positives": int(ft_fp),

        "Corrected_False_Positives": int(corrected_fp),

        "New_False_Positives": int(introduced_fp),

        "Base_Recall": (
            1 - base_fn / support
            if support > 0 else np.nan
        ),

        "Finetuned_Recall": (
            1 - ft_fn / support
            if support > 0 else np.nan
        ),

    })


class_analysis_df = pd.DataFrame(
    class_records
)

class_analysis_df.to_csv(
    OUTPUT_DIR / "class_level_improvements.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# OVERALL STATISTICS
# ============================================================

total_documents = len(texts)

summary = {

    "Dataset": dataset_name,

    "Total_Documents": int(total_documents),

    "Threshold": THRESHOLD,

    "Base_Exact_Match_Count": int(
        base_exact_match.sum()
    ),

    "Finetuned_Exact_Match_Count": int(
        finetuned_exact_match.sum()
    ),

    "Base_Exact_Match_Ratio": float(
        base_exact_match.mean()
    ),

    "Finetuned_Exact_Match_Ratio": float(
        finetuned_exact_match.mean()
    ),

    "Fully_Recovered_Documents": int(
        fully_recovered.sum()
    ),

    "Partially_Recovered_Documents": int(
        partially_recovered.sum()
    ),

    "Documents_With_At_Least_One_Recovered_Label": int(
        any_recovered.sum()
    ),

    "Documents_Correct_In_Both": int(
        both_correct.sum()
    ),

    "Documents_Incorrect_In_Both": int(
        both_incorrect.sum()
    ),

    "Regressed_Documents": int(
        regressed.sum()
    ),

    "Total_Recovered_False_Negatives": int(
        recovered_false_negatives.sum()
    ),

    "Total_Corrected_False_Positives": int(
        corrected_false_positives.sum()
    ),

    "Total_New_False_Positives": int(
        new_false_positives.sum()
    ),

    "Total_Remaining_False_Negatives": int(
        remaining_false_negatives.sum()
    ),

}


with open(
    OUTPUT_DIR / "overall_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


# ============================================================
# SAVE RECOMMENDED QUALITATIVE SAMPLE
# ============================================================

# Prioritize fully recovered examples, then partially
# recovered examples with the largest number of recovered labels.

qualitative_df = comparison_df[
    comparison_df["Status"].isin([
        "FULLY_RECOVERED",
        "PARTIALLY_RECOVERED",
    ])
].copy()

qualitative_df["Analysis_Priority"] = (
    qualitative_df["Number_Recovered_Labels"]
    + qualitative_df["Status"].map({
        "FULLY_RECOVERED": 100,
        "PARTIALLY_RECOVERED": 0,
    })
)

qualitative_df = qualitative_df.sort_values(
    by="Analysis_Priority",
    ascending=False,
)

qualitative_df.to_csv(
    OUTPUT_DIR / "qualitative_analysis_candidates.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# CLASS-SPECIFIC QUALITATIVE CANDIDATES
# ============================================================

for class_idx, class_name in enumerate(model_labels):

    class_mask = recovered_false_negatives[
        :, class_idx
    ]

    class_candidates = comparison_df[
        class_mask
    ].copy()

    if len(class_candidates) >= MIN_CLASS_EXAMPLES:

        safe_name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(class_name),
        )

        class_candidates.to_csv(
            OUTPUT_DIR
            / f"recovered_examples_{safe_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# SAVE RAW PROBABILITIES
# ============================================================

np.save(
    OUTPUT_DIR / "base_probabilities.npy",
    base_probabilities,
)

np.save(
    OUTPUT_DIR / "finetuned_probabilities.npy",
    finetuned_probabilities,
)

np.save(
    OUTPUT_DIR / "ground_truth.npy",
    ground_truth_binary,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)

print("COMPARATIVE QUALITATIVE ANALYSIS COMPLETED")

print("=" * 80)

print("\nOverall results:")

for key, value in summary.items():

    print(f"{key}: {value}")


print("\nClass-level improvements:")

print(
    class_analysis_df.to_string(index=False)
)


print("\nOutput files:")

for file in sorted(OUTPUT_DIR.iterdir()):

    if file.is_file():

        print(file.name)


print("\nOutput directory:")

print(OUTPUT_DIR)

print("\n" + "=" * 80)