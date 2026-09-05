import os
import json
import gc
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
    / "_Sliding_MultiCore_BGE_m3_finetuned_evaluation"
)


THRESHOLD_ROOT = Path(
    r"/projappl/project_462001491/nima/Hybrid_SP_ID_effect /Models/Multi_COre/Fine_tuned/BGEM3_Sliding_Threshold_tuning"
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
# SLIDING WINDOW SETTINGS
# MUST MATCH TRAINING
# ============================================================

MAX_LENGTH = 1024
STRIDE = 768


# ============================================================
# DATASETS
# ============================================================

DATASET_NAMES = [
    "Combined_hybrid_no_NA",
    "Combined_ID_hybrid_no_NA",
    "Combined_single_no_NA",
    "Combined_SP_hybrid_no_na",
]


# ============================================================
# AGGREGATION METHODS
# ============================================================

AGGREGATION_METHODS = [
    "max",
    "mean",
    "top2_mean",
]


# ============================================================
# SLURM ARRAY
# ============================================================

if "SLURM_ARRAY_TASK_ID" not in os.environ:
    raise RuntimeError(
        "SLURM_ARRAY_TASK_ID is not set."
    )

task_id = int(
    os.environ["SLURM_ARRAY_TASK_ID"]
)

if task_id < 0 or task_id >= len(DATASET_NAMES):
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
# PATHS
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
print("DEV threshold directory:", THRESHOLD_DIR)
print("Test output:", OUTPUT_DIR)


# ============================================================
# CHECK PATHS
# ============================================================

for path in [
    DATASET_PATH,
    MODEL_PATH,
    THRESHOLD_DIR,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n{path}"
        )


# ============================================================
# LOAD DEV BEST POSTHOC CONFIGURATION
# ============================================================
#
# EXPECTED:
#
# THRESHOLD_DIR/
#     best_posthoc_configuration.json
#
# Example:
#
# {
#   "selection_metric": "macro_f1",
#   "aggregation": "mean",
#   "threshold_method": "optimized_global",
#   "threshold": 0.32,
#   ...
# }
#
# This is the ONLY configuration used to select the
# aggregation + threshold.
#
# TEST DATA MUST NOT be used to make this choice.
# ============================================================

best_configuration_file = (
    THRESHOLD_DIR
    / "best_posthoc_configuration.json"
)

if not best_configuration_file.exists():

    raise FileNotFoundError(
        "\nCould not find DEV-selected configuration.\n"
        f"Expected:\n"
        f"{best_configuration_file}\n\n"
        "The test script expects one file named:\n"
        "best_posthoc_configuration.json\n"
        "inside the dataset threshold directory."
    )


with open(
    best_configuration_file,
    "r",
    encoding="utf-8",
) as f:

    best_configuration = json.load(f)


# ============================================================
# READ DEV SELECTION
# ============================================================

selection_metric = (
    best_configuration["selection_metric"]
)

dev_selected_aggregation = (
    best_configuration["aggregation"]
)

dev_selected_threshold_method = (
    best_configuration["threshold_method"]
)

dev_selected_threshold = float(
    best_configuration["threshold"]
)


# ============================================================
# VALIDATE DEV CONFIGURATION
# ============================================================

if (
    dev_selected_aggregation
    not in AGGREGATION_METHODS
):

    raise ValueError(
        "\nInvalid aggregation in DEV configuration:\n"
        f"{dev_selected_aggregation}\n"
        f"Allowed values: {AGGREGATION_METHODS}"
    )


if not np.isfinite(
    dev_selected_threshold
):

    raise ValueError(
        "\nDEV threshold is not finite:\n"
        f"{dev_selected_threshold}"
    )


if not (
    0.0
    <= dev_selected_threshold
    <= 1.0
):

    raise ValueError(
        "\nDEV threshold must be between 0 and 1:\n"
        f"{dev_selected_threshold}"
    )


# ============================================================
# PRINT LOCKED DEV CONFIGURATION
# ============================================================

print("\n" + "=" * 100)
print("DEV-SELECTED CONFIGURATION")
print("=" * 100)

print(
    "Configuration file:",
    best_configuration_file
)

print(
    "Selection metric:",
    selection_metric
)

print(
    "DEV-selected aggregation:",
    dev_selected_aggregation
)

print(
    "DEV-selected threshold method:",
    dev_selected_threshold_method
)

print(
    "DEV-selected threshold:",
    dev_selected_threshold
)

print(
    "DEV macro F1:",
    best_configuration.get(
        "macro_f1",
        "not available"
    )
)

print("=" * 100)

print(
    "\nIMPORTANT:"
)

print(
    "The aggregation and threshold above are LOCKED."
)

print(
    "They will be applied to TEST without optimization."
)

print("=" * 100)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = load_from_disk(
    str(DATASET_PATH)
)

print(dataset)


# ============================================================
# LOAD DATASET METADATA
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
# LOAD TOKENIZER
# MUST MATCH TRAINING
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-m3-retromae"
)

print("Tokenizer loaded.")


# ============================================================
# LOAD FINE-TUNED MODEL
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
        "Labels missing from dataset:\n"
        f"{missing_labels}"
    )


if extra_labels:

    raise ValueError(
        "Extra dataset labels:\n"
        f"{extra_labels}"
    )


# ============================================================
# LABEL ALIGNMENT
# SAME AS TRAINING
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
# WITH SLIDING WINDOWS
# ============================================================

def prepare_test_dataset(ds):

    print(
        "\nOriginal test documents:",
        len(ds)
    )


    # --------------------------------------------------------
    # Document IDs
    # --------------------------------------------------------

    ds = ds.add_column(
        "document_id",
        list(range(len(ds)))
    )


    # --------------------------------------------------------
    # Align labels
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Sliding-window tokenization
    # --------------------------------------------------------

    def tokenize_with_overflow(batch):

        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
        )


        sample_mapping = (
            tokenized.pop(
                "overflow_to_sample_mapping"
            )
        )


        # Copy document labels to every chunk

        tokenized["labels"] = [

            batch["labels"][sample_idx]

            for sample_idx
            in sample_mapping

        ]


        # Copy document IDs to every chunk

        tokenized["document_id"] = [

            batch["document_id"][sample_idx]

            for sample_idx
            in sample_mapping

        ]


        return tokenized


    ds = ds.map(
        tokenize_with_overflow,
        batched=True,
        remove_columns=ds.column_names,
    )


    print(
        "\nGenerated test chunks:",
        len(ds)
    )


    print(
        "Unique test documents:",
        len(
            set(
                ds["document_id"]
            )
        )
    )


    return ds


print(
    "\nPreparing TEST dataset with sliding windows..."
)


test_dataset = prepare_test_dataset(
    dataset["test"]
)


# ============================================================
# CHECK CHUNK/DOCUMENT STRUCTURE
# ============================================================

test_chunk_document_ids = np.asarray(
    test_dataset["document_id"],
    dtype=np.int64,
)

test_chunk_labels = np.asarray(
    test_dataset["labels"],
    dtype=np.float32,
)


print(
    "\nTest chunk-document mapping:"
)

print(
    "Chunks:",
    len(test_chunk_document_ids)
)

print(
    "Unique documents:",
    len(
        np.unique(
            test_chunk_document_ids
        )
    )
)

print(
    "Chunk labels shape:",
    test_chunk_labels.shape
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)


# ============================================================
# INFERENCE TRAINER
# ============================================================

args = TrainingArguments(

    output_dir=str(
        OUTPUT_DIR
        / "tmp_eval"
    ),

    per_device_eval_batch_size=16,

    report_to="none",

    bf16=True,

)


trainer = Trainer(

    model=model,

    args=args,

    tokenizer=tokenizer,

    data_collator=data_collator,

)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nRunning sliding-window prediction on TEST..."
)


test_output = trainer.predict(
    test_dataset
)


test_chunk_logits = (
    test_output.predictions
)


print(
    "\nTest chunk logits shape:",
    test_chunk_logits.shape
)


# ============================================================
# VALIDATE LOGIT / MAPPING ALIGNMENT
# ============================================================

if (
    len(test_chunk_logits)
    != len(test_chunk_document_ids)
):

    raise ValueError(
        "\nChunk/logit mismatch!\n"
        f"Logits: "
        f"{len(test_chunk_logits)}\n"
        f"Chunk document IDs: "
        f"{len(test_chunk_document_ids)}"
    )


if test_chunk_logits.ndim != 2:

    raise ValueError(
        f"Expected logits with shape "
        f"(n_chunks, n_classes), got "
        f"{test_chunk_logits.shape}"
    )


if (
    test_chunk_logits.shape[1]
    != len(model_labels)
):

    raise ValueError(
        f"Model output has "
        f"{test_chunk_logits.shape[1]} classes, "
        f"but expected "
        f"{len(model_labels)}."
    )


# ============================================================
# SAVE RAW TEST CHUNK OUTPUTS
# ============================================================

np.save(
    OUTPUT_DIR
    / "test_chunk_logits.npy",
    test_chunk_logits,
)

np.save(
    OUTPUT_DIR
    / "test_chunk_document_ids.npy",
    test_chunk_document_ids,
)

np.save(
    OUTPUT_DIR
    / "test_chunk_labels.npy",
    test_chunk_labels,
)


print(
    "\nSaved raw chunk-level test outputs."
)


# ============================================================
# SIGMOID
# ============================================================

test_chunk_probabilities = (
    1.0
    /
    (
        1.0
        +
        np.exp(
            -np.clip(
                test_chunk_logits,
                -50,
                50,
            )
        )
    )
)


# ============================================================
# SAVE CHUNK PROBABILITIES
# ============================================================

np.save(
    OUTPUT_DIR
    / "test_chunk_probabilities.npy",
    test_chunk_probabilities,
)


# ============================================================
# DOCUMENT-LEVEL AGGREGATION
# ============================================================

def aggregate_chunk_probabilities(
    chunk_probabilities,
    chunk_document_ids,
    chunk_labels,
    aggregation="max",
):

    chunk_document_ids = np.asarray(
        chunk_document_ids
    )

    chunk_labels = np.asarray(
        chunk_labels
    )


    unique_document_ids = pd.unique(
        chunk_document_ids
    )


    document_probabilities = []
    document_labels = []


    for document_id in unique_document_ids:

        mask = (
            chunk_document_ids
            == document_id
        )


        document_chunk_probabilities = (
            chunk_probabilities[mask]
        )


        document_chunk_labels = (
            chunk_labels[mask]
        )


        # ----------------------------------------------------
        # Aggregate probabilities
        # ----------------------------------------------------

        if aggregation == "max":

            document_probability = np.max(
                document_chunk_probabilities,
                axis=0,
            )


        elif aggregation == "mean":

            document_probability = np.mean(
                document_chunk_probabilities,
                axis=0,
            )


        elif aggregation == "top2_mean":

            sorted_probabilities = np.sort(
                document_chunk_probabilities,
                axis=0,
            )


            top_k = min(
                2,
                sorted_probabilities.shape[0],
            )


            document_probability = np.mean(
                sorted_probabilities[-top_k:],
                axis=0,
            )


        else:

            raise ValueError(
                f"Unknown aggregation: "
                f"{aggregation}"
            )


        # ----------------------------------------------------
        # Check labels
        # ----------------------------------------------------

        document_label = (
            document_chunk_labels[0]
        )


        if not np.all(
            document_chunk_labels
            == document_label
        ):

            raise ValueError(
                f"Labels differ between chunks "
                f"for document "
                f"{document_id}"
            )


        document_probabilities.append(
            document_probability
        )


        document_labels.append(
            document_label
        )


    return (

        np.asarray(
            unique_document_ids
        ),

        np.asarray(
            document_probabilities
        ),

        np.asarray(
            document_labels
        ),

    )


# ============================================================
# METRICS
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
# FUNCTION:
# EVALUATE ONE AGGREGATION
# ============================================================

def evaluate_aggregation(
    aggregation,
    threshold,
):

    print(
        "\n" + "-" * 100
    )

    print(
        "Evaluating aggregation:",
        aggregation
    )

    print(
        "Threshold:",
        threshold
    )

    print(
        "-" * 100
    )


    # --------------------------------------------------------
    # Aggregate TEST chunks
    # --------------------------------------------------------

    (
        document_ids,
        document_probabilities,
        document_labels,
    ) = aggregate_chunk_probabilities(

        chunk_probabilities=
            test_chunk_probabilities,

        chunk_document_ids=
            test_chunk_document_ids,

        chunk_labels=
            test_chunk_labels,

        aggregation=
            aggregation,
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        len(document_probabilities)
        != len(document_labels)
    ):

        raise ValueError(
            "Document probability/label "
            "length mismatch."
        )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = (
        document_probabilities
        >= threshold
    ).astype(np.int32)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        document_labels,
        predictions,
    )


    # --------------------------------------------------------
    # Save document-level arrays
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / (
            "test_document_probabilities_"
            f"{aggregation}.npy"
        ),
        document_probabilities,
    )


    np.save(
        OUTPUT_DIR
        / (
            "test_document_ids_"
            f"{aggregation}.npy"
        ),
        document_ids,
    )


    np.save(
        OUTPUT_DIR
        / (
            "test_document_labels_"
            f"{aggregation}.npy"
        ),
        document_labels,
    )


    np.save(
        OUTPUT_DIR
        / (
            "test_predictions_"
            f"{aggregation}.npy"
        ),
        predictions,
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "dataset":
            dataset_name,

        "aggregation":
            aggregation,

        "threshold":
            threshold,

        "selection_metric":
            selection_metric,

        "selection_set":
            "DEV",

        "evaluation_set":
            "TEST",

        "threshold_method":
            dev_selected_threshold_method,

        **metrics,

    }


    return (
        result,
        document_ids,
        document_probabilities,
        document_labels,
        predictions,
    )


# ============================================================
# MAIN TEST EVALUATION
# ============================================================
#
# We calculate all three aggregations for transparency.
#
# HOWEVER:
#
# ONLY the DEV-selected aggregation is the official
# Experiment A test result.
#
# The other two are NOT used to choose anything.
# ============================================================

all_test_results = []


aggregation_outputs = {}


for aggregation in AGGREGATION_METHODS:

    # --------------------------------------------------------
    # Important:
    #
    # For the DEV-selected aggregation:
    # use the DEV-selected threshold.
    #
    # For the other aggregations:
    # also use the SAME locked DEV threshold here only
    # as a descriptive comparison.
    #
    # No threshold is optimized on TEST.
    # --------------------------------------------------------

    result, document_ids, document_probabilities, document_labels, predictions = (
        evaluate_aggregation(
            aggregation=aggregation,
            threshold=dev_selected_threshold,
        )
    )


    all_test_results.append(
        result
    )


    aggregation_outputs[
        aggregation
    ] = {

        "document_ids":
            document_ids,

        "probabilities":
            document_probabilities,

        "labels":
            document_labels,

        "predictions":
            predictions,

    }


# ============================================================
# TEST RESULTS FOR ALL AGGREGATIONS
# USING THE LOCKED DEV THRESHOLD
# ============================================================

all_test_results_df = pd.DataFrame(
    all_test_results
)


print(
    "\n" + "=" * 120
)

print(
    "TEST RESULTS — ALL AGGREGATIONS"
)

print(
    "=" * 120
)

print(
    all_test_results_df[
        [
            "aggregation",
            "threshold",
            "macro_f1",
            "micro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "micro_precision",
            "micro_recall",
        ]
    ].to_string(
        index=False
    )
)


all_test_results_df.to_csv(
    OUTPUT_DIR
    / "test_all_aggregations_locked_DEV_threshold.csv",
    index=False,
)


# ============================================================
# OFFICIAL EXPERIMENT A
# ============================================================
#
# This is the ONLY result that should be treated as the
# final TEST result.
#
# It uses:
#
#     aggregation = selected on DEV
#     threshold   = selected on DEV
#
# and evaluates once on TEST.
# ============================================================

official_test_result = None


for result in all_test_results:

    if (
        result["aggregation"]
        == dev_selected_aggregation
    ):

        official_test_result = result

        break


if official_test_result is None:

    raise RuntimeError(
        "Could not find official TEST result "
        "for DEV-selected aggregation."
    )


official_results_df = pd.DataFrame(
    [official_test_result]
)


print(
    "\n" + "=" * 120
)

print(
    "EXPERIMENT A — OFFICIAL TEST RESULT"
)

print(
    "=" * 120
)

print(
    official_results_df[
        [
            "dataset",
            "aggregation",
            "threshold",
            "selection_metric",
            "selection_set",
            "evaluation_set",
            "macro_f1",
            "micro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "micro_precision",
            "micro_recall",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE OFFICIAL RESULT
# ============================================================

official_results_df.to_csv(
    OUTPUT_DIR
    / "experiment_A_official_test_result.csv",
    index=False,
)


# ============================================================
# SAVE DEV CONFIGURATION USED FOR TEST
# ============================================================

with open(
    OUTPUT_DIR
    / "dev_best_posthoc_configuration_used_for_test.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_configuration,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# PER-CLASS RESULTS
# OFFICIAL DEV-SELECTED CONFIGURATION ONLY
# ============================================================

official_output = aggregation_outputs[
    dev_selected_aggregation
]


official_document_labels = (
    official_output["labels"]
)

official_predictions = (
    official_output["predictions"]
)


per_class_results = []


for class_idx, label in enumerate(
    model_labels
):

    y_true = (
        official_document_labels[
            :, class_idx
        ]
    )


    y_pred = (
        official_predictions[
            :, class_idx
        ]
    )


    row = {

        "dataset":
            dataset_name,

        "experiment":
            "Experiment_A",

        "selection_set":
            "DEV",

        "evaluation_set":
            "TEST",

        "aggregation":
            dev_selected_aggregation,

        "threshold":
            dev_selected_threshold,

        "selection_metric":
            selection_metric,

        "class_index":
            class_idx,

        "label":
            label,

        "actual_positives":
            int(
                y_true.sum()
            ),

        "predicted_positives":
            int(
                y_pred.sum()
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

    }


    per_class_results.append(
        row
    )


per_class_df = pd.DataFrame(
    per_class_results
)


# ============================================================
# SAVE PER-CLASS RESULTS
# ============================================================

per_class_df.to_csv(
    OUTPUT_DIR
    / "experiment_A_official_test_per_class.csv",
    index=False,
)


# ============================================================
# CLASSIFICATION REPORT
# OFFICIAL CONFIGURATION ONLY
# ============================================================

classification_report_dict = (
    classification_report(

        official_document_labels,

        official_predictions,

        target_names=model_labels,

        zero_division=0,

        output_dict=True,

    )
)


classification_report_df = (
    pd.DataFrame(
        classification_report_dict
    ).transpose()
)


classification_report_df.to_csv(
    OUTPUT_DIR
    / "experiment_A_official_test_classification_report.csv",
    encoding="utf-8-sig",
)


# ============================================================
# FIXED 0.50 REFERENCE
# ============================================================
#
# This is optional and is NOT used for selection.
# It is simply a reference showing what happens with the
# same DEV-selected aggregation but threshold = 0.50.
# ============================================================

pred_05 = (
    official_output["probabilities"]
    >= 0.50
).astype(np.int32)


metrics_05 = calculate_metrics(
    official_document_labels,
    pred_05,
)


fixed_05_result = {

    "dataset":
        dataset_name,

    "experiment":
        "Experiment_A",

    "aggregation":
        dev_selected_aggregation,

    "threshold":
        0.50,

    "threshold_method":
        "fixed_reference",

    "selection_set":
        "NONE",

    "evaluation_set":
        "TEST",

    **metrics_05,

}


pd.DataFrame(
    [fixed_05_result]
).to_csv(
    OUTPUT_DIR
    / "experiment_A_test_fixed_0.50_reference.csv",
    index=False,
)


# ============================================================
# SAVE COMPLETE TEST CONFIGURATION
# ============================================================

test_configuration = {

    "experiment":
        "Experiment_A",

    "dataset":
        dataset_name,

    "max_length":
        MAX_LENGTH,

    "stride":
        STRIDE,

    "tokenizer":
        "BAAI/bge-m3-retromae",

    "model_path":
        str(MODEL_PATH),

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    "model_labels":
        model_labels,

    "dataset_labels":
        dataset_labels,

    "dataset_indices_for_model":
        dataset_indices_for_model,

    # --------------------------------------------------------
    # DEV selection
    # --------------------------------------------------------

    "dev_configuration_file":
        str(
            best_configuration_file
        ),

    "dev_configuration":
        best_configuration,

    "selection_metric":
        selection_metric,

    "dev_selected_aggregation":
        dev_selected_aggregation,

    "dev_selected_threshold_method":
        dev_selected_threshold_method,

    "dev_selected_threshold":
        dev_selected_threshold,

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    "test_aggregation":
        dev_selected_aggregation,

    "test_threshold":
        dev_selected_threshold,

    "selection_set":
        "DEV",

    "evaluation_set":
        "TEST",

    "threshold_was_optimized_on_test":
        False,

    "aggregation_was_selected_on_test":
        False,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    "all_aggregations_evaluated_descriptively":
        AGGREGATION_METHODS,

    "official_test_aggregation":
        dev_selected_aggregation,

    "official_test_threshold":
        dev_selected_threshold,

}


with open(
    OUTPUT_DIR
    / "test_evaluation_configuration.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        test_configuration,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 120
)

print(
    "TEST EVALUATION COMPLETE"
)

print(
    "=" * 120
)

print(
    "\nDataset:",
    dataset_name
)

print(
    "Documents:",
    len(
        np.unique(
            test_chunk_document_ids
        )
    )
)

print(
    "Chunks:",
    len(
        test_chunk_document_ids
    )
)

print(
    "\nDEV selected aggregation:",
    dev_selected_aggregation
)

print(
    "DEV selected threshold:",
    dev_selected_threshold
)

print(
    "\nOfficial TEST macro F1:",
    official_test_result["macro_f1"]
)

print(
    "Official TEST micro F1:",
    official_test_result["micro_f1"]
)

print(
    "Official TEST weighted F1:",
    official_test_result["weighted_f1"]
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\n" + "=" * 120
)


# ============================================================
# CLEANUP
# ============================================================

del trainer
del model
del test_dataset
del dataset

gc.collect()
