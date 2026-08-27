#---------------------------------------------
#AVOID
#this script is a mess #AVOID
#---------------------------------------------
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna

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
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path(
    r"/scratch/project_462001491/nima/Hybrid_SP_ID_effect/Dataset_Hugging_face/without_NA"
)

RESULTS_ROOT = DATASET_ROOT / "_MultiCore_finetuned_evaluation"

MODEL_ID = "TurkuNLP/web-register-classification-multilingual"

MAX_LENGTH = 512

SEED = 42

# ------------------------------------------------------------
# Initial baseline hyperparameters
# ------------------------------------------------------------
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 512
SEED = 42
bf16=True
dataloader_num_workers=0


# ------------------------------------------------------------
# Optuna hyperparameter search
# ------------------------------------------------------------

N_OPTUNA_TRIALS = 20

LEARNING_RATE_MIN = 5e-6
LEARNING_RATE_MAX = 3e-5

WEIGHT_DECAY_MIN = 0.0
WEIGHT_DECAY_MAX = 0.1

WARMUP_RATIO_MIN = 0.0
WARMUP_RATIO_MAX = 0.15

# Number of epochs used during hyperparameter search
OPTUNA_NUM_TRAIN_EPOCHS = 3

# Number of epochs for final training
FINAL_NUM_TRAIN_EPOCHS = 5

# Fixed threshold
THRESHOLD = 0.5



# ------------------------------------------------------------
# Hugging Face cache
# ------------------------------------------------------------

HF_CACHE = Path(
    "/scratch/project_462001491/nima/huggingface"
)


# ============================================================
# Reproducibility
# ============================================================

set_seed(SEED)


# ============================================================
# Device information
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
# Select dataset from SLURM array
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
        f"No Hugging Face datasets found in:\n"
        f"{DATASET_ROOT}"
    )


if "SLURM_ARRAY_TASK_ID" not in os.environ:
    raise RuntimeError(
        "SLURM_ARRAY_TASK_ID is not set."
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


print("\n" + "=" * 100)
print("SLURM ARRAY INFORMATION")
print("=" * 100)

print("Task ID:", task_id)
print("Dataset:", dataset_name)
print("Dataset path:", dataset_path)

print("=" * 100)


# ============================================================
# Load dataset
# ============================================================

print(
    f"\nLoading dataset: {dataset_name}"
)

dataset = load_from_disk(
    str(dataset_path)
)

print(dataset)


# ============================================================
# Load metadata
# ============================================================

metadata_file = (
    dataset_path
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

print(
    "Number of dataset labels:",
    len(dataset_labels)
)


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-large",
    cache_dir=str(HF_CACHE),
)

print("Tokenizer loaded.")


# ============================================================
# Load model
# ============================================================

def model_init(trial=None):

    print("\nLoading pretrained model for trial...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        cache_dir=str(HF_CACHE),
    )

    model.config.problem_type = (
        "multi_label_classification"
    )

    return model



# ============================================================
# Load model configuration for labels
# ============================================================

print("\nLoading model configuration...")

label_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    cache_dir=str(HF_CACHE),
)

id2label = label_model.config.id2label

model_labels = [
    id2label[i]
    for i in range(
        len(id2label)
    )
]

del label_model

print("\nModel labels:")
print(model_labels)

print(
    "Number of model labels:",
    len(model_labels)
)


# ============================================================
# Verify label space
# ============================================================

if len(dataset_labels) != len(model_labels):

    raise ValueError(
        f"\nLabel count mismatch.\n"
        f"Dataset labels: {len(dataset_labels)}\n"
        f"Model labels: {len(model_labels)}"
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
        f"\nLabels missing from dataset:\n"
        f"{missing_labels}"
    )


if extra_labels:

    raise ValueError(
        f"\nExtra labels in dataset:\n"
        f"{extra_labels}"
    )


# ============================================================
# Dataset label -> model label mapping
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
for model_index, label in enumerate(model_labels):

    dataset_index = (
        dataset_indices_for_model[
            model_index
        ]
    )

    print(
        f"{model_index:2d}: "
        f"{label:5s} <- "
        f"dataset column {dataset_index}"
    )


# ============================================================
# Important:
# Model must use multilabel classification
# ============================================================

model.config.problem_type = (
    "multi_label_classification"
)


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(ds, split_name):


    print(
        f"\nPreparing {split_name}: "
        f"{len(ds)} examples"
    )

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
            "labels": aligned_labels.tolist()
        }

    ds = ds.map(
        align_labels
    )


    def tokenize(example):

        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    ds = ds.map(
        tokenize,
        batched=False,
    )


    # Only keep what Trainer needs
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


train_dataset = prepare_dataset(
    dataset["train"],
    "train",
)

validation_dataset = prepare_dataset(
    dataset["dev"],
    "dev",
)

test_dataset = prepare_dataset(
    dataset["test"],
    "test",
)


print("\nPrepared datasets:")

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(validation_dataset)
)

print(
    "Test:",
    len(test_dataset)
)


# ============================================================
# Data collator
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_prediction):

    logits = eval_prediction.predictions

    labels = eval_prediction.label_ids

    probabilities = 1 / (
        1 + np.exp(-logits)
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int32)

    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    micro_precision = precision_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
    }

# ============================================================
# Optuna hyperparameter search space
# ============================================================

def hp_space(trial):

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            LEARNING_RATE_MIN,
            LEARNING_RATE_MAX,
            log=True,
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            WEIGHT_DECAY_MIN,
            WEIGHT_DECAY_MAX,
        ),

        "warmup_ratio": trial.suggest_float(
            "warmup_ratio",
            WARMUP_RATIO_MIN,
            WARMUP_RATIO_MAX,
        ),

        "num_train_epochs": OPTUNA_NUM_TRAIN_EPOCHS,
    }

# ============================================================
# Optuna objective
# ============================================================

def compute_objective(metrics):

    return metrics["eval_micro_f1"]

# ============================================================
# Output directories
# ============================================================

OUTPUT_DIR = (
    RESULTS_ROOT
    / dataset_name
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Training arguments
# ============================================================

def create_training_args():

    return TrainingArguments(

        output_dir=str(
            OUTPUT_DIR / "optuna"
        ),

        overwrite_output_dir=True,

        num_train_epochs=OPTUNA_NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=1e-5,

        weight_decay=0.01,

        warmup_ratio=0.05,

        lr_scheduler_type="linear",

        eval_strategy="epoch",

        logging_strategy="epoch",

        # IMPORTANT:
        # Do not save model checkpoints during Optuna.
        save_strategy="no",

        # Therefore this must also be False.
        load_best_model_at_end=False,

        bf16=True,

        seed=SEED,

        data_seed=SEED,

        report_to="none",

        dataloader_num_workers=0,

        remove_unused_columns=True,
    )


training_args = create_training_args()

trainer = Trainer(
    model_init=model_init,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


OPTUNA_ROOT = RESULTS_ROOT / "_optuna"

OPTUNA_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

OPTUNA_STORAGE_FILE = (
    OPTUNA_ROOT
    / f"{dataset_name}_optuna_journal.log"
)

OPTUNA_STUDY_NAME = (
    f"multilabel_{dataset_name}"
)

optuna_storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(
        str(OPTUNA_STORAGE_FILE)
    )
)

best_run = trainer.hyperparameter_search(

    hp_space=hp_space,

    compute_objective=compute_objective,

    n_trials=N_OPTUNA_TRIALS,

    direction="maximize",

    backend="optuna",

    storage=optuna_storage,

    study_name=OPTUNA_STUDY_NAME,

    load_if_exists=True,

)

print("\nBest Optuna run:")
print(best_run)

print("\nBest hyperparameters:")
print(best_run.hyperparameters)

best_hyperparameters = best_run.hyperparameters

BEST_LEARNING_RATE = best_hyperparameters["learning_rate"]
BEST_WEIGHT_DECAY = best_hyperparameters["weight_decay"]
BEST_WARMUP_RATIO = best_hyperparameters["warmup_ratio"]

# ============================================================
# FINAL MODEL TRAINING
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL TRAINING")
print("=" * 100)

print(
    "Using best Optuna hyperparameters:"
)

print(
    "Learning rate:",
    BEST_LEARNING_RATE
)

print(
    "Weight decay:",
    BEST_WEIGHT_DECAY
)

print(
    "Warmup ratio:",
    BEST_WARMUP_RATIO
)

print(
    "Epochs:",
    FINAL_NUM_TRAIN_EPOCHS
)

print("=" * 100)


# ------------------------------------------------------------
# Create a completely new pretrained model
# ------------------------------------------------------------

final_model = model_init()


# ------------------------------------------------------------
# Final training arguments
# ------------------------------------------------------------

final_training_args = TrainingArguments(

    output_dir=str(
        OUTPUT_DIR / "final_training"
    ),

    overwrite_output_dir=True,

    num_train_epochs=FINAL_NUM_TRAIN_EPOCHS,

    per_device_train_batch_size=(
        PER_DEVICE_TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        PER_DEVICE_EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=BEST_LEARNING_RATE,

    weight_decay=BEST_WEIGHT_DECAY,

    warmup_ratio=BEST_WARMUP_RATIO,

    lr_scheduler_type="linear",

    eval_strategy="epoch",

    logging_strategy="epoch",

    save_strategy="epoch",

    save_total_limit=1,

    load_best_model_at_end=True,

    metric_for_best_model="eval_micro_f1",

    greater_is_better=True,

    bf16=True,

    seed=SEED,

    data_seed=SEED,

    report_to="none",

    dataloader_num_workers=0,

    remove_unused_columns=True,
)


# ------------------------------------------------------------
# Final trainer
# ------------------------------------------------------------

final_trainer = Trainer(

    model=final_model,

    args=final_training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\nStarting final training...")

final_train_result = final_trainer.train()


# ------------------------------------------------------------
# Save best final model
# ------------------------------------------------------------

FINAL_MODEL_DIR = (
    OUTPUT_DIR / "best_model"
)

print(
    "\nSaving final best model to:"
)

print(FINAL_MODEL_DIR)


final_trainer.save_model(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

print(
    "\nEvaluating final best model "
    "on validation set..."
)

final_validation_metrics = (
    final_trainer.evaluate(
        eval_dataset=validation_dataset
    )
)

print("\nFinal validation metrics:")

for key, value in final_validation_metrics.items():

    print(
        f"{key}: {value}"
    )


# ------------------------------------------------------------
# Final test evaluation
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("FINAL TEST EVALUATION")
print("=" * 100)


final_test_output = final_trainer.predict(
    test_dataset
)


final_test_logits = (
    final_test_output.predictions
)

final_test_labels = (
    final_test_output.label_ids
)


final_test_probabilities = (
    1 / (
        1 + np.exp(
            -final_test_logits
        )
    )
)


final_test_predictions = (
    final_test_probabilities >= THRESHOLD
).astype(np.int32)


# ------------------------------------------------------------
# Final test metrics
# ------------------------------------------------------------

final_test_metrics = {

    "dataset": dataset_name,

    "threshold": THRESHOLD,

    "num_test_examples":
        len(test_dataset),

    "micro_f1_all":
        f1_score(
            final_test_labels,
            final_test_predictions,
            average="micro",
            zero_division=0,
        ),

    "macro_f1_all":
        f1_score(
            final_test_labels,
            final_test_predictions,
            average="macro",
            zero_division=0,
        ),

    "weighted_f1_all":
        f1_score(
            final_test_labels,
            final_test_predictions,
            average="weighted",
            zero_division=0,
        ),

    "micro_precision_all":
        precision_score(
            final_test_labels,
            final_test_predictions,
            average="micro",
            zero_division=0,
        ),

    "micro_recall_all":
        recall_score(
            final_test_labels,
            final_test_predictions,
            average="micro",
            zero_division=0,
        ),
}


print("\nFinal test metrics:")

for key, value in final_test_metrics.items():

    print(
        f"{key}: {value}"
    )


# ------------------------------------------------------------
# Save final test metrics
# ------------------------------------------------------------

final_test_metrics_file = (
    OUTPUT_DIR
    / "final_test_metrics.json"
)

with open(
    final_test_metrics_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_test_metrics,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# Save Optuna results
# ------------------------------------------------------------

optuna_results = {

    "dataset": dataset_name,

    "n_trials":
        N_OPTUNA_TRIALS,

    "search_epochs":
        OPTUNA_NUM_TRAIN_EPOCHS,

    "best_objective":
        best_run.objective,

    "best_hyperparameters":
        best_run.hyperparameters,
}


optuna_results_file = (
    OUTPUT_DIR
    / "optuna_results.json"
)

with open(
    optuna_results_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        optuna_results,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# Save final configuration
# ------------------------------------------------------------

configuration = {

    "model_id": MODEL_ID,

    "dataset": dataset_name,

    "task_id": task_id,

    "max_length": MAX_LENGTH,

    "seed": SEED,

    "optuna_num_trials":
        N_OPTUNA_TRIALS,

    "optuna_search_epochs":
        OPTUNA_NUM_TRAIN_EPOCHS,

    "final_num_train_epochs":
        FINAL_NUM_TRAIN_EPOCHS,

    "optuna_best_objective":
        best_run.objective,

    "optuna_best_hyperparameters":
        best_run.hyperparameters,

    "learning_rate":
        BEST_LEARNING_RATE,

    "weight_decay":
        BEST_WEIGHT_DECAY,

    "warmup_ratio":
        BEST_WARMUP_RATIO,

    "per_device_train_batch_size":
        PER_DEVICE_TRAIN_BATCH_SIZE,

    "per_device_eval_batch_size":
        PER_DEVICE_EVAL_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "effective_batch_size":
        (
            PER_DEVICE_TRAIN_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        ),

    "threshold":
        THRESHOLD,

    "problem_type":
        "multi_label_classification",

    "model_labels":
        model_labels,

    "dataset_labels":
        dataset_labels,
}


configuration_file = (
    OUTPUT_DIR
    / "configuration.json"
)

with open(
    configuration_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        configuration,
        f,
        indent=2,
    )


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 100)
print("FINE-TUNING COMPLETE")
print("=" * 100)

print(
    f"\nResults saved to:\n"
    f"  {OUTPUT_DIR}"
)

print("\nBest final model:")
print(
    f"  {FINAL_MODEL_DIR}"
)

print("\nOptuna results:")
print(
    f"  {optuna_results_file}"
)

print("\nFinal test metrics:")
print(
    f"  {final_test_metrics_file}"
)

print("\n" + "=" * 100)




print("\n" + "=" * 100)
print("BEST OPTUNA HYPERPARAMETERS")
print("=" * 100)

print(
    "Learning rate:",
    BEST_LEARNING_RATE
)

print(
    "Weight decay:",
    BEST_WEIGHT_DECAY
)

print(
    "Warmup ratio:",
    BEST_WARMUP_RATIO
)

print(
    "Optuna objective:",
    best_run.objective
)

print("=" * 100)


# ============================================================
# Print training configuration
# ============================================================

print("\n" + "=" * 100)
print("TRAINING CONFIGURATION")
print("=" * 100)

print(
    "Dataset:",
    dataset_name
)

print(
    "Train examples:",
    len(train_dataset)
)

print(
    "Validation examples:",
    len(validation_dataset)
)

print(
    "Test examples:",
    len(test_dataset)
)

print(
    "Train batch size:",
    PER_DEVICE_TRAIN_BATCH_SIZE
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Effective batch size:",
    PER_DEVICE_TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)


print(
    "Threshold:",
    THRESHOLD
)

print(
    "Learning rate:",
    BEST_LEARNING_RATE
)

print(
    "Weight decay:",
    BEST_WEIGHT_DECAY
)

print(
    "Warmup ratio:",
    BEST_WARMUP_RATIO
)

print(
    "Epochs:",
    FINAL_NUM_TRAIN_EPOCHS
)

print("=" * 100)





# ============================================================
# Test metrics
# ============================================================

test_metrics = {

    "dataset": dataset_name,

    "threshold": THRESHOLD,

    "num_test_examples": len(
        test_dataset
    ),

    "micro_f1_all":
        f1_score(
            test_labels,
            test_predictions,
            average="micro",
            zero_division=0,
        ),

    "macro_f1_all":
        f1_score(
            test_labels,
            test_predictions,
            average="macro",
            zero_division=0,
        ),

    "weighted_f1_all":
        f1_score(
            test_labels,
            test_predictions,
            average="weighted",
            zero_division=0,
        ),

    "micro_precision_all":
        precision_score(
            test_labels,
            test_predictions,
            average="micro",
            zero_division=0,
        ),

    "micro_recall_all":
        recall_score(
            test_labels,
            test_predictions,
            average="micro",
            zero_division=0,
        ),
}


# ============================================================
# Main labels
# ============================================================

main_labels = [
    label
    for label in model_labels
    if label.isupper()
]


main_indices = [
    i
    for i, label in enumerate(model_labels)
    if label in main_labels
]


test_labels_main = test_labels[
    :,
    main_indices
]

test_predictions_main = (
    test_predictions[
        :,
        main_indices
    ]
)


test_metrics[
    "micro_f1_main"
] = f1_score(
    test_labels_main,
    test_predictions_main,
    average="micro",
    zero_division=0,
)


test_metrics[
    "macro_f1_main"
] = f1_score(
    test_labels_main,
    test_predictions_main,
    average="macro",
    zero_division=0,
)


test_metrics[
    "weighted_f1_main"
] = f1_score(
    test_labels_main,
    test_predictions_main,
    average="weighted",
    zero_division=0,
)


# ============================================================
# Print test metrics
# ============================================================

print("\nTest metrics:")

for key, value in test_metrics.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# Save test metrics
# ============================================================

test_metrics_file = (
    OUTPUT_DIR
    / "test_metrics.json"
)

with open(
    test_metrics_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        test_metrics,
        f,
        indent=2,
    )


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=model_labels,
    zero_division=0,
    output_dict=True,
)


report_file = (
    OUTPUT_DIR
    / "test_classification_report.json"
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


report_df = pd.DataFrame(
    report
).transpose()


report_csv = (
    OUTPUT_DIR
    / "test_classification_report.csv"
)

report_df.to_csv(
    report_csv,
    encoding="utf-8-sig",
)


# ============================================================
# Save test predictions
# ============================================================

original_test_dataset = dataset["test"]

texts = original_test_dataset["text"]

original_labels = np.asarray(
    original_test_dataset["labels"],
    dtype=np.int32,
)


prediction_rows = []


for i, text in enumerate(texts):

    row = {
        "index": i,
        "text": text,
    }

    # --------------------------------------------------------
    # Probabilities and predictions
    # --------------------------------------------------------

    for j, label in enumerate(
        model_labels
    ):

        row[
            f"prob_{label}"
        ] = float(
            test_probabilities[i, j]
        )

        row[
            f"pred_{label}"
        ] = int(
            test_predictions[i, j]
        )

    # --------------------------------------------------------
    # Gold labels
    #
    # Original dataset ordering
    # --------------------------------------------------------

    gold_indices = np.where(
        original_labels[i] == 1
    )[0]

    row["gold_labels"] = " ".join(
        dataset_labels[j]
        for j in gold_indices
    )

    # --------------------------------------------------------
    # Predicted labels
    #
    # Model ordering
    # --------------------------------------------------------

    predicted_indices = np.where(
        test_predictions[i] == 1
    )[0]

    row["predicted_labels"] = " ".join(
        model_labels[j]
        for j in predicted_indices
    )

    prediction_rows.append(row)


predictions_df = pd.DataFrame(
    prediction_rows
)


prediction_file = (
    OUTPUT_DIR
    / "test_predictions.csv"
)


predictions_df.to_csv(
    prediction_file,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Save all configuration
# ============================================================

configuration = {

    "model_id": MODEL_ID,

    "dataset": dataset_name,

    "task_id": task_id,

    "max_length": MAX_LENGTH,

    "seed": SEED,

    "learning_rate": BEST_LEARNING_RATE,

    "weight_decay": BEST_WEIGHT_DECAY,

    "warmup_ratio": BEST_WARMUP_RATIO,

    "num_train_epochs":
        FINAL_NUM_TRAIN_EPOCHS,

    "optuna_num_trials":
        N_OPTUNA_TRIALS,

    "optuna_search_epochs":
        OPTUNA_NUM_TRAIN_EPOCHS,

    "optuna_best_objective":
        best_run.objective,

    "optuna_best_hyperparameters":
        best_run.hyperparameters,

    "per_device_train_batch_size":
        PER_DEVICE_TRAIN_BATCH_SIZE,

    "per_device_eval_batch_size":
        PER_DEVICE_EVAL_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "effective_batch_size":
        PER_DEVICE_TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS,

    "threshold": THRESHOLD,

    "problem_type":
        "multi_label_classification",

    "model_labels":
        model_labels,

    "dataset_labels":
        dataset_labels,
}


configuration_file = (
    OUTPUT_DIR
    / "configuration.json"
)


with open(
    configuration_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        configuration,
        f,
        indent=2,
    )




#Final model

final_model = model_init()
final_training_args = TrainingArguments(

    output_dir=str(
        OUTPUT_DIR / "final_training"
    ),

    overwrite_output_dir=True,

    num_train_epochs=FINAL_NUM_TRAIN_EPOCHS,

    per_device_train_batch_size=(
        PER_DEVICE_TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        PER_DEVICE_EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=BEST_LEARNING_RATE,

    weight_decay=BEST_WEIGHT_DECAY,

    warmup_ratio=BEST_WARMUP_RATIO,

    lr_scheduler_type="linear",

    eval_strategy="epoch",

    logging_strategy="epoch",

    save_strategy="no",

    save_total_limit=1,

    load_best_model_at_end=False,

    metric_for_best_model="eval_micro_f1",

    greater_is_better=True,

    bf16=True,

    seed=SEED,

    data_seed=SEED,

    report_to="none",

    dataloader_num_workers=0,

    remove_unused_columns=True,
)

final_trainer = Trainer(

    model=final_model,

    args=final_training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


train_result = final_trainer.train()

validation_metrics = final_trainer.evaluate(
    eval_dataset=validation_dataset
)

test_output = final_trainer.predict(
    test_dataset
)

final_trainer.save_model(
    str(OUTPUT_DIR / "best_model")
)

# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 100)
print("FINE-TUNING COMPLETE")
print("=" * 100)

print(
    f"\nResults saved to:\n"
    f"  {OUTPUT_DIR}"
)

print("\nBest model:")
print(
    f"  {OUTPUT_DIR / 'best_model'}"
)

print("\nTest metrics:")
print(
    f"  {test_metrics_file}"
)

print("\nTest predictions:")
print(
    f"  {prediction_file}"
)

print("\n" + "=" * 100)