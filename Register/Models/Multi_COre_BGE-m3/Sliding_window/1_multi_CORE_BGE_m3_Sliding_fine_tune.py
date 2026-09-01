#------------------------
#This script works perefectly
#--------------------------
#Imports
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
import random
from collections import Counter

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
    EarlyStoppingCallback,
    TrainerCallback,
    AutoConfig
)

from datasets import load_from_disk
import shutil
import gc

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

#Configs/paths
DATASET_ROOT = Path(
    r"/scratch/project_462001491/nima/Hybrid_SP_ID_effect/Dataset_Hugging_face/without_NA"
)

RESULTS_ROOT = DATASET_ROOT / "_Sliding_MultiCore_BGE_m3_finetuned_evaluation"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "TurkuNLP/web-register-classification-multilingual-bge"

MAX_LENGTH = 1024
STRIDE = 768 #25_% overlap

THRESHOLD = 0.5 #0.35

HF_CACHE = Path(
    "/scratch/project_462001491/nima/huggingface"
)
HF_CACHE.mkdir(parents=True, exist_ok=True)


# Select dataset from SLURM array

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

# Load data

print(
    f"\nLoading dataset: {dataset_name}"
)

dataset = load_from_disk(
    str(dataset_path)
)

print(dataset)

# Load metadata

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


#tokenizer loading

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-m3-retromae",
    cache_dir=str(HF_CACHE),
)

print("Tokenizer loaded.")

# Load model

print("\nLoading pretrained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    cache_dir=str(HF_CACHE),
)

print("Model loaded.")


# Model labels

config = AutoConfig.from_pretrained(
    MODEL_ID,
    cache_dir=str(HF_CACHE),
)
id2label = model.config.id2label

model_labels = [
    id2label[i]
    for i in range(
        len(id2label)
    )
]


print("\nModel labels:")
print(model_labels)

print(
    "Number of model labels:",
    len(model_labels)
)

# Verify label space

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

# Dataset label -> model label mapping

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


#Model Configuration

model.config.problem_type = (
    "multi_label_classification"
)

# Prepare dataset

def prepare_dataset(ds, split_name):

    print(
        f"\nPreparing {split_name}: "
        f"{len(ds)} original documents"
    )

    # --------------------------------------------------
    # Create document IDs
    # --------------------------------------------------
    #
    # Each row in the original dataset is one document.
    # These IDs allow us to reconstruct documents after
    # sliding-window tokenization creates multiple chunks.
    #

    ds = ds.add_column(
        "document_id",
        list(range(len(ds)))
    )

    print(
        f"Created {len(ds)} document IDs."
    )

    print(
        "First document IDs:",
        ds["document_id"][:10]
    )

    # --------------------------------------------------
    # Align labels to model label order
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Sliding-window tokenization
    # --------------------------------------------------

    def tokenize_with_overflow(batch):

        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
        )

        # Maps every generated chunk to its
        # original document within this batch.
        sample_mapping = tokenized.pop(
            "overflow_to_sample_mapping"
        )

        # --------------------------------------------------
        # Copy document-level labels to every chunk
        # --------------------------------------------------

        tokenized["labels"] = [
            batch["labels"][sample_idx]
            for sample_idx in sample_mapping
        ]

        # --------------------------------------------------
        # Copy original document ID to every chunk
        # --------------------------------------------------

        tokenized["document_id"] = [
            batch["document_id"][sample_idx]
            for sample_idx in sample_mapping
        ]

        return tokenized

    ds = ds.map(
        tokenize_with_overflow,
        batched=True,
        remove_columns=ds.column_names,
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    print(
        f"{split_name}: "
        f"{len(ds)} total chunks"
    )

    print(
        f"{split_name}: "
        f"{len(set(ds['document_id']))} unique documents"
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

validation_document_ids = np.asarray(
    validation_dataset["document_id"]
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


print(
    "\nExample document IDs:"
)

print(
    train_dataset["document_id"][:30]
)

print(
    "\nNumber of train chunks:",
    len(train_dataset)
)

print(
    "Number of unique train documents:",
    len(
        set(
            train_dataset["document_id"]
        )
    )
)



chunk_counts = Counter(
    train_dataset["document_id"]
)

print(
    "\nChunk distribution:"
)

print(
    "Average:",
    np.mean(
        list(chunk_counts.values())
    )
)

print(
    "Maximum:",
    max(
        chunk_counts.values()
    )
)

# Data collator padding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)

#Doc aggregation

def aggregate_chunk_predictions(
    logits,
    labels,
    document_ids,
    aggregation="max",
):
    """
    Convert chunk-level predictions into
    document-level predictions.

    Each document may have multiple chunks.
    """

    probabilities = 1 / (
        1 + np.exp(-np.clip(logits, -50, 50))
    )

    document_ids = np.asarray(
        document_ids
    )

    labels = np.asarray(
        labels
    )

    unique_document_ids = []
    document_probabilities = []
    document_labels = []

    for document_id in pd.unique(document_ids):

        mask = (
            document_ids == document_id
        )

        chunk_probs = probabilities[mask]

        chunk_labels = labels[mask]

        print(
            "Documents:",
            len(unique_document_ids)
        )
        print(
            "Chunks:",
            len(document_ids)
        )


        # --------------------------------------------------
        # Aggregate chunk probabilities
        # --------------------------------------------------

        if aggregation == "max":

            document_prob = np.max(
                chunk_probs,
                axis=0,
            )

        elif aggregation == "mean":

            document_prob = np.mean(
                chunk_probs,
                axis=0,
            )

        elif aggregation == "top2_mean":

            sorted_probs = np.sort(
                chunk_probs,
                axis=0,
            )

            top_k = min(
                2,
                sorted_probs.shape[0],
            )

            document_prob = np.mean(
                sorted_probs[-top_k:],
                axis=0,
            )

        else:

            raise ValueError(
                f"Unknown aggregation method: "
                f"{aggregation}"
            )

        # All chunks from the same document
        # should have the same labels.
        document_label = chunk_labels[0]

        if not np.all(
            chunk_labels == document_label
        ):
            raise ValueError(
                f"Labels differ between chunks "
                f"for document {document_id}"
            )

        unique_document_ids.append(
            document_id
        )

        document_probabilities.append(
            document_prob
        )

        document_labels.append(
            document_label
        )

    return (
        np.asarray(unique_document_ids),
        np.asarray(document_probabilities),
        np.asarray(document_labels),
    )



# Metrics

#def compute_metrics(eval_prediction):

def make_compute_metrics(eval_document_ids, aggregation="max"):

    def compute_metrics(eval_prediction):

        (
            _,
            document_probabilities,
            document_labels,
        ) = aggregate_chunk_predictions(
            logits=eval_prediction.predictions,
            labels=eval_prediction.label_ids,
            document_ids=eval_document_ids,
            aggregation=aggregation,
        )

        document_predictions = (
            document_probabilities >= THRESHOLD
        ).astype(np.int32)

        return {
            "micro_f1": f1_score(
                document_labels,
                document_predictions,
                average="micro",
                zero_division=0,
            ),
            "macro_f1": f1_score(
                document_labels,
                document_predictions,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                document_labels,
                document_predictions,
                average="weighted",
                zero_division=0,
            ),
            "micro_precision": precision_score(
                document_labels,
                document_predictions,
                average="micro",
                zero_division=0,
            ),
            "micro_recall": recall_score(
                document_labels,
                document_predictions,
                average="micro",
                zero_division=0,
            ),
        }

    return compute_metrics


compute_metrics=make_compute_metrics(
    validation_document_ids,
    aggregation="max",
)

def calculate_document_metrics(
    prediction_output,
    document_ids,
    aggregation="max",
):

    (
        unique_document_ids,
        document_probabilities,
        document_labels,
    ) = aggregate_chunk_predictions(
        logits=prediction_output.predictions,
        labels=prediction_output.label_ids,
        document_ids=document_ids,
        aggregation=aggregation,
    )

    document_predictions = (
        document_probabilities >= THRESHOLD
    ).astype(np.int32)

    macro_f1 = f1_score(
        document_labels,
        document_predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        document_labels,
        document_predictions,
        average="micro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        document_labels,
        document_predictions,
        average="weighted",
        zero_division=0,
    )

    micro_precision = precision_score(
        document_labels,
        document_predictions,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        document_labels,
        document_predictions,
        average="micro",
        zero_division=0,
    )

    return {
        "document_ids": unique_document_ids,
        "probabilities": document_probabilities,
        "labels": document_labels,
        "predictions": document_predictions,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
    }

# Output directories

OUTPUT_DIR = (
    RESULTS_ROOT
    / dataset_name
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

#OptunaPruningCallback
class OptunaPruningCallback(TrainerCallback):

    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs,
    ):
        if metrics is None:
            return control

        metric = metrics.get("eval_macro_f1")

        if metric is None:
            return control

        self.trial.report(
            metric,
            step=state.epoch,
        )

        if self.trial.should_prune():
            raise optuna.TrialPruned()

        return control


# Optuna objective
#Bayesian optimization (sample hyperparameters intelligently across trials)
def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 5e-6, 3e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)

    grad_accum = trial.suggest_categorical("grad_accum", [4, 8])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(model_labels),
        problem_type="multi_label_classification",
        cache_dir=str(HF_CACHE),
    )

    args = TrainingArguments(
        output_dir=str(
            HF_CACHE
            / "optuna"
            / dataset_name
            / f"trial_{trial.number}"
            ),
        overwrite_output_dir=True,
        #remove_unused_columns=False,

        num_train_epochs=10,
        per_device_eval_batch_size=16,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps=grad_accum,

        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,

        eval_strategy="epoch",
        logging_strategy="epoch",

        save_strategy="epoch",
        save_total_limit=1,   # keep only one checkpoint for better
        load_best_model_at_end=True,
        #save_strategy="best",
        metric_for_best_model="eval_macro_f1", #as we have imballence data
        report_to="none",

        seed=42,
        bf16=True,
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        #remove_unused_columns=False,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[
            OptunaPruningCallback(trial),
        ]
    )

    try:
        trainer.train()

        validation_output = trainer.predict(
            validation_dataset
        )

        (
            dev_document_ids,
            dev_document_probabilities,
            dev_document_labels,
        ) = aggregate_chunk_predictions(
            logits=validation_output.predictions,
            labels=validation_output.label_ids,
            document_ids=validation_document_ids,
            aggregation="max",
        )

        dev_predictions = (
            dev_document_probabilities >= THRESHOLD
        ).astype(np.int32)

        document_macro_f1 = f1_score(
            dev_document_labels,
            dev_predictions,
            average="macro",
            zero_division=0,
        )

        return document_macro_f1

    finally:
        del trainer
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(
            args.output_dir,
            ignore_errors=True,
        )
        gc.collect()
        torch.cuda.empty_cache()





# Get the best checkpoint from the Optuna trial
# run optuna
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=2,
        n_warmup_steps=2,
    ),
)
study.optimize(objective, n_trials=20)  # set n_trials

best_params = study.best_trial.params

print(best_params)


with open(
    OUTPUT_DIR / "best_params.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        best_params,
        f,
        indent=2,
    )

# Final best Trainer

print("\nFinal/best training...")

best_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(model_labels),
    problem_type="multi_label_classification",
    cache_dir=str(HF_CACHE),
)

final_args = TrainingArguments(
    output_dir=str(
        HF_CACHE
        / "final_model"
        / dataset_name
    ),

    #remove_unused_columns=False,
    num_train_epochs=10,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=best_params["grad_accum"],

    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    warmup_ratio=best_params["warmup_ratio"],

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,

    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,

    report_to="none",
    bf16=True,
    seed=42,
)

final_trainer = Trainer(
    model=best_model,
    args=final_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\nStarting final training...")
final_trainer.train()

pred = final_trainer.predict(
    validation_dataset
)

dev_results = calculate_document_metrics(
    prediction_output=pred,
    document_ids=validation_document_ids,
    aggregation="max",
)

dev_document_ids = dev_results["document_ids"]
dev_document_probabilities = dev_results["probabilities"]
dev_document_labels = dev_results["labels"]

print("\nSaving document-level Dev predictions...")

np.save( #To aevaluate all aggregation methods
    OUTPUT_DIR / "dev_chunk_logits.npy",
    pred.predictions,
)


np.save(
    OUTPUT_DIR / "dev_document_ids.npy",
    dev_document_ids,
)

#Avoid because it is only based on max_agg and I can do the same with other saved .npy
#np.save(
#    OUTPUT_DIR / "dev_document_probabilities.npy",
#    dev_document_probabilities,
#)

np.save(
    OUTPUT_DIR / "dev_document_labels.npy",
    dev_document_labels,
)


print("\nSaving best model...")
final_trainer.save_model(OUTPUT_DIR / "final_model")
tokenizer.save_pretrained(OUTPUT_DIR / "final_model")



print("\n===== BEST TRIAL =====")
print(study.best_trial.params)
print("Best F1_macro (Optuna):", study.best_value)
#final_metrics = final_trainer.evaluate(
#    eval_dataset=validation_dataset
#)
print("\n===== FINAL RETRAINING — DEV RESULTS =====")

print(
    f"Macro F1:      {dev_results['macro_f1']:.4f}"
)
print(
    f"Micro F1:      {dev_results['micro_f1']:.4f}"
)
print(
    f"Weighted F1:   {dev_results['weighted_f1']:.4f}"
)
print(
    f"Micro Precision: {dev_results['micro_precision']:.4f}"
)
print(
    f"Micro Recall:    {dev_results['micro_recall']:.4f}"
)


print(
    f"\nModel saved to: "
    f"{OUTPUT_DIR / 'final_model'}"
)
#shutil.rmtree(HF_CACHE, ignore_errors=True) #Do not as it is a prallel job

# Classification report
print("\nClassification report for a Threshold of 0.5...")

final_trainer.compute_metrics = None ## Disable compute_metrics because test metrics are calculated manually at document level below.
test_output = final_trainer.predict(
    test_dataset
)

test_document_ids = np.asarray(
    test_dataset["document_id"]
)

(
    test_document_ids,
    test_document_probabilities,
    test_document_labels,
) = aggregate_chunk_predictions(
    logits=test_output.predictions,
    labels=test_output.label_ids,
    document_ids=test_document_ids,
    aggregation="max",
)

test_document_predictions = (
    test_document_probabilities >= THRESHOLD
).astype(np.int32)

report = classification_report(
    test_document_labels,
    test_document_predictions,
    target_names=model_labels,
    zero_division=0,
    output_dict=True,
)

test_macro_f1 = f1_score(
    test_document_labels,
    test_document_predictions,
    average="macro",
    zero_division=0,
)

test_micro_f1 = f1_score(
    test_document_labels,
    test_document_predictions,
    average="micro",
    zero_division=0,
)

test_weighted_f1 = f1_score(
    test_document_labels,
    test_document_predictions,
    average="weighted",
    zero_division=0,
)

print("\n===== TEST DOCUMENT-LEVEL METRICS =====")
print("Macro F1:", test_macro_f1)
print("Micro F1:", test_micro_f1)
print("Weighted F1:", test_weighted_f1)


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

with open(
    OUTPUT_DIR / "model_labels.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        model_labels,
        f,
        indent=2,
        ensure_ascii=False,
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