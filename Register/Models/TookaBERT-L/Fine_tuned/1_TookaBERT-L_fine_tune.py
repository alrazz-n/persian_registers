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
)

from datasets import load_from_disk
import shutil

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

#Configs/paths
DATASET_ROOT = Path(
    r"/scratch/project_462001491/nima/Hybrid_SP_ID_effect/Dataset_Hugging_face/without_NA"
)

RESULTS_ROOT = DATASET_ROOT / "_TookaBERT-L_finetuned_evaluation"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "PartAI/TookaBERT-Large"

MAX_LENGTH = 512 #based on checking

THRESHOLD = 0.5 #0.35

HF_CACHE = Path(
    f"/scratch/project_462001491/nima/hf_test_{os.environ['SLURM_JOB_ID']}"
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
    MODEL_ID,
    cache_dir=str(HF_CACHE),
)


print("Tokenizer loaded.")

#Labels
model_labels = dataset_labels

label2id = {
    label: i
    for i, label in enumerate(model_labels)
}

id2label = {
    i: label
    for i, label in enumerate(model_labels)
}

# Load model

print("\nLoading pretrained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(model_labels),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
    cache_dir=str(HF_CACHE),
)

print(
    "INITIAL MODEL:",
    model.get_input_embeddings().weight.shape
)
print("Model loaded.")


print("Tokenizer vocab size:", len(tokenizer))
print("Tokenizer model max length:", tokenizer.model_max_length)

print("Model vocab size:", model.config.vocab_size)
print("Embedding shape:", model.get_input_embeddings().weight.shape)




# Prepare dataset

def prepare_dataset(ds, split_name):

    print(
        f"\nPreparing {split_name}: "
        f"{len(ds)} examples"
    )

    def tokenize(example):

        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
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


# Data collator padding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)

# Metrics

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

    macro_f1 = f1_score( #this is the most important for me
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
        id2label=id2label,
        label2id=label2id,
        cache_dir=str(HF_CACHE),
    )


    # CHECK MODEL
    print(
        f"Trial {trial.number}: "
        f"embedding = "
        f"{model.get_input_embeddings().weight.shape}"
    )

    print(
        f"Trial {trial.number}: "
        f"config vocab_size = "
        f"{model.config.vocab_size}"
    )

    print(
        f"Trial {trial.number}: "
        f"tokenizer vocab_size = "
        f"{len(tokenizer)}"
    )

    args = TrainingArguments(
        output_dir=str(
            HF_CACHE
            / "optuna"
            / dataset_name
            / f"trial_{trial.number}"
            ),
        overwrite_output_dir=True,

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
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3
            ),
            OptunaPruningCallback(trial),
        ]
    )

    try:
        trainer.train()

        best_metric = trainer.state.best_metric

        return best_metric

    finally:
        shutil.rmtree(
            args.output_dir,
            ignore_errors=True,
        )





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
    id2label=id2label,
    label2id=label2id,
    cache_dir=str(HF_CACHE),
)

final_args = TrainingArguments(
    output_dir=str(
        HF_CACHE
        / "final_model"
        / dataset_name
    ),


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

final_trainer.train()

pred = final_trainer.predict(validation_dataset) #save Dev predictions for Threshhold tuning afterwards

#Do not need to load the model this way for threshold tuning
print("\nSaving logits and npy...")

np.save(
    OUTPUT_DIR / "dev_logits.npy",
    pred.predictions,
)

np.save(
    OUTPUT_DIR / "dev_labels.npy",
    pred.label_ids,
)


print("\nSaving best model...")
final_trainer.save_model(OUTPUT_DIR / "final_model")
tokenizer.save_pretrained(OUTPUT_DIR / "final_model")



print("\n===== BEST TRIAL =====")
print(study.best_trial.params)
print("Best F1_macro:", study.best_value)
final_metrics = final_trainer.evaluate(
    eval_dataset=validation_dataset
)
print(final_metrics)


print(
    f"\nModel saved to: "
    f"{OUTPUT_DIR / 'final_model'}"
)
#shutil.rmtree(HF_CACHE, ignore_errors=True) #Do not as it is a prallel job

# Classification report
print("\nClassification report for a Threshold of 0.5...")
test_output = final_trainer.predict(
    test_dataset
)


test_logits = test_output.predictions

test_labels = test_output.label_ids


test_probabilities = 1 / (
    1 + np.exp(-test_logits)
)


test_predictions = (
    test_probabilities >= THRESHOLD  
).astype(np.int32)


report = classification_report(
    test_labels,
    test_predictions,
    target_names=model_labels,
    zero_division=0,
    output_dict=True,
)

np.save(
    OUTPUT_DIR / "test_logits.npy",
    test_output.predictions,
)

np.save(
    OUTPUT_DIR / "test_labels.npy",
    test_output.label_ids,
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