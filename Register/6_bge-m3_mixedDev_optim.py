# ============================================================
# bge-m3 — Optuna Hyperparameter Optimization
# ============================================================

import json
import gzip
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from scipy.special import expit as sigmoid
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
import optuna

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ------------------------------------------------
# Reproducibility
# ------------------------------------------------

torch.manual_seed(44)
np.random.seed(44)

# ------------------------------------------------
# Labels
# ------------------------------------------------

labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

all_valid_labels = sorted(
    list(labels_structure.keys())
    + [s for subs in labels_structure.values() for s in subs]
)

NUM_LABELS = len(all_valid_labels)

# ------------------------------------------------
# Data loading
# ------------------------------------------------

def load_multicore_tsv(root_dir):
    texts, labels = [], []
    root_dir = Path(root_dir)

    for lang_dir in root_dir.iterdir():
        if not lang_dir.is_dir() or lang_dir.name == "fa":
            continue

        files = list(lang_dir.glob("*.tsv.gz"))
        if not files:
            continue

        with gzip.open(files[0], "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    label_str, text = line.strip().split("\t", 1)
                except ValueError:
                    continue

                label_list = label_str.split()
                labels.append(
                    [1.0 if l in label_list else 0.0 for l in all_valid_labels]
                )
                texts.append(text)

    return np.array(texts), np.array(labels, dtype=np.float32)


def load_jsonl_data(filepath):
    texts, labels = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            label_list = r["label"].split()
            labels.append(
                [1.0 if l in label_list else 0.0 for l in all_valid_labels]
            )
            texts.append(r["text"])

    return np.array(texts), np.array(labels, dtype=np.float32)


def create_dataset(X, y):
    return Dataset.from_dict(
        {
            "text": X.tolist(),
            "labels": y.tolist(),
        }
    )

# ------------------------------------------------
# Metric
# ------------------------------------------------

def compute_metrics(p):
    y_true = p.label_ids
    y_prob = sigmoid(p.predictions)

    best_f1 = 0.0
    for t in np.arange(0.3, 0.7, 0.05):
        f1 = f1_score(y_true, y_prob > t, average="micro")
        best_f1 = max(best_f1, f1)

    return {"f1_micro": best_f1}

# ------------------------------------------------
# Load & split data (same as main training)
# ------------------------------------------------

X_jsonl, y_jsonl = load_jsonl_data("./data/persian_consolidated.jsonl")
X_tsv, y_tsv = load_multicore_tsv("data/multilingual-CORE")


# skmultilearn requires 2D X
X_jsonl_2d = X_jsonl.reshape(-1, 1)


# 1) Train vs temp (dev+test)
X_train_jsonl, y_train_jsonl, X_temp, y_temp = iterative_train_test_split(
    X_jsonl_2d,
    y_jsonl,
    test_size=0.5
)


# 2) Dev vs test
X_dev, y_dev, X_test, y_test = iterative_train_test_split(
    X_temp,
    y_temp,
    test_size=0.5
)


# Split TSV into train and a small dev portion
X_train_tsv, y_train_tsv, X_dev_tsv, y_dev_tsv = iterative_train_test_split(
    X_tsv.reshape(-1, 1), y_tsv, test_size=0.1 # Take 10% for dev
)


# Flatten
X_train_jsonl = X_train_jsonl.flatten()
X_dev = X_dev.flatten()
X_test = X_test.flatten()


# Combine JSONL training + CORE
X_train = np.concatenate([X_train_jsonl, X_train_tsv.flatten()]) #train on persian and core
y_train = np.concatenate([y_train_jsonl, y_train_tsv])
X_dev_final = np.concatenate([X_dev, X_dev_tsv.flatten()]) #dev on persian and core
y_dev_final = np.concatenate([y_dev, y_dev_tsv])


train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev_final, y_dev_final)
test_dataset = create_dataset(X_test, y_test)


print(
    f"Split: {len(train_dataset)} train, {len(dev_dataset)} dev, {len(test_dataset)} test"
) 

# ------------------------------------------------
# Tokenization
# ------------------------------------------------

MODEL_NAME = "BAAI/bge-m3-retromae"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

print("max_length = 1024") #keep track of what I am doing

train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)

for ds in (train_dataset, dev_dataset):
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ------------------------------------------------
# Optuna objective
# ------------------------------------------------

def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 5e-6, 3e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)

    per_device_batch = trial.suggest_categorical("batch_size", [4, 8])
    grad_accum = trial.suggest_categorical("grad_accum", [4, 8])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    args = TrainingArguments(
        output_dir=f"./optuna_xlmr/trial_{trial.number}",
        overwrite_output_dir=True,

        num_train_epochs=10,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=grad_accum,

        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,

        eval_strategy="epoch",
        logging_strategy="epoch",

        save_strategy="no",
        report_to="none",

        seed=42,
        bf16=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    metrics = trainer.evaluate()

    return metrics["eval_f1_micro"]

# ------------------------------------------------
# Run optimization
# ------------------------------------------------

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=8)

print("\n===== BEST TRIAL =====")
print(study.best_trial.params)
print("Best F1:", study.best_value)
