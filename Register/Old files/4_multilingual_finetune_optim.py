import json
import gzip
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, f1_score
from skmultilearn.model_selection import iterative_train_test_split
import optuna
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

torch.manual_seed(44)
np.random.seed(44)

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
    + [sublabel for sublabels in labels_structure.values() for sublabel in sublabels]
)

# ------------------------------------------------
# Data loading
# ------------------------------------------------

def load_multicore_tsv(root_dir):
    texts, labels = [], []
    root_dir = Path(root_dir)

    for lang_dir in root_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        if lang_dir.name == "fa":
            continue

        tsv_gz_files = list(lang_dir.glob("*.tsv.gz"))
        if not tsv_gz_files:
            continue

        with gzip.open(tsv_gz_files[0], "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    label_str, text = line.split("\t", 1)
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
            record = json.loads(line)
            label_list = record["label"].split()

            labels.append(
                [1.0 if l in label_list else 0.0 for l in all_valid_labels]
            )
            texts.append(record["text"])

    return np.array(texts), np.array(labels, dtype=np.float32)


def create_dataset(X, y):
    return Dataset.from_dict({"text": X.tolist(), "labels": y.tolist()})


# ------------------------------------------------
# Metrics
# ------------------------------------------------

def compute_metrics(p, verbose=False):
    y_true = p.label_ids
    y_pred = sigmoid(p.predictions)

    best_f1 = 0.0
    best_threshold = 0.5

    for t in np.arange(0.3, 0.7, 0.05):
        f1 = f1_score(y_true, y_pred > t, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    if verbose:
        print(
            classification_report(
                y_true,
                y_pred > best_threshold,
                target_names=all_valid_labels,
                zero_division=0,
            )
        )

    return {"f1_micro": best_f1}


# ------------------------------------------------
# Load data
# ------------------------------------------------

X_jsonl, y_jsonl = load_jsonl_data("./data/persian_consolidated.jsonl")
X_tsv, y_tsv = load_multicore_tsv("data/multilingual-CORE")

# ------------------------------------------------
# FIX: skmultilearn needs 2D X
# ------------------------------------------------

X_jsonl_2d = X_jsonl.reshape(-1, 1)

X_dev, y_dev, X_test, y_test = iterative_train_test_split(
    X_jsonl_2d, y_jsonl, test_size=0.5
)

X_dev = X_dev.flatten()
X_test = X_test.flatten()

X_train_jsonl, y_train_jsonl, _, _ = iterative_train_test_split(
    X_jsonl_2d,
    y_jsonl,
    test_size=(len(X_dev) + len(X_test)) / len(X_jsonl),
)

X_train_jsonl = X_train_jsonl.flatten()

# Combine training JSONL + CORE
X_train = np.concatenate([X_train_jsonl, X_tsv])
y_train = np.concatenate([y_train_jsonl, y_tsv])

train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev, y_dev)

# ------------------------------------------------
# Tokenization
# ------------------------------------------------

model_name = "BAAI/bge-m3-retromae"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)

for ds in (train_dataset, dev_dataset):
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# ------------------------------------------------
# Optuna objective
# ------------------------------------------------

def objective(trial):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(all_valid_labels),
        problem_type="multi_label_classification",
    )

    args = TrainingArguments(
        output_dir=f"./optuna_runs/trial_{trial.number}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=trial.suggest_float("learning_rate", 3e-6, 3e-5, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.1),
        eval_strategy="steps",
        eval_steps=1000,
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=42,

        # EarlyStoppingCallback
        metric_for_best_model="f1_micro",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_f1_micro"]


# ------------------------------------------------
# Run Optuna
# ------------------------------------------------

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("\n===== OPTUNA RESULTS =====")
print("Best F1 (dev):", study.best_value)
print("Best hyperparameters:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")

print("\nDone.")
