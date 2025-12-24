import json
import gzip
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, f1_score
from skmultilearn.model_selection import iterative_train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
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
    return Dataset.from_dict(
        {
            "text": X.tolist(),
            "labels": y.tolist(),
        }
    )

# ------------------------------------------------
# Metrics
# ------------------------------------------------

def compute_metrics(p, verbose=False):
    y_true = p.label_ids
    y_prob = sigmoid(p.predictions)

    best_f1 = 0.0
    best_threshold = 0.5

    for t in np.arange(0.3, 0.7, 0.05):
        f1 = f1_score(y_true, y_prob > t, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    if verbose:
        print(
            classification_report(
                y_true,
                y_prob > best_threshold,
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

# skmultilearn requires 2D X
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

# Combine JSONL training + CORE
X_train = np.concatenate([X_train_jsonl, X_tsv])
y_train = np.concatenate([y_train_jsonl, y_tsv])

train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev, y_dev)
test_dataset = create_dataset(X_test, y_test)

print(
    f"Split: {len(train_dataset)} train, {len(dev_dataset)} dev, {len(test_dataset)} test"
)
# ------------------------------------------------
# Tokenization
# ------------------------------------------------

model_name = "FacebookAI/xlm-roberta-large" #"BAAI/bge-m3-retromae"
print(
    f"Model name:{model_name}"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length= 512 #1024, #2048 #512 for XLMR
    )

train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)

for ds in (train_dataset, dev_dataset):
    ds.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

# ------------------------------------------------
# Model & Training
# ------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_valid_labels),
    problem_type="multi_label_classification",
)

training_args = TrainingArguments(
    output_dir="./single_run",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    #gradient_accumulation_steps=4,

    learning_rate=5e-5,
    lr_scheduler_type="linear", #"constant"
    weight_decay=0.0,
    warmup_ratio=0.0,

    # Disable gradient clipping
    max_grad_norm=0.0,

    eval_strategy="epoch",
    #eval_steps=1000,
    logging_strategy="epoch",
    save_strategy="no",
    report_to="none",
    seed=42,

    metric_for_best_model="f1_micro",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ------------------------------------------------
# Train & Evaluate
# ------------------------------------------------

trainer.train()
metrics = trainer.evaluate(test_dataset)

print("\n===== FINAL RESULTS =====")
print("Dev F1 (micro):", metrics["eval_f1_micro"])
print("Done.")
