import json
import gzip
from pathlib import Path
import os
import argparse

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

from transformers import TrainerCallback

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

    return {"f1_micro": best_f1,
    "best_threshold": best_threshold,
    }

class ClassificationReportCallback(TrainerCallback):
    def __init__(self, trainer, label_names, threshold=0.5):
        self.trainer = trainer
        self.label_names = label_names
        self.threshold = threshold

    def on_evaluate(self, args, state, control, **kwargs):
        # Get predictions on eval dataset
        preds = self.trainer.predict(self.trainer.eval_dataset)

        y_true = preds.label_ids
        y_prob = sigmoid(preds.predictions)

        # Optional: re-tune threshold per epoch
        best_f1 = 0.0
        best_threshold = self.threshold
        for t in np.arange(0.3, 0.7, 0.05):
            f1 = f1_score(y_true, y_prob > t, average="micro")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        y_pred = y_prob > best_threshold

        print("\n" + "=" * 80)
        print(f"Classification report — epoch {int(state.epoch)}")
        print(f"Best threshold: {best_threshold:.2f}")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=self.label_names,
                zero_division=0,
            )
        )
        print("=" * 80 + "\n")

# ------------------------------------------------
# Load data
# ------------------------------------------------

X_jsonl, y_jsonl = load_jsonl_data("./data/persian_consolidated.jsonl")

# skmultilearn requires 2D X
X_jsonl_2d = X_jsonl.reshape(-1, 1)

# 1) Train vs temp (dev+test)
X_train_jsonl, y_train_jsonl, X_temp, y_temp = iterative_train_test_split(
    X_jsonl_2d,
    y_jsonl,
    test_size=0.3
)

# 2) Dev vs test
X_dev, y_dev, X_test, y_test = iterative_train_test_split(
    X_temp,
    y_temp,
    test_size=0.5
)

# Flatten
X_train_jsonl = X_train_jsonl.flatten()
X_dev = X_dev.flatten()
X_test = X_test.flatten()

#Rename
X_train = X_train_jsonl
y_train = y_train_jsonl

train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev, y_dev)
test_dataset = create_dataset(X_test, y_test)

shuffled_train = train_dataset.shuffle(seed=42)
shuffled_dev = dev_dataset.shuffle(seed=42)
shuffled_test = test_dataset.shuffle(seed=42)

print(
    f"Split: {len(shuffled_train)} train, {len(shuffled_dev)} dev, {len(shuffled_test)} test"
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

shuffled_train = shuffled_train.map(tokenize, batched=True)
shuffled_dev = shuffled_dev.map(tokenize, batched=True)
shuffled_test = shuffled_test.map(tokenize, batched=True)

for ds in (shuffled_train, shuffled_dev, shuffled_test):
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# Get the Job ID from environment variables, default to 'debug' if running locally
job_id = os.getenv("SLURM_JOB_ID", "debug")
output_path = f"./results_{job_id}"

training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,

    learning_rate=1e-5,
    lr_scheduler_type="cosine", #linear #"constant"
    weight_decay=0.01,
    warmup_steps=100,

    # Disable gradient clipping
    max_grad_norm=1.0,

    eval_strategy="epoch",
    #eval_steps=1000,
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
    args=training_args,
    train_dataset=shuffled_train,
    eval_dataset=shuffled_dev,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
# Attach report callback
trainer.add_callback(
    ClassificationReportCallback(
        trainer=trainer,
        label_names=all_valid_labels,
        threshold=0.5, # starting threshold
    )
)

# ------------------------------------------------
# Train & Evaluate
# ------------------------------------------------

trainer.train()

test_preds = trainer.predict(shuffled_test)

y_true = test_preds.label_ids
y_prob = sigmoid(test_preds.predictions)
y_pred = y_prob > 0.5

print("\n===== TEST SET CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=all_valid_labels,
        zero_division=0,
    )
)


print("Done!")