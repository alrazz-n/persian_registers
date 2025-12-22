import json
import os
import gzip
from pathlib import Path

#os.environ["HF_HOME"] = "./hf_home"

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

def load_multicore_tsv(root_dir):
    texts = []
    labels = []

    root_dir = Path(root_dir)

    for lang_dir in root_dir.iterdir():
        if not lang_dir.is_dir():
            continue

        if lang_dir.name == "fa":
            continue  # 🚫 skip Persian

        tsv_gz_files = list(lang_dir.glob("*.tsv.gz"))
        if not tsv_gz_files:
            continue

        tsv_path = tsv_gz_files[0]

        print(f"Loading {tsv_path}")

        with gzip.open(tsv_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    label_str, text = line.split("\t", 1)
                except ValueError:
                    continue  # malformed line

                label_list = label_str.split()

                binary_vector = [
                    1.0 if label in label_list else 0.0
                    for label in all_valid_labels
                ]

                texts.append(text)
                labels.append(binary_vector)

    return np.array(texts), np.array(labels, dtype=np.float32)

def load_jsonl_data(filepath):
    texts, labels = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            text = record["text"]
            label_list = record["label"].split()

            binary_vector = [
                1.0 if label in label_list else 0.0
                for label in all_valid_labels
            ]

            texts.append(text)
            labels.append(binary_vector)

    return np.array(texts), np.array(labels, dtype=np.float32)



def create_dataset(X_split, y_split):
    return Dataset.from_dict(
        {
            "text": X_split.flatten().tolist(),
            "labels": y_split.tolist(),
        }
    )


def compute_metrics(p, verbose=False):
    true_labels = p.label_ids
    predictions = sigmoid(p.predictions)

    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.3, 0.7, 0.05):
        binary_predictions = predictions > threshold
        f1 = f1_score(true_labels, binary_predictions, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    binary_predictions = predictions > best_threshold

    if verbose:
        print(
            classification_report(
                true_labels, binary_predictions, target_names=all_valid_labels
            )
        )

    return {
        "f1_micro": best_f1,
    }


# Shuffle JSONL data
perm_jsonl = np.random.permutation(len(X_jsonl))
X_jsonl = X_jsonl[perm_jsonl]
y_jsonl = y_jsonl[perm_jsonl]

# Split JSONL into dev/test
X_dev, y_dev, X_test, y_test = iterative_train_test_split(
    X_jsonl, y_jsonl, test_size=0.5
)

# The remaining JSONL data for training
X_train_jsonl, y_train_jsonl, _, _ = iterative_train_test_split(
    X_jsonl, y_jsonl, test_size=(len(X_dev) + len(X_test)) / len(X_jsonl)
)

# Combine training JSONL with CORE data for the final training set
X_train = np.concatenate([X_train_jsonl, X_tsv], axis=0)
y_train = np.concatenate([y_train_jsonl, y_tsv], axis=0)

# Shuffle final training set
perm_train = np.random.permutation(len(X_train))
X_train = X_train[perm_train]
y_train = y_train[perm_train]

# Create datasets
train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev, y_dev)
test_dataset = create_dataset(X_test, y_test)

print(
    f"Split: {len(train_dataset)} train, {len(dev_dataset)} dev, {len(test_dataset)} test"
)


# Load model
model_name = "BAAI/bge-m3-retromae"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_valid_labels),
    problem_type="multi_label_classification",
)


# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=2048, padding="max_length"
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

def objective(trial):
    # --- Hyperparameters to search ---
    learning_rate = trial.suggest_float("learning_rate", 3e-6, 3e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)

    # --- Fresh model per trial ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(all_valid_labels),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
    output_dir=f"./optuna_runs/trial_{trial.number}",

    # 🔴 IMPORTANT: disable saving
    save_strategy="no",
    save_total_limit=0,
    load_best_model_at_end=False,

    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,

    eval_strategy="epoch",
    logging_strategy="epoch",
    metric_for_best_model="f1_micro",
    greater_is_better=True,

    report_to="none",
    seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    return eval_metrics["eval_f1_micro"]

#Run optim

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best trial:")
print(study.best_trial.params)
print("Best F1:", study.best_value)



#Best model

best_params = study.best_trial.params

final_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    warmup_ratio=best_params["warmup_ratio"],
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    tf32=True,
    seed=42,
)

final_trainer = Trainer(
    model=model,
    args=final_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=lambda p: compute_metrics(p, verbose=True),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

final_trainer.train()


# Evaluate and save
test_results = final_trainer.predict(test_dataset)

model.config.label2id = {label: i for i, label in enumerate(all_valid_labels)}
model.config.id2label = {i: label for i, label in enumerate(all_valid_labels)}

final_trainer.save_model("./persian_register_model")
tokenizer.save_pretrained("./persian_register_model")

with open("./persian_register_model/test_results.json", "w") as f:
    json.dump(
        {k.replace("test_", ""): float(v) for k, v in test_results.metrics.items()},
        f,
        indent=2,
    )

print("Done!")
