import json
import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.special import expit as sigmoid
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from skmultilearn.model_selection import iterative_train_test_split
import optuna


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from datasets import load_from_disk

dataset = load_from_disk("/scratch/project_2005092/nima/binary_dataset")

train_dataset = dataset["train"]
dev_dataset = dataset["validation"]
test_dataset = dataset["test"]

# rename target to labels
train_dataset = train_dataset.rename_column("Binary", "labels")
dev_dataset = dev_dataset.rename_column("Binary", "labels")
test_dataset = test_dataset.rename_column("Binary", "labels")

NUM_LABELS = 2

MODEL_NAME = "FacebookAI/xlm-roberta-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512, #for XLMR
    )

train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

for ds in (train_dataset, dev_dataset, test_dataset):
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

from sklearn.metrics import recall_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

#Do not want to leave any Junk behind
    return {"recall_0": recall_score(labels, predictions, pos_label=0)} #recall for class 0


# Optuna objective
#Bayesian optimization (sample hyperparameters intelligently across trials)
def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 5e-6, 3e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)

    per_device_batch = trial.suggest_categorical("batch_size", [4, 8])
    grad_accum = trial.suggest_categorical("grad_accum", [4, 8])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification",
    )

    args = TrainingArguments(
        output_dir=f"./XLMR_optuna_ham_spam/trial_{trial.number}",
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

        save_strategy="epoch",
        save_total_limit=1,   # keep only one checkpoint for better
        load_best_model_at_end=True,
        metric_for_best_model="eval_recall_0",
        report_to="none",

        seed=42,
        bf16=True,
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(f"./saved_models/trial_{trial.number}")
    tokenizer.save_pretrained(f"./saved_models/trial_{trial.number}")
    #metrics = trainer.evaluate()

    #return metrics["eval_recall_0"]
    return trainer.state.best_metric


# 1) Get the best checkpoint from your Optuna trial (example assumes you kept it)
# If you ran only objective() directly, you need to rerun best trial or store it.
# For illustration: load from some known best path
# --- run optuna ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=8)  # set n_trials

# --- load best checkpoint ---
#best_ckpt = f"./XLMR_optuna_ham_spam/trial_{study.best_trial.number}"
#best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
best_model = AutoModelForSequenceClassification.from_pretrained(
    f"./saved_models/trial_{study.best_trial.number}"
)

# --- evaluate on test and print classification table ---
trainer = Trainer(
    model=best_model,
    args=TrainingArguments(output_dir="./tmp", report_to="none"),
    tokenizer=tokenizer,
)

print("\n===== BEST TRIAL =====")
print(study.best_trial.params)
print("Best Recall:", study.best_value)

pred = trainer.predict(test_dataset)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=-1)

print("\n=============")
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


target_names = ["class_0", "class_1"]
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

df_cm = pd.DataFrame(
    confusion_matrix(y_true, y_pred),
    index=target_names,
    columns=target_names
)
print(df_cm)

