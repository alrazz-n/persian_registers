import json
import os

os.environ["HF_HOME"] = "./hf_home"

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
#Fro clean lab:
import random
import numpy as np
import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from cleanlab.multilabel_classification import find_label_issues

from cleanlab import Datalab
from cleanlab.internal.multilabel_utils import int2onehot, onehot2int
from sentence_transformers import SentenceTransformer



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


def load_jsonl_data(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append([record["text"]])
            label_list = record["label"].split()
            binary_vector = [
                1.0 if label in label_list else 0.0 for label in all_valid_labels
            ]  # Note: these are floats for compatibility with Trainer
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

# Load and split data
X, y = load_jsonl_data("data/persian_consolidated.jsonl")
print(f"Loaded {len(X)} documents with {len(all_valid_labels)} labels")

X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.3)
X_dev, y_dev, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev, y_dev)
test_dataset = create_dataset(X_test, y_test)

print(
    f"Split: {len(train_dataset)} train, {len(dev_dataset)} dev, {len(test_dataset)} test"
)

texts_train = X_train.flatten().tolist()


embedder = SentenceTransformer("BAAI/bge-m3-retromae")

X_train_emb = embedder.encode(
    texts_train,
    batch_size=32,
    show_progress_bar=True
)


clf = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )
)

pred_probs_train = cross_val_predict(
    clf,
    X_train_emb,
    y_train,
    cv=5,
    method="predict_proba"
)


label_issue_indices = find_label_issues(
    labels=y_train,
    pred_probs=pred_probs_train,
    return_indices_ranked_by="self_confidence",
)


for idx in label_issue_indices[:20]:
    print("TEXT:", texts_train[idx][:300])
    print("GIVEN LABELS:",
          [l for l, v in zip(all_valid_labels, y_train[idx]) if v == 1])
    print("MODEL CONFIDENCE:")
    for l, p in zip(all_valid_labels, pred_probs_train[idx]):
        if p > 0.3:
            print(f"  {l}: {p:.2f}")
    print("-" * 60)
