# =========================
# Environment setup
# =========================
import os
os.environ["HF_HOME"] = "./hf_home"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import numpy as np
import torch

from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from skmultilearn.model_selection import iterative_train_test_split

from sentence_transformers import SentenceTransformer

# Cleanlab multilabel API (THIS is the key)
from cleanlab.multilabel_classification.filter import find_label_issues
from cleanlab.multilabel_classification.rank import get_label_quality_scores

# =========================
# Reproducibility
# =========================
torch.manual_seed(44)
np.random.seed(44)
random.seed(44)

# =========================
# Label structure
# =========================
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

# =========================
# Data loading
# =========================
def load_jsonl_data(filepath):
    texts, labels = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])

            label_list = record["label"].split()
            binary_vector = [
                1 if lbl in label_list else 0
                for lbl in all_valid_labels
            ]
            labels.append(binary_vector)

    # X must be 2D for skmultilearn
    return np.array(texts).reshape(-1, 1), np.array(labels, dtype=int)

# =========================
# Load + split data
# =========================
X, y = load_jsonl_data("data/persian_consolidated.jsonl")
print(f"Loaded {len(X)} documents with {NUM_LABELS} labels")

X_train, y_train, X_temp, y_temp = iterative_train_test_split(
    X, y, test_size=0.3
)
X_dev, y_dev, X_test, y_test = iterative_train_test_split(
    X_temp, y_temp, test_size=0.5
)

texts_train = X_train.flatten().tolist()

print(
    f"Split: {len(X_train)} train, "
    f"{len(X_dev)} dev, "
    f"{len(X_test)} test"
)

# =========================
# Sentence embeddings
# =========================
embedder = SentenceTransformer("BAAI/bge-m3-retromae")

X_train_emb = embedder.encode(
    texts_train,
    batch_size=32,
    show_progress_bar=True,
)

# =========================
# Multilabel classifier
# =========================
clf = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
)

# Cross-validated predicted probabilities (REQUIRED by Cleanlab)
pred_probs_train = cross_val_predict(
    clf,
    X_train_emb,
    y_train,
    cv=5,
    method="predict_proba",
    n_jobs=-1,
)

# =========================
# Convert labels to Cleanlab multilabel format
# =========================
# Cleanlab expects: List[List[int]]
labels_list = [
    [i for i, v in enumerate(row) if v == 1]
    for row in y_train
]

# =========================
# Cleanlab: Find label issues
# =========================
issue_indices = find_label_issues(
    labels=labels_list,
    pred_probs=pred_probs_train,
    return_indices_ranked_by="self_confidence",
)

# Optional: per-example label quality scores
label_quality_scores = get_label_quality_scores(
    labels_list,
    pred_probs_train,
)

# =========================
# Inspect top suspicious examples
# =========================
for idx in issue_indices[:100]:
    print("TEXT:")
    print(texts_train[idx][:300])

    print("GIVEN LABELS:")
    print([all_valid_labels[i] for i in labels_list[idx]])

    print("LABEL QUALITY SCORE:")
    print(f"{label_quality_scores[idx]:.4f}")

    print("MODEL CONFIDENCE (>0.3):")
    for l, p in zip(all_valid_labels, pred_probs_train[idx]):
        if p > 0.3:
            print(f"  {l}: {p:.2f}")

    print("-" * 70)
print("DOne!")
