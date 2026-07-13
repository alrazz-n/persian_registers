from datasets import load_from_disk
from datasets import concatenate_datasets

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


dataset = load_from_disk("/scratch/project_2005092/nima/binary_dataset")
train_dataset = dataset["train"]
dev_dataset = dataset["validation"]
test_dataset = dataset["test"]



full_dataset = concatenate_datasets(
    [
        train_dataset,
        dev_dataset,
        test_dataset
    ]
)

#print(full_dataset)

#print(full_dataset["Turku_NLP"][:5])
#print(full_dataset["Turku_NLP_sub"][:5])
#print(full_dataset["Binary"][:5])

#print(full_dataset[0])

labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it", "os"],
    "ID": [],
    "NA": ["ne", "sr", "nb", "on"],
    "HI": ["re", "oh"],
    "IN": ["en", "ra", "dtp", "fi", "lt", "oi"],
    "OP": ["rv", "ob", "rs", "av", "oo"],
    "IP": ["ds", "ed", "oe"],
}


labels_to_remove = { #To make it like multiCore
    "oi",
    "os",
    "on",
    "oh",
    "oo",
    "oe"
}

all_valid_labels = sorted(
    [
        label
        for label in (
            list(labels_structure.keys())
            + [
                s
                for subs in labels_structure.values()
                for s in subs
            ]
        )
        if label not in labels_to_remove
    ]
)

#print(all_valid_labels)



def extract_labels(row):

    labels = set()

    for column in ["Turku_NLP", "Turku_NLP_sub"]:

        value = row[column]

        if value:
            for label in value.split(";"):

                label = label.strip()

                # ignore empty values and "-"
                if label == "" or label == "-":
                    continue

                # remove "other" categories
                if label in labels_to_remove:
                    continue

                # keep only valid labels
                if label in all_valid_labels:
                    labels.add(label)


    return sorted(labels)

cleaned_labels = [
    extract_labels(row)
    for row in full_dataset
]

#print(cleaned_labels[:5])

import numpy as np

label_to_index = {
    label: i
    for i, label in enumerate(all_valid_labels)
}


y = np.zeros(
    (len(cleaned_labels), len(all_valid_labels)),
    dtype=int
)


for row_idx, labels in enumerate(cleaned_labels):
    for label in labels:
        y[row_idx, label_to_index[label]] = 1

#print(y.shape)
#print(y[:5])

texts = full_dataset["text"]

#print(len(texts))
#print(texts[0][:300])

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(
    "BAAI/bge-m3-retromae"
)


X_emb = embedder.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict


clf = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
)

#print(X_emb.shape)


pred_probs = cross_val_predict(
    clf,
    X_emb,
    y,
    cv=5,
    method="predict_proba",
    n_jobs=-1,
)
#print(pred_probs.shape)

labels_list = [
    [
        i
        for i, value in enumerate(row)
        if value == 1
    ]
    for row in y
]

#print(labels_list[:5])

from cleanlab.multilabel_classification.filter import find_label_issues
from cleanlab.multilabel_classification.rank import get_label_quality_scores


issue_indices = find_label_issues(
    labels=labels_list,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence"
)

issue_indices = [int(i) for i in issue_indices]



quality_scores = get_label_quality_scores(
    labels_list,
    pred_probs,
)


print(
    f"Found {len(issue_indices)} suspicious examples"
)

import pandas as pd

issues_df = pd.DataFrame(
    {
        "index": issue_indices,
        "id": [
            full_dataset[i]["id"]
            for i in issue_indices
        ],
        "source": [
             full_dataset[i]["source"]
             for i in issue_indices
        ],
        "text": [
            full_dataset[i]["text"]
             for i in issue_indices
        ],
        "given_labels": [
            cleaned_labels[i]
            for i in issue_indices
        ],
        "quality_score": [
            quality_scores[i]
            for i in issue_indices
        ],
    }
)

def get_predicted_labels(idx, threshold=0.4): #based on multicore threshold

    return [
        label
        for label, prob in zip(
            all_valid_labels,
            pred_probs[idx]
        )
        if prob >= threshold
    ]


issues_df["model_labels"] = [
    get_predicted_labels(i)
    for i in issue_indices
]

def get_label_probabilities(idx):

    return {
        label: round(float(prob), 4)
        for label, prob in zip(
            all_valid_labels,
            pred_probs[idx]
        )
    }


issues_df["label_probabilities"] = [
    get_label_probabilities(i)
    for i in issue_indices
]


issues_df = issues_df.sort_values(
    "quality_score"
)

issues_df.to_csv(
    "cleanlab_annotation_review_bge-m3.csv",
    index=False,
    encoding="utf-8-sig"
)