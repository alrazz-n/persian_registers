

# ============================================================
# UMAP VISUALIZATION OF PRETRAINED MULTILINGUAL BGE-M3
# ============================================================
#
# This script:
# 1. Loads the pretrained web-register classification model
# 2. Loads the original test dataset
# 3. Extracts document embeddings from the BGE-M3 encoder
# 4. Applies UMAP
# 5. Creates overall and class-specific visualizations
#
# NO TRAINING OR RETRAINING REQUIRED
#
# Model:
# TurkuNLP/web-register-classification-multilingual-bge
#
# Tokenizer:
# BAAI/bge-m3-retromae
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import umap.umap_ as umap

from torch.utils.data import DataLoader

from datasets import load_from_disk

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CONFIGURATION / PATHS
# ============================================================

DATASET_ROOT = Path(
    r"/scratch/project_462001491/nima/Hybrid_SP_ID_effect/Dataset_Hugging_face/without_NA"
)

MODEL_ID = (
    "TurkuNLP/web-register-classification-multilingual-bge"
)

TOKENIZER_ID = (
    "BAAI/bge-m3-retromae"
)

RESULTS_ROOT = (
    DATASET_ROOT
    / "_MultiCore_BGE_m3_base_UMAP_Multilabel"
)

HF_CACHE = Path(
    "/scratch/project_462001491/nima/huggingface"
)

MAX_LENGTH = 1024

BATCH_SIZE = 8

# UMAP parameters
N_NEIGHBORS = 15
MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# Create output directory
RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SELECT DATASET USING SLURM ARRAY
# ============================================================

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
        f"No Hugging Face datasets found in:\n{DATASET_ROOT}"
    )


if "SLURM_ARRAY_TASK_ID" not in os.environ:

    raise RuntimeError(
        "SLURM_ARRAY_TASK_ID is not set."
    )


task_id = int(
    os.environ["SLURM_ARRAY_TASK_ID"]
)


if task_id < 0 or task_id >= len(dataset_dirs):

    raise IndexError(
        f"SLURM_ARRAY_TASK_ID={task_id}, "
        f"but only {len(dataset_dirs)} datasets were found."
    )


dataset_path = dataset_dirs[task_id]

dataset_name = dataset_path.name


print("\n" + "=" * 80)
print("UMAP VISUALIZATION - PRETRAINED BGE-M3")
print("=" * 80)

print("Task ID:", task_id)
print("Dataset:", dataset_name)
print("Dataset path:", dataset_path)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR = (
    RESULTS_ROOT
    / dataset_name
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("\nModel ID:", MODEL_ID)
print("Tokenizer ID:", TOKENIZER_ID)
print("Output path:", OUTPUT_DIR)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = load_from_disk(
    str(dataset_path)
)

test_dataset = dataset["test"]

print(test_dataset)


# ============================================================
# LOAD METADATA
# ============================================================

metadata_file = (
    dataset_path
    / "split_metadata.json"
)

if not metadata_file.exists():

    raise FileNotFoundError(
        f"Metadata file not found:\n{metadata_file}"
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


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    cache_dir=str(HF_CACHE),
)

print("Tokenizer loaded.")


# ============================================================
# LOAD PRETRAINED CLASSIFICATION MODEL
# ============================================================

print("\nLoading pretrained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    cache_dir=str(HF_CACHE),
)

model.to(device)

model.eval()

print("Pretrained model loaded.")


# ============================================================
# MODEL LABELS
# ============================================================

id2label = model.config.id2label

model_labels = [
    id2label.get(i, id2label.get(str(i)))
    for i in range(model.config.num_labels)
]

print("\nModel labels:")
print(model_labels)

print(
    "\nNumber of model labels:",
    len(model_labels)
)


# ============================================================
# VERIFY LABEL SPACE
# ============================================================

if len(dataset_labels) != len(model_labels):

    raise ValueError(
        "Label count mismatch.\n"
        f"Dataset labels: {len(dataset_labels)}\n"
        f"Model labels: {len(model_labels)}"
    )


missing_labels = [
    label
    for label in model_labels
    if label not in dataset_labels
]

extra_labels = [
    label
    for label in dataset_labels
    if label not in model_labels
]


if missing_labels or extra_labels:

    raise ValueError(
        f"Missing labels: {missing_labels}\n"
        f"Extra labels: {extra_labels}"
    )


# Dataset label -> dataset index
label_to_dataset_index = {
    label: i
    for i, label in enumerate(dataset_labels)
}


# Dataset columns in model label order
dataset_indices_for_model = [
    label_to_dataset_index[label]
    for label in model_labels
]


print("\nLabel mapping:")

for model_index, label in enumerate(model_labels):

    print(
        f"{model_index}: {label}"
    )


# ============================================================
# PREPARE TEST DATASET
# ============================================================

print("\nPreparing test dataset...")


def prepare_test_dataset(ds):

    # --------------------------------------------------------
    # Align labels with model label order
    # --------------------------------------------------------

    def align_labels(example):

        original_labels = np.asarray(
            example["labels"],
            dtype=np.float32,
        )

        aligned_labels = original_labels[
            dataset_indices_for_model
        ]

        return {
            "labels": aligned_labels.tolist()
        }


    ds = ds.map(
        align_labels
    )


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Keep only necessary columns
    # --------------------------------------------------------

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


test_dataset = prepare_test_dataset(
    test_dataset
)

print(
    "Prepared test examples:",
    len(test_dataset)
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)


# ============================================================
# EXTRACT DOCUMENT EMBEDDINGS
# ============================================================

print("\nExtracting test-set embeddings...")


# Extract underlying BGE-M3 transformer encoder
encoder = model.base_model

encoder.eval()


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
)


all_embeddings = []
all_labels = []


with torch.inference_mode():

    for batch_idx, batch in enumerate(test_loader):

        # ----------------------------------------------------
        # Keep labels separately
        # ----------------------------------------------------

        labels = batch.pop("labels")


        # ----------------------------------------------------
        # Move model inputs to device
        # ----------------------------------------------------

        inputs = {
            key: value.to(device)
            for key, value in batch.items()
        }


        # ----------------------------------------------------
        # Extract hidden representations
        # ----------------------------------------------------

        outputs = encoder(
            **inputs,
            return_dict=True,
        )


        # Last hidden layer
        token_embeddings = outputs.last_hidden_state


        # ----------------------------------------------------
        # Attention-mask-aware mean pooling
        # ----------------------------------------------------

        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).to(token_embeddings.dtype)


        summed_embeddings = torch.sum(
            token_embeddings * mask,
            dim=1,
        )


        summed_mask = torch.clamp(
            mask.sum(dim=1),
            min=1e-9,
        )


        sentence_embeddings = (
            summed_embeddings / summed_mask
        )


        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        sentence_embeddings = (
            torch.nn.functional.normalize(
                sentence_embeddings,
                p=2,
                dim=1,
            )
        )


        # ----------------------------------------------------
        # Save embeddings
        # ----------------------------------------------------

        all_embeddings.append(
            sentence_embeddings.cpu().numpy()
        )


        all_labels.append(
            labels.numpy()
        )


        if (batch_idx + 1) % 50 == 0:

            print(
                f"Processed batches: {batch_idx + 1}"
            )


# ============================================================
# COMBINE BATCHES
# ============================================================

embeddings = np.concatenate(
    all_embeddings,
    axis=0,
)

test_labels_array = np.concatenate(
    all_labels,
    axis=0,
)


print("\nEmbedding extraction completed.")

print("Embedding shape:", embeddings.shape)
print("Labels shape:", test_labels_array.shape)


# Verify alignment
assert len(embeddings) == len(test_dataset)

assert len(test_labels_array) == len(test_dataset)


# ============================================================
# VALIDATE MULTILABEL STRUCTURE
# ============================================================

positive_label_counts = test_labels_array.sum(axis=1)

print("\nMultilabel statistics:")

print(
    "Documents with no positive labels:",
    np.sum(positive_label_counts == 0)
)

print(
    "Documents with exactly one positive label:",
    np.sum(positive_label_counts == 1)
)

print(
    "Documents with multiple positive labels:",
    np.sum(positive_label_counts > 1)
)

print(
    "Maximum positive labels per document:",
    positive_label_counts.max()
)

# ============================================================
# SAVE DOCUMENT EMBEDDINGS
# ============================================================

print("\nSaving document embeddings...")

np.save(
    OUTPUT_DIR / "test_embeddings.npy",
    embeddings,
)

np.save(
    OUTPUT_DIR / "test_embedding_labels.npy",
    test_labels_array,
)

print("Embeddings saved.")


# ============================================================
# UMAP DIMENSIONALITY REDUCTION
# ============================================================

print("\nRunning UMAP...")


if len(embeddings) < 3:

    raise ValueError(
        "UMAP requires at least 3 test examples."
    )


effective_neighbors = min(
    N_NEIGHBORS,
    len(embeddings) - 1,
)


reducer = umap.UMAP(
    n_components=2,
    n_neighbors=effective_neighbors,
    min_dist=MIN_DIST,
    metric=UMAP_METRIC,
    random_state=SEED,
)


embedding_2d = reducer.fit_transform(
    embeddings
)


print("UMAP completed.")

print(
    "2D embedding shape:",
    embedding_2d.shape
)


# ============================================================
# SAVE UMAP COORDINATES
# ============================================================

np.save(
    OUTPUT_DIR / "test_umap_2d.npy",
    embedding_2d,
)


umap_df = pd.DataFrame({

    "UMAP_1": embedding_2d[:, 0],

    "UMAP_2": embedding_2d[:, 1],

})


umap_df.to_csv(
    OUTPUT_DIR / "test_umap_coordinates.csv",
    index=False,
)


# ============================================================
# OVERALL UMAP VISUALIZATION
# ============================================================

print("\nCreating overall UMAP visualization...")


# ============================================================
# ASSIGN ALL POSITIVE LABELS
# ============================================================

document_labels = []

for row in test_labels_array:

    positive_indices = np.where(row == 1)[0]

    if len(positive_indices) > 0:

        labels = [
            model_labels[i]
            for i in positive_indices
        ]

        document_labels.append(
            ", ".join(labels)
        )

    else:

        document_labels.append(
            "No positive label"
        )


plot_df = pd.DataFrame({

    "UMAP_1": embedding_2d[:, 0],

    "UMAP_2": embedding_2d[:, 1],

    "Label": document_labels,

})


plt.figure(figsize=(14, 10))


sns.scatterplot(
    data=plot_df,
    x="UMAP_1",
    y="UMAP_2",
    hue="Label",
    palette="tab20",
    s=35,
    alpha=0.75,
    linewidth=0,
)


plt.title(
    f"UMAP Visualization - Pretrained BGE-M3 - {dataset_name}",
    fontsize=16,
    fontweight="bold",
)


plt.xlabel(
    "UMAP Dimension 1",
    fontsize=12,
)

plt.ylabel(
    "UMAP Dimension 2",
    fontsize=12,
)


plt.legend(
    title="Label Combination",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR / "test_umap_visualization.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()

print("Overall multilabel UMAP visualization saved.")

# ============================================================
# CLASS-SPECIFIC UMAP VISUALIZATIONS
# ============================================================

print("\nCreating class-specific UMAP visualizations...")


for class_idx, class_name in enumerate(model_labels):

    plt.figure(figsize=(10, 8))


    positive_mask = (
        test_labels_array[:, class_idx] == 1
    )

    negative_mask = ~positive_mask


    # Negative documents
    plt.scatter(
        embedding_2d[negative_mask, 0],
        embedding_2d[negative_mask, 1],
        c="lightgray",
        s=20,
        alpha=0.35,
        label="Not belonging to class",
    )


    # Positive documents
    plt.scatter(
        embedding_2d[positive_mask, 0],
        embedding_2d[positive_mask, 1],
        c="red",
        s=30,
        alpha=0.75,
        label="Belonging to class",
    )


    plt.title(
        f"UMAP - {class_name}",
        fontsize=15,
        fontweight="bold",
    )


    plt.xlabel("UMAP Dimension 1")

    plt.ylabel("UMAP Dimension 2")


    plt.legend()

    plt.tight_layout()


    # Safe filename
    safe_class_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(class_name),
    )


    plt.savefig(
        OUTPUT_DIR / f"umap_class_{safe_class_name}.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


print("Class-specific visualizations saved.")


# ============================================================
# SAVE COMPLETE VISUALIZATION DATAFRAME
# ============================================================

umap_df["Labels"] = document_labels


for class_idx, class_name in enumerate(model_labels):

    umap_df[class_name] = (
        test_labels_array[:, class_idx]
    )


umap_df.to_csv(
    OUTPUT_DIR / "test_umap_complete.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)

print("UMAP VISUALIZATION COMPLETED SUCCESSFULLY!")

print("=" * 80)

print("\nSaved files:")

print("1. test_embeddings.npy")

print("2. test_embedding_labels.npy")

print("3. test_umap_2d.npy")

print("4. test_umap_coordinates.csv")

print("5. test_umap_visualization.png")

print("6. umap_class_*.png")

print("7. test_umap_complete.csv")


print("\nOutput directory:")

print(OUTPUT_DIR)

print("\n" + "=" * 80)