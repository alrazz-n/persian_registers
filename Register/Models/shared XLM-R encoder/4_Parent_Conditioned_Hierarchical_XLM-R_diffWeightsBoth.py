#===============================

#Child classifier will get

#                    ┌─────────────────┐
#                    │                 │
#                    ▼                 │
#                  h (1024)            │
#                    │                 │
#                    │          parent_hidden (256)
#                    │                 │
#                    └────────┬────────┘
#                             ▼
#                       child_input
#                         1280 dim
#                             │
#                             ▼
#                      Child classifier
#                             │
#                             ▼
#                         16 logits

# And the whole thing looks like:

#                         XLM-R
#                           │
#                           ▼
#                           h
#                           │
#                  ┌────────┴────────┐
#                  │                 │
#                  ▼                 │
#          parent representation     │
#                  │                 │
#                  ▼                 │
#           Parent classifier        │
#                  │                 │
#                  ▼                 │
#            parent predictions      │
#                                    │
#                  ┌─────────────────┘
#                  ▼
#            Child classifier

#Or in big picture

#                     Text
#                      │
#                      ▼
#                    XLM-R
#                      │
#                      ▼
#                      h
#                      │
#             ┌────────┴─────────┐
#             │                  │
#             ▼                  ▼
#      Parent projection       original h
#             │                  │
#             ▼                  │
#      parent_hidden             │
#             │                  │
#             ▼                  │
#      Parent classifier         │
#             │                  │
#             ▼                  │
#       Parent logits            │
#                                │
#             ┌──────────────────┘
#             │
#             ▼
#      concatenate(h,
#                  parent_hidden)
#             │
#             ▼
#       Child classifier
#             │
#             ▼
#        Child logits

# ============================================================
# Hierarchical XLM-R Multilabel Classification
#
# Architecture:
#
#                         XLM-R Encoder
#                               |
#                               v
#                         shared representation h
#                               |
#                     +---------+---------+
#                     |                   |
#                     v                   |
#              Parent projection          |
#                     |                   |
#                     v                   |
#               parent_hidden             |
#                  /        \              |
#                 /          \             |
#                v            v            |
#       Parent classifier   concatenate <--+
#                |            |
#                v            v
#         9 parent logits   Child classifier
#                               |
#                               v
#                         16 child logits
#
#
# Total output labels = 25
#
# Parent labels:
# MT LY SP ID NA HI IN OP IP
#
# Child labels:
# it ne sr nb re en ra dtp fi lt
# rv ob rs av ds ed
#
#
# Loss:
#
#   L = parent_weight * parent_loss
#       +
#       child_weight * child_loss
#
# Both parent_weight and child_weight are optimized by Optuna.
#
# ============================================================


# ============================================================
# Imports
# ============================================================

import os
import json
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.nn import BCEWithLogitsLoss

import optuna

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from datasets import load_from_disk

from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback,
)

from transformers import set_seed


# ============================================================
# GPU configuration
# ============================================================

print("\n" + "=" * 100)
print("GPU CONFIGURATION")
print("=" * 100)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "CUDA device count:",
    torch.cuda.device_count()
)

if torch.cuda.is_available():

    print(
        "Current device:",
        torch.cuda.current_device()
    )

    print(
        "Device name:",
        torch.cuda.get_device_name(
            torch.cuda.current_device()
        )
    )


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# Hierarchical label structure
# ============================================================

labels_structure = {

    "MT": [],

    "LY": [],

    "SP": [
        "it"
    ],

    "ID": [],

    "NA": [
        "ne",
        "sr",
        "nb",
    ],

    "HI": [
        "re"
    ],

    "IN": [
        "en",
        "ra",
        "dtp",
        "fi",
        "lt",
    ],

    "OP": [
        "rv",
        "ob",
        "rs",
        "av",
    ],

    "IP": [
        "ds",
        "ed",
    ],
}


# ============================================================
# Parent labels
# ============================================================

parents = list(
    labels_structure.keys()
)


# ============================================================
# Child labels
# ============================================================

children = [
    child
    for child_list in labels_structure.values()
    for child in child_list
]


# ============================================================
# Child -> parent mapping
# ============================================================

child_to_parent = {

    child: parent

    for parent, child_list
    in labels_structure.items()

    for child in child_list
}


# ============================================================
# Complete hierarchical label list
#
# IMPORTANT:
# This is the exact order used by the model output.
# ============================================================

hierarchical_labels = (
    parents + children
)


print("\n" + "=" * 100)
print("HIERARCHICAL LABEL STRUCTURE")
print("=" * 100)

print("\nParents:")

for i, label in enumerate(parents):

    print(
        f"{i:2d}: {label}"
    )


print("\nChildren:")

for i, label in enumerate(children):

    print(
        f"{len(parents) + i:2d}: "
        f"{label} -> "
        f"{child_to_parent[label]}"
    )


print(
    "\nTotal labels:",
    len(hierarchical_labels),
)


print("\nComplete label order:")

for i, label in enumerate(
    hierarchical_labels
):

    print(
        f"{i:2d}: {label}"
    )


print("=" * 100)


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path(
    r"/scratch/project_462001491/nima/Hybrid_SP_ID_effect/Dataset_Hugging_face/without_NA"
)


RESULTS_ROOT = (
    DATASET_ROOT
    / "_MultiCore_ParentConditionedtoChild_XLM-R_DiffWeightsBoth"
)


RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Pretrained multilingual web-register model
# ============================================================

MODEL_ID = (
    "TurkuNLP/web-register-classification-multilingual"
)


# ============================================================
# Tokenization
# ============================================================

MAX_LENGTH = 512


# ============================================================
# Initial threshold
#
# We save logits and probabilities later so thresholds
# can be tuned independently.
# ============================================================

THRESHOLD = 0.5


# ============================================================
# Parent representation size
#
# XLM-R-large hidden size is 1024.
#
# We transform:
#
#     h:             1024
#
# into:
#
#     parent_hidden: 256
#
# The child classifier then receives:
#
#     h + parent_hidden
#
#     1024 + 256 = 1280 dimensions
# ============================================================

PARENT_HIDDEN_SIZE = 256


# ============================================================
# Hugging Face cache
# ============================================================

HF_CACHE = Path(
    "/scratch/project_462001491/nima/huggingface"
)


HF_CACHE.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Select dataset using SLURM array
# ============================================================

dataset_dirs = sorted(
    [
        path
        for path in DATASET_ROOT.iterdir()

        if path.is_dir()

        and (
            path / "dataset_dict.json"
        ).exists()
    ]
)


if not dataset_dirs:

    raise RuntimeError(
        "No Hugging Face datasets found in:\n"
        f"{DATASET_ROOT}"
    )


if (
    "SLURM_ARRAY_TASK_ID"
    not in os.environ
):

    raise RuntimeError(
        "SLURM_ARRAY_TASK_ID is not set."
    )


task_id = int(
    os.environ[
        "SLURM_ARRAY_TASK_ID"
    ]
)


if task_id >= len(dataset_dirs):

    raise IndexError(
        f"SLURM_ARRAY_TASK_ID={task_id}, "
        f"but only {len(dataset_dirs)} "
        "datasets were found."
    )


dataset_path = (
    dataset_dirs[task_id]
)


dataset_name = (
    dataset_path.name
)


print("\n" + "=" * 100)
print("SLURM ARRAY INFORMATION")
print("=" * 100)

print(
    "Task ID:",
    task_id
)

print(
    "Dataset:",
    dataset_name
)

print(
    "Dataset path:",
    dataset_path
)

print("=" * 100)


# ============================================================
# Load dataset
# ============================================================

print(
    f"\nLoading dataset: {dataset_name}"
)


dataset = load_from_disk(
    str(dataset_path)
)


print(dataset)


# ============================================================
# Load metadata
# ============================================================

metadata_file = (
    dataset_path
    / "split_metadata.json"
)


if not metadata_file.exists():

    raise FileNotFoundError(
        "Metadata file not found:\n"
        f"{metadata_file}"
    )


with open(
    metadata_file,
    "r",
    encoding="utf-8",
) as f:

    metadata = json.load(f)


dataset_labels = (
    metadata["labels"]
)


print(
    "\nOriginal dataset labels:"
)

print(
    dataset_labels
)


print(
    "Number of original dataset labels:",
    len(dataset_labels),
)


# ============================================================
# Dataset label -> index
# ============================================================

dataset_label_to_index = {

    label: i

    for i, label
    in enumerate(dataset_labels)
}


# ============================================================
# Verify hierarchy against dataset
# ============================================================

missing_hierarchy_labels = [

    label

    for label in hierarchical_labels

    if label not in dataset_label_to_index
]


if missing_hierarchy_labels:

    raise ValueError(
        "\nThe following hierarchical labels "
        "are missing from the dataset:\n"
        f"{missing_hierarchy_labels}"
    )


print(
    "\nAll hierarchical labels were found "
    "in the dataset."
)


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR = (
    RESULTS_ROOT
    / dataset_name
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Save label order
# ============================================================

with open(
    OUTPUT_DIR
    / "hierarchical_labels.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        hierarchical_labels,
        f,
        indent=2,
    )


# ============================================================
# Tokenizer
# ============================================================

print(
    "\nLoading tokenizer..."
)


tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-large",
    cache_dir=str(HF_CACHE),
)


print(
    "Tokenizer loaded."
)


# ============================================================
# Convert original labels into hierarchical labels
# ============================================================

def make_hierarchical_labels(
    example
):

    original_labels = np.asarray(
        example["labels"],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Create dictionary:
    #
    # "IN"  -> 0/1
    # "en"  -> 0/1
    # etc.
    # --------------------------------------------------------

    label_values = {

        label: original_labels[
            dataset_label_to_index[label]
        ]

        for label in dataset_labels
    }


    hierarchical_target = []


    # --------------------------------------------------------
    # Parents
    #
    # If any child is positive,
    # the corresponding parent becomes positive.
    # --------------------------------------------------------

    for parent in parents:

        parent_value = (
            label_values[parent]
        )


        for child in labels_structure[parent]:

            if (
                label_values[child]
                > 0
            ):

                parent_value = 1.0


        hierarchical_target.append(
            parent_value
        )


    # --------------------------------------------------------
    # Children
    # --------------------------------------------------------

    for child in children:

        child_value = (
            label_values[child]
        )


        hierarchical_target.append(
            child_value
        )


    return {
        "labels":
            hierarchical_target
    }


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(
    ds,
    split_name,
):

    print(
        f"\nPreparing {split_name}: "
        f"{len(ds)} examples"
    )


    # --------------------------------------------------------
    # Create hierarchical labels
    # --------------------------------------------------------

    ds = ds.map(
        make_hierarchical_labels
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
    # Keep only Trainer-required columns
    # --------------------------------------------------------

    columns_to_keep = [

        "input_ids",

        "attention_mask",

        "labels",
    ]


    if (
        "token_type_ids"
        in ds.column_names
    ):

        columns_to_keep.append(
            "token_type_ids"
        )


    ds = ds.remove_columns(

        [
            column

            for column
            in ds.column_names

            if column
            not in columns_to_keep
        ]
    )


    return ds


# ============================================================
# Prepare train/dev/test
# ============================================================

train_dataset = prepare_dataset(
    dataset["train"],
    "train",
)


validation_dataset = prepare_dataset(
    dataset["dev"],
    "dev",
)


test_dataset = prepare_dataset(
    dataset["test"],
    "test",
)


print("\n" + "=" * 100)
print("PREPARED DATASETS")
print("=" * 100)

print(
    "Train:",
    len(train_dataset)
)

print(
    "Dev:",
    len(validation_dataset)
)

print(
    "Test:",
    len(test_dataset)
)

print("=" * 100)


# ============================================================
# Data collator
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)


# ============================================================
# Hierarchical Parent-Conditioned XLM-R
# ============================================================

class HierarchicalXLMR(
    nn.Module
):

    def __init__(
        self,
        encoder,
        num_parents,
        num_children,
        parent_weight=1.0,
        child_weight=1.0,
        parent_hidden_size=256,
    ):

        super().__init__()


        # ----------------------------------------------------
        # Shared pretrained XLM-R encoder
        # ----------------------------------------------------

        self.encoder = encoder


        hidden_size = (
            self.encoder.config.hidden_size
        )


        # ----------------------------------------------------
        # Parent projection
        #
        # h:
        #
        #     [batch, hidden_size]
        #
        # becomes:
        #
        #     parent_hidden:
        #     [batch, parent_hidden_size]
        #
        # This representation is learned to contain
        # information useful for the parent task.
        # ----------------------------------------------------

        self.parent_projection = nn.Linear(
            hidden_size,
            parent_hidden_size,
        )


        # ----------------------------------------------------
        # Optional non-linearity
        #
        # This gives the parent representation more
        # expressive power than a simple linear projection.
        # ----------------------------------------------------

        self.parent_activation = nn.GELU()


        # ----------------------------------------------------
        # Parent classifier
        #
        # parent_hidden
        #       ↓
        # 9 parent logits
        # ----------------------------------------------------

        self.parent_classifier = nn.Linear(
            parent_hidden_size,
            num_parents,
        )


        # ----------------------------------------------------
        # Child classifier
        #
        # Child classifier receives:
        #
        #     original h
        #
        # PLUS
        #
        #     parent_hidden
        #
        # Therefore input dimension is:
        #
        #     hidden_size + parent_hidden_size
        # ----------------------------------------------------

        child_input_size = (
            hidden_size
            + parent_hidden_size
        )


        self.child_classifier = nn.Linear(
            child_input_size,
            num_children,
        )


        # ----------------------------------------------------
        # Store dimensions
        # ----------------------------------------------------

        self.num_parents = (
            num_parents
        )

        self.num_children = (
            num_children
        )

        self.parent_hidden_size = (
            parent_hidden_size
        )


        # ----------------------------------------------------
        # Loss weights
        # ----------------------------------------------------

        self.parent_weight = (
            parent_weight
        )

        self.child_weight = (
            child_weight
        )


        # ----------------------------------------------------
        # Multilabel loss
        # ----------------------------------------------------

        self.loss_fn = (
            BCEWithLogitsLoss()
        )


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):

        # ====================================================
        # 1. XLM-R
        # ====================================================

        outputs = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask,
        )


        # ====================================================
        # 2. Shared representation h
        #
        # XLM-R first-token representation.
        #
        # Shape:
        #
        #     [batch_size, hidden_size]
        #
        # For XLM-R-large:
        #
        #     [batch_size, 1024]
        # ====================================================

        h = (
            outputs.last_hidden_state[:, 0]
        )


        # ====================================================
        # 3. Parent representation
        #
        # h
        #  ↓
        # Linear projection
        #  ↓
        # GELU
        #  ↓
        # parent_hidden
        #
        # Shape:
        #
        #     [batch_size, 256]
        # ====================================================

        parent_hidden = (
            self.parent_projection(h)
        )


        parent_hidden = (
            self.parent_activation(
                parent_hidden
            )
        )


        # ====================================================
        # 4. Parent logits
        #
        # parent_hidden
        #       ↓
        # parent classifier
        #       ↓
        # 9 parent logits
        # ====================================================

        parent_logits = (
            self.parent_classifier(
                parent_hidden
            )
        )


        # ====================================================
        # 5. Build child representation
        #
        # IMPORTANT:
        #
        # We do NOT use hard parent predictions.
        #
        # Instead, we concatenate:
        #
        #     h
        #
        # with:
        #
        #     parent_hidden
        #
        # Therefore the child classifier has access to:
        #
        #     - original XLM-R information
        #     - parent-task information
        # ====================================================

        child_input = torch.cat(

            [
                h,
                parent_hidden,
            ],

            dim=1,
        )


        # ====================================================
        # 6. Child logits
        #
        # child_input
        #      ↓
        # child classifier
        #      ↓
        # 16 child logits
        # ====================================================

        child_logits = (
            self.child_classifier(
                child_input
            )
        )


        # ====================================================
        # 7. Combine logits
        #
        # Output order:
        #
        #     [9 parent logits |
        #      16 child logits]
        #
        # Total = 25
        # ====================================================

        all_logits = torch.cat(

            [
                parent_logits,
                child_logits,
            ],

            dim=1,
        )


        # ====================================================
        # 8. Loss
        # ====================================================

        loss = None


        if labels is not None:

            # ------------------------------------------------
            # Parent labels
            # ------------------------------------------------

            parent_labels = labels[
                :,
                :self.num_parents
            ].float()


            # ------------------------------------------------
            # Child labels
            # ------------------------------------------------

            child_labels = labels[
                :,
                self.num_parents:
            ].float()


            # ------------------------------------------------
            # Parent loss
            # ------------------------------------------------

            parent_loss = self.loss_fn(

                parent_logits,

                parent_labels,
            )


            # ------------------------------------------------
            # Child loss
            # ------------------------------------------------

            child_loss = self.loss_fn(

                child_logits,

                child_labels,
            )


            # ------------------------------------------------
            # Weighted hierarchical loss
            #
            # L =
            #
            # parent_weight * parent_loss
            #
            # +
            #
            # child_weight * child_loss
            # ------------------------------------------------

            loss = (

                self.parent_weight
                * parent_loss

                +

                self.child_weight
                * child_loss
            )


        # ====================================================
        # Return
        # ====================================================

        return {

            "loss": loss,

            "logits": all_logits,
        }


# ============================================================
# Model factory
#
# Every Optuna trial gets a completely fresh encoder.
# ============================================================

def create_model(
    parent_weight=1.0,
    child_weight=1.0,
    seed=SEED,
):

    print(
        "\nLoading encoder for new model..."
    )


    # --------------------------------------------------------
    # Set seed before creating model
    # --------------------------------------------------------

    set_seed(seed)


    # --------------------------------------------------------
    # Load pretrained encoder
    # --------------------------------------------------------

    encoder = AutoModel.from_pretrained(

        MODEL_ID,

        cache_dir=str(HF_CACHE),
    )


    print(
        "Encoder model type:",
        encoder.config.model_type,
    )


    print(
        "Encoder hidden size:",
        encoder.config.hidden_size,
    )


    # --------------------------------------------------------
    # Create hierarchical model
    # --------------------------------------------------------

    model = HierarchicalXLMR(

        encoder=encoder,

        num_parents=len(parents),

        num_children=len(children),

        parent_weight=parent_weight,

        child_weight=child_weight,

        parent_hidden_size=(
            PARENT_HIDDEN_SIZE
        ),
    )


    return model


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    eval_prediction
):

    logits = (
        eval_prediction.predictions
    )

    labels = (
        eval_prediction.label_ids
    )


    # --------------------------------------------------------
    # Sigmoid
    #
    # Multilabel classification:
    #
    # DO NOT use softmax.
    # --------------------------------------------------------

    probabilities = 1 / (

        1
        + np.exp(-logits)
    )


    predictions = (

        probabilities
        >= THRESHOLD
    ).astype(np.int32)


    # ========================================================
    # Parent metrics
    # ========================================================

    parent_labels = labels[
        :,
        :len(parents)
    ]


    parent_predictions = predictions[
        :,
        :len(parents)
    ]


    parent_macro_f1 = f1_score(

        parent_labels,

        parent_predictions,

        average="macro",

        zero_division=0,
    )


    parent_micro_f1 = f1_score(

        parent_labels,

        parent_predictions,

        average="micro",

        zero_division=0,
    )


    # ========================================================
    # Child metrics
    # ========================================================

    child_labels = labels[
        :,
        len(parents):
    ]


    child_predictions = predictions[
        :,
        len(parents):
    ]


    child_macro_f1 = f1_score(

        child_labels,

        child_predictions,

        average="macro",

        zero_division=0,
    )


    child_micro_f1 = f1_score(

        child_labels,

        child_predictions,

        average="micro",

        zero_division=0,
    )


    # ========================================================
    # Overall metrics
    # ========================================================

    macro_f1 = f1_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,
    )


    micro_f1 = f1_score(

        labels,

        predictions,

        average="micro",

        zero_division=0,
    )


    return {

        "macro_f1":
            macro_f1,

        "micro_f1":
            micro_f1,

        "parent_macro_f1":
            parent_macro_f1,

        "parent_micro_f1":
            parent_micro_f1,

        "child_macro_f1":
            child_macro_f1,

        "child_micro_f1":
            child_micro_f1,
    }


# ============================================================
# Optuna pruning callback
# ============================================================

class OptunaPruningCallback(
    TrainerCallback
):

    def __init__(
        self,
        trial,
    ):

        self.trial = trial


    def on_evaluate(

        self,

        args,

        state,

        control,

        metrics=None,

        **kwargs,
    ):

        if metrics is None:

            return control


        # ----------------------------------------------------
        # Optimize child macro F1
        # ----------------------------------------------------

        metric = metrics.get(
            "eval_child_macro_f1"
        )


        if metric is None:

            return control


        self.trial.report(

            metric,

            step=state.epoch,
        )


        if self.trial.should_prune():

            raise optuna.TrialPruned()


        return control


# ============================================================
# Optuna objective
# ============================================================

def objective(trial):

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    learning_rate = (
        trial.suggest_float(
            "learning_rate",
            5e-6,
            3e-5,
            log=True,
        )
    )


    # --------------------------------------------------------
    # Weight decay
    # --------------------------------------------------------

    weight_decay = (
        trial.suggest_float(
            "weight_decay",
            0.0,
            0.1,
        )
    )


    # --------------------------------------------------------
    # Warmup ratio
    # --------------------------------------------------------

    warmup_ratio = (
        trial.suggest_float(
            "warmup_ratio",
            0.0,
            0.15,
        )
    )


    # --------------------------------------------------------
    # Parent loss weight
    # --------------------------------------------------------

    parent_weight = (
        trial.suggest_float(
            "parent_weight",
            0.25,
            2.0,
            log=True,
        )
    )


    # --------------------------------------------------------
    # Child loss weight
    # --------------------------------------------------------

    child_weight = (
        trial.suggest_float(
            "child_weight",
            0.25,
            2.0,
            log=True,
        )
    )


    # --------------------------------------------------------
    # Gradient accumulation
    # --------------------------------------------------------

    grad_accum = (
        trial.suggest_categorical(
            "grad_accum",
            [4, 8],
        )
    )


    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    set_seed(SEED)

    model = create_model(
        parent_weight=parent_weight,
        child_weight=child_weight,
        seed=SEED,
    )


    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    args = TrainingArguments(

        output_dir=str(
            HF_CACHE
            / "optuna"
            / dataset_name
            / f"trial_{trial.number}"
        ),

        overwrite_output_dir=True,

        num_train_epochs=10,

        per_device_train_batch_size=16,

        per_device_eval_batch_size=16,

        gradient_accumulation_steps=(
            grad_accum
        ),

        learning_rate=learning_rate,

        weight_decay=weight_decay,

        warmup_ratio=warmup_ratio,

        eval_strategy="epoch",

        logging_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=1,

        load_best_model_at_end=True,

        metric_for_best_model=(
            "eval_child_macro_f1"
        ),

        greater_is_better=True,

        report_to="none",

        seed=SEED,

        bf16=True,
    )


    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        tokenizer=tokenizer,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3
            ),

            OptunaPruningCallback(
                trial
            ),
        ],
    )


    try:

        trainer.train()

        best_metric = (
            trainer.state.best_metric
        )

        return best_metric


    finally:

        shutil.rmtree(
            args.output_dir,
            ignore_errors=True,
        )

# ============================================================
# Run Optuna
# ============================================================

print("\n" + "=" * 100)
print("STARTING OPTUNA")
print("=" * 100)


study = optuna.create_study(

    direction="maximize",

    pruner=optuna.pruners.MedianPruner(

        n_startup_trials=2,

        n_warmup_steps=2,
    ),
)


# ------------------------------------------------------------
# For the FIRST debugging run:
#
#     n_trials=1
#
# Once everything works:
#
#     n_trials=8
#
# ------------------------------------------------------------

study.optimize(
    objective,
    n_trials=8,
)


# ============================================================
# Best Optuna parameters
# ============================================================

best_params = (
    study.best_trial.params
)


print("\n" + "=" * 100)
print("BEST OPTUNA TRIAL")
print("=" * 100)

print(
    "Best trial:",
    study.best_trial.number,
)

print(
    "Best child macro F1:",
    study.best_value,
)

print("\nBest parameters:")

for key, value in best_params.items():

    print(
        f"{key}: {value}"
    )

print("=" * 100)


# ============================================================
# Save Optuna parameters
# ============================================================

with open(
    OUTPUT_DIR
    / "best_params.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_params,
        f,
        indent=2,
    )




# ============================================================
# Final training arguments
# ============================================================

final_args = TrainingArguments(

    output_dir=str(
        HF_CACHE
        / "final_model"
        / dataset_name
    ),


    # Training
    num_train_epochs=10,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    gradient_accumulation_steps=(
        best_params["grad_accum"]
    ),


    # Optimization
    learning_rate=(
        best_params["learning_rate"]
    ),

    weight_decay=(
        best_params["weight_decay"]
    ),

    warmup_ratio=(
        best_params["warmup_ratio"]
    ),


    # Evaluation
    eval_strategy="epoch",

    save_strategy="epoch",


    # Checkpoints
    save_total_limit=1,

    load_best_model_at_end=True,


    # Optimize child macro F1
    metric_for_best_model=(
        "eval_child_macro_f1"
    ),

    greater_is_better=True,


    # Other
    report_to="none",

    bf16=True,

    seed=SEED,
)

set_seed(SEED)

final_model = create_model(
    parent_weight=best_params["parent_weight"],
    child_weight=best_params["child_weight"],
    seed=SEED,
)

# ============================================================
# Final Trainer
# ============================================================

final_trainer = Trainer(

    model=create_model(
        parent_weight=best_params["parent_weight"],
        child_weight=best_params["child_weight"],
    ),

    args=final_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    data_collator=data_collator,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ],
)


# ============================================================
# Final training
# ============================================================

print("\n" + "=" * 100)
print("FINAL TRAINING")
print("=" * 100)

print("\nBest Optuna parameters:")
print(best_params)

print(
    "\nTraining final hierarchical model..."
)

final_trainer.train()


# ============================================================
# Evaluate final model on DEV
# ============================================================

print("\n" + "=" * 100)
print("FINAL DEV EVALUATION")
print("=" * 100)

dev_metrics = final_trainer.evaluate(
    eval_dataset=validation_dataset
)

print("\nDev metrics:")

for key, value in dev_metrics.items():

    if isinstance(value, (float, int)):
        print(
            f"{key}: {value:.4f}"
        )
    else:
        print(
            f"{key}: {value}"
        )


# ============================================================
# Save DEV predictions
# ============================================================

print("\nSaving DEV predictions...")

dev_output = final_trainer.predict(
    validation_dataset
)

dev_logits = dev_output.predictions
dev_labels = dev_output.label_ids


np.save(
    OUTPUT_DIR / "dev_logits.npy",
    dev_logits,
)

np.save(
    OUTPUT_DIR / "dev_labels.npy",
    dev_labels,
)


# Save label names/order
with open(
    OUTPUT_DIR / "hierarchical_labels.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        hierarchical_labels,
        f,
        indent=2,
    )


print(
    "\nDEV logits shape:",
    dev_logits.shape,
)

print(
    "DEV labels shape:",
    dev_labels.shape,
)


# ============================================================
# Save best model
# ============================================================

print("\nSaving final model...")

FINAL_MODEL_DIR = (
    OUTPUT_DIR / "final_model"
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


final_trainer.save_model(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)


print(
    "\nModel saved to:"
)

print(
    FINAL_MODEL_DIR
)


# ============================================================
# Final TEST prediction
# ============================================================

print("\n" + "=" * 100)
print("FINAL TEST EVALUATION")
print("=" * 100)

test_output = final_trainer.predict(
    test_dataset
)

test_logits = test_output.predictions
test_labels = test_output.label_ids


print(
    "\nTEST logits shape:",
    test_logits.shape,
)

print(
    "TEST labels shape:",
    test_labels.shape,
)


# ============================================================
# Convert logits → probabilities
# ============================================================

test_probabilities = 1 / (
    1 + np.exp(-test_logits)
)


# ============================================================
# Threshold
# ============================================================

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(np.int32)


# ============================================================
# Parent metrics
# ============================================================

parent_test_labels = (
    test_labels[:, :len(parents)]
)

parent_test_predictions = (
    test_predictions[:, :len(parents)]
)


parent_macro_f1 = f1_score(
    parent_test_labels,
    parent_test_predictions,
    average="macro",
    zero_division=0,
)


parent_micro_f1 = f1_score(
    parent_test_labels,
    parent_test_predictions,
    average="micro",
    zero_division=0,
)


# ============================================================
# Child metrics
# ============================================================

child_test_labels = (
    test_labels[:, len(parents):]
)

child_test_predictions = (
    test_predictions[:, len(parents):]
)


child_macro_f1 = f1_score(
    child_test_labels,
    child_test_predictions,
    average="macro",
    zero_division=0,
)


child_micro_f1 = f1_score(
    child_test_labels,
    child_test_predictions,
    average="micro",
    zero_division=0,
)


# ============================================================
# Overall hierarchical metrics
# ============================================================

overall_macro_f1 = f1_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0,
)


overall_micro_f1 = f1_score(
    test_labels,
    test_predictions,
    average="micro",
    zero_division=0,
)


overall_weighted_f1 = f1_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0,
)


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 100)
print("TEST RESULTS")
print("=" * 100)

print(
    f"\nParent Macro-F1:   {parent_macro_f1:.4f}"
)

print(
    f"Parent Micro-F1:   {parent_micro_f1:.4f}"
)

print(
    f"\nChild Macro-F1:    {child_macro_f1:.4f}"
)

print(
    f"Child Micro-F1:    {child_micro_f1:.4f}"
)

print(
    f"\nOverall Macro-F1:  {overall_macro_f1:.4f}"
)

print(
    f"Overall Micro-F1:  {overall_micro_f1:.4f}"
)

print(
    f"Overall Weighted-F1: {overall_weighted_f1:.4f}"
)


# ============================================================
# Classification report
# ============================================================

print(
    "\nGenerating classification report..."
)

report = classification_report(
    test_labels,
    test_predictions,
    target_names=hierarchical_labels,
    zero_division=0,
    output_dict=True,
)


# ============================================================
# Save classification report
# ============================================================

report_file = (
    OUTPUT_DIR
    / "test_classification_report.json"
)

with open(
    report_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


report_df = pd.DataFrame(
    report
).transpose()


report_csv = (
    OUTPUT_DIR
    / "test_classification_report.csv"
)

report_df.to_csv(
    report_csv,
    encoding="utf-8-sig",
)


# ============================================================
# Save test predictions
# ============================================================

np.save(
    OUTPUT_DIR / "test_logits.npy",
    test_logits,
)

np.save(
    OUTPUT_DIR / "test_labels.npy",
    test_labels,
)

np.save(
    OUTPUT_DIR / "test_probabilities.npy",
    test_probabilities,
)

np.save(
    OUTPUT_DIR / "test_predictions.npy",
    test_predictions,
)


# ============================================================
# Save summary
# ============================================================

summary = {
    "threshold": THRESHOLD,

    "parent_macro_f1": float(
        parent_macro_f1
    ),

    "parent_micro_f1": float(
        parent_micro_f1
    ),

    "child_macro_f1": float(
        child_macro_f1
    ),

    "child_micro_f1": float(
        child_micro_f1
    ),

    "overall_macro_f1": float(
        overall_macro_f1
    ),

    "overall_micro_f1": float(
        overall_micro_f1
    ),

    "overall_weighted_f1": float(
        overall_weighted_f1
    ),

    "num_parents": len(parents),

    "num_children": len(children),

    "hierarchical_labels": hierarchical_labels,

    "best_params": best_params,

    "parent_weight": float(
        best_params["parent_weight"]
    ),

    "child_weight": float(
        best_params["child_weight"]
    ),

    "parent_hidden_size": PARENT_HIDDEN_SIZE,

}


with open(
    OUTPUT_DIR / "test_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)
