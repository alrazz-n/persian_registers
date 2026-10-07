"""
Hierarchical XLM-R with a Parent-Specific
Mixture-of-Experts (MoE).

Architecture
    XLM-R encoder
          │
    h = last_hidden_state[:, 0]   (1024)
          │
     ┌────┴───────────────────────────────┐
     │                                     │
     ▼                                     ▼
 PARENT HEAD                         6 PARENT-SPECIFIC
 Linear 1024→256                     EXPERTS
      │                              SP, NA, HI, IN, OP, IP
    GELU                                  │
      │                                   │
 Linear 256→9                             │
      │                              each expert:
      ▼                              Linear 1024→256
 parent logits (9)                        │
      │                                  GELU
 sigmoid                                  │
      │                               Dropout(0.1)
      ▼                                   │
 parent probabilities p (9)           Linear 256→256
      │                                  │
      │                                 GELU
      │                                  │
      │                                  ▼
      │                           expert representation
      │                              (256 each)
      │                                  │
      ├───────────────┐                  │
      │               │                  │
      ▼               ▼                  │
 TRAINING          EVAL / TEST           │
 w = p             w = 1[p ≥ 0.5]        │
      │               │                  │
      │               │                  │
      └───────┬───────┘                  │
              │                          │
              ▼                          │
   Zero weights for parents              │
   without experts:                      │
   MT, LY, ID                            │
              │                          │
              ▼                          │
      normalize weights                  │
          w / Σw                         │
      (Σw clamped ≥ 1e-6)                │
              │                          │
              └──────────┬───────────────┘
                         ▼
                  WEIGHTED MIXTURE
                  h_mix = Σ w_p E_p(h)
                         │
                         ▼
               Shared child classifier
                    Linear 256→16
                         │
                         ▼
                  child logits (16)
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       parent logits             child logits
             │                       │
             └───────────┬───────────┘
                         ▼
                 25 total logits
                         │
             ┌───────────┴────────────┐
             ▼                        ▼
        TRAINING                 EVAL / TEST
             │                        │
       BCE(parent)              parents:
             │                  sigmoid ≥ 0.5
       BCE(child)                    │
             │                  children:
       parent_weight ×           sigmoid ≥ 0.5
       parent BCE               independently
             +                       │
       child_weight ×                │
       child BCE                     │
             │                       │
             ▼                       ▼
          LOSS                  PREDICTIONS
```
"""





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

print("=" * 100)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
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

    for child_list
    in labels_structure.values()

    for child
    in child_list
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
# Parent -> child index mapping
#
# This is useful when constructing child logits in the
# exact required order.
# ============================================================

parent_to_children = {

    parent: list(child_list)

    for parent, child_list
    in labels_structure.items()
}


# ============================================================
# Complete hierarchical label list
# ============================================================

hierarchical_labels = (
    parents + children
)


# ============================================================
# Print hierarchy
# ============================================================

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


print("\nParent -> children:")

for parent in parents:

    print(
        f"{parent}: "
        f"{parent_to_children[parent]}"
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
    / "_MultiCore_ParentMoE_XLMR_SoftTrain_HardInference"
)



RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Pretrained multilingual model
# ============================================================

MODEL_ID = (
    "TurkuNLP/web-register-classification-multilingual"
)


# ============================================================
# Tokenization
# ============================================================

MAX_LENGTH = 512


# ============================================================
# Default threshold
#
# Threshold tuning can be done later using saved
# probabilities.
# ============================================================

THRESHOLD = 0.5

# TRAINING:
#     Parent probabilities determine soft routing.
#
#     Every expert processes every example.
#     The predicted parent probability controls
#     the contribution of that expert.
#
#     This is differentiable, so child loss can
#     backpropagate into the parent classifier.
#
# DEV / TEST:
#     Predicted parent probabilities determine
#     hard routing.


PARENT_ROUTING_THRESHOLD = 0.5


# ============================================================
# Child prediction threshold
# ============================================================

CHILD_PREDICTION_THRESHOLD = 0.5


# ============================================================
# Representation sizes
# ============================================================

PARENT_HIDDEN_SIZE = 256

EXPERT_HIDDEN_SIZE = 256


# ============================================================
# Expert dropout
# ============================================================

EXPERT_DROPOUT = 0.1


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

        for path
        in DATASET_ROOT.iterdir()

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

    for label
    in hierarchical_labels

    if label
    not in dataset_label_to_index

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
# Save hierarchy metadata
# ============================================================

with open(

    OUTPUT_DIR
    / "labels_structure.json",

    "w",

    encoding="utf-8",

) as f:

    json.dump(

        labels_structure,

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
    # Convert original label vector into dictionary
    # --------------------------------------------------------

    label_values = {

        label:

            original_labels[
                dataset_label_to_index[label]
            ]

        for label
        in dataset_labels

    }


    hierarchical_target = []


    # ========================================================
    # Parents
    # ========================================================

    for parent in parents:

        parent_value = (
            label_values[parent]
        )


        # ----------------------------------------------------
        # Parent is positive if:
        #
        #   1. it is explicitly annotated positive
        #
        # OR
        #
        #   2. any of its children is positive
        # ----------------------------------------------------

        for child in labels_structure[parent]:

            if (
                label_values[child]
                > 0
            ):

                parent_value = 1.0


        hierarchical_target.append(
            parent_value
        )


    # ========================================================
    # Children
    # ========================================================

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
    # Hierarchical labels
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
# Parent-Specific Expert
# ============================================================

class ParentExpert(nn.Module):

    """
    Parent-specific child expert.

    Input:
        h: [batch_subset, encoder_hidden_size]

    Output:
        expert_hidden:
            [batch_subset, expert_hidden_size]
    """

    def __init__(
        self,
        input_size,
        expert_hidden_size=256,
        dropout=0.1,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                expert_hidden_size,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                expert_hidden_size,
                expert_hidden_size,
            ),

            nn.GELU(),

        )


    def forward(self, h):

        return self.network(h)


# ============================================================
# Hierarchical XLM-R with Soft Training Routing
# and Hard Inference Routing
# ============================================================


class HierarchicalParentMoEXLMR(nn.Module):

    """
Soft multi-expert routing during training and thresholded hard multi-expert routing during inference.

Training:
    parent probabilities -> differentiable routing

Evaluation/test:
    thresholded parent probabilities -> hard routing
    """



    def __init__(
        self,
        encoder,
        labels_structure,
        parent_weight=1.0,
        child_weight=1.0,
        parent_hidden_size=256,
        expert_hidden_size=256,
        expert_dropout=0.1,
        parent_routing_threshold=0.5,
    ):

        super().__init__()


        # ====================================================
        # Encoder
        # ====================================================

        self.encoder = encoder

        hidden_size = (
            encoder.config.hidden_size
        )


        # ====================================================
        # Hierarchy
        # ====================================================

        self.labels_structure = labels_structure


        self.parents = list(
            labels_structure.keys()
        )


        self.child_labels = [

            child

            for child_list
            in labels_structure.values()

            for child in child_list

        ]


        # ====================================================
        # Dimensions
        # ====================================================

        self.hidden_size = hidden_size

        self.expert_hidden_size = (
            expert_hidden_size
        )


        self.parent_routing_threshold = (
            parent_routing_threshold
        )


        # ====================================================
        # Parent classifier
        # ====================================================

        self.parent_projection = nn.Linear(

            hidden_size,

            parent_hidden_size,

        )


        self.parent_activation = nn.GELU()


        self.parent_classifier = nn.Linear(

            parent_hidden_size,

            len(self.parents),

        )


        # ====================================================
        # Experts
        # ====================================================

        self.experts = nn.ModuleDict()


        for parent, child_list in labels_structure.items():

            if len(child_list) == 0:
                continue


            self.experts[parent] = ParentExpert(

                input_size=hidden_size,

                expert_hidden_size=expert_hidden_size,

                dropout=expert_dropout,

            )


# ============================================================
# Child classifier
# ============================================================

        self.child_classifier = nn.Linear(
            expert_hidden_size,
            len(self.child_labels),
        )



        # ====================================================
        # Loss
        # ====================================================

        self.parent_weight = parent_weight

        self.child_weight = child_weight


        self.loss_fn = BCEWithLogitsLoss()



    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):


        # ====================================================
        # 1. Encoder
        # ====================================================

        outputs = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask,

        )


        h = outputs.last_hidden_state[:,0]


        batch_size = h.size(0)



        # ====================================================
        # 2. Parent classifier
        # ====================================================

        parent_hidden = self.parent_projection(h)

        parent_hidden = self.parent_activation(
            parent_hidden
        )


        parent_logits = self.parent_classifier(
            parent_hidden
        )


        parent_probabilities = torch.sigmoid(
            parent_logits
        )



        # ====================================================
        # 3. Compute every expert representation
        #
        # h_parent = Expert(h)
        # ====================================================

        expert_representations = {}


        for parent in self.experts:

            expert_representations[parent] = (
                self.experts[parent](h)
            )



        # ====================================================
        # 4. Router weights
        # ====================================================

        if self.training:


            # ----------------------------------------------
             #Soft multi-expert routing
            # ----------------------------------------------

            routing_weights = (
                parent_probabilities.clone()
            )


        else:


            # ----------------------------------------------
            # Hard multi-expert routing
            # ----------------------------------------------

            routing_weights = (

                parent_probabilities

                >= self.parent_routing_threshold

            ).float()



        # ====================================================
        # 5. Remove parents without experts
        # ====================================================

        for i, parent in enumerate(self.parents):

            if parent not in expert_representations:

                routing_weights[:,i] = 0.0



        # ====================================================
        # 6. Normalize routing weights
        #
        # sigmoid parents do not sum to 1
        # ====================================================

        weight_sum = (

            routing_weights.sum(

                dim=1,

                keepdim=True

            )

            .clamp_min(1e-6)

        )


        routing_weights = (

            routing_weights

            / weight_sum

        )



        # ====================================================
        # 7. Mixture of experts
        #
        # h_mix =
        # sum(weight * expert_representation)
        #
        # ====================================================

        mixture = torch.zeros(

            batch_size,

            self.expert_hidden_size,

            device=h.device,

            dtype=h.dtype,

        )


        for i, parent in enumerate(self.parents):


            if parent not in expert_representations:

                continue


            weight = (

                routing_weights[:,i]

                .unsqueeze(1)

            )


            mixture += (

                weight

                *

                expert_representations[parent]

            )



        # ====================================================
        # 8. Child logits
        # ====================================================

        child_logits = self.child_classifier(mixture)


        #child_logits = torch.full(

            #(

                #batch_size,

                #len(self.child_labels),

            #),

            #-5.0,

            #device=h.device,

            #dtype=parent_logits.dtype,

        #)


        #child_offset = 0



        #for parent, child_list in self.labels_structure.items():


            #if len(child_list)==0:

                #continue



            #logits = self.child_classifiers[parent](
               # expert_representations[parent]
            #)




          #  child_logits[
#
             #   :,

              #  child_offset:

                #child_offset + len(child_list)

           # ] = logits



          # child_offset += len(child_list)



        # ====================================================
        # 9. Combine logits
        # ====================================================

        all_logits = torch.cat(

            [

                parent_logits,

                child_logits,

            ],

            dim=1,

        )



        # ====================================================
        # 10. Loss
        # ====================================================

        loss = None


        if labels is not None:


            parent_labels = labels[

                :,

                :len(self.parents)

            ].float()



            child_labels = labels[

                :,

                len(self.parents):

            ].float()



            parent_loss = self.loss_fn(

                parent_logits,

                parent_labels,

            )



            child_loss = self.loss_fn(

                child_logits,

                child_labels,

            )



            loss = (

                self.parent_weight

                *

                parent_loss

                +

                self.child_weight

                *

                child_loss

            )



        return {

            "loss": loss,

            "logits": all_logits,

        }

# ============================================================
# Model factory
# ============================================================

def create_model(

    parent_weight=1.0,

    child_weight=1.0,

    seed=SEED,

):

    print(
        "\nLoading encoder for new hard-routed model..."
    )


    # --------------------------------------------------------
    # Seed
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
    # Create hard-routed model
    # --------------------------------------------------------

    model = HierarchicalParentMoEXLMR(

        encoder=encoder,

        labels_structure=labels_structure,

        parent_weight=parent_weight,

        child_weight=child_weight,

        parent_hidden_size=(
            PARENT_HIDDEN_SIZE
        ),

        expert_hidden_size=(
            EXPERT_HIDDEN_SIZE
        ),

        expert_dropout=(
            EXPERT_DROPOUT
        ),

        parent_routing_threshold=(
            PARENT_ROUTING_THRESHOLD
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
    # --------------------------------------------------------

    probabilities = (

        1.0
        /
        (
            1.0
            +
            np.exp(-logits)
        )

    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = (

        probabilities
        >= THRESHOLD

    ).astype(np.int32)

    parent_probabilities = probabilities[
        :,
        :len(parents)
    ]
        # ========================================================
    # Parent routing statistics
    # ========================================================

    routing_predictions = (
        parent_probabilities
        >= PARENT_ROUTING_THRESHOLD
    ).astype(np.int32)

    routing_metrics = {}

    for i, parent in enumerate(parents):

        routing_metrics[
            f"routing_{parent}"
        ] = float(
            routing_predictions[:, i].mean()
        )



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

        **routing_metrics,

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

    # ========================================================
    # Learning rate
    # ========================================================

    learning_rate = (

        trial.suggest_float(

            "learning_rate",

            5e-6,

            3e-5,

            log=True,

        )

    )


    # ========================================================
    # Weight decay
    # ========================================================

    weight_decay = (

        trial.suggest_float(

            "weight_decay",

            0.0,

            0.1,

        )

    )


    # ========================================================
    # Warmup
    # ========================================================

    warmup_ratio = (

        trial.suggest_float(

            "warmup_ratio",

            0.0,

            0.15,

        )

    )


    # ========================================================
    # Parent loss weight
    # ========================================================

    parent_weight = (

        trial.suggest_float(

            "parent_weight",

            0.25,

            2.0,

            log=True,

        )

    )


    # ========================================================
    # Child loss weight
    # ========================================================

    child_weight = (

        trial.suggest_float(

            "child_weight",

            0.25,

            2.0,

            log=True,

        )

    )


    # ========================================================
    # Gradient accumulation
    # ========================================================

    grad_accum = (

        trial.suggest_categorical(

            "grad_accum",

            [4, 8],

        )

    )


    # ========================================================
    # Fresh model
    # ========================================================

    set_seed(SEED)


    model = create_model(

        parent_weight=parent_weight,

        child_weight=child_weight,

        seed=SEED,

    )


    # ========================================================
    # Trial output directory
    # ========================================================

    trial_output_dir = (

        HF_CACHE

        / "optuna"

        / dataset_name

        / f"trial_{trial.number}"

    )


    # ========================================================
    # Training arguments
    # ========================================================

    args = TrainingArguments(

        output_dir=str(
            trial_output_dir
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


    # ========================================================
    # Trainer
    # ========================================================

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

        # ----------------------------------------------------
        # Remove trial checkpoint directory
        # ----------------------------------------------------

        shutil.rmtree(

            trial_output_dir,

            ignore_errors=True,

        )

        # ----------------------------------------------------
        # Release model
        # ----------------------------------------------------

        del trainer

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


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


# ============================================================
# Run trials
# ============================================================

study.optimize(

    objective,

    n_trials=8,

)


# ============================================================
# Best parameters
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
# Save Optuna study information
# ============================================================

study_summary = {

    "best_trial":

        study.best_trial.number,

    "best_value":

        float(study.best_value),

    "best_params":

        best_params,

}


with open(

    OUTPUT_DIR
    / "optuna_summary.json",

    "w",

    encoding="utf-8",

) as f:

    json.dump(

        study_summary,

        f,

        indent=2,

    )


# ============================================================
# Final Training Arguments
# ============================================================

final_args = TrainingArguments(

    output_dir=str(

        HF_CACHE

        / "final_model"

        / dataset_name

    ),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_train_epochs=10,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    gradient_accumulation_steps=(

        best_params["grad_accum"]

    ),

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    learning_rate=(

        best_params["learning_rate"]

    ),

    weight_decay=(

        best_params["weight_decay"]

    ),

    warmup_ratio=(

        best_params["warmup_ratio"]

    ),

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="epoch",

    logging_strategy="epoch",

    save_strategy="epoch",

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_total_limit=1,

    load_best_model_at_end=True,

    # --------------------------------------------------------
    # Optimize child macro F1
    # --------------------------------------------------------

    metric_for_best_model=(

        "eval_child_macro_f1"

    ),

    greater_is_better=True,

    # --------------------------------------------------------
    # Other
    # --------------------------------------------------------

    report_to="none",

    bf16=True,

    seed=SEED,

)


# ============================================================
# Create FINAL model
#
# IMPORTANT:
#
# This exact model is passed into Trainer below.
# ============================================================

set_seed(SEED)


final_model = create_model(

    parent_weight=(

        best_params["parent_weight"]

    ),

    child_weight=(

        best_params["child_weight"]

    ),

    seed=SEED,

)


# ============================================================
# Final Trainer
# ============================================================

final_trainer = Trainer(

    model=final_model,

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
    "\nTraining final hierarchical expert model..."
)


final_trainer.train()


# ============================================================
# Routing diagnostic
# ============================================================

def print_routing_statistics(

    labels,

    logits,

):

    parent_logits = logits[

        :,
        :len(parents)

    ]


    parent_probabilities = (

        1.0
        /
        (
            1.0
            +
            np.exp(-parent_logits)
        )

    )


    predicted_parent_active = (

        parent_probabilities
        >= PARENT_ROUTING_THRESHOLD

    )


    print(
        "\n" + "=" * 100
    )

    print(
        "PARENT ROUTING STATISTICS"
    )

    print(
        "=" * 100
    )


    for i, parent in enumerate(parents):

        count = int(

            predicted_parent_active[
                :,
                i
            ].sum()

        )


        percentage = (

            100.0
            *
            count
            /
            len(labels)

        )


        print(

            f"{parent:>3s}: "
            f"{count:6d} examples "
            f"({percentage:6.2f}%)"

        )


    print(
        "=" * 100
    )

# ============================================================
# Final DEV evaluation
# ============================================================

print("\n" + "=" * 100)
print("FINAL DEV EVALUATION")
print("=" * 100)


dev_metrics = final_trainer.evaluate(

    eval_dataset=validation_dataset

)


print("\nDev metrics:")


for key, value in dev_metrics.items():

    if isinstance(
        value,
        (float, int)
    ):

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

print(
    "\nSaving DEV predictions..."
)


dev_output = final_trainer.predict(

    validation_dataset

)


dev_logits = (
    dev_output.predictions
)


dev_labels = (
    dev_output.label_ids
)


np.save(

    OUTPUT_DIR
    / "dev_logits.npy",

    dev_logits,

)


np.save(

    OUTPUT_DIR
    / "dev_labels.npy",

    dev_labels,

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

print(
    "\nSaving final model..."
)


FINAL_MODEL_DIR = (

    OUTPUT_DIR
    / "final_model"

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


test_logits = (
    test_output.predictions
)


test_labels = (
    test_output.label_ids
)


print(

    "\nTEST logits shape:",

    test_logits.shape,

)


print(

    "TEST labels shape:",

    test_labels.shape,

)

print_routing_statistics(

    test_labels,

    test_logits,

)


# ============================================================
# Convert logits -> probabilities
# ============================================================

test_probabilities = (

    1.0
    /
    (
        1.0
        +
        np.exp(-test_logits)
    )

)


# ============================================================
# Threshold
# ============================================================

# ============================================================
# Parent predictions
# ============================================================

test_parent_probabilities = (

    test_probabilities[
        :,
        :len(parents)
    ]

)


test_parent_predictions = (

    test_parent_probabilities
    >= PARENT_ROUTING_THRESHOLD

).astype(np.int32)


# ============================================================
# Child predictions
#
# ============================================================

test_child_probabilities = (

    test_probabilities[
        :,
        len(parents):
    ]

)


test_child_predictions = (

    test_child_probabilities
    >= CHILD_PREDICTION_THRESHOLD

).astype(np.int32)


# ============================================================
# Reconstruct complete 25-label prediction matrix
# ============================================================

test_predictions = np.concatenate(

    [

        test_parent_predictions,

        test_child_predictions,

    ],

    axis=1,

).astype(np.int32)


# ============================================================
# Parent metrics
# ============================================================

parent_test_labels = (

    test_labels[
        :,
        :len(parents)
    ]

)


parent_test_predictions = (

    test_predictions[
        :,
        :len(parents)
    ]

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

    test_labels[
        :,
        len(parents):
    ]

)


child_test_predictions = (

    test_predictions[
        :,
        len(parents):
    ]

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
# Overall metrics
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

    f"\nParent Macro-F1:   "
    f"{parent_macro_f1:.4f}"

)


print(

    f"Parent Micro-F1:   "
    f"{parent_micro_f1:.4f}"

)


print(

    f"\nChild Macro-F1:    "
    f"{child_macro_f1:.4f}"

)


print(

    f"Child Micro-F1:    "
    f"{child_micro_f1:.4f}"

)


print(

    f"\nOverall Macro-F1:  "
    f"{overall_macro_f1:.4f}"

)


print(

    f"Overall Micro-F1:  "
    f"{overall_micro_f1:.4f}"

)


print(

    f"Overall Weighted-F1: "
    f"{overall_weighted_f1:.4f}"

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
# Save classification report JSON
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


# ============================================================
# Save classification report CSV
# ============================================================

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

    OUTPUT_DIR
    / "test_logits.npy",

    test_logits,

)


np.save(

    OUTPUT_DIR
    / "test_labels.npy",

    test_labels,

)


np.save(

    OUTPUT_DIR
    / "test_probabilities.npy",

    test_probabilities,

)


np.save(

    OUTPUT_DIR
    / "test_predictions.npy",

    test_predictions,

)

np.save(
    OUTPUT_DIR / "test_parent_probabilities.npy",
    test_parent_probabilities,
)

np.save(
    OUTPUT_DIR / "test_parent_predictions.npy",
    test_parent_predictions,
)



# ============================================================
# Save summary
# ============================================================

summary = {

    "model":

        "Hierarchical XLM-R + Parent-Specific Experts",

    "threshold":

        THRESHOLD,

    "parent_macro_f1":

        float(parent_macro_f1),

    "parent_micro_f1":

        float(parent_micro_f1),

    "child_macro_f1":

        float(child_macro_f1),

    "child_micro_f1":

        float(child_micro_f1),

    "overall_macro_f1":

        float(overall_macro_f1),

    "overall_micro_f1":

        float(overall_micro_f1),

    "overall_weighted_f1":

        float(overall_weighted_f1),

    "num_parents":

        len(parents),

    "num_children":

        len(children),

    "hierarchical_labels":

        hierarchical_labels,

    "labels_structure":

        labels_structure,

    "best_params":

        best_params,

    "parent_weight":

        float(
            best_params["parent_weight"]
        ),

    "child_weight":

        float(
            best_params["child_weight"]
        ),

    "parent_hidden_size":

        PARENT_HIDDEN_SIZE,

    "expert_hidden_size":

        EXPERT_HIDDEN_SIZE,

    "expert_dropout":

        EXPERT_DROPOUT,

}


with open(

    OUTPUT_DIR
    / "test_summary.json",

    "w",

    encoding="utf-8",

) as f:

    json.dump(

        summary,

        f,

        indent=2,

    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)


print(

    "\nResults saved to:"

)


print(
    OUTPUT_DIR
)
