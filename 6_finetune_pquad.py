import argparse
import json
import math
from pathlib import Path
from transformers import AutoConfig

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import PreTrainedModel, GPT2Config
from datasets import load_metric

BASE_DIR = Path(
    "/scratch/project_462001491/nima"
)

MODEL_DIR = (
    BASE_DIR
    / "corpus_experiment"
    / "lm_results"
)

QA_RESULT_DIR = (
    BASE_DIR
    / "corpus_experiment"
    / "qa_results"
)

QA_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TOKENIZER_NAME = (
    "Qwen/Qwen2.5-0.5B"
)


MAX_LENGTH = 512
DOC_STRIDE = 128
SEED = 1234

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "https://huggingface.co/datasets/SajjadAyoubi/persian_qa/resolve/refs%2Fconvert%2Fparquet/persian_qa/train/0000.parquet",
        "validation": "https://huggingface.co/datasets/SajjadAyoubi/persian_qa/resolve/refs%2Fconvert%2Fparquet/persian_qa/validation/0000.parquet",
    }
)

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--corpus",
    required=True,
    choices=[
        "hplt3",
        "random20",
        "perref",
    ]
)
args = parser.parse_args()


import torch.nn as nn


from transformers.modeling_outputs import QuestionAnsweringModelOutput

from transformers import PreTrainedModel, GPT2Config

class GPT2ForExtractiveQA(PreTrainedModel):
    config_class = GPT2Config
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2LMHeadModel(config).transformer
        hidden_size = config.n_embd
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits = logits[..., 0]
        end_logits = logits[..., 1]

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token





def prepare_validation_features(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    example_ids = []

    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]

        example_ids.append(
            examples["id"][sample_idx]
            if "id" in examples
            else str(sample_idx)
        )

        sequence_ids = tokenized.sequence_ids(i)

        offsets = tokenized["offset_mapping"][i]

        tokenized["offset_mapping"][i] = [
            offset if sequence_ids[k] == 1 else None
            for k, offset in enumerate(offsets)
        ]

    tokenized["example_id"] = example_ids

    return tokenized


train_features = train_dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=train_dataset.column_names,
)

output_dir = (
    QA_RESULT_DIR / args.corpus
)


training_args = TrainingArguments(
    output_dir=str(output_dir),

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    warmup_ratio=0.1,

    logging_steps=50,

    save_strategy="epoch",

    eval_strategy="epoch",

    fp16=torch.cuda.is_available(),

    report_to="none",

    seed=SEED,

    data_seed=SEED,

    remove_unused_columns=False,

    device="cuda" if torch.cuda.is_available() else "cpu",
)


# ============================================================
# LOAD PRETRAINED CHECKPOINT
# ============================================================

def find_final_checkpoint(corpus_name):

    corpus_dir = MODEL_DIR / corpus_name

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist:\n{corpus_dir}"
        )

    checkpoints = []

    for path in corpus_dir.glob("checkpoint-*"):

        if not path.is_dir():
            continue

        try:
            step = int(
                path.name.split("-")[1]
            )
        except (ValueError, IndexError):
            continue

        checkpoints.append(
            (step, path)
        )

    if not checkpoints:
        raise RuntimeError(
            f"No checkpoints found in:\n{corpus_dir}"
        )

    checkpoints.sort(
        key=lambda x: x[0]
    )

    return checkpoints[-1][1]


checkpoint = find_final_checkpoint(
    args.corpus
)

set_seed(SEED)

config = AutoConfig.from_pretrained(checkpoint)
model = GPT2ForExtractiveQA(config)

# Load pretrained weights
pretrained_model = GPT2LMHeadModel.from_pretrained(checkpoint)
model.transformer.load_state_dict(
    pretrained_model.transformer.state_dict()
)



print()
print("=" * 70)
print("LOADING PRETRAINED CHECKPOINT")
print("=" * 70)

print(
    f"Corpus: {args.corpus}"
)

print(
    f"Checkpoint: {checkpoint}"
)



validation_features = validation_dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=validation_dataset.column_names,
)

# ============================================================
# TRAINER
# ============================================================

from datasets import load_metric

metric = load_metric("squad")

def compute_metrics(p):
    predictions, label_ids = p
    
    # predictions is a tuple: (start_logits, end_logits)
    start_logits = predictions[0]
    end_logits = predictions[1]
    
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)
    
    # For SQuAD metric, you need actual answers from the dataset
    # This is complex - consider using a simpler metric first
    
    # For now, return F1 score calculation:
    f1_scores = []
    for i, (start, end) in enumerate(zip(start_preds, end_preds)):
        if start > end:
            f1_scores.append(0.0)
        else:
            f1_scores.append(1.0)  # Placeholder
    
    return {"f1": np.mean(f1_scores)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_features,
    eval_dataset=validation_features,
    compute_metrics=compute_metrics,
)

# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING QA TRAINING")
print("=" * 70)

trainer.train()
