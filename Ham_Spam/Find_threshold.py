#Avoid threshold tuning as it is now giving 0.005 and only focusing on one label for better results... maybe it works after adding more JUNKS
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import recall_score
import json

MODEL_NAME = "alrazz-n/bge-m3_spamham_best_model"


# -------------------------
# Load model and tokenizer
# -------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# -------------------------
# Load validation dataset
# -------------------------

dataset = load_from_disk(
    "/scratch/project_2005092/nima/binary_dataset"
)

dev_dataset = dataset["validation"]

dev_dataset = dev_dataset.rename_column(
    "Binary",
    "labels"
)


# -------------------------
# Tokenize validation data
# -------------------------

def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )


dev_dataset = dev_dataset.map(
    tokenize,
    batched=True
)


dev_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)


# -------------------------
# Run threshold tuning
# -------------------------

inf_args = TrainingArguments(
    output_dir="/scratch/project_2005092/nima/tmp_threshold",
    per_device_eval_batch_size=16,
    bf16=True,
    report_to="none",
    save_strategy="no",
)


dev_trainer = Trainer(
    model=model,
    args=inf_args,
    tokenizer=tokenizer,
)

def probs_class0_from_logits(logits):
    # logits: (N,2)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs[:, 0]

def tune_threshold_recall0_under_keep90(trainer, dev_dataset, thresholds=None, min_recall1=0.90):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    out = trainer.predict(dev_dataset)
    logits = out.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]  # take first element if needed

    labels = out.label_ids  # 0/1

    p0 = probs_class0_from_logits(logits)  # P(class=0=junk)

    best = None
    for t in thresholds:
        pred = np.where(p0 >= t, 0, 1)  # predict junk=0 if confidence junk is high

        recall1 = recall_score(labels, pred, pos_label=1)
        if recall1 >= min_recall1:
            recall0 = recall_score(labels, pred, pos_label=0)
            if best is None or recall0 > best["recall_0"]:
                best = {"threshold": float(t), "recall_0": float(recall0), "recall_1": float(recall1)}

    return best


best = tune_threshold_recall0_under_keep90(
    trainer=dev_trainer,
    dev_dataset=dev_dataset,
    thresholds=np.linspace(0,1,201),
    min_recall1=0.90
)

if best is None:
    raise RuntimeError(
        "No threshold achieved recall_1 >= 0.90"
    )

threshold = best["threshold"]

print(model.config.id2label)
print(best)


# Save for future HPLT annotation
with open("threshold.json", "w") as f:
    json.dump(best, f, indent=2)

#########################

from sklearn.metrics import confusion_matrix, classification_report

pred = dev_trainer.predict(dev_dataset)

y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=-1)

print(confusion_matrix(y_true, y_pred))

print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)

import numpy as np

print(np.bincount(y_true))


logits = pred.predictions

p0 = probs_class0_from_logits(logits)

for i in np.argsort(p0)[-10:]:
    print("P(class0):", p0[i])
    print(dev_dataset[i]["labels"])
    print(dev_dataset[i]["text"][:300])
    print("----")