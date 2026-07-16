#Avoid threshold tuning as it is now giving 0.005 and only focusing on one label for better results... maybe it works after adding more JUNKS
#from transformers import (
#    Trainer,
#    TrainingArguments,
#    AutoModelForSequenceClassification,
#    AutoTokenizer,
#)

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    classification_report,
)
import json
from scipy.special import softmax

#MODEL_NAME = "alrazz-n/bge-m3_spamham_best_model"

MODEL_NAME = "BAAI/bge-m3-retromae"

MODEL_ID = MODEL_NAME.split("/")[-1]
SAVE_NAME = f"{MODEL_ID}_spamham"

SAVE_DIR = f"/scratch/project_2005092/nima/saved_models/{SAVE_NAME}" #where they are stored

# Threshold tuning configuration
MIN_RECALL_1 = 0.90
NUM_THRESHOLDS = 1001 #how many possible threshold values to test

logits = np.load(f"{SAVE_DIR}/dev_logits.npy")
labels = np.load(f"{SAVE_DIR}/dev_labels.npy")


def probs_class0_from_logits(logits):
    return softmax(logits, axis=-1)[:, 0]

# def probs_class0_from_logits(logits):
    # logits: (N,2)
    #exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    #probs = exp / exp.sum(axis=-1, keepdims=True)
    #return probs[:, 0]

def tune_threshold_recall0_under_keep90(
    logits,
    labels,
    thresholds=None,
    min_recall1=MIN_RECALL_1,
):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, NUM_THRESHOLDS)


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
    logits=logits,
    labels=labels,
    thresholds=np.linspace(0,1,NUM_THRESHOLDS),
    min_recall1=MIN_RECALL_1
)

if best is None:
    raise RuntimeError(
        "No threshold achieved recall_1 >= 0.90"
    )

threshold = best["threshold"]

print(best)


# Save for future HPLT annotation

best["constraint"] =  f"recall_1 >= {MIN_RECALL_1}"
best["num_thresholds"] = NUM_THRESHOLDS
best["num_samples"] = len(labels)
best["class_distribution"] = {
    "class_0": int(np.sum(labels == 0)),
    "class_1": int(np.sum(labels == 1)),
}

with open(f"{SAVE_DIR}/threshold.json", "w") as f:
    json.dump(best, f, indent=2)

#########################


y_true = labels

p0 = probs_class0_from_logits(logits)
y_pred = np.where(p0 >= threshold, 0, 1)
#y_pred = np.argmax(pred.predictions, axis=-1)

print(confusion_matrix(y_true, y_pred))

print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)


print(np.bincount(y_true))

p0 = probs_class0_from_logits(logits)

dataset = load_from_disk(
    "/scratch/project_2005092/nima/binary_dataset"
)

dev_texts = dataset["validation"]["text"]

#Error Analysis
print("\nLowest P(class0) examples:")
for i in np.argsort(p0)[:10]:
    print("P(class0):", p0[i])
    print("Label:", labels[i])
    print(dev_texts[i][:300])
    print("----")