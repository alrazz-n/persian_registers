from pathlib import Path
import json
import zstandard as zstd
from datasets import IterableDataset
import evaluate
import numpy as np


DATA_DIR = Path("/scratch/project_2005092/nima/annotated_data/kept")


def examples():
    files = sorted(DATA_DIR.glob("*.jsonl.zst"))

    print(f"Found {len(files)} shards")

    for file in files:
        print(f"Reading {file.name}")

        with open(file, "rb") as fh:
            dctx = zstd.ZstdDecompressor()

            with dctx.stream_reader(fh) as reader:
                for line in reader:
                    if not line.strip():
                        continue

                    obj = json.loads(line)

                    text = obj.get("text", "")

                    if len(text) < 50:
                        continue

                    if obj.get("prediction", 1) != 1:
                        continue

                    if obj.get("junk_probability", 0.0) > 0.1:
                        continue

                    yield {"text": text}


dataset = IterableDataset.from_generator(examples)

print(next(iter(dataset)))

MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "parsbert": "HooshvareLab/bert-base-parsbert-uncased",
    "tookabert": "PartAI/TookaBERT-Base",
    "ariabert": "ViraIntelligentDataMining/AriaBERT",
}

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

model_name = MODELS["tookabert"]

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
)

from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=f"results/{model_name.split('/')[-1]}",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


results = {}

metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(
        predictions=preds,
        references=labels,
        average="macro",
    )


for short_name, model_name in MODELS.items():

    print(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results[short_name] = trainer.evaluate()

print(results)