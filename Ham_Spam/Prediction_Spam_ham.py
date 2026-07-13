import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)



MODEL_DIR = "/projappl/project_2005092/nima/binary/saved_models/bge-m3_best_model"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=2,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

device="cuda"
model.to(device)
model.eval()


# -------------------------
# 1. Tune threshold on dev
# -------------------------

#Should avoid with current results
#threshold = 0.73   # obtained from tune_threshold...


# -------------------------
# 2. Prediction function
# -------------------------

def predict_batch(texts):

    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=1024,
        return_tensors="pt"
    )

    inputs = {
        k:v.to(device)
        for k,v in inputs.items()
    }

    with torch.no_grad():

        logits = model(**inputs).logits

        probs = torch.softmax(
            logits,
            dim=-1
        )

    return probs[:,0].cpu().numpy()



# -------------------------
# 3. Process shards
# -------------------------

from datasets import load_dataset
import json


shards = [
            "https://data.hplt-project.org/three/sorted/pes_Arab/10_1.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/5_1.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/6_1.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/6_2.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/7_1.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/7_2.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/7_3.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/8_1.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/8_2.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/8_3.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/8_4.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/9_1.jsonl.zst"
            "https://data.hplt-project.org/three/sorted/pes_Arab/9_2.jsonl.zst"
]


for shard in shards:

    print("Processing:", shard)

    dataset = load_dataset(
        "json",
        data_files=shard,
        split="train",
        streaming=True
    )


    output_file = (
        shard.split("/")[-1]
        .replace(".jsonl.zst", "_annotated.jsonl")
    )


    buffer_text=[]
    buffer_docs=[]


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as fout:


        for example in dataset:

            buffer_text.append(example["text"])
            buffer_docs.append(example)


            if len(buffer_text)==32:


                scores = predict_batch(buffer_text)


                for doc,score in zip(buffer_docs,scores):

                    doc["prediction"] = (
                        0 if score >= threshold else 1
                    )

                    doc["prob_class0"] = float(score)


                    fout.write(
                        json.dumps(
                            doc,
                            ensure_ascii=False
                        )+"\n"
                    )


                buffer_text=[]
                buffer_docs=[]


        # leftover documents

        if buffer_text:

            scores=predict_batch(buffer_text)

            for doc,score in zip(buffer_docs,scores):

                doc["prediction"] = (
                    0 if score >= threshold else 1
                )

                doc["prob_class0"]=float(score)

                fout.write(
                    json.dumps(
                        doc,
                        ensure_ascii=False
                    )+"\n"
                )

