#import numpy as np
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import os
from tqdm import tqdm
from datetime import datetime
import zstandard as zstd

def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def open_zst(path):
    fh = open(path, "wb")
    cctx = zstd.ZstdCompressor(level=3)
    return cctx.stream_writer(fh, closefd=True)

MODEL_DIR = "/scratch/project_2005092/nima/saved_models/bge-m3-retromae_spamham"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    #num_labels=2,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

device="cuda"
model.to(device)
model.eval()

# Check BF16 support once
#if ture keep  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#if not change it to with torch.autocast(device_type="cuda", dtype=torch.float16):
#It was ture
#print("GPU:", torch.cuda.get_device_name())
#print("BF16 supported:", torch.cuda.is_bf16_supported())


model.config.use_cache = False


# -------------------------
# 1. Tune threshold on dev
# -------------------------

threshold = 0.3   # obtained from looking at plots


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

    inputs = {k: v.to(device, non_blocking=True)
          for k, v in inputs.items()}

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            logits = model(**inputs).logits

            probs = torch.softmax(
                logits,
                dim=-1
            )

        return probs[:,0].cpu().numpy()



# -------------------------
# 3. Process shards
# -------------------------


shards = [
            "https://data.hplt-project.org/three/sorted/pes_Arab/10_1.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/5_1.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/6_1.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/6_2.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/7_1.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/7_2.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/7_3.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/8_1.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/8_2.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/8_3.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/8_4.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/9_1.jsonl.zst",
            "https://data.hplt-project.org/three/sorted/pes_Arab/9_2.jsonl.zst"
]


OUTPUT_ROOT = "/scratch/project_2005092/nima/annotated_data"
KEPT_DIR = os.path.join(OUTPUT_ROOT, "kept")
JUNK_DIR  = os.path.join(OUTPUT_ROOT, "junk_records")
META_DIR = os.path.join(OUTPUT_ROOT, "metadata")

os.makedirs(KEPT_DIR, exist_ok=True)
os.makedirs(JUNK_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)



BATCH_SIZE = 128

for shard in shards:
    kept_count = 0
    removed_count = 0
    last_report = 0
    last_flush = 0

    print("Processing:", shard)

    dataset = load_dataset(
        "json",
        data_files=shard,
        split="train",
        streaming=True
    )

    #OUTPUT_DIR = f"/scratch/project_2005092/nima/annotated_data"

    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    #output_file = os.path.join(
    #    OUTPUT_DIR,
    #    shard.split("/")[-1].replace(".jsonl.zst",
    #                                "_annotated.jsonl")
    #)

    MAX_CHARS = 20000
    buffer_text=[]
    buffer_docs=[]



    basename = shard.split("/")[-1].replace(".jsonl.zst", ".jsonl")
    kept_file = os.path.join(KEPT_DIR,  basename.replace(".jsonl", ".jsonl.zst"))
    junk_file = os.path.join(JUNK_DIR,  basename.replace(".jsonl", ".junk.jsonl.zst"))

    #kept_count = 0
    #removed_count = 0

    with open_zst(kept_file) as fkeep, \
     open_zst(junk_file) as fjunk:


        for example in tqdm(dataset, desc=basename, unit="docs"):

            buffer_text.append(example["text"][:MAX_CHARS])
            buffer_docs.append(example)


            if len(buffer_text)>= BATCH_SIZE:


                scores = predict_batch(buffer_text)


                for doc,score in zip(buffer_docs,scores):

                    doc["prediction"] = (
                        0 if score >= threshold else 1
                    )

                    #doc["junk_probability"] = float(score)


                    if score < threshold:
                        kept_count += 1
                        fkeep.write(
                            (json.dumps(doc, ensure_ascii=False, default=json_serializer) + "\n").encode("utf-8")
                        )
                    else:
                        removed_count += 1
                        junk_record = {
                            "id": doc["id"],
                            "doc_scores": doc["doc_scores"],
                            "junk_probability": float(score),
                            "web-register": doc["web-register"],
                            "source_shard": basename
                        }
                        fjunk.write(
                            (json.dumps(junk_record, ensure_ascii=False, default=json_serializer) + "\n").encode("utf-8")
                        )

                total_count = kept_count + removed_count

                if total_count - last_report >= 10000:
                    print(
                        f"{basename}: "
                        f"kept={kept_count:,}, "
                        f"removed={removed_count:,}"
                    )
                    last_report = total_count

                total_count = kept_count + removed_count
                if total_count - last_flush >= 100000:
                    fkeep.flush()
                    fjunk.flush()
                    last_flush = total_count


                buffer_text=[]
                buffer_docs=[]


        # leftover documents

        if buffer_text:

            scores=predict_batch(buffer_text)

            for doc, score in zip(buffer_docs, scores):

                doc["prediction"] = 0 if score >= threshold else 1
                doc["junk_probability"] = float(score) #prob_class0

                if score < threshold:
                    kept_count += 1
                    fkeep.write(
                        (json.dumps(doc, ensure_ascii=False, default=json_serializer) + "\n").encode("utf-8")
                    )
                else:
                    removed_count += 1
                    junk_record = {
                        "id": doc["id"],
                            "doc_scores": doc["doc_scores"],
                            "junk_probability": float(score),
                            "web-register": doc["web-register"],
                            "source_shard": basename
                    }
                    fjunk.write(
                        (json.dumps(junk_record, ensure_ascii=False, default=json_serializer) + "\n").encode("utf-8")
                    )

            print(
                f"{basename}: "
                f"kept={kept_count:,}, "
                f"removed={removed_count:,}"
                )        
            fkeep.flush()
            fjunk.flush()

        total_count = kept_count + removed_count
        metadata = {
            "source_shard": shard,
            "source_filename": basename,
            "model": "alrazz-n/bge-m3_spamham_best_model",
            "model_path": MODEL_DIR, #https://huggingface.co/alrazz-n/bge-m3_spamham_best_model
            "threshold": threshold,
            "threshold_definition": "remove documents where junk_probability >= threshold",
            "threshold_source": "dev set precision-recall analysis",
            "dev_metrics": {
                "junk_precision": 0.9484,
                "junk_recall": 0.9439,
                "good_recall": 0.9725
            },
            "class_mapping": {
                "0": "junk",
                "1": "good"
            },
            "batch_size": BATCH_SIZE,
            "max_length": 1024,
            "max_chars_for_inference": MAX_CHARS,
            "kept_documents": kept_count,
            "junk_documents": removed_count,
            "junk_storage": "id + junk_probability only",
            "total_documents": total_count,
            "kept_ratio": kept_count / total_count if total_count else 0, #avoid devision 0
            "junk_ratio": removed_count / total_count if total_count else 0, #avoid devision 0
            "processed_at": datetime.now().isoformat()
        }

        fkeep.flush()
        fjunk.flush()

        metadata_file = os.path.join(
            META_DIR,
            basename.replace(".jsonl", ".stats.json")
        )

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=json_serializer)

