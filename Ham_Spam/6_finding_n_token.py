import os
import io
import json
import zstandard as zstd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_DIR = "/scratch/project_2005092/nima/saved_models/bge-m3-retromae_spamham"
KEPT_DIR = "/scratch/project_2005092/nima/annotated_data/kept"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

BATCH_SIZE = 256
token_counts = []

for filename in sorted(os.listdir(KEPT_DIR)):
    if not filename.endswith(".jsonl.zst"):
        continue

    path = os.path.join(KEPT_DIR, filename)
    print(f"Processing {filename}")

    dctx = zstd.ZstdDecompressor()

    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        texts = []

        for line in tqdm(text_stream, desc=filename):
            doc = json.loads(line)
            texts.append(doc["text"])

            if len(texts) == BATCH_SIZE:
                enc = tokenizer(
                    texts,
                    add_special_tokens=False,
                    truncation=False,
                )

                token_counts.extend(len(ids) for ids in enc["input_ids"])
                texts = []

        # Process leftover texts
        if texts:
            enc = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
            )

            token_counts.extend(len(ids) for ids in enc["input_ids"])

token_counts = np.array(token_counts)

print(f"Documents     : {len(token_counts):,}")
print(f"Total tokens  : {token_counts.sum():,}")
print(f"Mean          : {token_counts.mean():.1f}")
print(f"Median        : {np.median(token_counts):.1f}")
print(f"Min           : {token_counts.min()}")
print(f"Max           : {token_counts.max()}")

for p in [90, 95, 99]:
    print(f"P{p}: {np.percentile(token_counts, p):.0f}")