import os
import io
import json
import argparse

import numpy as np
import zstandard as zstd
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_DIR = "/scratch/project_2005092/nima/saved_models/bge-m3-retromae_spamham"

BATCH_SIZE = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    basename = os.path.basename(args.input)
    outfile = os.path.join(
        args.output_dir,
        basename.replace(".jsonl.zst", ".npy")
    )

    # Skip already processed shards
    if os.path.exists(outfile):
        print(f"{outfile} already exists. Skipping.")
        return

    token_counts = []

    dctx = zstd.ZstdDecompressor()

    with open(args.input, "rb") as fh:
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        texts = []

        for line in tqdm(text_stream, desc=basename):

            doc = json.loads(line)
            texts.append(doc["text"])

            if len(texts) == BATCH_SIZE:
                enc = tokenizer(
                    texts,
                    add_special_tokens=False,
                    truncation=False,
                )

                token_counts.extend(
                    len(ids) for ids in enc["input_ids"]
                )

                texts = []

        if texts:
            enc = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
            )

            token_counts.extend(
                len(ids) for ids in enc["input_ids"]
            )

    np.save(outfile, np.asarray(token_counts, dtype=np.int32))

    print(f"Saved {outfile}")
    print(f"Documents: {len(token_counts):,}")


if __name__ == "__main__":
    main()