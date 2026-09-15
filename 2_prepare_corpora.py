import os
import json
import io
import random
from pathlib import Path

import zstandard as zstd
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/scratch/project_462001491/nima"
)

EXPERIMENT_DIR = BASE_DIR / "corpus_experiment"

EXCLUDED_IDS_FILE = (
    EXPERIMENT_DIR / "excluded_ids.txt"
)

PERREF_DIR = (
    BASE_DIR / "annotated_data" / "kept"
)

# IMPORTANT:
# Separate directory for Qwen3.6 tokenized data.
# This does NOT overwrite your previous Qwen2.5 data.
OUTPUT_DIR = (
    EXPERIMENT_DIR / "tokenized_Qwen3_6"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# HPLT3 source shards
# ------------------------------------------------------------

HPLT3_SHARDS = [
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
    "https://data.hplt-project.org/three/sorted/pes_Arab/9_2.jsonl.zst",
]


# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

TOKENIZER_NAME = "Qwen/Qwen3.6-35B-A3B"


# ------------------------------------------------------------
# Experiment size
# ------------------------------------------------------------

TARGET_TOKENS = 100_000_000


# ------------------------------------------------------------
# Random control
# ------------------------------------------------------------

RANDOM_REMOVAL_RATE = 0.2012
RANDOM_SEED = 42


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading Qwen3.6 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True
)

tokenizer.model_max_length = 10**9

if tokenizer.eos_token_id is None:
    raise ValueError(
        "Tokenizer does not have an EOS token."
    )

print(
    f"Tokenizer: {TOKENIZER_NAME}"
)

print(
    f"Tokenizer vocabulary size: "
    f"{len(tokenizer):,}"
)

print(
    f"EOS token ID: "
    f"{tokenizer.eos_token_id}"
)


# ============================================================
# LOAD EXCLUDED IDS
# ============================================================

with open(
    EXCLUDED_IDS_FILE,
    "r",
    encoding="utf-8"
) as f:

    excluded_ids = {
        line.strip()
        for line in f
        if line.strip()
    }

print(
    f"Excluded classifier IDs: "
    f"{len(excluded_ids):,}"
)


# ============================================================
# HELPERS
# ============================================================

def tokenize_text(text):
    """
    Tokenize one document without adding special tokens.
    """
    return tokenizer(
        text,
        add_special_tokens=False
    )["input_ids"]


def save_tokens(tokens, output_path):
    """
    Save token IDs as uint32 binary.

    Qwen3.6 token IDs fit comfortably inside uint32.
    """
    import numpy as np

    arr = np.asarray(
        tokens,
        dtype=np.uint32
    )

    arr.tofile(output_path)

    print(
        f"Saved {len(arr):,} tokens -> "
        f"{output_path}"
    )


def stream_zst_jsonl(path):
    """
    Stream a .jsonl.zst file without decompressing
    the entire file into memory.
    """

    with open(path, "rb") as fh:

        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(fh) as reader:

            text_stream = io.TextIOWrapper(
                reader,
                encoding="utf-8"
            )

            for line in text_stream:

                line = line.strip()

                if not line:
                    continue

                yield json.loads(line)


# ============================================================
# DOWNLOAD / STREAM HPLT3 SHARD
# ============================================================

def stream_hplt3_documents():
    """
    Stream HPLT3 shards.

    Uses wget/curl to avoid loading the whole corpus.
    """

    import subprocess

    for shard_url in HPLT3_SHARDS:

        print()
        print("=" * 70)
        print("HPLT3 SHARD")
        print(shard_url)
        print("=" * 70)

        shard_name = shard_url.split("/")[-1]

        local_path = (
            EXPERIMENT_DIR
            / "hplt3_shards"
            / shard_name
        )

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not local_path.exists():

            print(
                "Downloading shard..."
            )

            subprocess.run(
                [
                    "wget",
                    "-c",
                    "-O",
                    str(local_path),
                    shard_url
                ],
                check=True
            )

        else:

            print(
                "Shard already exists; "
                "using local copy."
            )

        yield from stream_zst_jsonl(
            local_path
        )


# ============================================================
# PREPARE ORIGINAL HPLT3
# ============================================================

def prepare_hplt3():

    print("\nPreparing HPLT3 for Qwen3.6...")

    tokens = []

    documents = 0
    excluded = 0

    for example in stream_hplt3_documents():

        doc_id = str(
            example.get("id", "")
        ).strip()

        if doc_id in excluded_ids:

            excluded += 1
            continue

        text = example.get(
            "text",
            ""
        )

        if not text:
            continue

        doc_tokens = tokenize_text(text)

        if not doc_tokens:
            continue

        tokens.extend(doc_tokens)

        documents += 1

        if len(tokens) >= TARGET_TOKENS:

            tokens = tokens[:TARGET_TOKENS]

            print(
                f"HPLT3 documents used: "
                f"{documents:,}"
            )

            print(
                f"Excluded documents: "
                f"{excluded:,}"
            )

            break

        if documents % 10_000 == 0:

            print(
                f"HPLT3 docs: {documents:,} | "
                f"tokens: {len(tokens):,}"
            )

    output = (
        OUTPUT_DIR /
        f"hplt3_Qwen3_6_{TARGET_TOKENS}.bin"
    )

    save_tokens(
        tokens,
        output
    )


# ============================================================
# PREPARE RANDOM 20.12% CONTROL
# ============================================================

def prepare_random_hplt3():

    print(
        "\nPreparing random-filtered HPLT3 "
        "for Qwen3.6..."
    )

    tokens = []

    rng = random.Random(
        RANDOM_SEED
    )

    documents = 0
    excluded = 0
    randomly_removed = 0

    for example in stream_hplt3_documents():

        doc_id = str(
            example.get("id", "")
        ).strip()

        if doc_id in excluded_ids:

            excluded += 1
            continue

        # Randomly remove 20.12% of documents
        if rng.random() < RANDOM_REMOVAL_RATE:

            randomly_removed += 1
            continue

        text = example.get(
            "text",
            ""
        )

        if not text:
            continue

        doc_tokens = tokenize_text(text)

        if not doc_tokens:
            continue

        tokens.extend(doc_tokens)

        documents += 1

        if len(tokens) >= TARGET_TOKENS:

            tokens = tokens[:TARGET_TOKENS]

            print(
                f"Random-control docs used: "
                f"{documents:,}"
            )

            print(
                f"Randomly removed: "
                f"{randomly_removed:,}"
            )

            break

        if documents % 10_000 == 0:

            print(
                f"Random docs: {documents:,} | "
                f"tokens: {len(tokens):,}"
            )

    output = (
        OUTPUT_DIR /
        f"hplt3_random20_Qwen3_6_{TARGET_TOKENS}.bin"
    )

    save_tokens(
        tokens,
        output
    )


# ============================================================
# PREPARE PERREF
# ============================================================

def prepare_perref():

    print(
        "\nPreparing HPLT3-PerRef "
        "for Qwen3.6..."
    )

    tokens = []

    documents = 0
    excluded = 0

    files = sorted(
        PERREF_DIR.glob(
            "*.jsonl.zst"
        )
    )

    print(
        f"PerRef files: {len(files)}"
    )

    for file_path in files:

        print(
            f"Reading {file_path}"
        )

        for example in stream_zst_jsonl(
            file_path
        ):

            doc_id = str(
                example.get("id", "")
            ).strip()

            if doc_id in excluded_ids:

                excluded += 1
                continue

            text = example.get(
                "text",
                ""
            )

            if not text:
                continue

            doc_tokens = tokenize_text(
                text
            )

            if not doc_tokens:
                continue

            tokens.extend(
                doc_tokens
            )

            documents += 1

            if len(tokens) >= TARGET_TOKENS:

                tokens = tokens[:TARGET_TOKENS]

                print(
                    f"PerRef documents used: "
                    f"{documents:,}"
                )

                print(
                    f"Excluded documents: "
                    f"{excluded:,}"
                )

                output = (
                    OUTPUT_DIR /
                    f"perref_Qwen3_6_{TARGET_TOKENS}.bin"
                )

                save_tokens(
                    tokens,
                    output
                )

                return

            if documents % 10_000 == 0:

                print(
                    f"PerRef docs: {documents:,} | "
                    f"tokens: {len(tokens):,}"
                )

    output = (
        OUTPUT_DIR /
        f"perref_Qwen3_6_{TARGET_TOKENS}.bin"
    )

    save_tokens(
        tokens,
        output
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepare_hplt3()

    prepare_random_hplt3()

    prepare_perref()

    print("\nFinished Qwen3.6 tokenization.")