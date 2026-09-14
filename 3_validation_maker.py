import io
import json
import subprocess
from pathlib import Path

import numpy as np
import zstandard as zstd
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/scratch/project_462001491/nima"
)

EXPERIMENT_DIR = (
    BASE_DIR / "corpus_experiment"
)

# ------------------------------------------------------------
# Validation shard
# ------------------------------------------------------------

VALIDATION_SHARD_URL = (
    "https://data.hplt-project.org/three/sorted/"
    "pes_Arab/5_1.jsonl.zst"
)

SHARD_NAME = "5_1.jsonl.zst"

SHARD_DIR = (
    EXPERIMENT_DIR / "hplt3_shards"
)

SHARD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SHARD_PATH = (
    SHARD_DIR / SHARD_NAME
)

# ------------------------------------------------------------
# Validation output
# ------------------------------------------------------------

OUTPUT_DIR = (
    EXPERIMENT_DIR / "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR / "validation_5m.bin"
)

TARGET_TOKENS = 5_000_000

# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

TOKENIZER_NAME = (
    "Qwen/Qwen2.5-0.5B"
)


# ============================================================
# DOWNLOAD VALIDATION SHARD
# ============================================================

def download_shard():

    if SHARD_PATH.exists():

        print()
        print(
            "Validation shard already exists:"
        )

        print(
            SHARD_PATH
        )

        return

    print()
    print("=" * 70)
    print("DOWNLOADING VALIDATION SHARD")
    print("=" * 70)

    print(
        f"URL: {VALIDATION_SHARD_URL}"
    )

    print(
        f"Output: {SHARD_PATH}"
    )

    subprocess.run(
        [
            "wget",
            "-c",
            "-O",
            str(SHARD_PATH),
            VALIDATION_SHARD_URL,
        ],
        check=True
    )

    print()
    print(
        "Validation shard downloaded."
    )


# ============================================================
# TOKENIZER
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True
)

# We are using the tokenizer only to convert
# text into token IDs.
#
# We are NOT creating model sequences here.

tokenizer.model_max_length = 10**9


# ============================================================
# STREAM ZSTD JSONL
# ============================================================

def stream_zst_jsonl(path):

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
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Download shard if necessary
    # --------------------------------------------------------

    download_shard()


    # --------------------------------------------------------
    # Build validation set
    # --------------------------------------------------------

    tokens = []

    documents = 0

    print()
    print("=" * 70)
    print("CREATING VALIDATION SET")
    print("=" * 70)

    print(
        f"Source: {SHARD_PATH}"
    )

    print(
        f"Target: {TARGET_TOKENS:,} tokens"
    )


    for example in stream_zst_jsonl(
        SHARD_PATH
    ):

        text = example.get(
            "text",
            ""
        )

        if not text:
            continue


        # IMPORTANT:
        # No EOS is added here.
        #
        # Your existing three training files were
        # generated without EOS, so validation uses
        # the same representation.

        ids = tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]


        if not ids:
            continue


        tokens.extend(ids)

        documents += 1


        if documents % 1000 == 0:

            print(
                f"Documents: {documents:,} | "
                f"tokens: {len(tokens):,}"
            )


        if len(tokens) >= TARGET_TOKENS:

            tokens = tokens[:TARGET_TOKENS]

            break


    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if len(tokens) < TARGET_TOKENS:

        raise RuntimeError(
            f"Validation shard did not contain "
            f"enough tokens. "
            f"Got {len(tokens):,}, "
            f"needed {TARGET_TOKENS:,}."
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    arr = np.asarray(
        tokens,
        dtype=np.uint32
    )

    arr.tofile(
        OUTPUT_FILE
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION SET COMPLETE")
    print("=" * 70)

    print(
        f"Documents: {documents:,}"
    )

    print(
        f"Tokens:    {len(arr):,}"
    )

    print(
        f"File size: "
        f"{OUTPUT_FILE.stat().st_size / (1024**2):.2f} MiB"
    )

    print(
        f"Output:    {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
