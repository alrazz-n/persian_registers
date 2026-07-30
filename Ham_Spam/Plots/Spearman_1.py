import io
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import orjson
import zstandard as zstd

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

JUNK_DIR = "/scratch/project_2005092/nima/annotated_data/junk_records"
KEPT_DIR = "/scratch/project_2005092/nima/annotated_data/kept"

OUTDIR = "/scratch/project_2005092/nima/statistics/spearman_tmp"

os.makedirs(OUTDIR, exist_ok=True)

workers = int(os.environ["SLURM_CPUS_PER_TASK"])

# ---------------------------------------------------------
# Reader
# ---------------------------------------------------------

def read_jsonl_zst(filename):

    dctx = zstd.ZstdDecompressor()

    with open(filename, "rb") as fh:

        with dctx.stream_reader(fh) as reader:

            text = io.TextIOWrapper(reader, encoding="utf-8")

            for line in text:

                try:
                    yield orjson.loads(line)

                except Exception:
                    continue


# ---------------------------------------------------------
# Worker
# ---------------------------------------------------------

def process_file(args):

    filename, label, shard_id = args

    scores = []
    labels = []

    for obj in read_jsonl_zst(filename):

        scores.append(obj["doc_scores"])
        labels.append(label)

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)

    score_file = os.path.join(
        OUTDIR,
        f"shard_{shard_id:05d}_scores.npy"
    )

    label_file = os.path.join(
        OUTDIR,
        f"shard_{shard_id:05d}_labels.npy"
    )

    np.save(score_file, scores)

    np.save(label_file, labels)

    return {
        "shard": shard_id,
        "documents": len(labels),
        "score_file": score_file,
        "label_file": label_file,
    }


# ---------------------------------------------------------
# Build jobs
# ---------------------------------------------------------

jobs = []

idx = 0

for f in sorted(glob(os.path.join(JUNK_DIR, "*.jsonl.zst"))):

    jobs.append((f, 1, idx))
    idx += 1

for f in sorted(glob(os.path.join(KEPT_DIR, "*.jsonl.zst"))):

    jobs.append((f, 0, idx))
    idx += 1

print(f"{len(jobs)} shards")


# ---------------------------------------------------------
# Parallel extraction
# ---------------------------------------------------------

manifest = []

with ProcessPoolExecutor(max_workers=workers) as pool:

    for info in pool.map(process_file, jobs):

        manifest.append(info)

        print(
            f"Finished shard {info['shard']:5d} "
            f"({info['documents']:,} docs)"
        )


manifest = pd.DataFrame(manifest)

manifest.sort_values("shard", inplace=True)

manifest.to_csv(
    os.path.join(OUTDIR, "manifest.csv"),
    index=False,
)

print()

print("Total documents:", manifest.documents.sum())

print("Done.")