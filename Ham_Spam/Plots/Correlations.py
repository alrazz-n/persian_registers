import os
import io
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

OUTDIR = "/scratch/project_2005092/nima/statistics"
os.makedirs(OUTDIR, exist_ok=True)

workers = int(os.environ["SLURM_CPUS_PER_TASK"])

titles = [
    "WDS score",
    "lang_score",
    "url_score",
    "punctuation_score",
    "singular_chars_score",
    "numbers_score",
    "repeated_score",
    "informativeness_score",
    "n_long_segment_score",
    "great_segment_score",
]

# ---------------------------------------------------------
# Reader
# ---------------------------------------------------------

def read_jsonl_zst(file):

    dctx = zstd.ZstdDecompressor()

    with open(file, "rb") as fh:
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

    file, label = args

    # statistics for 10 scores
    n = np.zeros(10, dtype=np.int64)

    sum_x = np.zeros(10, dtype=np.float64)
    sum_x2 = np.zeros(10, dtype=np.float64)

    sum_y = np.zeros(10, dtype=np.float64)
    sum_y2 = np.zeros(10, dtype=np.float64)

    sum_xy = np.zeros(10, dtype=np.float64)

    for item in read_jsonl_zst(file):

        scores = item["doc_scores"]

        y = float(label)

        for i in range(10):

            x = float(scores[i])

            n[i] += 1

            sum_x[i] += x
            sum_x2[i] += x * x

            sum_y[i] += y
            sum_y2[i] += y

            sum_xy[i] += x * y

    return (
        n,
        sum_x,
        sum_x2,
        sum_y,
        sum_y2,
        sum_xy,
    )


# ---------------------------------------------------------
# Build jobs
# ---------------------------------------------------------

jobs = []

for f in glob(os.path.join(JUNK_DIR, "*.jsonl.zst")):
    jobs.append((f, 1))

for f in glob(os.path.join(KEPT_DIR, "*.jsonl.zst")):
    jobs.append((f, 0))

print(f"Processing {len(jobs)} shards with {workers} workers")

# ---------------------------------------------------------
# Global accumulators
# ---------------------------------------------------------

N = np.zeros(10, dtype=np.int64)

SX = np.zeros(10)
SX2 = np.zeros(10)

SY = np.zeros(10)
SY2 = np.zeros(10)

SXY = np.zeros(10)

# ---------------------------------------------------------
# Parallel
# ---------------------------------------------------------

with ProcessPoolExecutor(max_workers=workers) as pool:

    for partial in pool.map(process_file, jobs):

        n, sx, sx2, sy, sy2, sxy = partial

        N += n
        SX += sx
        SX2 += sx2

        SY += sy
        SY2 += sy2

        SXY += sxy

# ---------------------------------------------------------
# Compute Pearson
# ---------------------------------------------------------

rows = []

for i, title in enumerate(titles):

    numerator = (
        N[i] * SXY[i]
        - SX[i] * SY[i]
    )

    denominator = np.sqrt(

        (N[i] * SX2[i] - SX[i] ** 2)

        *

        (N[i] * SY2[i] - SY[i] ** 2)

    )

    r = numerator / denominator

    rows.append({

        "score": title,
        "pearson_r": r,

        "N": int(N[i]),

    })

results = pd.DataFrame(rows)

print(results)

results.to_csv(
    os.path.join(
        OUTDIR,
        "pearson_correlations.csv",
    ),
    index=False,
)

print("Finished.")