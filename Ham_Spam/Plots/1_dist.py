import os
import io
import json
import pickle
from glob import glob
from collections import defaultdict
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor
import orjson
import numpy as np


workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))

junk_path = "/scratch/project_2005092/nima/annotated_data/junk_records/"
kept_path = "/scratch/project_2005092/nima/annotated_data/kept"

save_file = "/scratch/project_2005092/nima/plots/dist/score_distributions.pkl"


titles = [
    'WDS score',
    'lang_score',
    'url_score',
    'punctuation_score',
    'singular_chars_score',
    'numbers_score',
    'repeated_score',
    'informativeness_score',
    'n_long_segment_score',
    'great_segment_score'
]




def read_jsonl_zst(file):

    dctx = zstd.ZstdDecompressor()

    with open(file, "rb") as fh:
        with dctx.stream_reader(fh) as reader:

            # convert bytes stream -> text stream
            text_stream = io.TextIOWrapper(
                reader,
                encoding="utf-8"
            )

            for line in text_stream:
                try:
                    yield orjson.loads(line)
                except ValueError:
                    continue



#def compute_distribution(path):
##    """
#    Compute histogram of the 10 doc_scores
#    """
#    hist = {i: defaultdict(int) for i in range(10)}
#
#    files = glob(os.path.join(path, "*.jsonl.zst"))
##   print(f"Found {len(files)} files in {path}")
#
#    for file in files:
    #    print("Processing:", os.path.basename(file))

        #try:
        #    for jdata in read_jsonl_zst(file):
        #        scores = jdata["doc_scores"]

         #       for i, score in enumerate(scores):
          #          score_bin = int(score * 10)
           #         hist[i][score_bin] += 1

        #except Exception as e:
         #   print("FAILED:", file)
          #  print(e)
           # continue

    #return {i: dict(hist[i]) for i in range(10)}

def process_file(file):
    hist = np.zeros((10, 101), dtype=np.int64)

    try:
        for jdata in read_jsonl_zst(file):
            scores = jdata["doc_scores"]

            for i in range(10):
                bin_idx = min(int(scores[i] * 10), 100)
                hist[i, bin_idx] += 1

    except Exception as e:
        print(f"FAILED: {file}")
        print(e)

    return hist


def compute_distribution(path, workers):
    files = glob(os.path.join(path, "*.jsonl.zst"))

    print(f"Found {len(files)} files in {path}")

    hist = np.zeros((10, 101), dtype=np.int64)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for partial in pool.map(process_file, files):
            hist += partial

    return hist


junk_dist = compute_distribution(junk_path, workers)
kept_dist = compute_distribution(kept_path, workers)


# combine junk + kept
#all_dist = {i: defaultdict(int) for i in range(10)}

all_dist = junk_dist + kept_dist

#for i in range(10):
#    for k,v in junk_dist[i].items():
#        all_dist[i][k] += v

#    for k,v in kept_dist[i].items():
#        all_dist[i][k] += v


#all_dist = {i: dict(all_dist[i]) for i in range(10)}


def hist_to_dict(hist):
    return {
        i: {j: int(hist[i, j]) for j in range(hist.shape[1]) if hist[i, j] > 0}
        for i in range(hist.shape[0])
    }

distribution_data = {
    "junk": hist_to_dict(junk_dist),
    "kept": hist_to_dict(kept_dist),
    "all": hist_to_dict(all_dist),
}


os.makedirs(os.path.dirname(save_file), exist_ok=True)
with open(save_file, "wb") as f:
    pickle.dump(distribution_data, f)


print("Saved:", save_file)