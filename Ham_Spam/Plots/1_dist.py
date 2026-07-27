import os
import io
import json
import pickle
from glob import glob
from collections import defaultdict
import zstandard as zstd


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
                    yield json.loads(line)
                except ValueError:
                    continue



def compute_distribution(path):
    """
    Compute histogram of the 10 doc_scores
    """
    hist = {i: defaultdict(int) for i in range(10)}

    files = glob(os.path.join(path, "*.jsonl.zst"))

    print(f"Found {len(files)} files in {path}")

    for file in files:
        print("Processing:", os.path.basename(file))

        try:
            for jdata in read_jsonl_zst(file):
                scores = jdata["doc_scores"]

                for i, score in enumerate(scores):
                    score_bin = int(score * 10)
                    hist[i][score_bin] += 1

        except Exception as e:
            print("FAILED:", file)
            print(e)
            continue



junk_dist = compute_distribution(junk_path)
kept_dist = compute_distribution(kept_path)


# combine junk + kept
all_dist = {i: defaultdict(int) for i in range(10)}

for i in range(10):
    for k,v in junk_dist[i].items():
        all_dist[i][k] += v

    for k,v in kept_dist[i].items():
        all_dist[i][k] += v


all_dist = {i: dict(all_dist[i]) for i in range(10)}


# save everything
distribution_data = {
    "junk": junk_dist,
    "kept": kept_dist,
    "all": all_dist
}

os.makedirs(os.path.dirname(save_file), exist_ok=True)
with open(save_file, "wb") as f:
    pickle.dump(distribution_data, f)


print("Saved:", save_file)