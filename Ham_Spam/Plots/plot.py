import os
import json
import io
import zstandard as zstd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


JUNK_DIR = "/scratch/project_2005092/nima/annotated_data/junk_records"
PLOT_DIR = "/scratch/project_2005092/nima/plots"
os.makedirs(
    PLOT_DIR,
    exist_ok=True
)

print(f"Saving results to: {PLOT_DIR}")


def read_zst_jsonl(path):

    dctx = zstd.ZstdDecompressor()

    try:
        with open(path, "rb") as fh:
            reader = dctx.stream_reader(fh)

            text = io.TextIOWrapper(
                reader,
                encoding="utf-8"
            )

            for line in text:
                yield json.loads(line)

            text.detach()
            reader.close()

    except zstd.ZstdError as e:
        print(
            f"WARNING: Corrupted zstd file: {path}"
        )
        print(e)



# -------------------------------------------------
# 1. Load all junk records
# -------------------------------------------------

records = []

files = sorted([
    os.path.join(JUNK_DIR, f)
    for f in os.listdir(JUNK_DIR)
    if f.endswith(".junk.jsonl.zst")
])[:1] #for test


print(f"Found {len(files)} shards")


for file in tqdm(files):

    print(f"\nProcessing: {file}", flush=True)

    shard = os.path.basename(file)

    for item in read_zst_jsonl(file):

        scores = item["doc_scores"]

        if len(scores) != 10:
            print(
                f"Skipping {file}: unexpected doc_scores length {len(scores)}"
            )
            continue

        row = {
            "shard": shard,
            "junk_probability": item["junk_probability"]
        }

        # add the 10 doc scores
        for i, s in enumerate(scores):
            row[f"doc_score_{i}"] = s

        records.append(row)


PARQUET_OUT = "/scratch/project_2005092/nima/plots/junk_doc_scores.parquet"

df = pd.DataFrame(records)
df.to_parquet(
    PARQUET_OUT,
    compression="snappy"
)
print(df.shape)
print(df.head())

corr = df[
    ["junk_probability"] +
    [f"doc_score_{i}" for i in range(10)]
].corr()

print(
    corr["junk_probability"]
    .sort_values(ascending=False)
)

corr.to_csv(
    os.path.join(
        PLOT_DIR,
        "pearson_doc_score_correlation.csv"
    )
)

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Correlation between junk_probability and doc_scores"
)

fig.write_html(
    os.path.join(
        PLOT_DIR,
        "correlation.html"
    )
)

plot_df = df.sample(
    min(100000, len(df)),
    random_state=42
)


for i in range(10):

    fig = px.scatter(
        plot_df,
        x=f"doc_score_{i}",
        y="junk_probability",
        opacity=0.3,
        trendline="ols",
        title=f"junk_probability vs doc_score_{i}",
        labels={
            f"doc_score_{i}": f"doc_score_{i}",
            "junk_probability": "Junk probability"
        }
    )

    fig.write_html(
    os.path.join(
        PLOT_DIR,
        f"scatter_doc_score_{i}.html"
    )
)


score_cols = [
    f"doc_score_{i}"
    for i in range(10)
]


hist_df = df.sample(
    min(500000, len(df)),
    random_state=42
)


long_df = hist_df.melt(
    id_vars=["junk_probability"],
    value_vars=score_cols,
    var_name="score_name",
    value_name="score"
)


fig = px.histogram(
    long_df,
    x="score",
    color="score_name",
    marginal="box",
    opacity=0.5,
    nbins=100,
    title="Distribution of doc_scores in junk documents"
)

fig.write_html(
    os.path.join(
        PLOT_DIR,
        "histogram.html"
    )
)



for i in range(10):

    #temp = df.copy()

    temp = df[
        [
            f"doc_score_{i}",
            "junk_probability"
        ]
    ].copy()

    temp["score_bin"] = pd.qcut(
        temp[f"doc_score_{i}"],
        q=20,
        duplicates="drop"
    )

    grouped = (
    temp
    .groupby("score_bin", observed=True)
    ["junk_probability"]
    .mean()
    .reset_index()
)

    grouped["score_bin"] = grouped["score_bin"].astype(str)

    fig = px.line(
        grouped,
        x="score_bin",
        y="junk_probability",
        markers=True,
        title=f"Average junk_probability by doc_score_{i} percentile"
    )

    fig.write_html(
    os.path.join(
        PLOT_DIR,
        f"junk_probability_doc_score_{i}_binned.html"
    )
)



spearman = df[
    ["junk_probability"] +
    [f"doc_score_{i}" for i in range(10)]
].corr(method="spearman")

print(
    spearman["junk_probability"]
    .sort_values(ascending=False)
)

spearman.to_csv(
    os.path.join(
        PLOT_DIR,
        "spearman_doc_score_correlation.csv"
    )
)


fig = px.imshow(
    spearman,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Spearman correlation"
)

fig.write_html(
    os.path.join(
        PLOT_DIR,
        "spearman_correlation.html"
    )
)


print(
    f"Finished. Results saved to {PLOT_DIR}"
)