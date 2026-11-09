
import os
import json
import gc
import zstandard as zstd

from pathlib import Path
from collections import Counter
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "/scratch/project_462001491/nima/register_annotated_data"
)

OUTPUT_DIR = Path(
    "/scratch/project_462001491/nima/register_distribution"
)

TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


SHARDS = [
    "10_1.jsonl.zst",
    "5_1.jsonl.zst",
    "6_1.jsonl.zst",
    "6_2.jsonl.zst",
    "7_1.jsonl.zst",
    "7_2.jsonl.zst",
    "7_3.jsonl.zst",
    "8_1.jsonl.zst",
    "8_2.jsonl.zst",
    "8_3.jsonl.zst",
    "8_4.jsonl.zst",
    "9_1.jsonl.zst",
    "9_2.jsonl.zst",
]


# ------------------------------------------------------------
# CLASSIFICATION THRESHOLD
# ------------------------------------------------------------

THRESHOLD = 0.31 #Global threshold


LABEL_THRESHOLDS = {
  "MT": 0.11,
  "LY": 0.24,
  "SP": 0.15,
  "ID": 0.31,
  "NA": 0.35,
  "HI": 0.14,
  "IN": 0.46,
  "OP": 0.24,
  "IP": 0.32,
  "it": 0.3,
  "ne": 0.56,
  "sr": 0.28,
  "nb": 0.35,
  "re": 0.37,
  "en": 0.35,
  "ra": 0.13,
  "dtp": 0.21,
  "fi": 0.19,
  "lt": 0.18,
  "rv": 0.21,
  "ob": 0.32,
  "rs": 0.22,
  "av": 0.22,
  "ds": 0.59,
  "ed": 0.18
}


# ------------------------------------------------------------
# PLOT CONFIGURATION
# ------------------------------------------------------------

TOP_N_COMBINATIONS = 50

DPI = 300

sns.set_theme(
    style="whitegrid",
    context="talk",
)


# ============================================================
# LOAD LABELS FROM METADATA
# ============================================================

metadata_file = (
    INPUT_DIR / "10_1_metadata.json"
)

if not metadata_file.exists():
    raise FileNotFoundError(
        f"Metadata not found: {metadata_file}"
    )

with open(
    metadata_file,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


LABELS = metadata["labels"]

NUM_LABELS = len(LABELS)

print("=" * 80)
print("MULTI-LABEL DISTRIBUTION ANALYSIS")
print("=" * 80)

print("Labels:", LABELS)
print("Number of labels:", NUM_LABELS)
print("Default threshold:", THRESHOLD)
print("Individual thresholds:", LABEL_THRESHOLDS)


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

for label, threshold in LABEL_THRESHOLDS.items():

    if label not in LABELS:
        raise ValueError(
            f"Unknown label in LABEL_THRESHOLDS: {label}"
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Invalid threshold for {label}: {threshold}"
        )


def get_threshold(label):

    return LABEL_THRESHOLDS.get(
        label,
        THRESHOLD,
    )


# ============================================================
# INITIALIZE COUNTERS
# ============================================================

individual_counts = Counter()

combination_counts = Counter()

total_documents = 0

documents_without_labels = 0

shard_document_counts = {}


# ============================================================
# PROCESS ALL SHARDS
# ============================================================

overall_start = datetime.now()

for shard_number, shard_name in enumerate(
    SHARDS,
    start=1,
):

    input_file = INPUT_DIR / shard_name

    print("\n" + "=" * 80)

    print(
        f"SHARD {shard_number}/{len(SHARDS)}: "
        f"{shard_name}"
    )

    print("=" * 80)

    if not input_file.exists():

        raise FileNotFoundError(
            f"Output shard does not exist: {input_file}"
        )

    shard_count = 0

    with zstd.open(
        input_file,
        mode="rt",
        encoding="utf-8",
    ) as reader:

        for line in reader:

            if not line.strip():
                continue

            document = json.loads(line)

            if "new_web_register" not in document:

                raise KeyError(
                    f"Missing new_web_register in "
                    f"{shard_name}"
                )

            probabilities = document[
                "new_web_register"
            ]

            if set(probabilities.keys()) != set(LABELS):

                raise ValueError(
                    f"Label mismatch in {shard_name}, "
                    f"document {shard_count}"
                )

            # ------------------------------------------------
            # IDENTIFY POSITIVE LABELS
            # ------------------------------------------------

            positive_labels = tuple(
                label
                for label in LABELS
                if float(probabilities[label])
                >= get_threshold(label)
            )

            # ------------------------------------------------
            # INDIVIDUAL LABEL COUNTS
            # ------------------------------------------------

            for label in positive_labels:

                individual_counts[label] += 1

            # ------------------------------------------------
            # EXACT COMBINATION COUNTS
            # ------------------------------------------------

            # Sorting ensures that combinations have a
            # consistent representation.

            combination = tuple(
                sorted(positive_labels)
            )

            combination_counts[combination] += 1

            # ------------------------------------------------
            # DOCUMENT COUNTS
            # ------------------------------------------------

            if len(positive_labels) == 0:

                documents_without_labels += 1

            total_documents += 1

            shard_count += 1

            if shard_count % 1_000_000 == 0:

                print(
                    f"Processed {shard_count:,} "
                    f"documents in this shard"
                )

    shard_document_counts[shard_name] = shard_count

    print(
        "Shard documents:",
        f"{shard_count:,}"
    )

    print(
        "Total documents processed:",
        f"{total_documents:,}"
    )

    gc.collect()


# ============================================================
# VALIDATE TOTAL COUNTS
# ============================================================

if total_documents == 0:

    raise ValueError(
        "No documents were processed."
    )

if sum(combination_counts.values()) != total_documents:

    raise RuntimeError(
        "Combination counts do not match total documents."
    )

print("\n" + "=" * 80)

print("PROCESSING COMPLETED")

print("=" * 80)

print(
    "Total documents:",
    f"{total_documents:,}"
)

print(
    "Documents without any positive labels:",
    f"{documents_without_labels:,}"
)

print(
    "Unique combinations:",
    f"{len(combination_counts):,}"
)


# ============================================================
# 1. INDIVIDUAL LABEL DISTRIBUTION
# ============================================================

individual_rows = []

for label in LABELS:

    count = individual_counts[label]

    percentage = (
        count / total_documents
    ) * 100

    individual_rows.append({

        "label": label,

        "threshold": get_threshold(label),

        "document_count": count,

        "percentage_of_all_documents": round(
            percentage,
            4,
        ),

    })


individual_df = pd.DataFrame(
    individual_rows
)

individual_df = individual_df.sort_values(
    "document_count",
    ascending=False,
).reset_index(drop=True)


individual_csv = (
    TABLE_DIR /
    "individual_label_distribution.csv"
)

individual_df.to_csv(
    individual_csv,
    index=False,
)


# ============================================================
# 2. LABEL COMBINATION DISTRIBUTION
# ============================================================

combination_rows = []

for combination, count in combination_counts.items():

    combination_name = (
        " + ".join(combination)
        if combination
        else "NO_LABEL"
    )

    percentage = (
        count / total_documents
    ) * 100

    combination_rows.append({

        "combination": combination_name,

        "labels": json.dumps(
            list(combination),
            ensure_ascii=False,
        ),

        "number_of_labels": len(combination),

        "document_count": count,

        "percentage_of_all_documents": round(
            percentage,
            6,
        ),

    })


combination_df = pd.DataFrame(
    combination_rows
)

combination_df = combination_df.sort_values(
    [
        "document_count",
        "combination",
    ],
    ascending=[False, True],
).reset_index(drop=True)


combination_df.insert(
    0,
    "rank",
    range(1, len(combination_df) + 1),
)


combination_csv = (
    TABLE_DIR /
    "label_combination_distribution.csv"
)

combination_df.to_csv(
    combination_csv,
    index=False,
)


# ============================================================
# 3. ADD PERCENTAGE AMONG CLASSIFIED DOCUMENTS
# ============================================================

classified_documents = (
    total_documents - documents_without_labels
)

combination_df["percentage_of_classified_documents"] = (

    combination_df["document_count"]
    .div(classified_documents)
    .mul(100)
    .round(6)

    if classified_documents > 0
    else 0.0
)


combination_percentage_csv = (
    TABLE_DIR /
    "label_combination_distribution_with_percentages.csv"
)

combination_df.to_csv(
    combination_percentage_csv,
    index=False,
)


# ============================================================
# 4. DISTRIBUTION BY NUMBER OF LABELS
# ============================================================

label_count_rows = []

for number_of_labels in range(
    NUM_LABELS + 1
):

    matching_documents = sum(

        count

        for combination, count
        in combination_counts.items()

        if len(combination) == number_of_labels

    )

    label_count_rows.append({

        "number_of_labels": number_of_labels,

        "document_count": matching_documents,

        "percentage_of_all_documents": round(
            matching_documents / total_documents * 100,
            6,
        ),

    })


label_count_df = pd.DataFrame(
    label_count_rows
)

label_count_df.to_csv(
    TABLE_DIR / "distribution_by_number_of_labels.csv",
    index=False,
)


# ============================================================
# 5. DISTRIBUTION BY SHARD
# ============================================================

shard_df = pd.DataFrame([

    {
        "shard": shard,
        "document_count": count,
        "percentage_of_all_documents": round(
            count / total_documents * 100,
            6,
        ),
    }

    for shard, count in shard_document_counts.items()

])

shard_df.to_csv(
    TABLE_DIR / "shard_document_counts.csv",
    index=False,
)


# ============================================================
# 6. PLOT INDIVIDUAL LABEL DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 8)
)

sns.barplot(
    data=individual_df,
    x="label",
    y="document_count",
    hue="label",
    palette="viridis",
    legend=False,
    ax=ax,
)

ax.set_title(
    f"Individual Label Distribution "
    f"(Threshold = {THRESHOLD})",
    fontsize=16,
    pad=15,
)

ax.set_xlabel(
    "Label",
    fontsize=13,
)

ax.set_ylabel(
    "Number of Documents",
    fontsize=13,
)

ax.tick_params(
    axis="x",
    rotation=45,
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, _: f"{x:,.0f}"
    )
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "individual_label_distribution.png",
    dpi=DPI,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 7. PLOT TOP N LABEL COMBINATIONS
# ============================================================

plot_df = combination_df.head(
    TOP_N_COMBINATIONS
).copy()

# Replace long combination names only for plotting.
plot_df = plot_df.sort_values(
    "document_count",
    ascending=True,
)

fig_height = max(
    8,
    len(plot_df) * 0.35,
)

fig, ax = plt.subplots(
    figsize=(16, fig_height)
)

sns.barplot(
    data=plot_df,
    x="document_count",
    y="combination",
    color="steelblue",
    ax=ax,
)

ax.set_title(
    f"Top {TOP_N_COMBINATIONS} "
    "Label Combinations",
    fontsize=16,
    pad=15,
)

ax.set_xlabel(
    "Number of Documents",
    fontsize=13,
)

ax.set_ylabel(
    "Label Combination",
    fontsize=13,
)

ax.xaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, _: f"{x:,.0f}"
    )
)

ax.grid(
    axis="x",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "label_combination_distribution_top50.png",
    dpi=DPI,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 8. PLOT LABEL COMBINATIONS USING LOG SCALE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, fig_height)
)

sns.barplot(
    data=plot_df,
    x="document_count",
    y="combination",
    color="darkorange",
    ax=ax,
)

ax.set_xscale("log")

ax.set_title(
    f"Top {TOP_N_COMBINATIONS} "
    "Label Combinations (Log Scale)",
    fontsize=16,
    pad=15,
)

ax.set_xlabel(
    "Number of Documents (Log Scale)",
    fontsize=13,
)

ax.set_ylabel(
    "Label Combination",
    fontsize=13,
)

ax.grid(
    axis="x",
    which="both",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "label_combination_distribution_log.png",
    dpi=DPI,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 9. SAVE SUMMARY
# ============================================================

elapsed_seconds = (
    datetime.now() - overall_start
).total_seconds()


summary = {

    "analysis_date": datetime.now().isoformat(),

    "input_directory": str(INPUT_DIR),

    "output_directory": str(OUTPUT_DIR),

    "shards": SHARDS,

    "total_shards": len(SHARDS),

    "total_documents": total_documents,

    "number_of_labels": NUM_LABELS,

    "labels": LABELS,

    "default_threshold": THRESHOLD,

    "individual_thresholds": LABEL_THRESHOLDS,

    "classified_documents": classified_documents,

    "documents_without_labels": documents_without_labels,

    "documents_without_labels_percentage": round(
        documents_without_labels / total_documents * 100,
        6,
    ),

    "unique_label_combinations": len(
        combination_counts
    ),

    "processing_time_seconds": round(
        elapsed_seconds,
        2,
    ),

    "shard_document_counts": shard_document_counts,

}


summary_file = (
    TABLE_DIR / "distribution_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 10. SAVE ANALYSIS CONFIGURATION
# ============================================================

configuration = {

    "threshold": THRESHOLD,

    "individual_thresholds": LABEL_THRESHOLDS,

    "labels": LABELS,

    "top_n_combinations_in_plot": TOP_N_COMBINATIONS,

    "probabilities_source": "new_web_register",

    "classification_rule": "probability >= threshold",

    "combination_rule": "Exact set of positive labels; "
                        "label order ignored",

    "no_label_documents": "Stored as NO_LABEL",

}

with open(
    OUTPUT_DIR / "configuration.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        configuration,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)

print("DISTRIBUTION ANALYSIS COMPLETED")

print("=" * 80)

print(
    "Total documents:",
    f"{total_documents:,}"
)

print(
    "Unique combinations:",
    f"{len(combination_counts):,}"
)

print("\nIndividual label distribution:")

print(
    individual_df.to_string(
        index=False
    )
)

print("\nTop 20 label combinations:")

print(
    combination_df.head(20)[
        [
            "rank",
            "combination",
            "document_count",
            "percentage_of_all_documents",
        ]
    ].to_string(
        index=False
    )
)

print("\nOutput directory:")

print(OUTPUT_DIR)

print("\nTables:")

print(TABLE_DIR)

print("\nPlots:")

print(PLOT_DIR)

print("\nProcessing time:")

print(
    f"{elapsed_seconds / 60:.2f} minutes"
)

print("\nAll analysis files saved successfully.")