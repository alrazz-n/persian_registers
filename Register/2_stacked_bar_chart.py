
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
# 1. CONFIGURATION
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


# ============================================================
# 2. CLASSIFICATION THRESHOLDS
# ============================================================

THRESHOLD = 0.31

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

    "it": 0.30,

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
    "ed": 0.18,
}


# ============================================================
# 3. HIERARCHICAL TAXONOMY
# ============================================================

LABELS_STRUCTURE = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}


# ============================================================
# 4. PLOT CONFIGURATION
# ============================================================

DPI = 300

sns.set_theme(
    style="whitegrid",
    context="talk",
)

# Colors assigned consistently to subcategories.
PALETTE = sns.color_palette(
    "tab20",
    20,
)

MAIN_ONLY_COLOR = "#BDBDBD"

# Labels displayed inside bars only above this threshold.
ANNOTATION_THRESHOLD = 0.04


# ============================================================
# 5. LOAD METADATA
# ============================================================

metadata_file = INPUT_DIR / "10_1_metadata.json"

if not metadata_file.exists():
    raise FileNotFoundError(metadata_file)

with open(
    metadata_file,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


LABELS = metadata["labels"]

print("=" * 80)
print("HIERARCHICAL LABEL DISTRIBUTION ANALYSIS")
print("=" * 80)

print("Labels:", LABELS)
print("Number of labels:", len(LABELS))


# ============================================================
# 6. VALIDATE CONFIGURATION
# ============================================================

# Every taxonomy label must exist in metadata.

for main_label, sub_labels in LABELS_STRUCTURE.items():

    if main_label not in LABELS:
        raise ValueError(
            f"Main category missing from metadata: {main_label}"
        )

    for sub_label in sub_labels:

        if sub_label not in LABELS:
            raise ValueError(
                f"Subcategory missing from metadata: {sub_label}"
            )


# Check that every label belongs to exactly one position
# in the hierarchy.

taxonomy_labels = []

for main_label, sub_labels in LABELS_STRUCTURE.items():

    taxonomy_labels.append(main_label)
    taxonomy_labels.extend(sub_labels)


if len(taxonomy_labels) != len(set(taxonomy_labels)):
    raise ValueError(
        "A label appears more than once in LABELS_STRUCTURE."
    )


if set(taxonomy_labels) != set(LABELS):
    raise ValueError(
        "LABELS_STRUCTURE does not match metadata labels. "
        f"Missing: {set(LABELS) - set(taxonomy_labels)}; "
        f"Extra: {set(taxonomy_labels) - set(LABELS)}"
    )


for label, threshold in LABEL_THRESHOLDS.items():

    if label not in LABELS:
        raise ValueError(
            f"Unknown threshold label: {label}"
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
# 7. INITIALIZE COUNTERS
# ============================================================

# Individual label counts (optional diagnostic).

individual_counts = Counter()


# Hierarchical counts.

# Example:
#
# hierarchical_counts["NA"]["NA"]
#     = NA without any positive subcategory
#
# hierarchical_counts["NA"]["ne"]
#     = NA with positive ne
#
# hierarchical_counts["NA"]["sr"]
#     = NA with positive sr

hierarchical_counts = {

    main_label: Counter()

    for main_label in LABELS_STRUCTURE

}


total_documents = 0

documents_without_labels = 0

shard_document_counts = {}

na_subcategory_count_distribution = Counter()


# ============================================================
# 8. PROCESS ALL SHARDS
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
        raise FileNotFoundError(input_file)

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
                    f"Missing new_web_register in {shard_name}"
                )

            probabilities = document["new_web_register"]

            if set(probabilities.keys()) != set(LABELS):

                raise ValueError(
                    f"Label mismatch in {shard_name}, "
                    f"document {shard_count}"
                )


            # ------------------------------------------------
            # IDENTIFY POSITIVE LABELS
            # ------------------------------------------------

            positive_labels = {

                label

                for label in LABELS

                if float(probabilities[label])
                >= get_threshold(label)

            }

            # Number of positive NA subcategories per NA-positive document

            if "NA" in positive_labels:

                na_subcategory_count = sum(
                    sub in positive_labels
                    for sub in LABELS_STRUCTURE["NA"]
                )

                na_subcategory_count_distribution[
                    na_subcategory_count
                ] += 1


            # ------------------------------------------------
            # INDIVIDUAL LABEL COUNTS
            # ------------------------------------------------

            for label in positive_labels:

                individual_counts[label] += 1


            # ------------------------------------------------
            # HIERARCHICAL LABEL COUNTS
            # ------------------------------------------------

            for main_label, sub_labels in LABELS_STRUCTURE.items():

                # Parent category is not positive.
                if main_label not in positive_labels:
                    continue


                # Main category has no subcategories.
                if not sub_labels:

                    hierarchical_counts[
                        main_label
                    ][main_label] += 1

                    continue


                # Identify positive subcategories.
                positive_subcategories = [

                    sub_label

                    for sub_label in sub_labels

                    if sub_label in positive_labels

                ]


                # Parent is positive, but no subcategory
                # is positive.
                if not positive_subcategories:

                    hierarchical_counts[
                        main_label
                    ][main_label] += 1


                # Only count positive subcategories.
                # Do not count the parent separately.
                else:

                    for sub_label in positive_subcategories:

                        hierarchical_counts[
                            main_label
                        ][sub_label] += 1


            # ------------------------------------------------
            # DOCUMENT COUNTS
            # ------------------------------------------------

            if not positive_labels:

                documents_without_labels += 1

            total_documents += 1

            shard_count += 1


            if shard_count % 1_000_000 == 0:

                print(
                    f"Processed {shard_count:,} "
                    "documents in this shard"
                )


    shard_document_counts[shard_name] = shard_count

    print(
        "Shard documents:",
        f"{shard_count:,}"
    )

    print(
        "Total documents:",
        f"{total_documents:,}"
    )

    gc.collect()


# ============================================================
# 9. VALIDATE TOTAL DOCUMENTS
# ============================================================

if total_documents == 0:

    raise ValueError(
        "No documents were processed."
    )


print("\n" + "=" * 80)
print("PROCESSING COMPLETED")
print("=" * 80)

print(
    "Total documents:",
    f"{total_documents:,}"
)

print(
    "Documents without positive labels:",
    f"{documents_without_labels:,}"
)

# ============================================================
# 9.1. NA SUBCATEGORY COUNT DISTRIBUTION
# ============================================================

na_distribution_df = pd.DataFrame([

    {
        "number_of_positive_subcategories": number,
        "document_count": count,
        "percentage_of_NA_documents": round(
            count / individual_counts["NA"] * 100,
            4,
        ),
    }

    for number, count in sorted(
        na_subcategory_count_distribution.items()
    )

])

na_distribution_df.to_csv(
    TABLE_DIR / "NA_subcategory_count_distribution.csv",
    index=False,
)

print("\nNA SUBCATEGORY COUNT DISTRIBUTION")

print(
    na_distribution_df.to_string(index=False)
)


# ============================================================
# 10. VALIDATE HIERARCHICAL COUNTS
# ============================================================

print("\nHIERARCHICAL COUNT VALIDATION")
print("-" * 80)

validation_rows = []

for main_label, sub_labels in LABELS_STRUCTURE.items():

    expected = individual_counts[main_label]

    parent_only = hierarchical_counts[
        main_label
    ][main_label]

    subcategory_assignments = sum(
        hierarchical_counts[main_label][sub]
        for sub in sub_labels
    )

    observed = parent_only + subcategory_assignments

    # Documents with at least one positive subcategory
    # are counted once here, regardless of how many
    # subcategories they have.

    documents_with_subcategories = sum(
        hierarchical_counts[main_label][sub]
        for sub in sub_labels
    )

    # This is the number of subcategory assignments
    # beyond the first assignment per document.
    #
    # We cannot calculate this from aggregate counters alone.
    # Therefore, the difference below is NOT a valid
    # document-level validation.

    if observed < expected:

        raise RuntimeError(
            f"Hierarchical counts are unexpectedly lower "
            f"than parent counts for {main_label}."
        )

    validation_rows.append({

        "main_category": main_label,

        "parent_documents": expected,

        "parent_only_documents": parent_only,

        "subcategory_assignments": subcategory_assignments,

        "total_hierarchical_assignments": observed,

        "additional_assignments": observed - expected,

    })


validation_df = pd.DataFrame(
    validation_rows
)

print(
    validation_df.to_string(
        index=False
    )
)

print("\nHierarchical counting completed.")



# ============================================================
# 11. CREATE HIERARCHICAL DISTRIBUTION TABLE
# ============================================================

hierarchical_rows = []

for main_label, sub_labels in LABELS_STRUCTURE.items():

    segments = [main_label] + sub_labels

    for segment in segments:

        count = hierarchical_counts[
            main_label
        ][segment]

        hierarchical_rows.append({

            "main_category": main_label,

            "segment": segment,

            "document_count": count,

            "percentage_of_all_documents": round(
                count / total_documents * 100,
                6,
            ),

        })


hierarchical_df = pd.DataFrame(
    hierarchical_rows
)


# Save CSV.

hierarchical_csv = (
    TABLE_DIR /
    "hierarchical_label_distribution.csv"
)

hierarchical_df.to_csv(
    hierarchical_csv,
    index=False,
)


# ============================================================
# 12. PREPARE PIVOT TABLE
# ============================================================

pivot_df = hierarchical_df.pivot(
    index="main_category",
    columns="segment",
    values="document_count",
).fillna(0)


# Preserve taxonomy order.

category_order = list(
    LABELS_STRUCTURE.keys()
)

pivot_df = pivot_df.reindex(
    category_order
)


# Ensure every expected segment exists,
# including zero-count segments.

all_segments = []

for main_label, sub_labels in LABELS_STRUCTURE.items():

    all_segments.append(main_label)

    all_segments.extend(sub_labels)


pivot_df = pivot_df.reindex(
    columns=all_segments,
    fill_value=0,
)


# ============================================================
# 13. DEFINE CONSISTENT COLORS
# ============================================================

segment_colors = {}

color_index = 0

for main_label, sub_labels in LABELS_STRUCTURE.items():

    # Main-only segment.
    segment_colors[main_label] = MAIN_ONLY_COLOR

    for sub_label in sub_labels:

        segment_colors[sub_label] = PALETTE[
            color_index % len(PALETTE)
        ]

        color_index += 1


# ============================================================
# 14. ABSOLUTE COUNT STACKED BAR CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 9)
)

bottom = pd.Series(
    0.0,
    index=pivot_df.index,
)


for segment in all_segments:

    values = pivot_df[segment]

    if values.sum() == 0:
        continue

    ax.bar(
        pivot_df.index,
        values,
        bottom=bottom,
        label=segment,
        color=segment_colors[segment],
        edgecolor="white",
        linewidth=0.7,
    )


    # Add segment labels only when sufficiently large.

    for x_position, (value, current_bottom) in enumerate(
        zip(values, bottom)
    ):

        if value == 0:
            continue

        if value / total_documents < ANNOTATION_THRESHOLD:
            continue

        ax.text(
            x_position,
            current_bottom + value / 2,
            f"{value:,.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )


    bottom += values


ax.set_title(
    "Hierarchical Label Distribution "
    "(Absolute Document Counts)",
    fontsize=17,
    pad=20,
)

ax.set_xlabel(
    "Main Categories",
    fontsize=13,
)

ax.set_ylabel(
    "Number of Documents",
    fontsize=13,
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, _: f"{x:,.0f}"
    )
)

ax.legend(
    title="Labels",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=10,
)

ax.grid(
    axis="y",
    alpha=0.3,
)

ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "hierarchical_stacked_bar_counts.png",
    dpi=DPI,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 15. 100% STACKED BAR CHART
# ============================================================

# Normalize each main category by its own total.
# Each bar therefore represents 100% of documents
# assigned to that main category.

category_totals = pivot_df.sum(
    axis=1
)


percentage_pivot = pivot_df.div(
    category_totals.replace(0, float("nan")),
    axis=0,
).mul(100).fillna(0)


fig, ax = plt.subplots(
    figsize=(16, 9)
)

bottom = pd.Series(
    0.0,
    index=percentage_pivot.index,
)


for segment in all_segments:

    values = percentage_pivot[segment]

    if values.sum() == 0:
        continue

    ax.bar(
        percentage_pivot.index,
        values,
        bottom=bottom,
        label=segment,
        color=segment_colors[segment],
        edgecolor="white",
        linewidth=0.7,
    )


    # Add percentage labels.

    for x_position, (value, current_bottom) in enumerate(
        zip(values, bottom)
    ):

        if value < 3:
            continue

        ax.text(
            x_position,
            current_bottom + value / 2,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )


    bottom += values


ax.set_title(
    "Hierarchical Label Distribution "
    "(100% Stacked)",
    fontsize=17,
    pad=20,
)

ax.set_xlabel(
    "Main Categories",
    fontsize=13,
)

ax.set_ylabel(
    "Percentage of Documents Within Main Category",
    fontsize=13,
)

ax.set_ylim(
    0,
    100,
)

ax.set_yticks(
    range(0, 101, 10)
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, _: f"{x:.0f}%"
    )
)

ax.legend(
    title="Labels",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=10,
)

ax.grid(
    axis="y",
    alpha=0.3,
)

ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "hierarchical_stacked_bar_percentage.png",
    dpi=DPI,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 16. SAVE SUMMARY
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

    "documents_without_labels": documents_without_labels,

    "classified_documents": (
        total_documents - documents_without_labels
    ),

    "labels": LABELS,

    "taxonomy": LABELS_STRUCTURE,

    "default_threshold": THRESHOLD,

    "individual_thresholds": LABEL_THRESHOLDS,

    "classification_rule": "probability >= threshold",

    "hierarchical_counting_rule": (
        "Count positive subcategories instead of their "
        "positive parent; count parent alone when no "
        "subcategory is positive."
    ),

    "processing_time_seconds": round(
        elapsed_seconds,
        2,
    ),

    "shard_document_counts": shard_document_counts,

}


with open(
    TABLE_DIR / "hierarchical_distribution_summary.json",
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
# 17. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)

print("HIERARCHICAL DISTRIBUTION ANALYSIS COMPLETED")

print("=" * 80)

print(
    "Total documents:",
    f"{total_documents:,}"
)

print(
    "Documents without positive labels:",
    f"{documents_without_labels:,}"
)


print("\nHIERARCHICAL DISTRIBUTION:")

print(
    hierarchical_df.to_string(
        index=False
    )
)


print("\nOUTPUT FILES:")

print("\nCSV:")

print(
    hierarchical_csv
)

print("\nPLOTS:")

print(
    PLOT_DIR / "hierarchical_stacked_bar_counts.png"
)

print(
    PLOT_DIR / "hierarchical_stacked_bar_percentage.png"
)

print("\nSUMMARY:")

print(
    TABLE_DIR / "hierarchical_distribution_summary.json"
)


print("\nProcessing time:")

print(
    f"{elapsed_seconds / 60:.2f} minutes"
)

print("\nAll files saved successfully.")

print("=" * 80)
