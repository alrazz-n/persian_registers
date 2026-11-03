
import os
import json
import gc
import io
import time
import shutil
import sys

from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import zstandard as zstd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path(
    "/scratch/project_462001491/nima/Hybrid_SP_ID_effect/"
    "Dataset_Hugging_face/without_NA/"
    "_MultiCore_BGE_m3_finetuned_evaluation/"
    "Combined_single_no_NA/final_model"
)

DATA_DIR = Path(
    "/scratch/project_462001491/nima/annotated_data/kept"
)

OUTPUT_ROOT = Path(
    "/scratch/project_462001491/nima/register_annotated_data"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

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

TEXT_COLUMN = "text"

OUTPUT_COLUMN = "new_web_register"

MAX_LENGTH = 1024

BATCH_SIZE = 128

PROBABILITY_DECIMALS = 3

USE_BF16 = True

# Skip only when final output AND completed metadata exist.
SKIP_EXISTING = True

# Zstandard compression level.
COMPRESSION_LEVEL = 3

# Progress reporting.
LOG_EVERY_BATCHES = 100


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    USE_BF16 = (
        USE_BF16
        and torch.cuda.is_bf16_supported()
    )

else:

    DEVICE = torch.device("cpu")

    USE_BF16 = False


print("=" * 80)
print("INFERENCE CONFIGURATION")
print("=" * 80)

print("Device:", DEVICE)
print("BF16:", USE_BF16)
print("Batch size:", BATCH_SIZE)
print("Maximum length:", MAX_LENGTH)
print("Probability decimals:", PROBABILITY_DECIMALS)
print("Model:", MODEL_DIR)
print("Input:", DATA_DIR)
print("Output:", OUTPUT_ROOT)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)

print("Tokenizer loaded.")


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================

print("\nLoading fine-tuned model...")

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR)
)

model.to(DEVICE)

model.eval()

if USE_BF16:

    model.to(dtype=torch.bfloat16)

print("Model loaded.")


# ============================================================
# LABELS
# ============================================================

id2label = model.config.id2label

NUM_LABELS = model.config.num_labels

LABELS = [
    id2label.get(i, id2label.get(str(i)))
    for i in range(NUM_LABELS)
]

if any(label is None for label in LABELS):

    raise ValueError(
        "Could not retrieve all labels from model configuration."
    )

generic_labels = [
    f"LABEL_{i}"
    for i in range(NUM_LABELS)
]

if LABELS == generic_labels:

    raise ValueError(
        "The saved model has generic LABEL_0, LABEL_1, etc. "
        "Please supply the original model_labels in training order."
    )

if len(set(LABELS)) != NUM_LABELS:

    raise ValueError(
        "Duplicate label names detected."
    )

print("\nModel labels:")

print(LABELS)

print("Number of labels:", NUM_LABELS)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_duration(seconds):

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    remaining_seconds = seconds % 60

    return (
        f"{hours}h "
        f"{minutes}m "
        f"{remaining_seconds:.2f}s"
    )


def save_checkpoint(checkpoint_file, checkpoint):

    """
    Save checkpoint atomically.

    First write to a temporary JSON file.
    Then replace the original checkpoint.
    """

    temporary_file = Path(
        str(checkpoint_file) + ".tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint,
            f,
            indent=4,
            ensure_ascii=False,
        )

        f.flush()

        os.fsync(f.fileno())

    os.replace(
        temporary_file,
        checkpoint_file,
    )


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def predict_batch(texts):

    """
    Predict multi-label probabilities.

    Returns:
        numpy array with shape:
        (batch_size, number_of_labels)
    """

    texts = [
        text if isinstance(text, str) else ""
        for text in texts
    ]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        if USE_BF16:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                outputs = model(**encoded)

        else:

            outputs = model(**encoded)

        # Multi-label classification.
        probabilities = torch.sigmoid(
            outputs.logits.float()
        )

    return probabilities.cpu().numpy()


# ============================================================
# COMMIT ONE BATCH
# ============================================================

def commit_batch(
    documents,
    spool_file,
    checkpoint_file,
    checkpoint,
    attempt_start,
    accumulated_seconds,
):

    """
    Save a completed inference batch.

    Order of operations:

    1. Serialize documents.
    2. Compress the batch independently.
    3. Append compressed frame to spool.
    4. Flush and fsync spool.
    5. Update checkpoint atomically.

    A batch is considered committed only after step 5.
    """

    if not documents:

        return checkpoint

    # Serialize documents.
    batch_text = "".join(
        json.dumps(
            doc,
            ensure_ascii=False,
        ) + "\n"
        for doc in documents
    )

    batch_bytes = batch_text.encode("utf-8")

    # Independently compress this batch.
    compressor = zstd.ZstdCompressor(
        level=COMPRESSION_LEVEL
    )

    compressed_batch = compressor.compress(
        batch_bytes
    )

    # Append compressed frame.
    with open(
        spool_file,
        "ab",
    ) as f:

        f.write(compressed_batch)

        f.flush()

        os.fsync(f.fileno())

    # Update checkpoint only after successful spool write.
    checkpoint["documents_processed"] += len(
        documents
    )

    checkpoint["spool_bytes"] = spool_file.stat().st_size

    checkpoint["processing_time_seconds"] = round(
        accumulated_seconds
        + time.perf_counter()
        - attempt_start,
        2,
    )

    checkpoint["last_checkpoint_at"] = (
        datetime.now().isoformat()
    )

    save_checkpoint(
        checkpoint_file,
        checkpoint,
    )

    return checkpoint


# ============================================================
# VALIDATE FINAL OUTPUT
# ============================================================

def validate_output(
    output_file,
    expected_documents,
):

    """
    Validate the complete final Zstandard file.

    Checks:

    - Zstandard decompression.
    - JSON validity.
    - Document count.
    - Presence of new_web_register.
    - Presence of all model labels.
    - Probability values between 0 and 1.
    """

    validated_documents = 0

    with open(
        output_file,
        "rb",
    ) as compressed_file:

        decompressor = zstd.ZstdDecompressor()

        with decompressor.stream_reader(
            compressed_file,
            read_across_frames=True,
        ) as reader:

            with io.TextIOWrapper(
                reader,
                encoding="utf-8",
            ) as text_reader:

                for line in text_reader:

                    if not line.strip():

                        continue

                    document = json.loads(line)

                    if OUTPUT_COLUMN not in document:

                        raise ValueError(
                            f"Missing {OUTPUT_COLUMN} "
                            f"in document {validated_documents}"
                        )

                    probabilities = document[
                        OUTPUT_COLUMN
                    ]

                    if set(probabilities.keys()) != set(LABELS):

                        raise ValueError(
                            f"Label mismatch in document "
                            f"{validated_documents}"
                        )

                    for label, probability in probabilities.items():

                        if not isinstance(
                            probability,
                            (int, float),
                        ):

                            raise ValueError(
                                f"Invalid probability for {label}"
                            )

                        if not 0 <= probability <= 1:

                            raise ValueError(
                                f"Probability outside [0, 1]: "
                                f"{label}={probability}"
                            )

                    validated_documents += 1

    if validated_documents != expected_documents:

        raise ValueError(
            f"Document count mismatch! "
            f"Expected: {expected_documents}, "
            f"Found: {validated_documents}"
        )

    return validated_documents


# ============================================================
# PROCESS ONE SHARD
# ============================================================

def process_shard(shard_name):

    input_file = DATA_DIR / shard_name

    output_file = OUTPUT_ROOT / shard_name

    # Temporary files.
    spool_file = OUTPUT_ROOT / (
        shard_name.replace(
            ".jsonl.zst",
            "_spool.zst",
        )
    )

    checkpoint_file = OUTPUT_ROOT / (
        shard_name.replace(
            ".jsonl.zst",
            "_checkpoint.json",
        )
    )

    final_partial = Path(
        str(output_file) + ".partial"
    )

    metadata_file = OUTPUT_ROOT / (
        shard_name.replace(
            ".jsonl.zst",
            "_metadata.json",
        )
    )

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input shard does not exist: {input_file}"
        )

    input_stat = input_file.stat()

    input_identity = {
        "input_size_bytes": input_stat.st_size,
        "input_mtime_ns": input_stat.st_mtime_ns,
        "model": str(MODEL_DIR),
        "labels": LABELS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "probability_decimals": PROBABILITY_DECIMALS,
    }

    # --------------------------------------------------------
    # SKIP ONLY COMPLETED SHARDS
    # --------------------------------------------------------

    if (
        SKIP_EXISTING
        and output_file.exists()
        and metadata_file.exists()
    ):

        with open(
            metadata_file,
            "r",
            encoding="utf-8",
        ) as f:

            existing_metadata = json.load(f)

        if existing_metadata.get("status") == "completed":

            print(
                f"\nAlready completed: {shard_name}"
            )

            print("Skipping shard.")

            return

    # --------------------------------------------------------
    # RESTORE OR CREATE CHECKPOINT
    # --------------------------------------------------------

    if checkpoint_file.exists():

        print("\nExisting checkpoint found!")

        with open(
            checkpoint_file,
            "r",
            encoding="utf-8",
        ) as f:

            checkpoint = json.load(f)

        # Prevent accidental resume with changed inputs/config.
        for key, value in input_identity.items():

            if checkpoint.get(key) != value:

                raise RuntimeError(
                    f"Checkpoint mismatch for {key} "
                    f"in {shard_name}. "
                    f"Do not resume with changed inputs "
                    f"or configuration."
                )

        processed_documents = checkpoint[
            "documents_processed"
        ]

        committed_spool_bytes = checkpoint[
            "spool_bytes"
        ]

        accumulated_seconds = checkpoint.get(
            "processing_time_seconds",
            0.0,
        )

        print(
            "Previously processed:",
            f"{processed_documents:,}",
        )

        print(
            "Previously accumulated time:",
            format_duration(accumulated_seconds),
        )

        if not spool_file.exists():

            raise RuntimeError(
                "Checkpoint exists but spool file is missing. "
                "Cannot safely resume."
            )

        if spool_file.stat().st_size < committed_spool_bytes:

            raise RuntimeError(
                "Spool file is smaller than the committed "
                "checkpoint size. Cannot safely resume."
            )

        # Remove uncommitted spool data.
        with open(
            spool_file,
            "r+b",
        ) as f:

            f.truncate(
                committed_spool_bytes
            )

            f.flush()

            os.fsync(f.fileno())

        print(
            "Committed spool successfully recovered."
        )

    else:

        print("\nStarting a new shard.")

        processed_documents = 0

        accumulated_seconds = 0.0

        # Create empty spool.
        with open(
            spool_file,
            "wb",
        ) as f:

            f.flush()

            os.fsync(f.fileno())

        checkpoint = {

            "shard": shard_name,

            **input_identity,

            "documents_processed": 0,

            "spool_bytes": 0,

            "started_at": datetime.now().isoformat(),

            "processing_time_seconds": 0.0,

            "resume_count": 0,

        }

        save_checkpoint(
            checkpoint_file,
            checkpoint,
        )

    # --------------------------------------------------------
    # START CURRENT ATTEMPT
    # --------------------------------------------------------

    checkpoint["resume_count"] = (
        checkpoint.get("resume_count", 0) + 1
    )

    save_checkpoint(
        checkpoint_file,
        checkpoint,
    )

    attempt_start = time.perf_counter()

    documents = []

    total_documents = processed_documents

    batch_number = 0

    input_documents_seen = 0

    print("\n" + "=" * 80)

    print("PROCESSING:", shard_name)

    print("Previously processed:", processed_documents)

    print("Attempt:", checkpoint["resume_count"])

    print("=" * 80)

    # --------------------------------------------------------
    # STREAM INPUT AND RESUME
    # --------------------------------------------------------

    with zstd.open(
        input_file,
        mode="rt",
        encoding="utf-8",
    ) as reader:

        for line in reader:

            if not line.strip():

                continue

            input_documents_seen += 1

            # Skip documents already committed.
            if input_documents_seen <= processed_documents:

                continue

            document = json.loads(line)

            if TEXT_COLUMN not in document:

                raise KeyError(
                    f"Missing '{TEXT_COLUMN}' in "
                    f"{shard_name}, document "
                    f"{input_documents_seen}"
                )

            documents.append(document)

            # ------------------------------------------------
            # PROCESS FULL BATCH
            # ------------------------------------------------

            if len(documents) >= BATCH_SIZE:

                batch_number += 1

                probabilities = predict_batch(
                    [
                        doc[TEXT_COLUMN]
                        for doc in documents
                    ]
                )

                # Add probability dictionaries.
                for doc, probs in zip(
                    documents,
                    probabilities,
                ):

                    doc[OUTPUT_COLUMN] = {

                        label: round(
                            float(prob),
                            PROBABILITY_DECIMALS,
                        )

                        for label, prob in zip(
                            LABELS,
                            probs,
                        )
                    }

                # Commit completed batch.
                checkpoint = commit_batch(
                    documents=documents,
                    spool_file=spool_file,
                    checkpoint_file=checkpoint_file,
                    checkpoint=checkpoint,
                    attempt_start=attempt_start,
                    accumulated_seconds=accumulated_seconds,
                )

                total_documents = checkpoint[
                    "documents_processed"
                ]

                documents.clear()

                if batch_number % LOG_EVERY_BATCHES == 0:

                    print(
                        f"Processed: "
                        f"{total_documents:,} documents"
                    )

    # --------------------------------------------------------
    # PROCESS REMAINING DOCUMENTS
    # --------------------------------------------------------

    if documents:

        probabilities = predict_batch(
            [
                doc[TEXT_COLUMN]
                for doc in documents
            ]
        )

        for doc, probs in zip(
            documents,
            probabilities,
        ):

            doc[OUTPUT_COLUMN] = {

                label: round(
                    float(prob),
                    PROBABILITY_DECIMALS,
                )

                for label, prob in zip(
                    LABELS,
                    probs,
                )
            }

        checkpoint = commit_batch(
            documents=documents,
            spool_file=spool_file,
            checkpoint_file=checkpoint_file,
            checkpoint=checkpoint,
            attempt_start=attempt_start,
            accumulated_seconds=accumulated_seconds,
        )

        total_documents = checkpoint[
            "documents_processed"
        ]

        documents.clear()

    # --------------------------------------------------------
    # VERIFY INPUT COUNT
    # --------------------------------------------------------

    if input_documents_seen != total_documents:

        raise RuntimeError(
            f"Input/output count mismatch for {shard_name}. "
            f"Input records: {input_documents_seen}, "
            f"Committed records: {total_documents}"
        )

    print(
        "\nAll documents processed:",
        f"{total_documents:,}",
    )

    # --------------------------------------------------------
    # FINALIZE COMPRESSED OUTPUT
    # --------------------------------------------------------

    print("\nFinalizing compressed output...")

    final_compressor = zstd.ZstdCompressor(
        level=COMPRESSION_LEVEL
    )

    with open(
        spool_file,
        "rb",
    ) as spool_input:

        with open(
            final_partial,
            "wb",
        ) as output_stream:

            with zstd.ZstdDecompressor().stream_reader(
                spool_input,
                read_across_frames=True,
            ) as decompressed_stream:

                with final_compressor.stream_writer(
                    output_stream,
                    closefd=False,
                ) as compressed_writer:

                    shutil.copyfileobj(
                        decompressed_stream,
                        compressed_writer,
                        length=8 * 1024 * 1024,
                    )

            output_stream.flush()

            os.fsync(
                output_stream.fileno()
            )

    # --------------------------------------------------------
    # VALIDATE FINAL OUTPUT
    # --------------------------------------------------------

    print("\nValidating final output...")

    validated_documents = validate_output(
        final_partial,
        total_documents,
    )

    print(
        "Validation successful:",
        f"{validated_documents:,} documents",
    )

    # --------------------------------------------------------
    # ATOMICALLY PUBLISH OUTPUT
    # --------------------------------------------------------

    os.replace(
        final_partial,
        output_file,
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    processing_seconds = (
        accumulated_seconds
        + time.perf_counter()
        - attempt_start
    )

    processed_at = datetime.now()

    metadata = {

        "shard": shard_name,

        "input_file": str(input_file),

        "output_file": str(output_file),

        "model": str(MODEL_DIR),

        "started_at": checkpoint["started_at"],

        "processed_at": processed_at.isoformat(),

        "processing_time_seconds": round(
            processing_seconds,
            2,
        ),

        "processing_time_human": format_duration(
            processing_seconds
        ),

        "total_documents": total_documents,

        "validated_documents": validated_documents,

        "batch_size": BATCH_SIZE,

        "max_length": MAX_LENGTH,

        "probability_decimals": PROBABILITY_DECIMALS,

        "number_of_labels": NUM_LABELS,

        "labels": LABELS,

        "device": str(DEVICE),

        "bf16": USE_BF16,

        "resume_count": checkpoint["resume_count"],

        "input_size_bytes": input_file.stat().st_size,

        "output_size_bytes": output_file.stat().st_size,

        "output_size_gb": round(
            output_file.stat().st_size / (1024 ** 3),
            3,
        ),

        "status": "completed",

    }

    save_checkpoint(
        metadata_file,
        metadata,
    )

    # --------------------------------------------------------
    # CLEAN UP TEMPORARY FILES
    # --------------------------------------------------------

    # Cleanup only after:
    # 1. Final output is published.
    # 2. Final output is validated.
    # 3. Metadata is saved.

    spool_file.unlink(
        missing_ok=True
    )

    checkpoint_file.unlink(
        missing_ok=True
    )

    print("\n" + "=" * 80)

    print("SHARD COMPLETED SUCCESSFULLY!")

    print("=" * 80)

    print("Documents:", f"{total_documents:,}")

    print(
        "Processing time:",
        format_duration(processing_seconds),
    )

    print("Output:", output_file)

    print("Metadata:", metadata_file)

    print("Recovery files cleaned up.")


# ============================================================
# PROCESS ONE SHARD PER SLURM ARRAY TASK
# ============================================================

print("\nStarting HPLT3 inference...")

overall_start = time.perf_counter()

# Get SLURM array task ID.
array_task_id = int(
    os.environ.get("SLURM_ARRAY_TASK_ID", "0")
)

# Validate array index.
if array_task_id < 0 or array_task_id >= len(SHARDS):

    raise ValueError(
        f"Invalid SLURM_ARRAY_TASK_ID: {array_task_id}. "
        f"Expected 0 to {len(SHARDS) - 1}."
    )

# Select exactly one shard.
shard = SHARDS[array_task_id]

print("\n" + "#" * 80)

print(
    f"ARRAY TASK: {array_task_id}"
)

print(
    f"PROCESSING SHARD: {shard}"
)

print("#" * 80)

try:

    process_shard(shard)

except Exception as e:

    print("\n" + "!" * 80)

    print("ERROR PROCESSING SHARD:", shard)

    print("ERROR:", str(e))

    print("Checkpoint files have been retained if created.")

    print("!" * 80)

    raise

finally:

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# FINAL SUMMARY
# ============================================================

overall_seconds = (
    time.perf_counter() - overall_start
)

print("\n" + "=" * 80)

print("HPLT3 INFERENCE FINISHED")

print("=" * 80)

print("Array task ID:", array_task_id)

print("Processed shard:", shard)

print(
    "Current run duration:",
    format_duration(overall_seconds),
)

print("\nOutput directory:")

print(OUTPUT_ROOT)

print("\nInference pipeline finished.")