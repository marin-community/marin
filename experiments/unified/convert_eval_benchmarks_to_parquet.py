#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Convert multimodal evaluation benchmarks to parquet format for the unified image-text model.

Downloads benchmarks from HuggingFace and produces sharded parquet files with the same
base schema as training data (messages, images, source) plus extra evaluation columns
(answer, choices, task_type, question_id, benchmark, split, metadata).

Supported benchmarks:
  Understanding: VQAv2, TextVQA, GQA, ChartQA, AI2D, MMMU.
  Generation:    CIFAR-10 (small/full), ImageNet (small/full).

Usage:
    # Convert all benchmarks (defaults: --output-gcs gs://marin-vlm/eval_benchmarks
    #                                    --output-local /tmp/eval_benchmarks)
    uv run experiments/unified/convert_eval_benchmarks_to_parquet.py

    # Convert specific benchmarks
    uv run experiments/unified/convert_eval_benchmarks_to_parquet.py \
        --benchmarks vqav2 textvqa

    # Convert generation benchmarks (small variants for fast iteration)
    uv run experiments/unified/convert_eval_benchmarks_to_parquet.py \
        --benchmarks cifar10_small imagenet_small

    # Test with small subset, local only
    uv run experiments/unified/convert_eval_benchmarks_to_parquet.py \
        --benchmarks chartqa --max-rows 10 --output-gcs ""
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

# Letter labels for multiple choice formatting
MC_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Default rows per output shard (eval datasets are smaller than training)
DEFAULT_ROWS_PER_SHARD = 500

# Batch size for processing large datasets (VQAv2)
BATCH_SIZE = 10_000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_image_bytes(image) -> bytes | None:
    """Extract PNG bytes from various image formats (PIL Image, dict, raw bytes)."""
    if image is None:
        return None
    if hasattr(image, "tobytes"):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        result = buf.getvalue()
        buf.close()
        return result
    elif isinstance(image, dict) and "bytes" in image:
        return image["bytes"]
    elif isinstance(image, bytes):
        return image
    return None


def build_eval_messages(
    question_text: str,
    answer_text: str,
    num_images: int = 1,
) -> list[dict]:
    """Build Levanter-format messages for an eval example.

    The user message contains image placeholder(s) followed by the question.
    The assistant message contains the ground truth answer.
    """
    user_content: list[dict] = []
    for _ in range(num_images):
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": question_text})

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
    ]


def format_mc_options(options: list[str]) -> str:
    """Format multiple choice options as A. / B. / C. / D. lines."""
    lines = []
    for i, opt in enumerate(options):
        lines.append(f"{MC_LETTERS[i]}. {opt}")
    return "\n".join(lines)


def build_generation_messages(prompt_text: str) -> list[dict]:
    """Build Levanter-format messages for a generation eval example.

    The user message contains only a text prompt (no images).
    The assistant message contains an image placeholder referencing images[0].
    """
    return [
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        {"role": "assistant", "content": [{"type": "image"}]},
    ]


def class_balanced_sample(dataset: Dataset, label_column: str, n_per_class: int) -> Dataset:
    """Select exactly n_per_class examples per class from a classification dataset."""
    labels = np.array(dataset[label_column])
    indices = []
    for class_idx in range(labels.max() + 1):
        class_indices = np.where(labels == class_idx)[0][:n_per_class]
        indices.extend(class_indices.tolist())
    return dataset.select(indices)


# ---------------------------------------------------------------------------
# Base converter
# ---------------------------------------------------------------------------


class BenchmarkConverter:
    """Base class for benchmark-specific conversion logic."""

    name: str
    hf_dataset_id: str
    hf_split: str
    task_type: str
    hf_config: str | None = None

    def load(self, max_rows: int | None = None) -> Dataset:
        """Load the dataset from HuggingFace."""
        ds = load_dataset(self.hf_dataset_id, name=self.hf_config, split=self.hf_split, trust_remote_code=True)
        if max_rows is not None:
            ds = ds.select(range(min(max_rows, len(ds))))
        return ds

    def convert_row(self, item: dict, index: int) -> dict | None:
        """Convert a single HF row to the output schema.

        Returns None if the row should be skipped (e.g., missing image).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# VQAv2
# ---------------------------------------------------------------------------


class VQAv2Converter(BenchmarkConverter):
    name = "vqav2"
    hf_dataset_id = "HuggingFaceM4/VQAv2"
    hf_split = "validation"
    task_type = "open_ended"

    def convert_row(self, item: dict, index: int) -> dict | None:
        image_bytes = extract_image_bytes(item.get("image"))
        if image_bytes is None:
            return None

        question = item["question"]
        answer = item["multiple_choice_answer"]
        all_answers = item.get("answers", [])

        question_text = f"{question}\nAnswer the question using a single word or phrase."
        messages = build_eval_messages(question_text, answer)

        return {
            "messages": messages,
            "images": [{"bytes": image_bytes}],
            "source": self.name,
            "answer": answer,
            "choices": None,
            "task_type": self.task_type,
            "question_id": f"vqav2_val_{item['question_id']}",
            "benchmark": self.name,
            "split": "validation",
            "metadata": json.dumps({"answers": all_answers}),
        }


# ---------------------------------------------------------------------------
# TextVQA
# ---------------------------------------------------------------------------


class TextVQAConverter(BenchmarkConverter):
    name = "textvqa"
    hf_dataset_id = "facebook/textvqa"
    hf_split = "validation"
    task_type = "open_ended"

    def convert_row(self, item: dict, index: int) -> dict | None:
        image_bytes = extract_image_bytes(item.get("image"))
        if image_bytes is None:
            return None

        question = item["question"]
        answers = item.get("answers", [])
        # Most common answer across 10 annotators
        answer = max(set(answers), key=answers.count) if answers else ""
        if not answer:
            return None

        question_text = f"{question}\nAnswer the question using a single word or phrase."
        messages = build_eval_messages(question_text, answer)

        return {
            "messages": messages,
            "images": [{"bytes": image_bytes}],
            "source": self.name,
            "answer": answer,
            "choices": None,
            "task_type": self.task_type,
            "question_id": f"textvqa_val_{item.get('question_id', index)}",
            "benchmark": self.name,
            "split": "validation",
            "metadata": json.dumps({"answers": answers}),
        }


# ---------------------------------------------------------------------------
# GQA (requires joining images + instructions configs)
# ---------------------------------------------------------------------------


class GQAConverter(BenchmarkConverter):
    name = "gqa"
    hf_dataset_id = "lmms-lab/GQA"
    hf_split = "testdev"
    task_type = "open_ended"

    def __init__(self):
        self._image_map: dict[str, Any] | None = None

    def load(self, max_rows: int | None = None) -> Dataset:
        """Load both images and instructions configs and join them."""
        logger.info("Loading GQA images config (testdev_balanced_images)...")
        images_ds = load_dataset(self.hf_dataset_id, "testdev_balanced_images", split=self.hf_split, trust_remote_code=True)

        # Build image lookup: imageId -> PIL image
        logger.info("Building image lookup map (%d images)...", len(images_ds))
        self._image_map = {}
        for row in images_ds:
            self._image_map[row["id"]] = row["image"]
        logger.info("Image map built with %d entries.", len(self._image_map))

        logger.info("Loading GQA instructions config (testdev_balanced_instructions)...")
        instructions_ds = load_dataset(self.hf_dataset_id, "testdev_balanced_instructions", split=self.hf_split, trust_remote_code=True)

        if max_rows is not None:
            instructions_ds = instructions_ds.select(range(min(max_rows, len(instructions_ds))))

        return instructions_ds

    def convert_row(self, item: dict, index: int) -> dict | None:
        image_id = item.get("imageId", "")
        image = self._image_map.get(image_id) if self._image_map else None
        image_bytes = extract_image_bytes(image)
        if image_bytes is None:
            logger.warning("Missing image for GQA question id=%s, imageId=%s", item.get("id"), image_id)
            return None

        question = item["question"]
        answer = item["answer"]

        question_text = f"{question}\nAnswer the question using a single word or phrase."
        messages = build_eval_messages(question_text, answer)

        return {
            "messages": messages,
            "images": [{"bytes": image_bytes}],
            "source": self.name,
            "answer": answer,
            "choices": None,
            "task_type": self.task_type,
            "question_id": f"gqa_testdev_{item.get('id', index)}",
            "benchmark": self.name,
            "split": "testdev",
            "metadata": json.dumps({"fullAnswer": item.get("fullAnswer", "")}),
        }


# ---------------------------------------------------------------------------
# ChartQA
# ---------------------------------------------------------------------------


class ChartQAConverter(BenchmarkConverter):
    name = "chartqa"
    hf_dataset_id = "lmms-lab/ChartQA"
    hf_split = "test"
    task_type = "open_ended"

    def convert_row(self, item: dict, index: int) -> dict | None:
        image_bytes = extract_image_bytes(item.get("image"))
        if image_bytes is None:
            return None

        question = item["question"]
        answer = str(item["answer"])

        question_text = f"{question}\nAnswer the question about the chart concisely."
        messages = build_eval_messages(question_text, answer)

        return {
            "messages": messages,
            "images": [{"bytes": image_bytes}],
            "source": self.name,
            "answer": answer,
            "choices": None,
            "task_type": self.task_type,
            "question_id": f"chartqa_test_{index}",
            "benchmark": self.name,
            "split": "test",
            "metadata": json.dumps({"type": item.get("type", "")}),
        }


# ---------------------------------------------------------------------------
# AI2D
# ---------------------------------------------------------------------------


class AI2DConverter(BenchmarkConverter):
    name = "ai2d"
    hf_dataset_id = "lmms-lab/ai2d"
    hf_split = "test"
    task_type = "multiple_choice"

    def convert_row(self, item: dict, index: int) -> dict | None:
        image_bytes = extract_image_bytes(item.get("image"))
        if image_bytes is None:
            return None

        question = item["question"]
        options = item["options"]
        answer_idx = int(item["answer"])
        answer_letter = MC_LETTERS[answer_idx]

        formatted_options = format_mc_options(options)
        question_text = f"{question}\n{formatted_options}\nAnswer with the option letter."
        messages = build_eval_messages(question_text, answer_letter)

        return {
            "messages": messages,
            "images": [{"bytes": image_bytes}],
            "source": self.name,
            "answer": answer_letter,
            "choices": options,
            "task_type": self.task_type,
            "question_id": f"ai2d_test_{index}",
            "benchmark": self.name,
            "split": "test",
            "metadata": json.dumps({}),
        }


# ---------------------------------------------------------------------------
# MMMU (multi-image, placeholder parsing)
# ---------------------------------------------------------------------------


class MMMUConverter(BenchmarkConverter):
    name = "mmmu"
    hf_dataset_id = "lmms-lab/MMMU"
    hf_split = "validation"
    task_type = "multiple_choice"

    def convert_row(self, item: dict, index: int) -> dict | None:
        # Collect all non-None images (image_1 through image_7)
        image_bytes_list: list[bytes] = []
        for img_idx in range(1, 8):
            img = item.get(f"image_{img_idx}")
            if img is not None:
                img_bytes = extract_image_bytes(img)
                if img_bytes is not None:
                    image_bytes_list.append(img_bytes)

        if not image_bytes_list:
            return None

        # Parse options (JSON-encoded string in lmms-lab/MMMU)
        options_raw = item.get("options", "[]")
        if isinstance(options_raw, str):
            options = json.loads(options_raw)
        else:
            options = list(options_raw)

        question = item["question"]
        answer = item["answer"]

        # Build user content with interleaved image markers at placeholder positions
        user_content = self._build_user_content(question, options, len(image_bytes_list))

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        return {
            "messages": messages,
            "images": [{"bytes": b} for b in image_bytes_list],
            "source": self.name,
            "answer": answer,
            "choices": options,
            "task_type": self.task_type,
            "question_id": f"mmmu_val_{item.get('id', index)}",
            "benchmark": self.name,
            "split": "validation",
            "metadata": json.dumps(
                {
                    "subject": item.get("subject", ""),
                    "subfield": item.get("subfield", ""),
                    "topic_difficulty": item.get("topic_difficulty", ""),
                    "question_type": item.get("question_type", ""),
                    "img_type": item.get("img_type", ""),
                }
            ),
        }

    @staticmethod
    def _build_user_content(question: str, options: list[str], num_images: int) -> list[dict]:
        """Build user content for MMMU, handling <image N> placeholders.

        MMMU questions may contain <image 1>, <image 2>, etc. These are replaced
        with {"type": "image"} entries. If no placeholders are found, all images
        are prepended.
        """
        formatted_options = format_mc_options(options)
        full_text = f"{question}\n{formatted_options}\nAnswer with the option letter."

        # Find all <image N> placeholders
        pattern = r"<image\s+(\d+)>"
        placeholders = list(re.finditer(pattern, full_text))

        if not placeholders:
            # No placeholders found: prepend all images before the text
            content: list[dict] = [{"type": "image"} for _ in range(num_images)]
            content.append({"type": "text", "text": full_text})
            return content

        # Split text around placeholders and interleave image markers
        content = []
        last_end = 0
        for match in placeholders:
            before = full_text[last_end : match.start()].strip()
            if before:
                content.append({"type": "text", "text": before})
            content.append({"type": "image"})
            last_end = match.end()

        # Remaining text after last placeholder
        after = full_text[last_end:].strip()
        if after:
            content.append({"type": "text", "text": after})

        return content


# ---------------------------------------------------------------------------
# CIFAR-10 (generation benchmark)
# ---------------------------------------------------------------------------


class CIFAR10Converter(BenchmarkConverter):
    """Class-conditional image generation benchmark using CIFAR-10.

    Stores reference images alongside "a photo of a {class_name}" prompts.
    Supports class-balanced subsampling via n_per_class.
    """

    hf_dataset_id = "cifar10"
    hf_split = "test"
    task_type = "generation"

    def __init__(self, name: str = "cifar10", n_per_class: int | None = None):
        self.name = name
        self._n_per_class = n_per_class
        self._class_names: list[str] = []

    def load(self, max_rows: int | None = None) -> Dataset:
        ds = load_dataset(self.hf_dataset_id, split=self.hf_split, trust_remote_code=True)
        self._class_names = ds.features["label"].names

        if self._n_per_class is not None:
            ds = class_balanced_sample(ds, "label", self._n_per_class)

        if max_rows is not None:
            ds = ds.select(range(min(max_rows, len(ds))))
        return ds

    def convert_row(self, item: dict, index: int) -> dict | None:
        image_bytes = extract_image_bytes(item.get("img"))
        if image_bytes is None:
            return None

        label = item["label"]
        class_name = self._class_names[label]
        prompt = f"a photo of a {class_name}"
        messages = build_generation_messages(prompt)

        return {
            "messages": messages,
            "images": [{"bytes": image_bytes}],
            "source": "cifar10",
            "answer": class_name,
            "choices": None,
            "task_type": self.task_type,
            "question_id": f"cifar10_test_{index}",
            "benchmark": self.name,
            "split": "test",
            "metadata": json.dumps({"class_idx": label, "class_name": class_name}),
        }


# ---------------------------------------------------------------------------
# ImageNet (generation benchmark)
# ---------------------------------------------------------------------------


class ImageNetConverter(BenchmarkConverter):
    """Class-conditional image generation benchmark using ImageNet-1K.

    Stores reference images alongside "a photo of a {class_name}" prompts.
    Supports class-balanced subsampling via n_per_class.
    Requires HuggingFace authentication (dataset is gated).
    """

    hf_dataset_id = "ILSVRC/imagenet-1k"
    hf_split = "validation"
    task_type = "generation"

    def __init__(self, name: str = "imagenet", n_per_class: int | None = None):
        self.name = name
        self._n_per_class = n_per_class
        self._class_names: list[str] = []

    def load(self, max_rows: int | None = None) -> Dataset:
        ds = load_dataset(self.hf_dataset_id, split=self.hf_split, token=True, trust_remote_code=True)
        self._class_names = ds.features["label"].names

        if self._n_per_class is not None:
            ds = class_balanced_sample(ds, "label", self._n_per_class)

        if max_rows is not None:
            ds = ds.select(range(min(max_rows, len(ds))))
        return ds

    def convert_row(self, item: dict, index: int) -> dict | None:
        image_bytes = extract_image_bytes(item.get("image"))
        if image_bytes is None:
            return None

        label = item["label"]
        class_name = self._class_names[label]
        prompt = f"a photo of a {class_name}"
        messages = build_generation_messages(prompt)

        return {
            "messages": messages,
            "images": [{"bytes": image_bytes}],
            "source": "imagenet",
            "answer": class_name,
            "choices": None,
            "task_type": self.task_type,
            "question_id": f"imagenet_val_{index}",
            "benchmark": self.name,
            "split": "validation",
            "metadata": json.dumps({"class_idx": label, "class_name": class_name}),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, BenchmarkConverter] = {
    # Understanding
    "vqav2": VQAv2Converter(),
    "textvqa": TextVQAConverter(),
    "gqa": GQAConverter(),
    "chartqa": ChartQAConverter(),
    "ai2d": AI2DConverter(),
    "mmmu": MMMUConverter(),
    # Generation
    "cifar10_small": CIFAR10Converter(name="cifar10_small", n_per_class=100),
    "cifar10": CIFAR10Converter(name="cifar10"),
    "imagenet_small": ImageNetConverter(name="imagenet_small", n_per_class=5),
    "imagenet": ImageNetConverter(name="imagenet"),
}

ALL_BENCHMARKS = list(BENCHMARK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Shard writing and GCS upload
# ---------------------------------------------------------------------------


def write_eval_shard(
    rows: list[dict[str, Any]],
    shard_idx: int,
    output_dir: str,
    benchmark_name: str,
) -> tuple[str, int]:
    """Write a shard of eval data to parquet.

    Uses Dataset.from_list().to_parquet() following the pattern from
    convert_llava_onevision_to_levanter.py.
    """
    if not rows:
        return "", 0

    shard_name = f"eval-{benchmark_name}-{shard_idx:05d}.parquet"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    local_path = str(Path(output_dir) / shard_name)

    dataset = Dataset.from_list(rows)
    dataset.to_parquet(local_path)

    # Ensure data is fully flushed to disk before any upload
    with open(local_path, "rb") as f:
        os.fsync(f.fileno())

    del dataset
    gc.collect()

    logger.info("Written shard %s: %d rows", shard_name, len(rows))
    return local_path, len(rows)


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a local file to GCS using gcloud storage cp."""
    result = subprocess.run(
        ["gcloud", "storage", "cp", "--quiet", local_path, gcs_path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GCS upload failed for {local_path}: {result.stderr.strip()}")
    logger.info("Uploaded %s → %s", Path(local_path).name, gcs_path)


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------


def convert_benchmark(
    converter: BenchmarkConverter,
    output_dir: str,
    output_gcs: str | None,
    rows_per_shard: int,
    max_rows: int | None,
) -> dict[str, int]:
    """Convert a single benchmark to parquet shards."""
    logger.info(
        "Loading %s from %s (split=%s)...",
        converter.name,
        converter.hf_dataset_id,
        converter.hf_split,
    )
    ds = converter.load(max_rows=max_rows)
    logger.info("Loaded %d examples for %s", len(ds), converter.name)

    rows: list[dict] = []
    shard_idx = 0
    total_rows = 0
    skipped = 0

    for i in range(len(ds)):
        item = ds[i]
        row = converter.convert_row(item, i)
        if row is None:
            skipped += 1
            continue

        rows.append(row)

        if len(rows) >= rows_per_shard:
            local_path, num_rows = write_eval_shard(rows, shard_idx, output_dir, converter.name)
            if output_gcs and local_path:
                gcs_shard = f"{output_gcs.rstrip('/')}/{converter.name}/eval-{converter.name}-{shard_idx:05d}.parquet"
                upload_to_gcs(local_path, gcs_shard)
            total_rows += num_rows
            shard_idx += 1
            rows = []

    # Write remaining rows
    if rows:
        local_path, num_rows = write_eval_shard(rows, shard_idx, output_dir, converter.name)
        if output_gcs and local_path:
            gcs_shard = f"{output_gcs.rstrip('/')}/{converter.name}/eval-{converter.name}-{shard_idx:05d}.parquet"
            upload_to_gcs(local_path, gcs_shard)
        total_rows += num_rows
        shard_idx += 1

    stats = {
        "benchmark": converter.name,
        "total_rows": total_rows,
        "total_shards": shard_idx,
        "skipped": skipped,
    }
    logger.info(
        "Completed %s: %d rows in %d shards (%d skipped)",
        converter.name,
        total_rows,
        shard_idx,
        skipped,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert multimodal evaluation benchmarks to parquet format.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=ALL_BENCHMARKS,
        choices=ALL_BENCHMARKS,
        help=f"Benchmarks to convert (default: all). Choices: {ALL_BENCHMARKS}",
    )
    parser.add_argument(
        "--output-gcs",
        type=str,
        default="gs://marin-vlm/eval_benchmarks",
        help="GCS output path (default: gs://marin-vlm/eval_benchmarks)",
    )
    parser.add_argument(
        "--output-local",
        type=str,
        default="/tmp/eval_benchmarks",
        help="Local output directory (default: /tmp/eval_benchmarks)",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=DEFAULT_ROWS_PER_SHARD,
        help=f"Rows per output parquet shard (default: {DEFAULT_ROWS_PER_SHARD})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows per benchmark (for testing)",
    )
    args = parser.parse_args()

    output_dir = args.output_local
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    for benchmark_name in args.benchmarks:
        converter = BENCHMARK_REGISTRY[benchmark_name]
        benchmark_output_dir = str(Path(output_dir) / benchmark_name)
        stats = convert_benchmark(
            converter=converter,
            output_dir=benchmark_output_dir,
            output_gcs=args.output_gcs,
            rows_per_shard=args.rows_per_shard,
            max_rows=args.max_rows,
        )
        all_stats.append(stats)

    # Print summary
    logger.info("=" * 60)
    logger.info("Conversion Summary:")
    total_all = 0
    for stats in all_stats:
        logger.info(
            "  %s: %d rows, %d shards, %d skipped",
            stats["benchmark"],
            stats["total_rows"],
            stats["total_shards"],
            stats["skipped"],
        )
        total_all += stats["total_rows"]
    logger.info("Total: %d rows across %d benchmarks", total_all, len(all_stats))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
