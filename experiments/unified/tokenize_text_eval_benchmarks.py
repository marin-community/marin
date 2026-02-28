# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenize text eval benchmarks (HellaSwag, WinoGrande, ARC, MMLU) into Levanter cache format.

Loads each benchmark from HuggingFace, formats as MCQ text (question + options + answer),
tokenizes, and writes a Levanter-compatible cache with input_ids and loss_weights.
Loss is computed on all tokens (standard LM loss).

Output cache directory structure:

    {output_path}/{benchmark}/validation/
        input_ids/{offsets, data}
        loss_weights/{offsets, data}
        shard_ledger.json

Usage:
    # Tokenize all text eval benchmarks
    uv run experiments/unified/tokenize_text_eval_benchmarks.py

    # Tokenize specific benchmarks
    uv run experiments/unified/tokenize_text_eval_benchmarks.py \
        --benchmarks hellaswag arc_challenge

    # Custom output path
    uv run experiments/unified/tokenize_text_eval_benchmarks.py \
        --output_path gs://marin-vlm/text_eval_cache
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator

import numpy as np
import transformers

from experiments.unified.unified_pretrain import UNIFIED_TOKENIZER_PATH
from experiments.unified.vlm_tokenize_captions import ENDOFTEXT_ID, gcs_upload

logger = logging.getLogger(__name__)

# --- Default paths ---

DEFAULT_OUTPUT_PATH = "gs://marin-vlm/text_eval_cache"

ALL_TEXT_BENCHMARKS = [
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
]


# --- Formatting functions ---


def _format_hellaswag(example: dict) -> str | None:
    """Format a HellaSwag example as MCQ text.

    HellaSwag has: ctx (context), endings (list of 4 endings), label (correct index).
    """
    ctx = example.get("ctx", "")
    endings = example.get("endings", [])
    label = example.get("label", "")

    if not ctx or not endings:
        return None

    # label can be string or int
    label_idx = int(label) if isinstance(label, (str, int)) else 0

    labels = ["A", "B", "C", "D"]
    lines = [ctx]
    for i, ending in enumerate(endings):
        if i < len(labels):
            lines.append(f"{labels[i]}. {ending}")
    correct_label = labels[label_idx] if label_idx < len(labels) else labels[0]
    correct_text = endings[label_idx] if label_idx < len(endings) else endings[0]
    lines.append(f"Answer: {correct_label}. {correct_text}")
    return "\n".join(lines)


def _format_winogrande(example: dict) -> str | None:
    """Format a WinoGrande example as MCQ text.

    WinoGrande has: sentence (with _ blank), option1, option2, answer (1 or 2).
    """
    sentence = example.get("sentence", "")
    option1 = example.get("option1", "")
    option2 = example.get("option2", "")
    answer = example.get("answer", "")

    if not sentence or not option1 or not option2:
        return None

    lines = [sentence, f"1. {option1}", f"2. {option2}"]
    if str(answer) == "1":
        lines.append(f"Answer: 1. {option1}")
    else:
        lines.append(f"Answer: 2. {option2}")
    return "\n".join(lines)


def _format_arc(example: dict) -> str | None:
    """Format an ARC (Easy or Challenge) example as MCQ text.

    ARC has: question, choices (dict with text and label lists), answerKey.
    """
    question = example.get("question", "")
    choices = example.get("choices", {})
    answer_key = example.get("answerKey", "")

    if not question or not choices:
        return None

    choice_texts = choices.get("text", [])
    choice_labels = choices.get("label", [])

    lines = [question]
    correct_text = ""
    for label, text in zip(choice_labels, choice_texts):
        lines.append(f"{label}. {text}")
        if label == answer_key:
            correct_text = text
    lines.append(f"Answer: {answer_key}. {correct_text}")
    return "\n".join(lines)


def _format_mmlu(example: dict) -> str | None:
    """Format an MMLU example as MCQ text.

    MMLU has: question, choices (list of 4), answer (0-3 index).
    """
    question = example.get("question", "")
    choices = example.get("choices", [])
    answer = example.get("answer")

    if not question or not choices:
        return None

    labels = ["A", "B", "C", "D"]
    answer_idx = int(answer) if answer is not None else 0

    lines = [question]
    for i, choice in enumerate(choices):
        if i < len(labels):
            lines.append(f"{labels[i]}. {choice}")
    correct_label = labels[answer_idx] if answer_idx < len(labels) else labels[0]
    correct_text = choices[answer_idx] if answer_idx < len(choices) else choices[0]
    lines.append(f"Answer: {correct_label}. {correct_text}")
    return "\n".join(lines)


FORMATTERS = {
    "hellaswag": _format_hellaswag,
    "winogrande": _format_winogrande,
    "arc_easy": _format_arc,
    "arc_challenge": _format_arc,
    "mmlu": _format_mmlu,
}


# --- Dataset loading ---


def _load_benchmark_dataset(benchmark: str):
    """Load a benchmark dataset from HuggingFace."""
    from datasets import load_dataset

    if benchmark == "hellaswag":
        return load_dataset("Rowan/hellaswag", split="validation")
    elif benchmark == "winogrande":
        return load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    elif benchmark == "arc_easy":
        return load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    elif benchmark == "arc_challenge":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    elif benchmark == "mmlu":
        return load_dataset("cais/mmlu", "all", split="test")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# --- Tokenization ---


def process_text_benchmark(
    benchmark: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Iterator[dict[str, np.ndarray]]:
    """Load and tokenize a text benchmark, yielding Levanter cache records."""
    dataset = _load_benchmark_dataset(benchmark)
    formatter = FORMATTERS[benchmark]

    total = len(dataset)
    skipped = 0

    for i, example in enumerate(dataset):
        text = formatter(example)
        if text is None:
            skipped += 1
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.append(ENDOFTEXT_ID)

        input_ids = np.array(token_ids, dtype=np.int32)
        loss_weights = np.ones(len(token_ids), dtype=np.float32)

        yield {"input_ids": input_ids, "loss_weights": loss_weights}

        if (i + 1) % 1000 == 0:
            logger.info("  ... processed %d/%d examples", i + 1, total)

    if skipped > 0:
        logger.info("Skipped %d/%d examples (missing data)", skipped, total)


# --- Benchmark tokenization ---


def _cache_tokenizer_locally(tokenizer_path: str) -> str:
    """Download a (possibly GCS-hosted) tokenizer to a local temp directory."""
    from levanter.compat.hf_checkpoints import load_tokenizer

    local_dir = tempfile.mkdtemp(prefix="tokenizer_cache_")
    tok = load_tokenizer(tokenizer_path)
    tok.save_pretrained(local_dir)
    logger.info("Tokenizer cached locally (vocab_size=%d)", len(tok))
    return local_dir


def tokenize_text_benchmark(
    benchmark: str,
    output_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Tokenize a single text eval benchmark into a Levanter cache.

    Writes to {output_path}/{benchmark}/validation/.
    """
    from zephyr.writers import write_levanter_cache

    logger.info("Processing benchmark: %s", benchmark)

    work_dir = tempfile.mkdtemp(prefix=f"text_eval_tok_{benchmark}_")
    local_cache = os.path.join(work_dir, f"cache-{benchmark}")

    records = process_text_benchmark(benchmark, tokenizer)
    metadata = {"benchmark": benchmark, "format": "text_eval_benchmark"}
    result = write_levanter_cache(records, local_cache, metadata)

    logger.info(
        "Benchmark %s: %d records, %d tokens",
        benchmark,
        result["count"],
        result["token_count"],
    )

    if result["count"] == 0:
        logger.warning("No records for %s, skipping upload", benchmark)
        shutil.rmtree(work_dir)
        return {"benchmark": benchmark, "total_records": 0, "total_tokens": 0}

    # Upload to GCS
    gcs_cache_path = f"{output_path}/{benchmark}/validation"
    logger.info("Uploading cache to %s", gcs_cache_path)
    gcs_upload(local_cache, gcs_cache_path)

    shutil.rmtree(work_dir)

    stats = {
        "benchmark": benchmark,
        "total_records": result["count"],
        "total_tokens": result["token_count"],
    }
    logger.info(
        "Completed %s: %d records, %d tokens",
        benchmark,
        result["count"],
        result["token_count"],
    )
    return stats


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(description="Tokenize text eval benchmarks into Levanter cache format.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=ALL_TEXT_BENCHMARKS,
        choices=ALL_TEXT_BENCHMARKS,
        help=f"Benchmarks to tokenize (default: all). Choices: {ALL_TEXT_BENCHMARKS}",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="GCS path for output Levanter caches",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=UNIFIED_TOKENIZER_PATH,
        help="Path to the tokenizer",
    )
    args = parser.parse_args()

    # Cache tokenizer locally
    local_tokenizer_path = _cache_tokenizer_locally(args.tokenizer)
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_tokenizer_path)

    all_stats = []
    for benchmark in args.benchmarks:
        logger.info("=" * 60)
        stats = tokenize_text_benchmark(
            benchmark=benchmark,
            output_path=args.output_path,
            tokenizer=tokenizer,
        )
        all_stats.append(stats)

    # Print summary
    logger.info("=" * 60)
    logger.info("Tokenization Summary:")
    total_records = 0
    total_tokens = 0
    for stats in all_stats:
        logger.info(
            "  %s: %d records, %d tokens",
            stats["benchmark"],
            stats["total_records"],
            stats["total_tokens"],
        )
        total_records += stats["total_records"]
        total_tokens += stats["total_tokens"]
    logger.info(
        "Total: %d records, %d tokens across %d benchmarks",
        total_records,
        total_tokens,
        len(all_stats),
    )

    # Clean up tokenizer cache
    shutil.rmtree(local_tokenizer_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
