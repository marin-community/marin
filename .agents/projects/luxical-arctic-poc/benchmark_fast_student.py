# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark random-weight fast-transformer student inference."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import fsspec
import jax
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from evaluate_ladder import (
    BASELINE_FILE,
    BASELINE_REPO,
    BASELINE_REVISION,
    CPU_THREADS,
    MANIFEST_URL,
    SPEED_DOCUMENTS,
    SPEED_REPEATS,
    SPEED_WARMUP_DOCUMENTS,
    read_json,
)
from fast_student import (
    E5_TOKENIZER_NAME,
    TIKTOKEN_NAME,
    FastStudent,
    provisional_remap,
    tokenizer_vocab_size,
)
from huggingface_hub import hf_hub_download
from luxical.embedder import Embedder
from rigging.filesystem import atomic_rename
from threadpoolctl import threadpool_limits

from experiments.datakit.cluster.quality.fast_transformer.model import count_params

RESULT_FILE = Path("/tmp/luxical-fast-student-speed")
DEFAULT_OUTPUT_ROOT = "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluation_texts() -> list[str]:
    """Load the fixed held-out document views without teacher vectors."""
    manifest = read_json(MANIFEST_URL)
    texts = []
    for source, result in sorted(manifest["sources"].items()):
        filesystem, path = fsspec.core.url_to_fs(result["output_url"])
        table = pq.read_table(path, filesystem=filesystem, columns=["split", "eval_rank", "text"])
        table = table.filter(pc.equal(table["split"], "eval")).sort_by("eval_rank")
        logger.info("Loaded %d speed candidates from %s", len(table), source)
        texts.extend(table["text"].to_pylist())
    if len(texts) < SPEED_DOCUMENTS:
        raise ValueError(f"Only {len(texts)} evaluation texts are available")
    indices = np.linspace(0, len(texts) - 1, num=SPEED_DOCUMENTS, dtype=np.int64)
    return [texts[index] for index in indices]


def timed_rate(model: Any, texts: list[str], batch_size: int) -> tuple[float, float]:
    started = time.perf_counter()
    vectors = model(texts, batch_size=batch_size)
    duration = time.perf_counter() - started
    if vectors.ndim != 2 or len(vectors) != len(texts):
        raise ValueError(f"Model returned an unexpected shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Model returned non-finite speed vectors")
    return duration, len(texts) / duration


def benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(CPU_THREADS)
    remap = provisional_remap(tokenizer_vocab_size(arguments.tokenizer))
    student = FastStudent.random(arguments.config, remap, seed=42, tokenizer_name=arguments.tokenizer)
    texts = evaluation_texts()
    warmup = texts[:SPEED_WARMUP_DOCUMENTS]
    student(warmup, batch_size=arguments.batch_size)

    student_durations = []
    student_rates = []
    baseline_durations = []
    baseline_rates = []
    baseline = None
    if arguments.mode == "cpu":
        baseline_path = hf_hub_download(
            repo_id=BASELINE_REPO,
            filename=BASELINE_FILE,
            revision=BASELINE_REVISION,
        )
        baseline = Embedder.load(baseline_path)
        baseline(warmup, batch_size=arguments.batch_size)

    with threadpool_limits(limits=CPU_THREADS):
        for repeat in range(SPEED_REPEATS):
            order = ("baseline", "student") if repeat % 2 == 0 else ("student", "baseline")
            if baseline is None:
                order = ("student",)
            for name in order:
                if name == "student":
                    duration, rate = timed_rate(student, texts, arguments.batch_size)
                    student_durations.append(duration)
                    student_rates.append(rate)
                else:
                    duration, rate = timed_rate(baseline, texts, arguments.batch_size)
                    baseline_durations.append(duration)
                    baseline_rates.append(rate)
                logger.info("repeat=%d model=%s duration=%.3f rate=%.2f", repeat, name, duration, rate)

    student_rate = float(np.median(student_rates))
    baseline_rate = float(np.median(baseline_rates)) if baseline_rates else None
    config = student.model.backbone.config
    return {
        "mode": arguments.mode,
        "jax_backend": jax.default_backend(),
        "config_name": arguments.config,
        "config": student.metadata(),
        "parameters": count_params(student.model),
        "flops_per_token": config.flops_per_token(),
        "documents": len(texts),
        "batch_size": arguments.batch_size,
        "repeats": SPEED_REPEATS,
        "student_documents_per_second": student_rate,
        "student_rates": student_rates,
        "student_durations": student_durations,
        "baseline_documents_per_second": baseline_rate,
        "baseline_rates": baseline_rates,
        "baseline_durations": baseline_durations,
        "student_to_baseline_ratio": student_rate / baseline_rate if baseline_rate else None,
        "vocabulary": "speed-only provisional remap",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "accelerator"), required=True)
    parser.add_argument("--config", choices=("full", "slim"), required=True)
    parser.add_argument("--tokenizer", choices=(E5_TOKENIZER_NAME, TIKTOKEN_NAME), required=True)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = benchmark(arguments)
    tokenizer_label = arguments.tokenizer.replace("/", "--")
    output_url = f"{arguments.output_root}/{arguments.mode}-{arguments.config}-{tokenizer_label}.json"
    filesystem, path = fsspec.core.url_to_fs(output_url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(report, file, indent=2, sort_keys=True)
    report["output_url"] = output_url
    RESULT_FILE.write_text(json.dumps(report, sort_keys=True))
    logger.info("FAST_STUDENT_SPEED=%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
