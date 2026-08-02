# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a paired CPU benchmark on an exact trained Fast Transformer artifact."""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import fsspec
import jax
import numpy as np
import torch
from benchmark_fast_student import evaluation_texts, timed_rate
from evaluate_fast_student import load_student
from evaluate_ladder import (
    BASELINE_FILE,
    BASELINE_REPO,
    BASELINE_REVISION,
    CPU_THREADS,
    SPEED_REPEATS,
    SPEED_WARMUP_DOCUMENTS,
)
from huggingface_hub import hf_hub_download
from luxical.embedder import Embedder
from rigging.filesystem import atomic_rename
from threadpoolctl import threadpool_limits

RESULT_FILE = Path("/tmp/luxical-trained-fast-student-speed")
DEFAULT_OUTPUT_ROOT = "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed"

logger = logging.getLogger(__name__)


def paired_rates(student: Any, baseline: Any, texts: list[str], batch_size: int) -> dict[str, Any]:
    """Return alternating paired throughput measurements."""
    warmup = texts[:SPEED_WARMUP_DOCUMENTS]
    student(warmup, batch_size=batch_size)
    baseline(warmup, batch_size=batch_size)
    student_durations = []
    student_rates = []
    baseline_durations = []
    baseline_rates = []
    with threadpool_limits(limits=CPU_THREADS):
        for repeat in range(SPEED_REPEATS):
            order = ("baseline", "student") if repeat % 2 == 0 else ("student", "baseline")
            for name in order:
                model = student if name == "student" else baseline
                duration, rate = timed_rate(model, texts, batch_size)
                if name == "student":
                    student_durations.append(duration)
                    student_rates.append(rate)
                else:
                    baseline_durations.append(duration)
                    baseline_rates.append(rate)
                logger.info("repeat=%d model=%s duration=%.3f rate=%.2f", repeat, name, duration, rate)
    student_rate = float(np.median(student_rates))
    baseline_rate = float(np.median(baseline_rates))
    return {
        "student_documents_per_second": student_rate,
        "student_rates": student_rates,
        "student_durations": student_durations,
        "baseline_documents_per_second": baseline_rate,
        "baseline_rates": baseline_rates,
        "baseline_durations": baseline_durations,
        "student_to_baseline_ratio": student_rate / baseline_rate,
    }


def benchmark(config: str, teacher: str, rung: str, batch_size: int) -> dict[str, Any]:
    """Load exact model artifacts and return a paired CPU report."""
    torch.set_num_threads(CPU_THREADS)
    texts = evaluation_texts()
    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    baseline = Embedder.load(baseline_path)
    with tempfile.TemporaryDirectory() as temporary_directory:
        student, training_report = load_student(config, teacher, rung, Path(temporary_directory))
        rates = paired_rates(student, baseline, texts, batch_size)
        metadata = student.metadata()
    return {
        "mode": "cpu",
        "jax_backend": jax.default_backend(),
        "config_name": config,
        "teacher": teacher,
        "rung": rung,
        "model_metadata": metadata,
        "training_report": training_report,
        "baseline": {
            "repo": BASELINE_REPO,
            "file": BASELINE_FILE,
            "revision": BASELINE_REVISION,
        },
        "documents": len(texts),
        "batch_size": batch_size,
        "repeats": SPEED_REPEATS,
        **rates,
    }


def main() -> None:
    """Parse arguments, run the paired test, and write its private report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    logging.basicConfig(level=logging.INFO)
    report = benchmark(args.config, args.teacher, args.rung, args.batch_size)
    output_url = f"{args.output_root}/cpu-trained-{args.config}-{args.teacher}-{args.rung}.json"
    filesystem, path = fsspec.core.url_to_fs(output_url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(report, file, indent=2, sort_keys=True)
    report["output_url"] = output_url
    RESULT_FILE.write_text(json.dumps(report, sort_keys=True))
    logger.info("TRAINED_FAST_STUDENT_SPEED=%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
