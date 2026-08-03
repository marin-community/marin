# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark one staged FastTransformer runtime on an accelerator."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fsspec
import jax
import numpy as np
from benchmark_fast_student import evaluation_texts, timed_rate
from benchmark_trained_fast_student import rate_stability
from evaluate_ladder import SPEED_REPEATS
from rigging.filesystem import StoragePath, atomic_rename

from experiments.datakit.embeddings.fast_transformer.embedder import FastEmbeddingModel, payload_sha256

DEFAULT_OUTPUT_ROOT = (
    "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed-runtime"
)
RESULT_FILE = Path("/tmp/luxical-fast-embedding-accelerator-speed")

logger = logging.getLogger(__name__)


def accelerator_rates(student: Any, texts: list[str], batch_size: int) -> dict[str, object]:
    """Return stable accelerator rates after one full-workload warmup."""
    student(texts, batch_size=batch_size)
    durations = []
    rates = []
    for repeat in range(SPEED_REPEATS):
        duration, rate = timed_rate(student, texts, batch_size)
        durations.append(duration)
        rates.append(rate)
        logger.info("repeat=%d model=student duration=%.3f rate=%.2f", repeat, duration, rate)
    stability = rate_stability(rates)
    return {
        "student_documents_per_second": float(np.median(rates)),
        "student_rates": rates,
        "student_durations": durations,
        "student_stability": stability,
        "measurement_valid": stability["passed"] is True,
    }


def main() -> None:
    """Load one runtime and write its accelerator throughput report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-name", required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    logging.basicConfig(level=logging.INFO)

    backend = jax.default_backend()
    if backend == "cpu":
        raise ValueError("The accelerator benchmark received a CPU-only JAX backend")
    student = FastEmbeddingModel.load_runtime(args.runtime_root, args.runtime_manifest_sha256)
    training_payload = StoragePath(student.manifest.training_report_url).read_bytes()
    if payload_sha256(training_payload) != student.manifest.training_report_sha256:
        raise ValueError("The runtime training-report digest does not match")
    training_report = json.loads(training_payload)
    expected_identity = {
        "config_name": args.config,
        "training_name": args.training_name,
        "rung": args.rung,
        "final_model_sha256": student.manifest.model_sha256,
    }
    if any(training_report.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("The runtime does not identify the benchmarked student")

    texts = evaluation_texts()
    report = {
        "mode": "accelerator",
        "jax_backend": backend,
        "compute_dtype": student.manifest.accelerator_compute_dtype,
        "config_name": args.config,
        "teacher": args.training_name,
        "rung": args.rung,
        "runtime_bundle_root": args.runtime_root,
        "runtime_manifest_sha256": args.runtime_manifest_sha256,
        "model_metadata": student.metadata(),
        "training_report": training_report,
        "documents": len(texts),
        "batch_size": args.batch_size,
        "repeats": SPEED_REPEATS,
        "warmup_documents": len(texts),
        **accelerator_rates(student, texts, args.batch_size),
    }
    output_url = (
        f"{args.output_root}/accelerator-runtime-{args.config}-{args.training_name}-{args.rung}-"
        f"{args.runtime_manifest_sha256[:12]}.json"
    )
    filesystem, path = fsspec.core.url_to_fs(output_url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(report, file, indent=2, sort_keys=True)
    report["output_url"] = output_url
    RESULT_FILE.write_text(json.dumps(report, sort_keys=True))
    logger.info("FAST_EMBEDDING_ACCELERATOR_SPEED=%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
