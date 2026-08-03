# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the paired CPU benchmark through a staged production runtime."""

import argparse
import json
import logging
from pathlib import Path

import fsspec
from benchmark_trained_fast_student import benchmark_loaded_student
from rigging.filesystem import StoragePath, atomic_rename

from experiments.datakit.embeddings.fast_transformer.embedder import FastEmbeddingModel, payload_sha256

DEFAULT_OUTPUT_ROOT = (
    "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed-runtime"
)
RESULT_FILE = Path("/tmp/luxical-fast-embedding-runtime-speed")

logger = logging.getLogger(__name__)


def main() -> None:
    """Load one runtime, benchmark it, and write its identity-bound report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-name", required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    logging.basicConfig(level=logging.INFO)

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

    report = benchmark_loaded_student(
        student,
        training_report,
        args.config,
        args.training_name,
        args.rung,
        args.batch_size,
    )
    report.update(
        {
            "runtime_bundle_root": args.runtime_root,
            "runtime_manifest_sha256": args.runtime_manifest_sha256,
        }
    )
    output_url = (
        f"{args.output_root}/cpu-runtime-{args.config}-{args.training_name}-{args.rung}-"
        f"{args.runtime_manifest_sha256[:12]}.json"
    )
    filesystem, path = fsspec.core.url_to_fs(output_url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(report, file, indent=2, sort_keys=True)
    report["output_url"] = output_url
    RESULT_FILE.write_text(json.dumps(report, sort_keys=True))
    logger.info("FAST_EMBEDDING_RUNTIME_SPEED=%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
