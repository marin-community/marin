# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the bounded fast-student training loader on saved data."""

import argparse
import hashlib
import json
import math
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
from fast_student import MAX_TOKENS
from fast_student_training_data import StagedTrainingRows
from ladder_config import MANIFEST_ROOT, read_json, write_json
from train_fast_student import (
    BATCH_SIZE,
    PREPARED_MANIFEST_URL,
    SEED,
    TEACHERS,
    TRAINING_BLOCK_ROWS,
    TrainingLayout,
    training_rows,
)

OUTPUT_ROOT = f"{MANIFEST_ROOT}/fast-student/loader-canary"
RESULT_FILE = Path("/tmp/luxical-fast-student-loader-canary")
PEAK_RSS_LIMIT_BYTES = 8 * 1024**3


def peak_rss_bytes() -> int:
    """Return the Linux process peak resident memory."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-rows", type=int, required=True)
    parser.add_argument("--prepared-manifest-url", default=PREPARED_MANIFEST_URL)
    parser.add_argument("--teacher", choices=tuple(TEACHERS), default="arctic-medium-256")
    return parser.parse_args()


def main() -> None:
    """Stage and scan one fixed training sample without model training."""
    arguments = parse_args()
    if arguments.training_rows < BATCH_SIZE:
        raise ValueError(f"The canary needs at least one full batch of {BATCH_SIZE} rows")
    prepared = read_json(arguments.prepared_manifest_url)
    teacher_spec = TEACHERS[arguments.teacher]
    started = time.perf_counter()
    initial_peak_rss = peak_rss_bytes()
    with tempfile.TemporaryDirectory(prefix="luxical-fast-student-loader-canary-") as directory:
        staged, quotas = training_rows(
            prepared,
            arguments.training_rows,
            teacher_spec,
            TrainingLayout.STAGED,
            Path(directory),
            MAX_TOKENS,
        )
        if not isinstance(staged, StagedTrainingRows):
            raise TypeError(f"The canary received training layout {type(staged).__name__}")
        staging_seconds = time.perf_counter() - started
        staging_peak_rss = peak_rss_bytes()
        digest = hashlib.sha256()
        batches = 0
        scanned_rows = 0
        seen_sources = set()
        scan_started = time.perf_counter()
        for batch in staged.epoch_batches(0, BATCH_SIZE, TRAINING_BLOCK_ROWS, SEED):
            if batch.ids.shape != (BATCH_SIZE, staged.id_width):
                raise ValueError(f"The canary ID batch has shape {batch.ids.shape}")
            if batch.teacher.shape != (BATCH_SIZE, teacher_spec.dimension):
                raise ValueError(f"The canary teacher batch has shape {batch.teacher.shape}")
            if not np.isfinite(batch.teacher).all():
                raise ValueError("The canary teacher batch contains non-finite values")
            digest.update(batch.ids.tobytes())
            digest.update(batch.teacher.tobytes())
            digest.update(batch.source_ids.tobytes())
            seen_sources.update(int(value) for value in np.unique(batch.source_ids))
            batches += 1
            scanned_rows += len(batch.ids)
        scan_seconds = time.perf_counter() - scan_started
        final_peak_rss = peak_rss_bytes()
        expected_batches = math.ceil(arguments.training_rows / BATCH_SIZE)
        if batches != expected_batches:
            raise ValueError(f"The canary returned {batches} batches; expected {expected_batches}")
        if len(seen_sources) != len(quotas):
            raise ValueError(f"The canary saw {len(seen_sources)} sources; expected {len(quotas)}")
        memory = staged.memory_report(TRAINING_BLOCK_ROWS)
    gates = {
        "all_batches_scanned": batches == expected_batches,
        "all_sources_seen": len(seen_sources) == len(quotas),
        "peak_rss_within_limit": final_peak_rss <= PEAK_RSS_LIMIT_BYTES,
        "full_rows_not_materialized_in_host_report": memory["layout"] == "local-numpy-memmap",
    }
    report = {
        "run_id": arguments.run_id,
        "prepared_manifest_url": arguments.prepared_manifest_url,
        "prepared_manifest_sha256": prepared["manifest_sha256"],
        "teacher": arguments.teacher,
        "training_rows": arguments.training_rows,
        "sources": len(quotas),
        "batches": batches,
        "scanned_rows_with_padding": scanned_rows,
        "batch_digest_sha256": digest.hexdigest(),
        "staging_seconds": staging_seconds,
        "scan_seconds": scan_seconds,
        "initial_peak_rss_bytes": initial_peak_rss,
        "staging_peak_rss_bytes": staging_peak_rss,
        "final_peak_rss_bytes": final_peak_rss,
        "peak_rss_limit_bytes": PEAK_RSS_LIMIT_BYTES,
        "memory": memory,
        "gates": gates,
        "passes": all(gates.values()),
    }
    output_url = f"{OUTPUT_ROOT}/{arguments.run_id}.json"
    write_json(output_url, report)
    summary = {
        "run_id": arguments.run_id,
        "output_url": output_url,
        "training_rows": arguments.training_rows,
        "final_peak_rss_bytes": final_peak_rss,
        "passes": report["passes"],
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    print(f"FAST_STUDENT_LOADER_CANARY={json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
