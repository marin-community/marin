# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from collections.abc import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl_gate as gate
from experiments.post_training.task_curation.selection import TRAIN_FILENAME


def selection_summary(full_rows: int) -> dict:
    return {
        "recipe": {"heldout": [{"version": "2026.10.23.1", "rows": 910}]},
        "training_rows": 100000,
        "validation_rows": 100,
        "full_validation_rows": full_rows,
        "grader_kind_counts": {"train": {}, "validation": {}, "full_validation": {}},
    }


def write_training_families(root: StoragePath, counts: Mapping[str, int]) -> None:
    families = [family for family, count in counts.items() for _ in range(count)]
    with (root / TRAIN_FILENAME).open("wb") as stream:
        pq.write_table(pa.table({"family": families}), stream)


def run_native_test(module: str, args: Sequence[str], *, uv_options: tuple[str, ...] = ()):
    """Run a checked-in test module in the locked SkyRL CPU environment."""
    try:
        return gate.run_native_json(*uv_options, "python", "-I", "-m", module, *args, timeout=170)
    except gate.InvalidComparison as error:
        cause = error.__cause__
        if isinstance(cause, subprocess.CalledProcessError) and cause.stderr:
            raise AssertionError(cause.stderr.strip()) from error
        raise
