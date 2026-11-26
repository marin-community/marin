# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.config_discovery import find_project_root
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl_gate as gate
from experiments.post_training.rl_presets import ModelPresets
from experiments.post_training.task_curation.selection import TRAIN_FILENAME


def with_current_control_revisions(table: ModelPresets) -> ModelPresets:
    return replace(
        table,
        control_pins={
            name: replace(pin, revision=table.by_name[name].revision) for name, pin in table.control_pins.items()
        },
    )


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


def native_json(*uv_options: str):
    def execute(script, paths):
        source = str(find_project_root(Path(__file__)) / "vendor/skyrl")
        program = gate._native_program(source, ("cloud.iris.launch_config", "skyrl_train"), script)
        try:
            return gate._native_json(*uv_options, "python", "-c", program, *paths, timeout=170)
        except gate.InvalidComparison as error:
            cause = error.__cause__
            if isinstance(cause, subprocess.CalledProcessError) and cause.stderr:
                raise AssertionError(cause.stderr.strip()) from error
            raise

    return execute
