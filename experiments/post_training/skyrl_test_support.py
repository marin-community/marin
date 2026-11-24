# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from marin.rl.skyrl import SKYRL_PROJECT, skyrl_cpu_command, skyrl_cpu_environment
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl_compose as recipe
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
        with TemporaryDirectory(prefix="skyrl-test-") as directory:
            program = Path(directory) / "native_test.py"
            program.write_text(script)
            runner = recipe.REPO_ROOT / "experiments/post_training/rl_tests/native_script.py"
            args = [str(runner), str(recipe.REPO_ROOT / SKYRL_PROJECT), str(program), *paths]
            try:
                if uv_options:
                    command = skyrl_cpu_command(*uv_options, "python", "-I", *args)
                    return gate._native_command_json(
                        "Run isolated native test program", command, env=skyrl_cpu_environment(), timeout=170
                    )
                return gate._native_json("Run native test program", *args, timeout=170)
            except gate.InvalidComparison as error:
                cause = error.__cause__
                if isinstance(cause, subprocess.CalledProcessError) and cause.stderr:
                    raise AssertionError(cause.stderr.strip()) from error
                raise

    return execute
