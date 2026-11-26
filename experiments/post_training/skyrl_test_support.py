# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
from marin.rl.skyrl import prepare_skyrl_cpu_python, run_skyrl_cpu, skyrl_project_root
from rigging.filesystem.storage_path import StoragePath

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


def run_native_test(module: str, args: Sequence[str]):
    """Run a checked-in test module in the isolated SkyRL runtime."""
    checkout = skyrl_project_root()
    interpreter = prepare_skyrl_cpu_python(170, project_root=checkout)
    result = run_skyrl_cpu([str(interpreter), "-I", "-m", module, str(checkout), *args], timeout=170)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
