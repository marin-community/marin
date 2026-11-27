# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.run_records import read_attempt_metrics
from marin.rl.skyrl import prepare_skyrl_cpu_python, run_skyrl_cpu, skyrl_project_root
from marin.rl.skyrl_records import read_skyrl_launch_config
from rigging.filesystem.storage_path import StoragePath

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


def run_native_test(module: str, args: Sequence[str]):
    """Run a checked-in test module in the isolated SkyRL runtime."""
    checkout = skyrl_project_root()
    interpreter = prepare_skyrl_cpu_python(170, project_root=checkout)
    result = run_skyrl_cpu([str(interpreter), "-I", "-m", module, str(checkout), *args], timeout=170)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def read_skyrl_metrics(run_output: str, attempt_id: str | None = None) -> dict[int, dict]:
    """Read one native attempt's metrics and bounded checkpoint ancestors."""
    config = read_skyrl_launch_config(run_output)
    return read_attempt_metrics(config["artifacts"]["export_root"], attempt_id or config["run"]["attempt_id"])
