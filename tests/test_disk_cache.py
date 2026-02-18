# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

from marin.execution.artifact import Artifact
from marin.execution.disk_cache import disk_cached
from marin.execution.distributed_lock import distributed_lock
from marin.execution.executor_step_status import STATUS_SUCCESS, StatusFile
from marin.execution.step_model import StepSpec


def _make_fn():
    """Return a fn that writes/reads a JSON file at output_path, tracking compute count."""
    compute_count = 0

    def fn(output_path: str) -> dict:
        nonlocal compute_count
        result_file = os.path.join(output_path, "result.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                return json.load(f)
        compute_count += 1
        os.makedirs(output_path, exist_ok=True)
        result = {"value": 42, "computed": True}
        with open(result_file, "w") as f:
            json.dump(result, f)
        return result

    return fn, lambda: compute_count


def test_disk_cached_runs_and_caches(tmp_path: Path):
    fn, get_count = _make_fn()
    output_path = StepSpec(name="step", output_path_prefix=tmp_path.as_posix()).output_path

    result1 = disk_cached(fn, output_path)
    assert get_count() == 1
    assert result1 == {"value": 42, "computed": True}

    # Cache hit: fn is called to load but no recomputation
    result2 = disk_cached(fn, output_path)
    assert get_count() == 1
    assert result2 == result1


def test_disk_cached_skips_when_another_worker_completed(tmp_path: Path):
    fn, get_count = _make_fn()

    # Simulate another worker having completed the step
    spec = StepSpec(name="race", output_path_prefix=tmp_path.as_posix())
    os.makedirs(spec.output_path, exist_ok=True)
    with open(os.path.join(spec.output_path, "result.json"), "w") as f:
        json.dump({"value": 99, "from_other": True}, f)
    StatusFile(spec.output_path, "other-worker").write_status(STATUS_SUCCESS)

    result = disk_cached(fn, spec.output_path)

    assert get_count() == 0
    assert result == {"value": 99, "from_other": True}


def test_composition_with_save_load(tmp_path: Path):
    """disk_cached + distributed_lock + save/load (the StepRunner pattern)."""
    call_count = 0

    def counting_fn(output_path: str) -> dict:
        nonlocal call_count
        call_count += 1
        os.makedirs(output_path, exist_ok=True)
        return {"value": 42}

    output_path = StepSpec(name="comp", output_path_prefix=tmp_path.as_posix()).output_path

    result1 = disk_cached(
        distributed_lock(counting_fn),
        output_path,
        save=Artifact.save,
        load=Artifact.load,
    )
    assert call_count == 1
    assert result1 == {"value": 42}

    result2 = disk_cached(
        distributed_lock(counting_fn),
        output_path,
        save=Artifact.save,
        load=Artifact.load,
    )
    assert call_count == 1
    assert result2 == result1
