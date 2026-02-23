# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import cloudpickle

from marin.execution.artifact import Artifact
from marin.execution.disk_cache import disk_cache
from marin.execution.distributed_lock import distributed_lock
from marin.execution.executor_step_status import STATUS_SUCCESS, StatusFile
from marin.execution.step_spec import StepSpec


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

    cached_fn = disk_cache(fn, output_path=output_path)

    result1 = cached_fn(output_path)
    assert get_count() == 1
    assert result1 == {"value": 42, "computed": True}

    # Cache hit: fn is called to load but no recomputation
    result2 = cached_fn(output_path)
    assert get_count() == 1
    assert result2 == result1


def test_disk_cached_skips_when_another_worker_completed(tmp_path: Path):
    fn, get_count = _make_fn()

    # Simulate another worker having completed the step: write
    # data.pkl (what disk_cache reads) and mark STATUS_SUCCESS.
    spec = StepSpec(name="race", output_path_prefix=tmp_path.as_posix())
    expected = {"value": 99, "from_other": True}
    os.makedirs(spec.output_path, exist_ok=True)
    with open(os.path.join(spec.output_path, "data.pkl"), "wb") as f:
        f.write(cloudpickle.dumps(expected))
    StatusFile(spec.output_path, "other-worker").write_status(STATUS_SUCCESS)

    cached_fn = disk_cache(fn, output_path=spec.output_path)
    result = cached_fn(spec.output_path)

    assert get_count() == 0
    assert result == expected


def test_composition_with_save_load(tmp_path: Path):
    """disk_cached + distributed_lock + save/load (the StepRunner pattern)."""
    call_count = 0

    def counting_fn(output_path: str) -> dict:
        nonlocal call_count
        call_count += 1
        os.makedirs(output_path, exist_ok=True)
        return {"value": 42}

    output_path = StepSpec(name="comp", output_path_prefix=tmp_path.as_posix()).output_path

    cached_fn = disk_cache(
        distributed_lock(counting_fn),
        output_path=output_path,
        save_fn=Artifact.save,
        load_fn=Artifact.load,
    )

    result1 = cached_fn(output_path)
    assert call_count == 1
    assert result1 == {"value": 42}

    result2 = cached_fn(output_path)
    assert call_count == 1
    assert result2 == result1


def test_decorator_with_cloudpickle(tmp_path: Path):
    """@disk_cache as a decorator uses cloudpickle for serialization by default."""
    call_count = 0
    output_path = str(tmp_path / "cache")

    @disk_cache(output_path=output_path)
    def expensive(x, y):
        nonlocal call_count
        call_count += 1
        return {"sum": x + y, "product": x * y}

    result1 = expensive(3, 7)
    assert call_count == 1
    assert result1 == {"sum": 10, "product": 21}

    # Second call should hit the cloudpickle cache on disk
    result2 = expensive(3, 7)
    assert call_count == 1
    assert result2 == result1

    # Verify the cloudpickle file was written
    assert os.path.exists(os.path.join(output_path, "data.pkl"))


def test_decorator_auto_path_from_marin_prefix(tmp_path: Path, monkeypatch):
    """When no output_path is given, disk_cache derives one from MARIN_PREFIX."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    # Ensure get_temp_bucket_path returns None so MARIN_PREFIX is used
    monkeypatch.setattr("marin.execution.disk_cache.get_temp_bucket_path", lambda *a, **kw: None)

    call_count = 0

    @disk_cache
    def compute(x):
        nonlocal call_count
        call_count += 1
        return x * 10

    result1 = compute(5)
    assert call_count == 1
    assert result1 == 50

    # Second call hits the cache
    result2 = compute(5)
    assert call_count == 1
    assert result2 == 50

    # Verify the cache landed under MARIN_PREFIX
    prefix_dir = tmp_path / "prefix"
    cache_dirs = list(prefix_dir.glob("disk_cache_*"))
    assert len(cache_dirs) == 1
    assert (cache_dirs[0] / "data.pkl").exists()


def test_functools_cache_with_disk_cache(tmp_path: Path, monkeypatch):
    """@cache + @disk_cache: in-memory cache avoids repeated disk reads."""
    from functools import cache

    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setattr("marin.execution.disk_cache.get_temp_bucket_path", lambda *a, **kw: None)

    call_count = 0

    @cache
    @disk_cache
    def load_model(name: str) -> dict:
        nonlocal call_count
        call_count += 1
        return {"model": name, "weights": [1, 2, 3]}

    # First call: disk miss, runs the function, writes cloudpickle
    r1 = load_model("bert")
    assert call_count == 1
    assert r1 == {"model": "bert", "weights": [1, 2, 3]}

    # Second call: @cache returns the in-memory result, no disk read
    r2 = load_model("bert")
    assert call_count == 1
    assert r2 is r1  # same object identity from functools.cache

    # Clear the in-memory cache — next call should hit disk_cache
    load_model.cache_clear()
    r3 = load_model("bert")
    assert call_count == 1  # still 1: disk_cache served the result
    assert r3 == r1

    # Verify the cache landed under MARIN_PREFIX
    prefix_dir = tmp_path / "prefix"
    cache_dirs = list(prefix_dir.glob("disk_cache_*"))
    assert len(cache_dirs) == 1
    assert (cache_dirs[0] / "data.pkl").exists()
