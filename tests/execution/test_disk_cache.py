# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import threading
from functools import cache
from pathlib import Path

import cloudpickle
import marin.execution.step_runner as step_runner_module
import marin.execution.step_status as step_status_module
import pytest
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact, read_record, write_artifact
from marin.execution.disk_cache import disk_cache
from marin.execution.step_runner import check_cache, run_step
from marin.execution.step_spec import StepSpec
from marin.execution.step_status import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    StatusFile,
    StepLeaseLostError,
    distributed_lock,
    get_status_path,
)
from pydantic import BaseModel
from rigging.cancellation import current_cancellation_token
from rigging.filesystem.distributed_lock import LeaseLostError


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


class _Val(BaseModel):
    value: int


def test_composition_with_save_load(tmp_path: Path):
    """disk_cached + distributed_lock + save/load (the StepRunner pattern)."""
    call_count = 0

    def counting_fn(output_path: str) -> _Val:
        nonlocal call_count
        call_count += 1
        os.makedirs(output_path, exist_ok=True)
        return _Val(value=42)

    output_path = StepSpec(name="comp", output_path_prefix=tmp_path.as_posix()).output_path

    cached_fn = disk_cache(
        distributed_lock(counting_fn),
        output_path=output_path,
        save_fn=write_artifact,
        load_fn=lambda path: read_artifact(path, _Val),
    )

    result1 = cached_fn(output_path)
    assert call_count == 1
    assert result1 == _Val(value=42)

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
    """When no output_path is given, disk_cache derives one from MARIN_PREFIX via marin_temp_bucket."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))

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

    # marin_temp_bucket places local caches under {MARIN_PREFIX}/tmp/
    tmp_dir = tmp_path / "prefix" / "tmp"
    cache_dirs = list(tmp_dir.glob("disk_cache_*"))
    assert len(cache_dirs) == 1
    assert (cache_dirs[0] / "data.pkl").exists()


def test_run_step_with_cache_and_lock(tmp_path: Path):
    """run_step acquires a lock, runs the function, saves the artifact, and writes STATUS_SUCCESS."""

    call_count = 0

    def counting_fn(output_path: str) -> dict:
        nonlocal call_count
        call_count += 1
        os.makedirs(output_path, exist_ok=True)
        return {"value": 42}

    spec = StepSpec(name="run-step", output_path_prefix=tmp_path.as_posix(), fn=counting_fn)

    # First run: should execute
    run_step(spec)
    assert call_count == 1
    assert check_cache(spec.output_path)
    assert StatusFile(spec.output_path, "check").status == STATUS_SUCCESS

    # Second run: cache hit, should not re-execute
    run_step(spec)
    assert call_count == 1


def test_run_step_waits_for_winner_after_lease_loss(tmp_path: Path, monkeypatch):
    """A losing worker does not overwrite the winner's terminal status."""
    monkeypatch.setattr(step_status_module, "HEARTBEAT_INTERVAL", 0.01)
    call_count = 0

    def lose_lease(output_path: str) -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        token = current_cancellation_token()
        assert token is not None

        Path(f"{get_status_path(output_path)}.lock").unlink()
        winner = StatusFile(output_path, "winner")
        assert winner.try_acquire_lock()
        assert token.wait(timeout=5.0)
        write_artifact({"owner": "winner"}, output_path)
        winner.write_status(STATUS_SUCCESS)
        winner.release_lock()
        return {"owner": "loser"}

    spec = StepSpec(name="lose-lease", output_path_prefix=tmp_path.as_posix(), fn=lose_lease)

    run_step(spec)

    assert call_count == 1
    assert StatusFile(spec.output_path, "check").status == STATUS_SUCCESS
    record = read_record(spec.output_path)
    assert record is not None
    assert record.result == {"owner": "winner"}


def test_run_step_requests_remote_job_termination_after_lease_loss(tmp_path: Path, monkeypatch):
    """A lease loss requests job termination before waiting for the winner."""
    monkeypatch.setattr(step_status_module, "HEARTBEAT_INTERVAL", 0.01)
    submitted = threading.Event()
    terminated = threading.Event()

    class BlockingHandle:
        def wait(self, raise_on_failure: bool = False) -> None:
            submitted.set()
            assert terminated.wait(timeout=5.0)
            raise RuntimeError("remote job terminated")

        def terminate(self) -> None:
            terminated.set()

    class BlockingClient:
        def submit(self, _request):
            return BlockingHandle()

    monkeypatch.setattr(step_runner_module, "current_client", lambda: BlockingClient())

    spec = StepSpec(
        name="remote-lease-loss",
        output_path_prefix=tmp_path.as_posix(),
        fn=lambda _output_path: None,
        resources=ResourceConfig.with_cpu(cpu=1, ram="1g"),
    )

    def complete_as_winner() -> None:
        assert submitted.wait(timeout=5.0)
        Path(f"{get_status_path(spec.output_path)}.lock").unlink()
        winner = StatusFile(spec.output_path, "winner")
        assert winner.try_acquire_lock()
        assert terminated.wait(timeout=5.0)
        winner.write_status(STATUS_SUCCESS)
        winner.release_lock()

    winner_thread = threading.Thread(target=complete_as_winner)
    winner_thread.start()
    run_step(spec)
    winner_thread.join(timeout=5.0)

    assert not winner_thread.is_alive()
    assert terminated.is_set()
    assert StatusFile(spec.output_path, "check").status == STATUS_SUCCESS


def test_run_step_treats_nested_lease_loss_as_step_failure(tmp_path: Path):
    def fail_nested_lock(_output_path: str) -> None:
        raise LeaseLostError("nested lock lost")

    spec = StepSpec(name="nested-lock-loss", output_path_prefix=tmp_path.as_posix(), fn=fail_nested_lock)

    with pytest.raises(LeaseLostError, match="nested lock lost"):
        run_step(spec)

    assert StatusFile(spec.output_path, "check").status == STATUS_FAILED


def test_run_step_does_not_retry_nested_step_lease_loss(tmp_path: Path):
    nested_output_path = (tmp_path / "nested").as_posix()
    call_count = 0

    def fail_nested_step_lock(_output_path: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise AssertionError("run_step retried after a nested lease loss")
        raise StepLeaseLostError(nested_output_path, "nested lock lost")

    spec = StepSpec(name="nested-step-lock-loss", output_path_prefix=tmp_path.as_posix(), fn=fail_nested_step_lock)

    with pytest.raises(StepLeaseLostError, match="nested lock lost"):
        run_step(spec)

    assert call_count == 1
    assert StatusFile(spec.output_path, "check").status == STATUS_FAILED


def test_should_run_waits_for_active_holder_before_accepting_success(tmp_path: Path, monkeypatch):
    output_path = (tmp_path / "mutable").as_posix()
    holder = StatusFile(output_path, "winner")
    holder.write_status(STATUS_SUCCESS)
    assert holder.try_acquire_lock()
    released = False

    def release_holder(_seconds: float) -> None:
        nonlocal released
        released = True
        holder.release_lock()

    monkeypatch.setattr(step_status_module, "sleep", release_holder)

    assert not step_status_module.should_run(StatusFile(output_path, "loser"), "mutable")
    assert released


def test_functools_cache_with_disk_cache(tmp_path: Path, monkeypatch):
    """@cache + @disk_cache: in-memory cache avoids repeated disk reads."""

    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))

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

    # marin_temp_bucket places local caches under {MARIN_PREFIX}/tmp/
    tmp_dir = tmp_path / "prefix" / "tmp"
    cache_dirs = list(tmp_dir.glob("disk_cache_*"))
    assert len(cache_dirs) == 1
    assert (cache_dirs[0] / "data.pkl").exists()
