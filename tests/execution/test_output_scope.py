# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``OutputScope`` write-target selection and plan-time read-fallback.

The overriding contract is zero behavior change for ``SHARED`` scope (the
executor default): the computed ``output_path``, override handling, and the
absence of any filesystem I/O during planning must match the historical
behavior. Only the new ``USER`` branch may differ.
"""

import os
from dataclasses import dataclass

import pytest
from marin.execution.executor import Executor, compute_output_path
from marin.execution.executor_step_status import STATUS_SUCCESS, StatusFile
from marin.execution.types import ExecutorStep, OutputName, OutputScope, output_path_of, this_output_path


@dataclass(frozen=True)
class StepConfig:
    output_path: OutputName | str
    input_path: str | OutputName | None = None


def _noop(config: StepConfig | None) -> None:
    return None


@pytest.fixture(autouse=True)
def clear_marin_user(monkeypatch):
    monkeypatch.delenv("MARIN_USER", raising=False)
    monkeypatch.delenv("MARIN_CLUSTER", raising=False)
    monkeypatch.delenv("MARIN_PREFIX", raising=False)


def _make_executor(prefix: str, *, default_scope: OutputScope = OutputScope.SHARED) -> Executor:
    return Executor(prefix=prefix, executor_info_base_path=prefix, default_scope=default_scope)


def _seed_success(output_path: str) -> None:
    """Write a SUCCESS ``.executor_status`` at ``output_path``."""
    StatusFile(output_path, worker_id="seed").write_status(STATUS_SUCCESS)


def test_shared_scope_output_path_is_byte_identical(tmp_path):
    """A SHARED step resolves to exactly ``{prefix}/{name}-{hash}`` as before."""
    prefix = str(tmp_path)
    step = ExecutorStep(name="train", fn=_noop, config=StepConfig(output_path=this_output_path()))

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)

    output_path = executor.output_paths[step]
    hashed = output_path.rsplit("-", 1)[-1]
    assert output_path == os.path.join(prefix, f"train-{hashed}")


def test_shared_scope_does_no_filesystem_io(tmp_path, monkeypatch):
    """SHARED-scope planning never probes ``.executor_status``."""
    prefix = str(tmp_path)
    step = ExecutorStep(name="train", fn=_noop, config=StepConfig(output_path=this_output_path()))

    def _explode(*args, **kwargs):
        raise AssertionError("SHARED scope must not construct a StatusFile during planning")

    monkeypatch.setattr("marin.execution.executor.StatusFile", _explode)

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)
    assert executor.output_paths[step].startswith(prefix + "/train-")


def test_shared_absolute_override_used_verbatim(tmp_path):
    prefix = str(tmp_path)
    absolute = str(tmp_path / "elsewhere" / "pinned")
    step = ExecutorStep(
        name="train",
        fn=_noop,
        config=StepConfig(output_path=this_output_path()),
        override_output_path=absolute,
    )

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)
    assert executor.output_paths[step] == absolute


def test_shared_relative_override_joined_against_prefix(tmp_path):
    prefix = str(tmp_path)
    step = ExecutorStep(
        name="train",
        fn=_noop,
        config=StepConfig(output_path=this_output_path()),
        override_output_path="custom/sub",
    )

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)
    assert executor.output_paths[step] == os.path.join(prefix, "custom/sub")


def test_compute_output_path_shared_matches_executor(tmp_path):
    """The pure helper agrees with ``Executor`` for SHARED scope."""
    prefix = str(tmp_path)
    config = StepConfig(output_path=this_output_path())
    helper_path = compute_output_path("train", config, prefix=prefix)

    step = ExecutorStep(name="train", fn=_noop, config=config)
    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)
    assert helper_path == executor.output_paths[step]


def test_user_scope_writes_under_user_home(tmp_path, monkeypatch):
    """A USER-scoped step writes under ``{prefix}/users/{user}/...``."""
    monkeypatch.setenv("MARIN_USER", "alice")
    prefix = str(tmp_path)
    step = ExecutorStep(
        name="train",
        fn=_noop,
        config=StepConfig(output_path=this_output_path()),
        output_scope=OutputScope.USER,
    )

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)

    output_path = executor.output_paths[step]
    user_home = os.path.join(prefix, "users", "alice")
    assert output_path.startswith(user_home + "/train-")


def test_user_scope_falls_back_to_shared_success(tmp_path, monkeypatch):
    """With a SUCCESS at the shared path and no user-home output, resolve shared."""
    monkeypatch.setenv("MARIN_USER", "alice")
    prefix = str(tmp_path)
    step = ExecutorStep(
        name="train",
        fn=_noop,
        config=StepConfig(output_path=this_output_path()),
        output_scope=OutputScope.USER,
    )

    # Compute the hash once via SHARED to learn the name-hash, then seed SUCCESS
    # at the shared location.
    shared_path = compute_output_path("train", StepConfig(output_path=this_output_path()), prefix=prefix)
    _seed_success(shared_path)

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)
    assert executor.output_paths[step] == shared_path


def test_user_scope_prefers_user_home_success(tmp_path, monkeypatch):
    """A SUCCESS at the user home wins over the shared/legacy candidate."""
    monkeypatch.setenv("MARIN_USER", "alice")
    prefix = str(tmp_path)
    config = StepConfig(output_path=this_output_path())
    step = ExecutorStep(name="train", fn=_noop, config=config, output_scope=OutputScope.USER)

    shared_path = compute_output_path("train", config, prefix=prefix)
    name_hash = os.path.basename(shared_path)
    user_home_path = os.path.join(prefix, "users", "alice", name_hash)
    # Seed SUCCESS at BOTH; user home must win.
    _seed_success(user_home_path)
    _seed_success(shared_path)

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)
    assert executor.output_paths[step] == user_home_path


def test_user_to_user_dependency_resolves_upstream_effective_path(tmp_path, monkeypatch):
    """A USER step depending on a USER step sees the upstream's resolved path.

    The upstream's effective path is its shared fallback (seeded SUCCESS, no
    user-home output). The downstream config must reference that resolved path,
    not the upstream's user-home write target.
    """
    monkeypatch.setenv("MARIN_USER", "alice")
    prefix = str(tmp_path)

    upstream = ExecutorStep(
        name="upstream",
        fn=_noop,
        config=StepConfig(output_path=this_output_path()),
        output_scope=OutputScope.USER,
    )
    downstream = ExecutorStep(
        name="downstream",
        fn=_noop,
        config=StepConfig(output_path=this_output_path(), input_path=output_path_of(upstream, "artifact")),
        output_scope=OutputScope.USER,
    )

    # Seed SUCCESS at the upstream's SHARED candidate so it resolves there.
    upstream_shared = compute_output_path("upstream", StepConfig(output_path=this_output_path()), prefix=prefix)
    _seed_success(upstream_shared)

    executor = _make_executor(prefix)
    executor.compute_version(downstream, is_pseudo_dep=False)

    assert executor.output_paths[upstream] == upstream_shared
    resolved_input = executor.configs[downstream].input_path
    assert resolved_input == os.path.join(upstream_shared, "artifact")


def test_user_relative_override_joined_against_user_home(tmp_path, monkeypatch):
    """A relative override on a USER step lands under the user's home."""
    monkeypatch.setenv("MARIN_USER", "alice")
    prefix = str(tmp_path)
    step = ExecutorStep(
        name="train",
        fn=_noop,
        config=StepConfig(output_path=this_output_path()),
        override_output_path="custom/sub",
        output_scope=OutputScope.USER,
    )

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)
    assert executor.output_paths[step] == os.path.join(prefix, "users", "alice", "custom/sub")


def test_identity_resolution_is_lazy_for_shared_runs(tmp_path, monkeypatch):
    """A SHARED-only run with a generic OS user never resolves identity.

    ``resolve_marin_user(for_user_scope=True)`` would raise on ``root``; a
    SHARED run must not call it.
    """
    monkeypatch.delenv("MARIN_USER", raising=False)
    monkeypatch.setattr("getpass.getuser", lambda: "root")
    prefix = str(tmp_path)
    step = ExecutorStep(name="train", fn=_noop, config=StepConfig(output_path=this_output_path()))

    executor = _make_executor(prefix)
    executor.compute_version(step, is_pseudo_dep=False)
    assert executor._user is None
    assert executor.output_paths[step].startswith(prefix + "/train-")
