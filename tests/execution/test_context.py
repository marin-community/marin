# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the executor build-phase construction guard."""

import contextlib
import copy
import logging
import pickle

import pytest
from marin.execution import context as ctx_mod
from marin.execution.context import current_executor_context, executor_context
from marin.execution.types import ExecutorStep


@contextlib.contextmanager
def _no_executor_context():
    """Suspend the autouse build context so contextless construction can be tested."""
    token = ctx_mod._active_context.set(None)
    try:
        yield
    finally:
        ctx_mod._active_context.reset(token)


@pytest.fixture(autouse=True)
def _reset_warned_sites():
    """Per-site warn dedup is process-global; clear it so each test starts fresh."""
    ctx_mod._warned_sites.clear()
    yield
    ctx_mod._warned_sites.clear()


def _guard_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "marin.execution.context" and r.levelno == logging.WARNING]


def test_context_active_inside_block():
    assert current_executor_context() is not None  # autouse fixture
    with executor_context(prefix="gs://marin-us-central1") as c:
        assert c.prefix == "gs://marin-us-central1"
    with _no_executor_context():
        assert current_executor_context() is None


def test_construction_outside_context_warns(caplog):
    with _no_executor_context(), caplog.at_level(logging.WARNING, logger="marin.execution.context"):
        step = ExecutorStep(name="warns", fn=None, config=None)
    assert step.name == "warns"
    assert len(_guard_warnings(caplog)) == 1


def test_construction_outside_context_warns_once_per_site(caplog):
    with _no_executor_context(), caplog.at_level(logging.WARNING, logger="marin.execution.context"):
        for _ in range(3):
            ExecutorStep(name="dup", fn=None, config=None)  # same source line every iteration
    assert len(_guard_warnings(caplog)) == 1


def test_strict_mode_raises(monkeypatch):
    monkeypatch.setenv("MARIN_EXECUTOR_STRICT", "1")
    with _no_executor_context(), pytest.raises(RuntimeError):
        ExecutorStep(name="boom", fn=None, config=None)


def test_construction_inside_context_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="marin.execution.context"):
        with executor_context():
            ExecutorStep(name="quiet", fn=None, config=None)
    assert _guard_warnings(caplog) == []


def test_pickle_and_deepcopy_do_not_trip_guard(monkeypatch):
    """Workers unpickle steps with no active context; that must never raise."""
    monkeypatch.setenv("MARIN_EXECUTOR_STRICT", "1")
    with executor_context():
        step = ExecutorStep(name="ship", fn=None, config=None)

    with _no_executor_context():
        # Standard frozen-dataclass pickling/deepcopy bypass __post_init__.
        assert pickle.loads(pickle.dumps(step)).name == "ship"
        assert copy.deepcopy(step).name == "ship"
