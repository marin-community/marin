# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``experiments.defaults.prepare_train`` / ``run_train`` and the
generalised ``TrainingPlan`` shape used by both the Levanter and Grug
training paths.
"""
from __future__ import annotations

import os
import pickle
import threading
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from fray import client as fray_client_module
from fray.types import Entrypoint, ResourceConfig

from experiments.defaults import (
    TrainingPlan,
    _run_training_on_worker,
    prepare_train,
    run_train,
)
from experiments.llama import llama_30m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    OutputName,
    this_output_path,
    versioned,
)
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.processing.tokenize.tokenize import TokenizeConfig

# ---------------------------------------------------------------------------
# Fixtures and tiny helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LeafCfg:
    output_path: str


@pytest.fixture
def fake_tokenized_step():
    """A minimal upstream ``ExecutorStep`` shaped like a real tokenize step.

    ``prepare_train`` reads the tokenize step's ``tokenizer`` field via
    ``_get_tokenizer_for_train``; we never invoke ``fn`` here so the fake's
    body is a no-op.
    """
    return ExecutorStep(
        name="tokenized/marker",
        fn=lambda c: None,
        config=TokenizeConfig(
            train_paths=["dummy://train"],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer="meta-llama/Llama-3.2-1B",
            tags=[],
        ),
    )


@pytest.fixture
def small_train_config():
    return SimpleTrainConfig(
        # CPU resources avoid `_check_for_wandb_key` in `resolve_training_env`
        # — that gate fires only on TpuConfig and demands WANDB_API_KEY.
        resources=ResourceConfig.with_cpu(),
        train_batch_size=8,
        num_train_steps=4,
        learning_rate=1e-4,
        weight_decay=0.0,
    )


@pytest.fixture(autouse=True)
def _local_marin_prefix(tmp_path, monkeypatch):
    """Force a local prefix so output_path computation doesn't try to read
    GCS metadata for ``gs://marin-{region}``."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))


# ---------------------------------------------------------------------------
# prepare_train
# ---------------------------------------------------------------------------


def test_prepare_train_returns_training_plan(small_train_config, fake_tokenized_step):
    plan = prepare_train(
        name="my-run",
        tokenized=fake_tokenized_step,
        model_config=versioned(llama_30m),
        train_config=small_train_config,
        eval_harness_tasks=[],
        use_default_validation=False,
    )

    assert isinstance(plan, TrainingPlan)
    assert plan.resources == small_train_config.resources
    # The plan's name is the `checkpoints/<name>` key the executor would have
    # used; the resolved path is concrete and contains the run name.
    assert "my-run" in plan.output_path
    assert plan.output_path.startswith("/")  # tmp_path-rooted


def test_prepare_train_output_path_is_concrete(small_train_config, fake_tokenized_step):
    """``plan.output_path`` is a fully-resolved string (no placeholders)."""
    plan = prepare_train(
        name="resolved-path",
        tokenized=fake_tokenized_step,
        model_config=versioned(llama_30m),
        train_config=small_train_config,
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    assert isinstance(plan.output_path, str)
    assert "OutputName" not in plan.output_path
    # Both checkpointer paths have been baked from `output_path`.
    assert plan.train_config.trainer.checkpointer.base_path.startswith(plan.output_path)
    assert plan.train_config.hf_save_path.startswith(plan.output_path)


def test_prepare_train_preserves_input_name_placeholders(small_train_config, fake_tokenized_step):
    """Upstream ``InputName(step=...)`` references stay unresolved on the
    plan; ``materialize`` resolves them on the worker in the worker's region."""
    plan = prepare_train(
        name="upstream-deferred",
        tokenized=fake_tokenized_step,
        model_config=versioned(llama_30m),
        train_config=small_train_config,
        eval_harness_tasks=[],
        use_default_validation=False,
    )

    # Walk the embedded data config and assert at least one InputName remains.
    found_input_names: list[InputName] = []

    def walk(obj):
        if isinstance(obj, InputName):
            found_input_names.append(obj)
            return
        if hasattr(obj, "__dataclass_fields__"):
            for f in obj.__dataclass_fields__:
                walk(getattr(obj, f))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(plan.train_config)
    assert any(
        n.step is fake_tokenized_step for n in found_input_names
    ), "Expected at least one InputName referencing the upstream tokenize step."


def test_prepare_train_no_iris_submission(small_train_config, fake_tokenized_step):
    """``prepare_train`` must not submit any Iris jobs by itself."""

    class _BangClient:
        def submit(self, request, adopt_existing: bool = True):  # pragma: no cover - test guard
            raise AssertionError("prepare_train should not call IrisClient.submit")

    with patch.object(fray_client_module, "current_client", lambda: _BangClient()):
        prepare_train(
            name="no-submit",
            tokenized=fake_tokenized_step,
            model_config=versioned(llama_30m),
            train_config=small_train_config,
            eval_harness_tasks=[],
            use_default_validation=False,
        )


# ---------------------------------------------------------------------------
# run_train
# ---------------------------------------------------------------------------


class _RecordingClient:
    def __init__(self) -> None:
        self.requests: list = []

    def submit(self, request, adopt_existing: bool = True):
        self.requests.append(request)
        return _RecordingHandle()


class _RecordingHandle:
    def wait(self, raise_on_failure: bool = True):
        return None


def _trivial_worker(config) -> None:  # pragma: no cover - runs on worker
    return None


def test_run_train_submits_one_job(monkeypatch, tmp_path):
    """``run_train`` submits exactly one JobRequest with the plan's resources."""
    plan = TrainingPlan(
        name="job-shape",
        output_path=str(tmp_path / "out"),
        train_config=_LeafCfg(output_path=str(tmp_path / "out")),
        worker_fn=_trivial_worker,
        resources=ResourceConfig.with_cpu(),
        env_vars={"USER_KEY": "user-value"},
    )

    fake = _RecordingClient()
    monkeypatch.setattr(fray_client_module, "current_client", lambda: fake)

    run_train(plan)

    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.resources == plan.resources

    # The user-supplied env var is carried into the job environment after env
    # resolution. (Other run-metadata keys like GIT_COMMIT may also be added.)
    env = request.environment.env_vars
    assert env.get("USER_KEY") == "user-value"


def test_run_train_passes_tpu_extras_for_tpu_resources(monkeypatch, tmp_path):
    """TPU plans must request the ``tpu`` uv extra so jax[tpu] is installed."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-tpu-resolve")

    plan = TrainingPlan(
        name="tpu-extras",
        output_path=str(tmp_path / "out"),
        train_config=_LeafCfg(output_path=str(tmp_path / "out")),
        worker_fn=_trivial_worker,
        resources=ResourceConfig.with_tpu("v4-8"),
        env_vars={},
    )

    fake = _RecordingClient()
    monkeypatch.setattr(fray_client_module, "current_client", lambda: fake)

    run_train(plan)

    assert list(fake.requests[0].environment.extras) == ["tpu"]


def test_run_train_passes_no_extras_for_cpu_resources(monkeypatch, tmp_path):
    """CPU plans must not request accelerator extras."""
    plan = TrainingPlan(
        name="cpu-extras",
        output_path=str(tmp_path / "out"),
        train_config=_LeafCfg(output_path=str(tmp_path / "out")),
        worker_fn=_trivial_worker,
        resources=ResourceConfig.with_cpu(),
        env_vars={},
    )

    fake = _RecordingClient()
    monkeypatch.setattr(fray_client_module, "current_client", lambda: fake)

    run_train(plan)

    assert list(fake.requests[0].environment.extras) == []


def test_run_train_worker_entrypoint_calls_materialize_then_worker_fn(monkeypatch, tmp_path):
    """The worker's captured callable runs `materialize` followed by
    `worker_fn(materialised_config)`."""
    received: list = []

    def fake_materialize(config, *, output_path=None, prefix=None):
        # Tag the config so the assertion downstream can prove this ran.
        return ("materialized", config, output_path)

    def worker_fn(config):
        received.append(config)

    monkeypatch.setattr("experiments.defaults.materialize", fake_materialize)

    plan = TrainingPlan(
        name="worker-flow",
        output_path=str(tmp_path / "out"),
        train_config=_LeafCfg(output_path=str(tmp_path / "out")),
        worker_fn=worker_fn,
        resources=ResourceConfig.with_cpu(),
        env_vars={"FAKE_ENV": "1"},
    )

    fake = _RecordingClient()
    monkeypatch.setattr(fray_client_module, "current_client", lambda: fake)

    run_train(plan)

    request = fake.requests[0]
    callable_ep = request.entrypoint.callable_entrypoint
    assert callable_ep is not None
    callable_ep.callable(*callable_ep.args, **callable_ep.kwargs)

    assert len(received) == 1
    tag, original_cfg, materialise_out_path = received[0]
    assert tag == "materialized"
    assert original_cfg == plan.train_config
    assert materialise_out_path == plan.output_path


def test_worker_entrypoint_is_picklable(tmp_path):
    """``Entrypoint.from_callable(_run_training_on_worker, args=...)`` must be
    serialisable so Fray can ship it to the worker."""
    # Use a module-level worker function — locals don't pickle, but the actual
    # production worker_fns (`levanter.main.train_lm.main`, `_run_grug_local`)
    # are top-level functions just like this one.
    args = [_trivial_worker, _LeafCfg(output_path=str(tmp_path)), str(tmp_path), {"K": "V"}]
    ep = Entrypoint.from_callable(_run_training_on_worker, args=args)
    # Pickling is the constraint that matters for Fray; if it survives the
    # round-trip the harness can pickle the JobRequest containing it too.
    pickle.dumps(ep)


# ---------------------------------------------------------------------------
# Sweep integration — TrainingPlan + claim_and_run
# ---------------------------------------------------------------------------


def test_sweep_runs_each_plan_once(tmp_path):
    """Three plans, three SweepTargets, each ``run_train`` substitute fires
    exactly once across the sweep."""
    plans = [
        TrainingPlan(
            name=f"sweep-{i}",
            output_path=str(tmp_path / f"out-{i}"),
            train_config=_LeafCfg(output_path=str(tmp_path / f"out-{i}")),
            worker_fn=_trivial_worker,
            resources=ResourceConfig.with_cpu(),
            env_vars={},
        )
        for i in range(3)
    ]

    targets = [SweepTarget(target_id=p.name, config=p) for p in plans]

    invocations: list[TrainingPlan] = []
    invocations_lock = threading.Lock()

    def fake_run_train(target: SweepTarget) -> None:
        with invocations_lock:
            invocations.append(target.config)

    sweep_root = os.path.join(str(tmp_path), "sweep-root")
    claim_and_run(sweep_root, targets, fake_run_train)

    assert len(invocations) == 3
    assert {p.name for p in invocations} == {f"sweep-{i}" for i in range(3)}


# ---------------------------------------------------------------------------
# Sanity: materialize's new `output_path=` kwarg
# ---------------------------------------------------------------------------


def test_materialize_accepts_explicit_output_path(tmp_path):
    """``materialize(config, output_path=...)`` resolves OutputName placeholders
    against the explicit argument when the config has no `output_path` field."""
    from marin.execution.dag import materialize

    @dataclass(frozen=True)
    class _NoOutputPath:
        # No `output_path` field — Levanter's TrainLmConfig is in the same shape.
        a: object
        b: int = 0

    cfg = _NoOutputPath(a=this_output_path(name="foo"))

    resolved = materialize(cfg, prefix=str(tmp_path), output_path=str(tmp_path / "concrete"))

    assert resolved.a == os.path.join(str(tmp_path / "concrete"), "foo")


def test_materialize_rejects_output_name_in_explicit_argument(tmp_path):
    """If the caller passes an OutputName as the explicit ``output_path``,
    fail loudly — that means the launcher forgot to resolve it."""
    from marin.execution.dag import materialize

    @dataclass(frozen=True)
    class _NoOutputPath:
        a: object = 0

    with pytest.raises(TypeError, match="OutputName"):
        materialize(_NoOutputPath(), prefix=str(tmp_path), output_path=OutputName(name="foo"))
