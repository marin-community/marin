# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``experiments.defaults.train`` and the helpers it delegates to.

Coverage areas
--------------
- ``_prepare_lm_train``: output path is concrete (no OutputName placeholders),
  checkpointer paths are baked from it, tokenizer comes from the upstream
  tokenize step, and no Iris job is submitted.
- ``train``: submits exactly one job via ``_submit_train_job`` with the expected
  job name, config, output_path, resources, env_vars, and worker_fn.
- ``_submit_train_job`` integration: exercises the fray-client path via a
  recording stub.
- ``materialize`` boundary: explicit ``output_path=`` kwarg resolves
  ``OutputName`` placeholders correctly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from fray import client as fray_client_module
from fray.types import ResourceConfig

import experiments.defaults as defaults_module
from experiments.defaults import (
    _prepare_lm_train,
    _submit_train_job,
    train,
)
from experiments.llama import llama_30m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, OutputName, materialize, this_output_path, versioned
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

    ``_prepare_lm_train`` reads the tokenize step's ``tokenizer`` field via
    ``_prepare_data_config``; we never invoke ``fn`` here so the fake's body
    is a no-op.
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
# _prepare_lm_train
# ---------------------------------------------------------------------------


@pytest.fixture
def make_prepared(small_train_config, fake_tokenized_step):
    """Call ``_prepare_lm_train`` with the standard small-CPU shape."""

    def _make(name: str = "my-run") -> tuple[str, object, str]:
        return _prepare_lm_train(
            name=name,
            tokenized=fake_tokenized_step,
            model_config=versioned(llama_30m),
            train_config=small_train_config,
            eval_harness_tasks=[],
            use_default_validation=False,
        )

    return _make


def test_prepare_lm_train_output_path_is_concrete(make_prepared):
    """``output_path`` is a fully-resolved string with no OutputName placeholders,
    and the checkpointer/HF save paths are baked from it."""
    _job_name, inner_config, output_path = make_prepared(name="resolved-path")

    assert "resolved-path" in output_path
    assert "OutputName" not in output_path
    assert inner_config.trainer.checkpointer.base_path.startswith(output_path)
    assert inner_config.hf_save_path.startswith(output_path)


def test_prepare_lm_train_job_name_has_checkpoints_prefix(make_prepared):
    """``job_name`` is ``checkpoints/<truncated_name>``."""
    job_name, _cfg, _path = make_prepared(name="some-run")
    assert job_name.startswith("checkpoints/")
    assert "some-run" in job_name


def test_prepare_lm_train_tokenizer_from_upstream_step(make_prepared, fake_tokenized_step):
    """The tokenizer string is read from the upstream tokenize step config."""
    _job_name, inner_config, _output_path = make_prepared(name="tok-test")
    # The data config should reference the tokenizer from the upstream step.
    assert inner_config.data.tokenizer == fake_tokenized_step.config.tokenizer


def test_prepare_lm_train_no_iris_submission(make_prepared):
    """``_prepare_lm_train`` must not submit any Iris jobs."""

    class _BangClient:
        def submit(self, request, adopt_existing: bool = True):  # pragma: no cover
            raise AssertionError("_prepare_lm_train should not call IrisClient.submit")

    with patch.object(fray_client_module, "current_client", lambda: _BangClient()):
        make_prepared(name="no-submit")


# ---------------------------------------------------------------------------
# train() — stub _submit_train_job, assert it's called exactly once
# ---------------------------------------------------------------------------


def test_train_calls_submit_train_job_once(tmp_path, small_train_config, fake_tokenized_step, monkeypatch):
    """``train()`` calls ``_submit_train_job`` exactly once with expected args."""
    captured: list[dict] = []

    def fake_submit(name, train_config, output_path, resources, env_vars, worker_fn):
        captured.append(
            dict(
                name=name,
                train_config=train_config,
                output_path=output_path,
                resources=resources,
                env_vars=env_vars,
                worker_fn=worker_fn,
            )
        )

    monkeypatch.setattr(defaults_module, "_submit_train_job", fake_submit)

    train(
        name="submit-once",
        tokenized=fake_tokenized_step,
        model_config=versioned(llama_30m),
        train_config=small_train_config,
        eval_harness_tasks=[],
        use_default_validation=False,
    )

    assert len(captured) == 1
    call = captured[0]
    assert "submit-once" in call["name"]
    assert call["resources"] == small_train_config.resources
    # output_path must be a concrete local or GCS path — no OutputName placeholders
    assert "OutputName" not in call["output_path"]
    assert "submit-once" in call["output_path"]
    # worker_fn must be the levanter entry point
    import levanter.main.train_lm as levanter_train_lm

    assert call["worker_fn"] is levanter_train_lm.main


def test_train_output_path_matches_prepare(small_train_config, fake_tokenized_step, monkeypatch):
    """``train()`` and ``_prepare_lm_train()`` resolve to the same output_path."""
    captured_path: list[str] = []

    def fake_submit(name, train_config, output_path, resources, env_vars, worker_fn):
        captured_path.append(output_path)

    monkeypatch.setattr(defaults_module, "_submit_train_job", fake_submit)

    train(
        name="path-parity",
        tokenized=fake_tokenized_step,
        model_config=versioned(llama_30m),
        train_config=small_train_config,
        eval_harness_tasks=[],
        use_default_validation=False,
    )

    _job_name, _cfg, expected_path = _prepare_lm_train(
        name="path-parity",
        tokenized=fake_tokenized_step,
        model_config=versioned(llama_30m),
        train_config=small_train_config,
        eval_harness_tasks=[],
        use_default_validation=False,
    )

    assert captured_path[0] == expected_path


# ---------------------------------------------------------------------------
# _submit_train_job — recording fray client
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


@pytest.fixture
def recording_client(monkeypatch):
    fake = _RecordingClient()
    monkeypatch.setattr(fray_client_module, "current_client", lambda: fake)
    return fake


def test_submit_train_job_submits_one_job(tmp_path, recording_client):
    """``_submit_train_job`` submits exactly one JobRequest with the given
    resources and carries user-supplied env vars through env resolution."""
    _submit_train_job(
        name="job-shape",
        train_config=_LeafCfg(output_path=str(tmp_path / "out")),
        output_path=str(tmp_path / "out"),
        resources=ResourceConfig.with_cpu(),
        env_vars={"USER_KEY": "user-value"},
        worker_fn=_trivial_worker,
    )

    assert len(recording_client.requests) == 1
    request = recording_client.requests[0]
    assert request.resources == ResourceConfig.with_cpu()
    assert request.environment.env_vars.get("USER_KEY") == "user-value"


@pytest.mark.parametrize(
    "resources, expected_extras",
    [
        pytest.param(ResourceConfig.with_cpu(), [], id="cpu_no_extras"),
        # WANDB_API_KEY is required by `_check_for_wandb_key` for TpuConfig.
        pytest.param(ResourceConfig.with_tpu("v4-8"), ["tpu"], id="tpu_jax_extra"),
    ],
)
def test_submit_train_job_extras_match_resource_class(
    tmp_path, recording_client, monkeypatch, resources, expected_extras
):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-tpu-resolve")
    _submit_train_job(
        name="extras-test",
        train_config=_LeafCfg(output_path=str(tmp_path / "out")),
        output_path=str(tmp_path / "out"),
        resources=resources,
        env_vars={},
        worker_fn=_trivial_worker,
    )

    assert list(recording_client.requests[0].environment.extras) == expected_extras


def test_submit_train_job_worker_entrypoint_calls_materialize_then_worker_fn(monkeypatch, tmp_path, recording_client):
    """The worker's captured callable runs `materialize` followed by
    `worker_fn(materialised_config)`."""
    received: list = []

    def fake_materialize(config, *, output_path=None, prefix=None):
        return ("materialized", config, output_path)

    def worker_fn(config):
        received.append(config)

    monkeypatch.setattr("experiments.defaults.materialize", fake_materialize)

    out_path = str(tmp_path / "out")
    train_config = _LeafCfg(output_path=out_path)

    _submit_train_job(
        name="worker-flow",
        train_config=train_config,
        output_path=out_path,
        resources=ResourceConfig.with_cpu(),
        env_vars={"FAKE_ENV": "1"},
        worker_fn=worker_fn,
    )

    request = recording_client.requests[0]
    callable_ep = request.entrypoint.callable_entrypoint
    assert callable_ep is not None
    callable_ep.callable(*callable_ep.args, **callable_ep.kwargs)

    assert len(received) == 1
    tag, original_cfg, materialise_out_path = received[0]
    assert tag == "materialized"
    assert original_cfg == train_config
    assert materialise_out_path == out_path


# ---------------------------------------------------------------------------
# materialize boundary tests
# ---------------------------------------------------------------------------


def test_materialize_accepts_explicit_output_path(tmp_path):
    """``materialize(config, output_path=...)`` resolves OutputName placeholders
    against the explicit argument when the config has no `output_path` field."""

    @dataclass(frozen=True)
    class _NoOutputPath:
        a: object
        b: int = 0

    cfg = _NoOutputPath(a=this_output_path(name="foo"))

    resolved = materialize(cfg, prefix=str(tmp_path), output_path=str(tmp_path / "concrete"))

    assert resolved.a == os.path.join(str(tmp_path / "concrete"), "foo")


def test_materialize_rejects_output_name_in_explicit_argument(tmp_path):
    """If the caller passes an OutputName as the explicit ``output_path``,
    fail loudly — that means the launcher forgot to resolve it."""

    @dataclass(frozen=True)
    class _NoOutputPath:
        a: object = 0

    with pytest.raises(TypeError, match="OutputName"):
        materialize(_NoOutputPath(), prefix=str(tmp_path), output_path=OutputName(name="foo"))
