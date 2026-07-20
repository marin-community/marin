# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from datetime import timedelta
from typing import cast

import pytest
from fray.client import Client
from fray.cluster import GpuConfig
from fray.current_client import set_current_client
from fray.types import JobRequest, JobStatus
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import StepContext

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.launch_datakit_moe_mix import _val_component, datakit_data_config
from experiments.grug.moe.launch_mxfp8_quality import (
    build_quality_checkpoint,
    quality_cell,
    quality_model,
)
from experiments.grug.moe.model import GrugFp8Config

_PINNED_TRAIN_ENV = {
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    "SCALE_MUON_INTRA_RACK": "1",
    "SCALE_MUON_DIST_NONEXPERT": "1",
    "SCALE_MUON_PAD_NONEXPERT": "1",
    "NCCL_SOCKET_IFNAME": "^ibs,ibp,lo,docker,veth,cilium,lxc",
    "CE_IMPL": "liger",
}
_UNSUPPORTED_TRAIN_ENV = ("SCALE_MUON_NO_NS", "SCALE_MUON_DROP_NS_MATMULS", "SCALE_MUON_SYRK")


class _RecordedJob:
    @property
    def job_id(self) -> str:
        return "recorded-quality-job"

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        return JobStatus.SUCCEEDED

    def status(self) -> JobStatus:
        return JobStatus.SUCCEEDED

    def terminate(self) -> None:
        pass


class _RecordingFrayClient:
    def __init__(self) -> None:
        self.requests: list[JobRequest] = []

    def submit(self, request: JobRequest, adopt_existing: bool = True) -> _RecordedJob:
        self.requests.append(request)
        return _RecordedJob()


def _ignore_config(_: object) -> None:
    pass


def _clear_quality_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*_PINNED_TRAIN_ENV, *_UNSUPPORTED_TRAIN_ENV):
        monkeypatch.delenv(key, raising=False)


def test_quality_cell_matches_fixed_compute_budget() -> None:
    model, _, batch_size, steps = quality_cell()

    assert (batch_size, steps, batch_size * steps * model.max_seq_len) == (512, 31_474, 66_005_762_048)
    assert (
        model.hidden_dim,
        model.num_layers,
        model.num_experts,
        model.num_experts_per_token,
        model.max_seq_len,
    ) == (2560, 26, 128, 4, 4096)
    assert model.attention_implementation == "gpu_fa4_cute"
    assert model.moe_implementation == "ring"
    assert model.use_array_stacked_blocks is True


def test_quality_models_differ_only_by_mxfp8_config() -> None:
    bf16 = quality_model("bf16")
    mxfp8 = quality_model("mxfp8")

    assert bf16.fp8 is None
    assert mxfp8.fp8 == GrugFp8Config(
        dense=True,
        grouped=True,
        recipe="mxfp8",
        mxfp8_producer="xla",
    )
    assert bf16 == dataclasses.replace(mxfp8, fp8=None)


def test_quality_model_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown quality arm"):
        quality_model("fp16")


def test_quality_checkpoint_exposes_durable_pair_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_quality_runtime_env(monkeypatch)
    monkeypatch.setenv("MXFP8_QUALITY_ARM", "mxfp8")
    monkeypatch.setenv("MXFP8_QUALITY_STEPS", "20")
    monkeypatch.setenv("MXFP8_QUALITY_PAIR_ID", "MXFP8Q-000")

    step = build_quality_checkpoint(version="dev")
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    resources = step.runtime_args["train_resources"]

    assert config.run_id == "MXFP8Q-000-mxfp8-s20"
    assert config.steps == 20
    assert config.batch_size == 512
    assert config.seed == 0
    assert config.model == quality_model("mxfp8")
    assert config.optimizer == quality_cell()[1]
    assert config.grug_trainer.expert_axis_size == 8
    assert config.grug_trainer.replica_axis_size == 2
    assert config.checkpointer is not None
    assert config.checkpointer.save_interval == timedelta(hours=1)
    assert config.checkpointer.keep is None
    assert config.eval is not None
    assert config.eval.eval_batch_size == 512
    assert config.eval.steps_per_eval == 1000
    assert config.eval.max_eval_batches == 8
    assert isinstance(config.tracker, WandbConfig)
    assert config.tracker.project == "marin_moe"
    assert config.tracker.group == "mxfp8-quality-7271"
    assert config.tracker.name == "MXFP8Q-000-mxfp8-s20"
    assert set(config.tracker.tags) >= {"mxfp8-quality", "7271", "mxfp8"}
    assert resources.device == GpuConfig(variant="GB200", count=4)
    assert resources.replicas == 8
    assert resources.preemptible is False
    assert config.data.target_budget == 10_372_343_704_053
    assert config.data.experiment_budget == 20 * 512 * 4096
    assert all(dep.name in config.data.components for dep in step.deps)
    assert {key: os.environ[key] for key in _PINNED_TRAIN_ENV} == _PINNED_TRAIN_ENV


def test_quality_checkpoint_identity_is_stable_for_exact_reruns_and_isolates_arm_and_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_quality_runtime_env(monkeypatch)
    monkeypatch.setenv("MXFP8_QUALITY_PAIR_ID", "MXFP8Q-identity")

    def identity(arm: str, steps: int) -> tuple[str, str, str | None]:
        monkeypatch.setenv("MXFP8_QUALITY_ARM", arm)
        monkeypatch.setenv("MXFP8_QUALITY_STEPS", str(steps))
        step = build_quality_checkpoint(version="dev")
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        assert isinstance(config.tracker, WandbConfig)
        return step.name, config.run_id, config.tracker.name

    bf16_smoke = identity("bf16", 20)
    exact_rerun = identity("bf16", 20)
    mxfp8_smoke = identity("mxfp8", 20)
    bf16_full = identity("bf16", quality_cell()[3])

    assert bf16_smoke == exact_rerun
    distinct_runs = (bf16_smoke, mxfp8_smoke, bf16_full)
    assert all(len({run[field] for run in distinct_runs}) == 3 for field in range(3))
    assert bf16_smoke[1] == "MXFP8Q-identity-bf16-s20"


def test_quality_checkpoint_requires_pair_id_instead_of_legacy_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_quality_runtime_env(monkeypatch)
    monkeypatch.setenv("MXFP8_QUALITY_ARM", "bf16")
    monkeypatch.setenv("MXFP8_QUALITY_RUN_ID", "legacy-run-id")
    monkeypatch.delenv("MXFP8_QUALITY_PAIR_ID", raising=False)

    with pytest.raises(ValueError, match="MXFP8_QUALITY_PAIR_ID"):
        build_quality_checkpoint()


def test_quality_runtime_recipe_reaches_gpu_job_request_identically_for_both_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_quality_runtime_env(monkeypatch)
    monkeypatch.setenv("MXFP8_QUALITY_PAIR_ID", "MXFP8Q-dispatch")
    monkeypatch.setenv("MXFP8_QUALITY_STEPS", "20")
    submitted_envs: list[dict[str, str]] = []

    for arm in ("bf16", "mxfp8"):
        monkeypatch.setenv("MXFP8_QUALITY_ARM", arm)
        step = build_quality_checkpoint()
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        client = _RecordingFrayClient()
        with set_current_client(cast(Client, client)):
            dispatch_grug_training_run(
                run_id=config.run_id,
                config=config,
                local_entrypoint=_ignore_config,
                resources=step.runtime_args["train_resources"],
            )
        assert len(client.requests) == 1
        assert client.requests[0].environment is not None
        submitted_envs.append(dict(client.requests[0].environment.env_vars))

    assert submitted_envs[0] == submitted_envs[1]
    assert {key: submitted_envs[0][key] for key in _PINNED_TRAIN_ENV} == _PINNED_TRAIN_ENV
    assert all(key not in submitted_envs[0] for key in _UNSUPPORTED_TRAIN_ENV)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("SCALE_MUON_NO_NS", "1"),
        ("SCALE_MUON_DROP_NS_MATMULS", "1"),
        ("SCALE_MUON_SYRK", "1"),
        ("CE_IMPL", "triton"),
    ],
)
def test_quality_runtime_recipe_rejects_conflicts_before_mutation_or_submission(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    _clear_quality_runtime_env(monkeypatch)
    monkeypatch.setenv("MXFP8_QUALITY_ARM", "bf16")
    monkeypatch.setenv("MXFP8_QUALITY_PAIR_ID", "MXFP8Q-conflict")
    monkeypatch.setenv(key, value)
    client = _RecordingFrayClient()

    with set_current_client(cast(Client, client)), pytest.raises(ValueError, match=key):
        step = build_quality_checkpoint()
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        dispatch_grug_training_run(
            run_id=config.run_id,
            config=config,
            local_entrypoint=_ignore_config,
            resources=step.runtime_args["train_resources"],
        )

    assert client.requests == []
    assert all(pinned_key not in os.environ for pinned_key in _PINNED_TRAIN_ENV if pinned_key != key)


def test_datakit_config_resolves_train_and_validation_caches_under_cluster_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster_prefix = "s3://marin-us-east-02a/marin"
    monkeypatch.setenv("MARIN_PREFIX", cluster_prefix)
    validation = _val_component(f"{cluster_prefix}/tokenized/paloma")

    data = datakit_data_config(
        total_steps=20,
        batch_size=512,
        max_seq_len=4096,
        enable_simulated_epoching=True,
        val_components={"paloma": validation},
    )

    assert data.components["c01q0"].cache_dir == f"{cluster_prefix}/datakit/store_8ac06c74/cluster=1/quality=0"
    assert data.components["paloma"].cache_dir == f"{cluster_prefix}/tokenized/paloma"
