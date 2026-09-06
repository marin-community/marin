# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import io
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import IO, cast

import pytest
from marin.evaluation.model_config import ModelConfig, ResourceHint
from marin.execution.artifact import Artifact, ArtifactRecord, write_record
from marin.execution.lazy import ArtifactStep, StepContext
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.skyrl import (
    SKYRL_POLICY_LOCATION,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    NativeCheckpointFile,
    ResolvedDataLocator,
    ResolvedModelLocator,
    SkyRLCheckpoint,
    SkyRLEvaluationModel,
    SkyRLExportConfig,
    SkyRLExportOutputPaths,
    SkyRLExportRequest,
    SkyRLLaunchRequest,
    SkyRLModel,
    SkyRLOutputPaths,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRunConfig,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    run_skyrl_training,
    skyrl_checkpoint_step,
    skyrl_export_step,
    skyrl_metrics_step,
    skyrl_step,
)
from marin.rl.skyrl import _run_export as run_skyrl_export
from marin.rl.skyrl import _run_launcher as run_launcher_for_test
from marin.training.training import LevanterCheckpoint

from experiments.evaluation.pipeline import eval_step


def _model_step() -> ArtifactStep[LevanterCheckpoint]:
    return ArtifactStep.adopt(
        "tests/iceball-sft",
        "2026.08.01",
        "s3://test/iceball-sft",
        kind=LevanterCheckpoint,
    )


def _data_step() -> ArtifactStep[Artifact]:
    return ArtifactStep.adopt(
        "tests/iceball-gsm8k",
        "2026.08.01",
        "s3://test/iceball-gsm8k",
    )


class _FakeLauncherProcess:
    """A launcher subprocess: its terminal response lands in the caller's file, its logs on stderr."""

    def __init__(self, *, response: str, logs: str = "", returncode: int = 1, stdout: IO[str] | None = None) -> None:
        if stdout is not None:
            stdout.write(response)
        self.stderr = io.StringIO(logs)
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        return None

    def __enter__(self) -> _FakeLauncherProcess:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _role_plan() -> SkyRLRolePlan:
    return SkyRLRolePlan(
        colocate_all=True,
        policy_num_nodes=1,
        policy_num_gpus_per_node=4,
        num_inference_engines=4,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=16,
        policy_mini_batch_size=16,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=4,
    )


def _spec() -> SkyRLSpec:
    return SkyRLSpec(
        name="users/tester/tests/iceball-rl",
        version="2026.08.01",
        config_yaml="trainer:\n  max_steps: 8\n",
        runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.FSDP),
        model=ArtifactHfModel(
            step=_model_step(),
            tokenizer_uri="Qwen/Qwen3-0.6B-Base",
            tokenizer_revision="da87bfb",
            relative_path="hf/global_step-32",
        ),
        train_data=(ArtifactDataSource(_data_step(), relative_path="train.parquet"),),
        validation_data=(),
        topology=SkyRLTopology(
            num_nodes=1,
            gpus_per_node=4,
            gpu_variant="GB200",
            role_plan=_role_plan(),
        ),
        retention=SkyRLRetentionPolicy(),
        seed=17,
        overrides=("++trainer.max_steps=8",),
    )


def _execution(cluster: str = "cw-us-east-08a") -> IrisSkyRLExecution:
    return IrisSkyRLExecution(
        cluster=cluster,
        cluster_config=f"lib/iris/config/{cluster}.yaml",
        cpu=128,
        memory="800GB",
        disk="4TB",
        priority="interactive",
        max_retries=3,
    )


def test_skyrl_retention_allows_explicit_rollback_depth_up_to_five() -> None:
    policy = SkyRLRetentionPolicy(resume_checkpoint_count=5)

    assert policy.resume_checkpoint_count == 5
    with pytest.raises(ValueError, match="between one and five"):
        SkyRLRetentionPolicy(resume_checkpoint_count=6)


def test_skyrl_step_fingerprint_includes_runtime_identity_and_excludes_placement() -> None:
    spec = _spec()
    base = skyrl_step(spec, _execution())
    moved = skyrl_step(spec, _execution("cw-us-east-02a"))
    resized = skyrl_step(
        spec,
        dataclasses.replace(_execution(), cpu=64, memory="400GB", disk="2TB"),
    )
    changed_profile = skyrl_step(
        dataclasses.replace(
            spec,
            runtime=dataclasses.replace(spec.runtime, profile=SkyRLRuntimeProfile.MEGATRON),
        ),
        _execution(),
    )
    changed_roles = skyrl_step(
        dataclasses.replace(
            spec,
            topology=dataclasses.replace(
                spec.topology,
                role_plan=dataclasses.replace(spec.topology.role_plan, train_batch_size=32),
            ),
        ),
        _execution(),
    )

    assert base.fingerprint() == moved.fingerprint()
    assert base.fingerprint() == resized.fingerprint()
    assert base.fingerprint() != changed_profile.fingerprint()
    assert base.fingerprint() != changed_roles.fingerprint()


def test_skyrl_step_declares_model_and_data_dependencies() -> None:
    step = skyrl_step(_spec(), _execution())

    assert [(dep.name, dep.version) for dep in step.deps] == [(f"{step.name}-training", "2026.08.01")]
    assert [(dep.name, dep.version) for dep in step.deps[0].deps] == [
        ("tests/iceball-sft", "2026.08.01"),
        ("tests/iceball-gsm8k", "2026.08.01"),
    ]


def test_completion_modes_have_distinct_identity_and_model_reuses_checkpoint_training() -> None:
    spec = _spec()
    metrics = skyrl_metrics_step(spec, _execution())
    checkpoint = skyrl_checkpoint_step(spec, _execution())
    model = skyrl_step(spec, _execution())

    assert metrics.name.endswith("-metrics")
    assert checkpoint.name.endswith("-training")
    assert metrics.fingerprint() != checkpoint.fingerprint()
    assert model.name == spec.name
    assert model.deps[0].name == checkpoint.name
    assert model.deps[0].fingerprint() == checkpoint.fingerprint()

    metrics_config = metrics.build_config(StepContext.for_fingerprint(metrics.runtime_args, metrics.deps))
    checkpoint_config = checkpoint.build_config(StepContext.for_fingerprint(checkpoint.runtime_args, checkpoint.deps))
    assert metrics_config.request.completion_mode == "metrics"
    assert metrics_config.request.checkpoint_retention_days is None
    assert metrics_config.request.overrides[-2:] == (
        "++trainer.ckpt_interval=-1",
        "++trainer.hf_save_interval=-1",
    )
    assert checkpoint_config.request.completion_mode == "checkpoint"
    assert checkpoint_config.request.checkpoint_retention_days == 14
    assert checkpoint_config.request.overrides[-1] == "++trainer.hf_save_interval=-1"


def test_export_step_accepts_an_adopted_checkpoint_handle(tmp_path: Path) -> None:
    source = tmp_path / "existing-training"
    checkpoint = ArtifactStep.adopt(
        "users/tester/checkpoints/existing-training",
        "2026.08.01",
        str(source),
        kind=SkyRLCheckpoint,
    )
    recorded_checkpoint = SkyRLCheckpoint(
        path=str(source),
        global_step=17,
        receipt_uri="s3://archive/training/receipt.json",
        resolved_config_uri="s3://archive/training/resolved.json",
        terminal_manifest_uri="s3://archive/training/terminal.json",
        iris_job_id="01KOLDTRAIN",
        checkpoint_path="s3://archive/training/checkpoints/global_step_17",
        trainer_state_sha256="a" * 64,
        files=(NativeCheckpointFile(path="trainer_state.pt", size=10),),
        checkpoint_retention_days=14,
        runtime_commit="old-runtime-commit",
        runtime_profile=SkyRLRuntimeProfile.MEGATRON,
        tokenizer_uri="Qwen/Qwen3-0.6B-Base",
        tokenizer_revision="old-tokenizer-revision",
    )
    recorded_result = recorded_checkpoint.model_dump(mode="json", exclude={"path"})
    write_record(
        ArtifactRecord(
            output_path=str(source),
            result=recorded_result,
        )
    )

    export = skyrl_export_step(checkpoint, _execution())
    fingerprint_config = export.build_config(StepContext.for_fingerprint(export.runtime_args, export.deps))
    config = export.build_config(
        StepContext.for_run(
            str(tmp_path / "export"),
            str(tmp_path),
            runtime_args=export.runtime_args,
            deps=export.deps,
        )
    )

    assert fingerprint_config.source_runtime_commit == "<from-checkpoint-artifact>"
    assert config.request.training_manifest_uri == "s3://archive/training/terminal.json"
    assert config.source_runtime_commit == "old-runtime-commit"
    assert config.source_runtime_profile is SkyRLRuntimeProfile.MEGATRON
    assert export.fingerprint()


@pytest.mark.parametrize(
    ("mode", "override"),
    [
        ("metrics", "++trainer.ckpt_interval=5"),
        ("metrics", "++trainer.hf_save_interval=5"),
        ("checkpoint", "++trainer.hf_save_interval=5"),
    ],
)
def test_completion_mode_rejects_contradictory_positive_save_overrides(mode: str, override: str) -> None:
    spec = dataclasses.replace(_spec(), overrides=(override,))
    build = skyrl_metrics_step if mode == "metrics" else skyrl_checkpoint_step

    with pytest.raises(ValueError, match="forbids positive"):
        build(spec, _execution())


def test_skyrl_step_routes_disposable_state_to_ttl_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "marin.rl.skyrl.temporary_storage_base_path",
        lambda _output_path, *, ttl_days, category: f"s3://temp/ttl={ttl_days}d/{category}/users/alice/run",
    )
    spec = dataclasses.replace(_spec(), name="users/alice/tests/iceball-rl", version="dev")
    step = skyrl_checkpoint_step(spec, _execution())
    output_path = "s3://durable/users/alice/tests/iceball-rl/dev"
    config = step.build_config(
        StepContext.for_run(
            output_path=output_path,
            prefix="s3://durable",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )

    assert step.name == "users/alice/tests/iceball-rl-training"
    assert config.request.output == SkyRLOutputPaths(
        checkpoint_root="s3://temp/ttl=14d/skyrl/users/alice/run/checkpoints",
        export_root=f"{output_path}/exports",
        attempts_root="s3://temp/ttl=14d/skyrl/users/alice/run/attempts",
        resolved_config_uri=f"{output_path}/resolved-skyrl.json",
        terminal_manifest_uri=f"{output_path}/terminal.json",
    )
    # The path values are single-quoted because they carry a ``ttl=<n>d`` segment and Hydra's
    # override grammar rejects a bare value containing ``=``.
    assert config.request.overrides[-4:-1] == (
        "++trainer.max_ckpts_to_keep=2",
        "++terminal_bench_config.trials_dir='s3://temp/ttl=14d/skyrl/users/alice/run/attempts/trace_jobs'",
        "++generator.trajectory_retention.output_path='s3://temp/ttl=14d/skyrl/users/alice/run/attempts/trajectories'",
    )


def test_terminal_policy_composes_into_shared_evaluation_step() -> None:
    rl = skyrl_step(_spec(), _execution())
    model = SkyRLEvaluationModel(
        step=rl,
        model=ModelConfig(
            name="iceball-micro",
            location=SKYRL_POLICY_LOCATION,
            tokenizer="Qwen/Qwen3-0.6B-Base",
            resource_hint=ResourceHint(gpu={"GB200": 1}),
        ),
    )

    evaluation = eval_step(model, "gsm8k", version="2026.08.01", accelerator="GB200x1")

    assert evaluation.deps == (rl,)
    assert evaluation.name == "evals/iceball-micro/gsm8k"
    assert rl.fingerprint() in evaluation.fingerprint_payload()


def test_evaluation_uses_the_validated_training_tokenizer() -> None:
    rl = skyrl_step(_spec(), _execution())
    terminal = SkyRLModel(
        policy_export_uri="s3://test/iceball-rl/exports/global_step_8/policy",
        global_step=8,
        tokenizer_uri="Qwen/Qwen3-0.6B-Base",
        tokenizer_revision="da87bfb",
        checkpoint_root="s3://test/iceball-rl/checkpoints",
        terminal_manifest_uri="s3://test/iceball-rl/terminal.json",
        iris_job_id="/tester/iceball-rl",
    )

    class ResolvedContext:
        is_fingerprint = False

        def resolved(self, step):
            assert step is rl
            return terminal

    source = SkyRLEvaluationModel(
        step=rl,
        model=ModelConfig(
            name="iceball-micro",
            location=SKYRL_POLICY_LOCATION,
            tokenizer="Qwen/Qwen3-0.6B-Base",
            resource_hint=ResourceHint(gpu={"GB200": 1}),
        ),
    )

    model = source.resolve(cast(StepContext, ResolvedContext()))

    assert model.location == terminal.policy_export_uri
    assert model.tokenizer == terminal.tokenizer_uri


@pytest.mark.parametrize("timeout_seconds", [0, 1800])
def test_run_skyrl_returns_external_checkpoint(monkeypatch: pytest.MonkeyPatch, timeout_seconds: int) -> None:
    output = SkyRLOutputPaths(
        checkpoint_root="s3://test/run/checkpoints",
        export_root="s3://test/run/exports",
        attempts_root="s3://test/run/attempts",
        resolved_config_uri="s3://test/run/resolved.json",
        terminal_manifest_uri="s3://test/run/terminal.json",
    )
    request = SkyRLLaunchRequest(
        run_id="checkpoints/iceball-rl-2026.08.01",
        attempt_id="attempt-1",
        config_yaml="trainer: {}\n",
        runtime=_spec().runtime,
        model=ResolvedModelLocator(
            uri="s3://test/sft/hf",
            identity="sft@version:fingerprint",
            local_path="/tmp/model",
            tokenizer_uri="Qwen/Qwen3-0.6B-Base",
            tokenizer_revision="da87bfb",
        ),
        train_data=(
            ResolvedDataLocator(
                uri="s3://test/gsm8k",
                identity="gsm8k@version:fingerprint",
                local_path="/tmp/data",
                relative_path="train.parquet",
            ),
        ),
        validation_data=(),
        topology=_spec().topology,
        output=output,
        seed=17,
        overrides=(),
        completion_mode="checkpoint",
        checkpoint_retention_days=14,
    )
    response = {
        "run_id": request.run_id,
        "attempt_id": request.attempt_id,
        "state": "succeeded",
        "iris_job_id": "01KTEST",
        "iris_job_state": "succeeded",
        "runtime": asdict(request.runtime),
        "failure": None,
        "training": {
            "global_step": 8,
            "receipt_uri": "s3://test/run/receipt.json",
            "resolved_config_uri": output.resolved_config_uri,
            "checkpoint": {
                "checkpoint_path": "s3://test/run/checkpoints/global_step_8",
                "global_step": 8,
                "trainer_state_sha256": "abc123",
                "files": [{"path": "policy/a.distcp", "size": 7}],
            },
        },
    }

    launch_envelopes = []

    def fake_popen(command, **_kwargs) -> _FakeLauncherProcess:
        request_path = command[command.index("--request") + 1]
        launch_envelopes.append(json.loads(Path(request_path).read_text()))
        return _FakeLauncherProcess(response=json.dumps(response), returncode=0, stdout=_kwargs["stdout"])

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    checkpoint = run_skyrl_training(
        SkyRLRunConfig(
            request=request,
            execution=dataclasses.replace(_execution(), timeout_seconds=timeout_seconds),
            launcher_requirement=MARIN_SKYRL.requirement(),
        )
    )

    assert checkpoint.checkpoint_path.endswith("global_step_8")
    assert checkpoint.global_step == 8
    assert checkpoint.iris_job_id == "01KTEST"
    assert launch_envelopes[0]["request"]["runtime"] == {
        "commit": MARIN_SKYRL.commit,
        "profile": SkyRLRuntimeProfile.FSDP.value,
    }
    assert launch_envelopes[0]["execution"]["job_name"] == "checkpoints-iceball-rl-2026.08.01-attempt-1"
    assert launch_envelopes[0]["execution"]["timeout_seconds"] == timeout_seconds


def test_export_uses_the_training_manifest_runtime_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    response = {
        "run_id": "run-1",
        "attempt_id": "export-1",
        "state": "succeeded",
        "runtime": {"commit": "source-commit", "profile": "megatron"},
        "training_iris_job_id": "01KTRAIN",
        "failure": None,
        "reused_export": False,
        "model": {
            "policy_export_uri": "s3://test/export/global_step_8/policy",
            "global_step": 8,
            "tokenizer_uri": "Qwen/Qwen3-0.6B-Base",
            "tokenizer_revision": "da87bfb",
            "checkpoint_root": "s3://test/checkpoints",
            "terminal_manifest_uri": "s3://test/export/terminal.json",
        },
    }
    commands = []

    def fake_popen(command, **kwargs) -> _FakeLauncherProcess:
        commands.append(command)
        return _FakeLauncherProcess(response=json.dumps(response), returncode=0, stdout=kwargs["stdout"])

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    model = run_skyrl_export(
        SkyRLExportConfig(
            request=SkyRLExportRequest(
                training_manifest_uri="s3://test/training/terminal.json",
                attempt_id="export-1",
                output=SkyRLExportOutputPaths(
                    export_root="s3://test/export",
                    attempts_root="s3://test/export/attempts",
                    terminal_manifest_uri="s3://test/export/terminal.json",
                ),
            ),
            execution=_execution(),
            source_runtime_commit="source-commit",
            source_runtime_profile=SkyRLRuntimeProfile.MEGATRON,
        )
    )

    requirement = commands[0][commands[0].index("--with") + 1]
    assert requirement.endswith("@source-commit")
    assert commands[0][commands[0].index("iris") + 1] == "export"
    assert model.iris_job_id == "01KTRAIN"


def test_launcher_failure_reports_the_launcher_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """A launcher that dies before printing its terminal response must still say why."""

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **_kwargs: _FakeLauncherProcess(response="", logs="entrypoint must be a registered name\n"),
    )
    step = skyrl_metrics_step(_spec(), _execution())
    config = step.build_config(
        StepContext.for_run(
            output_path="s3://durable/users/alice/tests/iceball-rl/2026.08.01",
            prefix="s3://durable",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )

    with pytest.raises(RuntimeError, match="entrypoint must be a registered name"):
        run_skyrl_training(config)


def test_launcher_logs_reach_stderr_while_the_run_is_live(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Buffering the launcher's logs until it exits would silence a multi-hour run."""
    observed = tmp_path / "observed"

    class MarkFirstWrite:
        def write(self, text: str) -> int:
            observed.touch()
            return len(text)

        def flush(self) -> None:
            return None

    # The child reports whether the parent had already forwarded its line while it was still
    # running, so an implementation that replays stderr after exit fails here.
    child = (
        "import pathlib, sys, time\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "sys.stderr.write('launcher line\\n')\n"
        "sys.stderr.flush()\n"
        "deadline = time.monotonic() + 5\n"
        "while time.monotonic() < deadline and not marker.exists():\n"
        "    time.sleep(0.01)\n"
        "sys.stdout.write('saw' if marker.exists() else 'missed')\n"
    )
    monkeypatch.setattr(sys, "stderr", MarkFirstWrite())

    completed = run_launcher_for_test([sys.executable, "-c", child, str(observed)])

    assert completed.stdout == "saw"
    assert "launcher line" in completed.stderr


def test_launcher_survives_undecodable_bytes_on_stderr() -> None:
    """Native CUDA and NCCL layers emit non-UTF-8 bytes; strict decoding would wedge the run."""
    completed = run_launcher_for_test(
        [sys.executable, "-c", "import sys; sys.stderr.buffer.write(b'\\xff bad\\n'); sys.exit(4)"]
    )

    assert completed.returncode == 4
