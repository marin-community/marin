# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import io
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO

import pytest
import yaml
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    TaskTroveDataSource,
    TaskTroveSelection,
    TaskTroveTagMatch,
    run_skyrl,
    skyrl_step,
    skyrl_temporary_run_path,
)
from marin.rl.skyrl import _run_launcher as run_launcher_for_test
from marin.training.training import LevanterCheckpoint


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
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=1,
        train_batch_size=16,
        policy_mini_batch_size=16,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=4,
    )


def _config_yaml(*, strategy: str | None = None) -> str:
    strategy_line = f"  strategy: {strategy}\n" if strategy is not None else ""
    return f"""\
trainer:
{strategy_line}  max_steps: 8
  algorithm:
    use_kl_loss: false
generator:
  backend: vllm
  run_engines_locally: true
"""


def _spec() -> SkyRLSpec:
    return SkyRLSpec(
        name="users/tester/tests/iceball-rl",
        version="2026.08.01",
        config_yaml=_config_yaml(),
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
        target_cluster=None,
        parent_cluster_config=None,
        coordinator_timeout_hours=12,
    )


def test_skyrl_retention_allows_explicit_rollback_depth_up_to_five() -> None:
    policy = SkyRLRetentionPolicy(resume_checkpoint_count=5)

    assert policy.resume_checkpoint_count == 5
    with pytest.raises(ValueError, match="between one and five"):
        SkyRLRetentionPolicy(resume_checkpoint_count=6)


def test_skyrl_topology_rejects_unassigned_gpus() -> None:
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=8,
        num_inference_engines=4,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=1,
    )

    with pytest.raises(ValueError, match=r"policy \+ rollout=36 GPUs, topology=64 GPUs"):
        SkyRLTopology(num_nodes=8, gpus_per_node=8, gpu_variant="H100", role_plan=plan)


def test_skyrl_topology_accepts_node_local_dp8_engines() -> None:
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=8,
        num_inference_engines=4,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    )

    SkyRLTopology(num_nodes=8, gpus_per_node=8, gpu_variant="H100", role_plan=plan)


def test_skyrl_topology_rejects_a_colocated_slice_that_does_not_tile_the_node() -> None:
    plan = dataclasses.replace(
        _role_plan(),
        policy_num_gpus_per_node=8,
        num_inference_engines=2,
        inference_engine_tensor_parallel_size=3,
    )

    with pytest.raises(ValueError, match=r"TP\*PP slice must divide gpus_per_node"):
        SkyRLTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100", role_plan=plan)


def test_skyrl_topology_rejects_unequal_colocated_role_sizes() -> None:
    plan = dataclasses.replace(
        _role_plan(),
        policy_num_gpus_per_node=8,
        num_inference_engines=3,
        inference_engine_tensor_parallel_size=2,
    )

    with pytest.raises(ValueError, match="colocated SkyRL roles must use the same GPUs: policy=8, rollout=6"):
        SkyRLTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100", role_plan=plan)


def test_skyrl_spec_rejects_config_that_disagrees_with_role_plan() -> None:
    with pytest.raises(ValueError, match=r"trainer\.train_batch_size=32"):
        dataclasses.replace(
            _spec(),
            config_yaml=_config_yaml().replace("  max_steps: 8", "  max_steps: 8\n  train_batch_size: 32"),
        )


def test_skyrl_spec_rejects_distinct_batch_sizes_for_fully_async() -> None:
    spec = _spec()
    plan = dataclasses.replace(_role_plan(), train_batch_size=32, policy_mini_batch_size=16)

    with pytest.raises(
        ValueError,
        match="fully_async entrypoint requires train_batch_size == policy_mini_batch_size; got 32 and 16",
    ):
        dataclasses.replace(
            spec,
            config_yaml=f"entrypoint: fully_async\n{_config_yaml()}",
            topology=dataclasses.replace(spec.topology, role_plan=plan),
        )


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
    changed_plan = dataclasses.replace(spec.topology.role_plan, train_batch_size=32)
    changed_roles = skyrl_step(
        dataclasses.replace(
            spec,
            config_yaml=_config_yaml(),
            topology=dataclasses.replace(spec.topology, role_plan=changed_plan),
        ),
        _execution(),
    )

    assert base.fingerprint() == moved.fingerprint()
    assert base.fingerprint() == resized.fingerprint()
    assert base.fingerprint() != changed_profile.fingerprint()
    assert base.fingerprint() != changed_roles.fingerprint()


def test_skyrl_step_declares_model_and_data_dependencies() -> None:
    step = skyrl_step(_spec(), _execution())

    assert [(dep.name, dep.version) for dep in step.deps] == [
        ("tests/iceball-sft", "2026.08.01"),
        ("tests/iceball-gsm8k", "2026.08.01"),
    ]


def test_skyrl_step_routes_disposable_state_to_ttl_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "marin.rl.skyrl.skyrl_temporary_run_path",
        lambda _output_path, *, ttl_days: f"s3://temp/ttl={ttl_days}d/skyrl/users/alice/run",
    )
    spec = dataclasses.replace(_spec(), name="users/alice/tests/iceball-rl", version="dev")
    step = skyrl_step(spec, _execution())
    output_path = "s3://durable/users/alice/tests/iceball-rl/dev"
    config = step.build_config(
        StepContext.for_run(
            output_path=output_path,
            prefix="s3://durable",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )

    assert step.name == "users/alice/tests/iceball-rl"
    assert config.output.checkpoint_root == "s3://temp/ttl=14d/skyrl/users/alice/run/checkpoints"
    assert config.output.export_root == f"{output_path}/exports"
    launch = yaml.safe_load(config.launch_config_yaml)
    assert launch["artifacts"] == {
        "checkpoint_root": "s3://temp/ttl=14d/skyrl/users/alice/run/checkpoints",
        "export_root": f"{output_path}/exports",
        "attempts_root": "s3://temp/ttl=14d/skyrl/users/alice/run/attempts",
        "resolved_config_uri": f"{output_path}/resolved-launch.yaml",
        "terminal_manifest_uri": f"{output_path}/terminal.json",
        "resume_checkpoint_count": 2,
    }


def test_skyrl_temporary_run_path_does_not_repeat_bucket_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")

    assert (
        skyrl_temporary_run_path(
            "s3://marin-us-east-02a/marin/users/alice/run",
            ttl_days=14,
        )
        == "s3://marin-us-east-02a/tmp/ttl=14d/skyrl/marin/users/alice/run"
    )


def test_run_skyrl_returns_external_terminal_model(monkeypatch: pytest.MonkeyPatch) -> None:
    step = skyrl_step(_spec(), _execution())
    config = step.build_config(
        StepContext.for_run(
            output_path="s3://test/run",
            prefix="s3://test",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )
    output = config.output
    response = {
        "run_id": config.run_id,
        "attempt_id": config.attempt_id,
        "state": "succeeded",
        "iris_job_id": "01KTEST",
        "iris_job_state": "succeeded",
        "failure": None,
        "model": {
            "policy_export_uri": "s3://test/run/exports/global_step_8/policy",
            "global_step": 8,
            "tokenizer_uri": config.model.tokenizer_uri,
            "tokenizer_revision": config.model.tokenizer_revision,
            "checkpoint_root": output.checkpoint_root,
            "terminal_manifest_uri": output.terminal_manifest_uri,
        },
    }

    launch_configs = []

    def fake_popen(command, **_kwargs) -> _FakeLauncherProcess:
        config_path = command[command.index("--config") + 1]
        launch_configs.append(yaml.safe_load(Path(config_path).read_text()))
        return _FakeLauncherProcess(response=json.dumps(response), returncode=0, stdout=_kwargs["stdout"])

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    catalog_rows = []
    monkeypatch.setattr("marin.rl.skyrl.record_rollout_run", catalog_rows.append)

    model = run_skyrl(config)

    assert model.policy_export_uri.endswith("global_step_8/policy")
    assert model.global_step == 8
    assert model.iris_job_id == "01KTEST"
    launch = launch_configs[0]
    assert launch["schema_version"] == 1
    assert launch["runtime"]["launcher_commit"] == MARIN_SKYRL.commit
    assert launch["iris"]["allocation"] == {
        "num_nodes": 1,
        "gpus_per_node": 4,
        "gpu_variant": "GB200",
        "cpu": 128,
        "memory": "800GB",
        "disk": "4TB",
    }
    assert launch["inputs"]["train_data"][0]["kind"] == "directory"
    assert launch["skyrl"]["trainer"]["max_steps"] == 8
    assert launch["skyrl"]["trainer"]["train_batch_size"] == 16
    assert launch["skyrl"]["trainer"]["placement"] == {
        "colocate_all": True,
        "colocate_policy_ref": True,
        "policy_num_nodes": 1,
        "policy_num_gpus_per_node": 4,
        "ref_num_nodes": 1,
        "ref_num_gpus_per_node": 4,
    }
    assert launch["skyrl"]["generator"]["num_inference_engines"] == 4
    assert launch["skyrl"]["generator"]["inference_engine_data_parallel_size"] == 1
    assert len(catalog_rows) == 1
    assert catalog_rows[0].run_id == config.run_id
    assert catalog_rows[0].attempt_id == config.attempt_id
    assert catalog_rows[0].status == "succeeded"
    assert catalog_rows[0].rollout_uri == f"{output.attempts_root}/trajectories"
    assert catalog_rows[0].job_id == "01KTEST"


def test_tasktrove_data_source_resolves_exact_file_and_verifier(tmp_path: Path) -> None:
    release = ArtifactStep.adopt("tasktrove/clean", "2026.09.10", str(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps({"verify_tool_ref": "tasktrove-verify@abc123"}))
    source = TaskTroveDataSource(
        release,
        TaskTroveSelection(
            sources=("source-b", "source-a"),
            tags=("terminal", "bash"),
            modes=("script",),
            tag_match=TaskTroveTagMatch.ALL,
            limit=160,
            seed=17,
        ),
    )
    context = StepContext.for_run(
        output_path=str(tmp_path / "output"),
        prefix=str(tmp_path),
        runtime_args={},
        deps=(release,),
    )

    resolved = source.resolve(context)

    assert resolved.uri == str(tmp_path / "tasks/part-00000.parquet")
    assert resolved.relative_path == "part-00000.parquet"
    assert resolved.verifier_ref == "tasktrove-verify@abc123"
    assert resolved.selection.sources == ("source-a", "source-b")
    assert resolved.selection.tags == ("bash", "terminal")
    assert resolved.kind == "tasktrove_parquet"


@pytest.mark.parametrize(
    "selection, message",
    [
        (TaskTroveSelection, "requires at least one"),
        (lambda: TaskTroveSelection(tags=("bash", "bash")), "duplicate"),
        (lambda: TaskTroveSelection(sources=("source-a",), limit=0), "positive"),
    ],
)
def test_tasktrove_selection_rejects_ambiguous_inputs(selection: Callable[[], TaskTroveSelection], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        selection()


def test_launcher_failure_reports_the_launcher_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """A launcher that dies before printing its terminal response must still say why."""

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **_kwargs: _FakeLauncherProcess(response="", logs="entrypoint must be a registered name\n"),
    )
    catalog_rows = []
    monkeypatch.setattr("marin.rl.skyrl.record_rollout_run", catalog_rows.append)
    step = skyrl_step(_spec(), _execution())
    config = step.build_config(
        StepContext.for_run(
            output_path="s3://durable/users/alice/tests/iceball-rl/2026.08.01",
            prefix="s3://durable",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )

    with pytest.raises(RuntimeError, match="entrypoint must be a registered name"):
        run_skyrl(config)

    assert len(catalog_rows) == 1
    assert catalog_rows[0].status == "failed"
    assert catalog_rows[0].rollout_uri.endswith("/attempts/trajectories")


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


def test_a_runtime_profile_that_contradicts_the_config_strategy_is_refused() -> None:
    """The mismatch is otherwise silent until the pod has its GPUs: the launcher installs one
    backend's closure, the trainer asks for the other, and the run dies on an import error naming
    neither the profile nor the strategy."""
    spec = _spec()

    with pytest.raises(ValueError, match="megatron"):
        dataclasses.replace(
            spec,
            config_yaml=_config_yaml(strategy="megatron"),
            runtime=dataclasses.replace(spec.runtime, profile=SkyRLRuntimeProfile.FSDP),
        )


@pytest.mark.parametrize(
    "config_yaml",
    [
        pytest.param(_config_yaml(strategy="megatron"), id="names the matching strategy"),
        pytest.param(_config_yaml(), id="names no strategy"),
    ],
)
def test_a_config_that_does_not_contradict_the_profile_is_accepted(config_yaml: str) -> None:
    spec = _spec()

    dataclasses.replace(
        spec,
        config_yaml=config_yaml,
        runtime=dataclasses.replace(spec.runtime, profile=SkyRLRuntimeProfile.MEGATRON),
    )
