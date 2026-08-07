# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
from marin.evaluation.model_config import ModelConfig, ResourceHint
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.skyrl import (
    SKYRL_POLICY_LOCATION,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    ResolvedDataLocator,
    ResolvedModelLocator,
    SkyRLEvaluationModel,
    SkyRLLaunchRequest,
    SkyRLModel,
    SkyRLOutputPaths,
    SkyRLRolePlan,
    SkyRLRunConfig,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    run_skyrl,
    skyrl_step,
)
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
        name="tests/iceball-rl",
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

    assert [(dep.name, dep.version) for dep in step.deps] == [
        ("tests/iceball-sft", "2026.08.01"),
        ("tests/iceball-gsm8k", "2026.08.01"),
    ]


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


def test_run_skyrl_returns_external_terminal_model(monkeypatch: pytest.MonkeyPatch) -> None:
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
    )
    response = {
        "run_id": request.run_id,
        "attempt_id": request.attempt_id,
        "state": "succeeded",
        "iris_job_id": "01KTEST",
        "iris_job_state": "succeeded",
        "runtime": asdict(request.runtime),
        "failure": None,
        "model": {
            "policy_export_uri": "s3://test/run/exports/global_step_8/policy",
            "global_step": 8,
            "tokenizer_uri": request.model.tokenizer_uri,
            "tokenizer_revision": request.model.tokenizer_revision,
            "checkpoint_root": output.checkpoint_root,
            "terminal_manifest_uri": output.terminal_manifest_uri,
        },
    }

    launch_envelopes = []

    def fake_run(command, **_kwargs) -> subprocess.CompletedProcess[str]:
        request_path = command[command.index("--request") + 1]
        launch_envelopes.append(json.loads(Path(request_path).read_text()))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(response))

    monkeypatch.setattr(subprocess, "run", fake_run)

    model = run_skyrl(
        SkyRLRunConfig(
            request=request,
            execution=_execution(),
            launcher_requirement=MARIN_SKYRL.requirement(),
        )
    )

    assert model.policy_export_uri.endswith("global_step_8/policy")
    assert model.global_step == 8
    assert model.iris_job_id == "01KTEST"
    assert launch_envelopes[0]["request"]["runtime"] == {
        "commit": MARIN_SKYRL.commit,
        "profile": SkyRLRuntimeProfile.FSDP.value,
    }
    assert launch_envelopes[0]["execution"]["job_name"] == "checkpoints-iceball-rl-2026.08.01-attempt-1"
