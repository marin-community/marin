# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import IO

import pytest
import yaml
from finestore.reader import ReadView
from finestore.rl.run_records import (
    AttemptRecord,
    LaunchState,
    read_attempt,
    read_attempt_metrics,
    run_records_root,
    update_attempt,
)
from finestore.store import DataStore
from iris.rpc import controller_pb2, job_pb2
from marin.evaluation.model_config import ModelConfig
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLGenerationPlan,
    SkyRLPolicyModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    TaskTroveDataSource,
    TaskTroveSelection,
    TaskTroveTagMatch,
    read_skyrl_launch_receipt,
    run_skyrl,
    skyrl_cpu_command,
    skyrl_step,
    skyrl_temporary_run_path,
    validate_megatron_geometry,
)
from marin.rl.skyrl import _run_launcher as run_launcher_for_test
from marin.rl.skyrl_records import (
    attempt_tracking,
    read_skyrl_metrics,
    read_stored_skyrl_attempts,
    read_stored_skyrl_receipt,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import cluster_config
from skyrlrecipe import RLEntrypoint

from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, resolve_skyrl_model


def _model_step() -> ArtifactStep[LevanterCheckpoint]:
    return ArtifactStep.adopt(
        "tests/iceball-sft",
        "2026.08.01",
        "s3://test/iceball-sft",
        kind=LevanterCheckpoint,
    )


@pytest.mark.parametrize("relative_path", [".", "hf/.", "hf/../policy"])
def test_artifact_hf_model_rejects_literal_object_store_dot_segments(relative_path: str) -> None:
    # A trailing '/.' caused a real export to appear empty during GPU startup.
    with pytest.raises(ValueError, match="dot segments"):
        ArtifactHfModel(_model_step(), "test/model", "revision", relative_path=relative_path)


@pytest.mark.parametrize(
    ("relative_path", "expected_path"),
    [("", ""), ("hf/checkpoint-10", "hf/checkpoint-10"), (None, "hf/checkpoint-20")],
)
def test_artifact_hf_model_resolves_root_explicit_path_and_discovery(
    tmp_path: Path, relative_path: str | None, expected_path: str
) -> None:
    root = tmp_path / "model"
    for name, modified in (("checkpoint-10", 10), ("checkpoint-20", 20)):
        export = root / "hf" / name
        export.mkdir(parents=True)
        (export / "config.json").write_text("{}")
        (export / "tokenizer_config.json").write_text("{}")
        os.utime(export / "config.json", (modified, modified))
    # A config without tokenizer metadata is not a discoverable HF export.
    (root / "config.json").write_text("{}")
    model = ArtifactHfModel(
        ArtifactStep.adopt("tests/model", "2026.08.01", str(root), kind=LevanterCheckpoint),
        "test/tokenizer",
        "revision",
        relative_path=relative_path,
    )
    ctx = StepContext.for_run(str(tmp_path / "output"), str(tmp_path), deps=model.deps())

    resolved = model.resolve(ctx)

    assert resolved.uri == str(root / expected_path)


def _data_step() -> ArtifactStep[Artifact]:
    return ArtifactStep.adopt(
        "tests/iceball-gsm8k",
        "2026.08.01",
        "s3://test/iceball-gsm8k",
    )


def test_nested_launcher_does_not_inherit_parent_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_uv = shutil.which("uv")
    assert real_uv is not None
    real_uv = str(Path(real_uv).resolve())
    python_version = list(sys.version_info[:2])
    workspace = tmp_path / "workspace"
    project = workspace / "vendor" / "skyrl"
    project.mkdir(parents=True)
    (workspace / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers=[]\nexclude=["vendor/skyrl"]\n')
    (project / "pyproject.toml").write_text(
        '[project]\nname="launcher-environment-test"\nversion="0.0.0"\n'
        f'requires-python="=={python_version[0]}.{python_version[1]}.*"\ndependencies=[]\n'
        "[project.optional-dependencies]\ncpu=[]\n[dependency-groups]\ndev=[]\n[tool.uv]\npackage=false\n"
    )
    (project / ".uv-version").write_bytes(
        (Path(__file__).resolve().parents[2] / "vendor/skyrl/.uv-version").read_bytes()
    )
    (workspace / ".python-version").write_text("3.12.999\n")
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    subprocess.run(
        [real_uv, "lock", "--project", str(project), "--python", sys.executable, "--no-python-downloads"],
        check=True,
        capture_output=True,
    )
    # Isolate tool installation; real UV still selects both interpreters and
    # executes the frozen project, with no index or Python downloads.
    binaries = tmp_path / "bin"
    binaries.mkdir()
    dispatcher = binaries / "uv"
    dispatcher.write_text(
        "#!/bin/sh\nset -eu\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        f'    --python) {shlex.quote(real_uv)} python find "$2" --no-python-downloads >/dev/null; shift 2 ;;\n'
        f'    uv) shift; exec {shlex.quote(real_uv)} "$@" ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
    )
    dispatcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr("marin.rl.skyrl.__file__", str(workspace / "lib/marin/src/marin/rl/skyrl.py"))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "incompatible-parent-python"))
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, "/parent/environment")
    monkeypatch.chdir(workspace)
    result = run_launcher_for_test(
        skyrl_cpu_command(
            "python",
            "-c",
            "import json, os, sys; from pathlib import Path; "
            "print(json.dumps({'python': list(sys.version_info[:2]), 'cwd': str(Path.cwd()), "
            "'environment': {key: os.environ.get(key) for key in "
            "['PYTHONPATH', 'PYTHONHOME', 'UV_PROJECT_ENVIRONMENT', 'VIRTUAL_ENV']}}))",
        )
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "python": python_version,
        "cwd": str(workspace),
        "environment": {
            "PYTHONPATH": None,
            "PYTHONHOME": None,
            "UV_PROJECT_ENVIRONMENT": None,
            "VIRTUAL_ENV": str(project / ".venv"),
        },
    }


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


def _config_yaml(*, strategy: str = "megatron") -> str:
    return f"""\
trainer:
  strategy: {strategy}
  max_steps: 8
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
        job_timeout_seconds=1800,
    )


@pytest.mark.parametrize("key", ["use_conversation_multi_turn", "require_exact_chat_transport"])
@pytest.mark.parametrize("value", [False, True])
def test_skyrl_spec_rejects_retired_transport_options_before_submission(key: str, value: bool) -> None:
    recipe = yaml.safe_load(_config_yaml())
    recipe["generator"][key] = value

    with pytest.raises(ValueError, match=f"generator.{key}"):
        dataclasses.replace(_spec(), config_yaml=yaml.safe_dump(recipe))


def test_skyrl_retention_allows_ten_checkpoint_recovery_window() -> None:
    # The Smartie continuation failed preflight despite its requested ten-checkpoint window.
    policy = SkyRLRetentionPolicy(resume_checkpoint_count=10)

    assert policy.resume_checkpoint_count == 10
    with pytest.raises(ValueError, match="between one and ten"):
        SkyRLRetentionPolicy(resume_checkpoint_count=11)


def test_skyrl_launch_reserves_capacity_for_config_derived_draft_trainer() -> None:
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=8,
        num_inference_engines=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    )
    recipe = yaml.safe_load(_config_yaml(strategy="megatron"))
    recipe["generator"]["speculative_decoding"] = {
        "method": "eagle3",
        "model": {
            "source_uri": "hf://test/draft",
            "source_identity": "0" * 40,
        },
        "num_speculative_tokens": 3,
        "training": {},
    }
    spec = dataclasses.replace(
        _spec(),
        config_yaml=yaml.safe_dump(recipe),
        topology=SkyRLTopology(
            num_nodes=6,
            gpus_per_node=8,
            gpu_variant="H100",
            role_plan=plan,
        ),
    )
    step = skyrl_step(spec, _execution())
    launch_config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    launch = yaml.safe_load(launch_config.launch_config_yaml)

    assert launch["iris"]["allocation"]["num_nodes"] == 6
    assert launch["skyrl"]["generator"]["speculative_decoding"]["training"] == {}
    assert launch["run"]["export_hf"] is False
    assert launch_config.draft_checkpoint_root == "<temporary_output_path>/checkpoints/drafts"


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


@pytest.mark.parametrize(("reference_nodes", "allocated_nodes"), [(None, 2), (1, 3)])
def test_static_preference_launch_reserves_policy_and_reference_without_rollout_nodes(reference_nodes, allocated_nodes):
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        policy_num_nodes=2,
        num_inference_engines=0,
        reference_num_nodes=reference_nodes,
    )
    recipe = yaml.safe_load(_config_yaml())
    recipe["trainer"]["evaluation_runner"] = "training"
    recipe["trainer"]["algorithm"].update(policy_loss_type="dpo", loss_reduction="pair_mean")
    recipe["environment"] = {"env_class": "preference_pair"}
    spec = dataclasses.replace(
        _spec(),
        topology=SkyRLTopology(num_nodes=allocated_nodes, gpus_per_node=4, gpu_variant="GB200", role_plan=plan),
        config_yaml=yaml.safe_dump(recipe),
    )
    step = skyrl_step(spec, _execution())
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    launch = yaml.safe_load(config.launch_config_yaml)
    assert launch["iris"]["allocation"]["num_nodes"] == allocated_nodes
    assert launch["skyrl"]["generator"]["num_inference_engines"] == 0
    assert plan.num_nodes == allocated_nodes


def test_skyrl_spec_rejects_config_that_disagrees_with_role_plan() -> None:
    with pytest.raises(ValueError, match=r"trainer\.train_batch_size=32"):
        dataclasses.replace(
            _spec(),
            config_yaml=_config_yaml().replace("  max_steps: 8", "  max_steps: 8\n  train_batch_size: 32"),
        )


def test_skyrl_separate_reference_reserves_capacity_and_renders_distinct_roles() -> None:
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=8,
        reference_num_nodes=2,
        num_inference_engines=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    )
    with pytest.raises(ValueError, match="core role plan exceeds the allocated topology"):
        SkyRLTopology(num_nodes=5, gpus_per_node=8, gpu_variant="H100", role_plan=plan)
    topology = SkyRLTopology(num_nodes=7, gpus_per_node=8, gpu_variant="H100", role_plan=plan)
    recipe = yaml.safe_load(_config_yaml(strategy="megatron"))
    recipe["trainer"]["algorithm"]["use_kl_loss"] = True
    recipe["trainer"]["policy"] = {
        "megatron_config": {"pipeline_model_parallel_size": 2, "expert_model_parallel_size": 8}
    }
    recipe["trainer"]["ref"] = {"megatron_config": {"pipeline_model_parallel_size": 2, "expert_model_parallel_size": 16}}
    with pytest.raises(ValueError, match=r"Megatron ref has 16 GPUs.*ETP\*EP\*PP"):
        dataclasses.replace(_spec(), topology=topology, config_yaml=yaml.safe_dump(recipe))
    recipe["trainer"]["ref"]["megatron_config"]["expert_model_parallel_size"] = 8
    spec = dataclasses.replace(
        _spec(),
        config_yaml=yaml.safe_dump(recipe),
        topology=topology,
    )
    step = skyrl_step(spec, _execution())
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    launch = yaml.safe_load(config.launch_config_yaml)
    assert launch["iris"]["allocation"]["num_nodes"] * launch["iris"]["allocation"]["gpus_per_node"] == 56
    assert launch["skyrl"]["trainer"]["placement"] == {
        "colocate_all": False,
        "colocate_policy_ref": False,
        "policy_num_nodes": 4,
        "policy_num_gpus_per_node": 8,
        "ref_num_nodes": 2,
        "ref_num_gpus_per_node": 8,
    }


@pytest.mark.parametrize("failure", ["render", "extra-node", "node-width", "entrypoint"])
def test_generation_plan_renders_native_geometry_and_rejects_conflicts(failure, tmp_path):
    spec = _spec()
    geometry = dataclasses.replace(
        spec.topology.role_plan, colocate_all=False, policy_num_nodes=4, entrypoint=RLEntrypoint.TASKCOMPENDIUM
    )
    plan = SkyRLGenerationPlan(geometry)
    topology = dataclasses.replace(spec.topology, num_nodes=plan.num_nodes, role_plan=plan)
    if failure == "render":
        spec = dataclasses.replace(
            spec,
            topology=topology,
            train_data=(),
            validation_data=spec.train_data,
            config_yaml=(
                "context_budget: {request_window_tokens: 1024, max_new_tokens_per_turn: 256, max_turns: 1}\n"
                "model_num_attention_heads: 8\n" + spec.config_yaml
            ),
        )
        step = skyrl_step(spec, _execution())
        config = step.build_config(
            StepContext.for_run(str(tmp_path / "run"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps)
        )
        path = tmp_path / "generation.yaml"
        path.write_text(config.launch_config_yaml)
        result = run_launcher_for_test(
            skyrl_cpu_command(
                "python",
                "-c",
                """
import json, sys
from cloud.iris.launch_config import load_launch_config
from cloud.iris.rl_config_translation import RL_ENTRYPOINTS
from skyrlrecipe import RLEntrypoint
config = load_launch_config(sys.argv[1])
assert config.runtime.entrypoint == RL_ENTRYPOINTS[RLEntrypoint.GENERATE]
print(json.dumps({"nodes": config.iris.allocation.num_nodes,
    "policy_nodes": config.skyrl.trainer.placement.policy_num_nodes,
    "batch": config.skyrl.trainer.train_batch_size,
    "engines": config.skyrl.generator.num_inference_engines}))
""",
                str(path),
            )
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout) == {"nodes": 1, "policy_nodes": 4, "batch": 16, "engines": 4}
        return
    with pytest.raises(
        ValueError,
        match={
            "extra-node": "exactly the inference engine nodes",
            "node-width": "whole-node topology width",
            "entrypoint": "entrypoint is derived",
        }[failure],
    ):
        if failure == "extra-node":
            dataclasses.replace(topology, num_nodes=topology.num_nodes + 1)
        elif failure == "node-width":
            dataclasses.replace(
                topology,
                role_plan=SkyRLGenerationPlan(
                    dataclasses.replace(plan.geometry, policy_num_gpus_per_node=2, num_inference_engines=2)
                ),
            )
        else:
            dataclasses.replace(spec, topology=topology, config_yaml="entrypoint: generate\n" + spec.config_yaml)


def _colocated_expert_block_spec(entrypoint: str | None = None) -> SkyRLSpec:
    plan = dataclasses.replace(
        _role_plan(),
        num_inference_engines=1,
        inference_engine_data_parallel_size=4,
        inference_engine_expert_parallel_size=4,
    )
    recipe = yaml.safe_load(_config_yaml())
    recipe["trainer"]["policy"] = {
        "megatron_config": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 4,
        }
    }
    recipe["generator"].update(
        weight_sync_transport="expert_block",
        weight_sync_backend="nccl",
        expert_block_sync={"verify": True},
        engine_init_kwargs={"kernel_config": {"moe_backend": "triton"}},
    )
    if entrypoint == "generate":
        plan = SkyRLGenerationPlan(plan)
    elif entrypoint is not None:
        plan = dataclasses.replace(plan, entrypoint=entrypoint)

    return dataclasses.replace(
        _spec(),
        topology=dataclasses.replace(_spec().topology, role_plan=plan),
        config_yaml=yaml.safe_dump(recipe),
    )


@pytest.mark.parametrize("entrypoint", ["generate", "taskcompendium"])
def test_expert_block_colocation_fails_before_job_submission(entrypoint: str) -> None:
    with pytest.raises(ValueError, match="colocated expert_block requires entrypoint=standard"):
        _colocated_expert_block_spec(entrypoint)


def test_expert_block_colocated_four_gpu_recipe_passes_preflight() -> None:
    spec = _colocated_expert_block_spec()
    step = skyrl_step(spec, _execution())
    launch = yaml.safe_load(
        step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).launch_config_yaml
    )
    assert launch["iris"]["allocation"]["num_nodes"] == 1
    assert launch["skyrl"]["generator"]["inference_engine_data_parallel_size"] == 4


def test_skyrl_step_fingerprint_includes_roles_and_excludes_placement() -> None:
    spec = _spec()
    base = skyrl_step(spec, _execution())
    moved = skyrl_step(spec, _execution("cw-us-east-02a"))
    resized = skyrl_step(
        spec,
        dataclasses.replace(_execution(), cpu=64, memory="400GB", disk="2TB"),
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
    assert base.fingerprint() != changed_roles.fingerprint()

    recipe = yaml.safe_load(spec.config_yaml)
    recipe["trainer"]["callbacks"] = [{"type": "evaluation", "additional_evaluations": {"quality": {}}}]
    panel = dataclasses.replace(_data_step(), name="tests/heldout-panel", version="2026.11.05")
    source = ArtifactDataSource(panel, relative_path="validation.parquet")
    profiled_spec = dataclasses.replace(
        spec, config_yaml=yaml.safe_dump(recipe), evaluation_data=(("quality", (source,)),)
    )
    profiled = skyrl_step(profiled_spec, _execution())
    changed_panel = dataclasses.replace(source, step=dataclasses.replace(panel, version="2026.11.05.1"))
    assert (
        profiled.fingerprint()
        != skyrl_step(
            dataclasses.replace(profiled_spec, evaluation_data=(("quality", (changed_panel,)),)), _execution()
        ).fingerprint()
    )


def test_skyrl_step_declares_model_and_data_dependencies() -> None:
    spec = _spec()
    recipe = yaml.safe_load(spec.config_yaml)
    recipe["trainer"]["callbacks"] = [{"type": "evaluation", "additional_evaluations": {"quality": {}}}]
    panel = dataclasses.replace(_data_step(), name="tests/heldout-panel", version="2026.11.05")
    step = skyrl_step(
        dataclasses.replace(
            spec,
            config_yaml=yaml.safe_dump(recipe),
            evaluation_data=(("quality", (ArtifactDataSource(panel, relative_path="validation.parquet"),)),),
        ),
        _execution(),
    )

    assert [(dep.name, dep.version) for dep in step.deps] == [
        ("tests/iceball-sft", "2026.08.01"),
        ("tests/iceball-gsm8k", "2026.08.01"),
        ("tests/heldout-panel", "2026.11.05"),
    ]


def test_skyrl_step_routes_disposable_state_to_ttl_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "marin.rl.skyrl.skyrl_temporary_run_path",
        lambda _output_path, *, ttl_days: f"s3://temp/ttl={ttl_days}d/skyrl/users/alice/run",
    )
    recipe = yaml.safe_load(_spec().config_yaml)
    recipe["trainer"]["callbacks"] = [{"type": "best_checkpoint", "initial_model_uri": "wrong-model"}]
    spec = dataclasses.replace(
        _spec(), name="users/alice/tests/iceball-rl", version="dev", config_yaml=yaml.safe_dump(recipe)
    )
    monkeypatch.setattr("marin.rl.skyrl.get_job_info", lambda: None)
    outside = skyrl_step(spec, _execution()).fingerprint()
    monkeypatch.setattr("marin.rl.skyrl.get_job_info", lambda: SimpleNamespace(job_id="/tests/coordinator"))
    step = skyrl_step(spec, _execution())
    assert step.fingerprint() == outside
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
    assert launch["run"]["coordinator_iris_job_id"] == "/tests/coordinator"
    assert launch["skyrl"]["trainer"]["callbacks"][0]["initial_model_uri"] == config.model.uri
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
    monkeypatch.delenv("MARIN_TEMP_PREFIX", raising=False)
    monkeypatch.setattr(
        cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", (str(Path(__file__).resolve().parents[2] / "config"),)
    )
    cluster_config.reset_data_config_cache()
    try:
        assert (
            skyrl_temporary_run_path(
                "s3://marin-us-east-02a/marin/users/alice/run",
                ttl_days=14,
            )
            == "s3://marin-us-east-08a/tmp/ttl=14d/skyrl/marin/users/alice/run"
        )
    finally:
        cluster_config.reset_data_config_cache()


def test_run_skyrl_returns_exported_policy_and_records_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    step = skyrl_step(_spec(), _execution(), export_hf=True)
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

    assert model.hf_model_uri is not None and model.hf_model_uri.endswith("global_step_8/policy")
    assert model.global_step == 8
    assert model.iris_job_id == "01KTEST"
    launch = launch_configs[0]
    assert launch["iris"]["timeout"] == 1800
    assert launch["schema_version"] == 1
    assert launch["run"]["export_hf"] is True
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


@pytest.mark.timeout(120)
def test_native_writers_round_trip_through_stored_readers(tmp_path, monkeypatch):
    spec = _spec()
    spec = dataclasses.replace(
        spec,
        config_yaml=(
            "context_budget: {request_window_tokens: 1024, max_new_tokens_per_turn: 256, max_turns: 1}\n"
            "model_num_attention_heads: 8\n" + spec.config_yaml
        ),
    )
    step = skyrl_step(spec, dataclasses.replace(_execution(), max_retries=0), export_hf=True)
    config = step.build_config(
        StepContext.for_run(str(tmp_path), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps)
    )
    launch = yaml.safe_load(config.launch_config_yaml)
    launch["artifacts"] = {
        key: str(tmp_path / value)
        for key, value in {
            "checkpoint_root": "checkpoints",
            "export_root": "exports",
            "attempts_root": "attempts",
            "resolved_config_uri": "resolved.yaml",
            "terminal_manifest_uri": "terminal.json",
        }.items()
    }
    launch["iris"]["wandb_entity"] = "tests-team"
    launch["skyrl"]["trainer"].update(project_name="tests-project", logger=["records", "wandb"])
    launch_path = tmp_path / "launch.yaml"
    launch_path.write_text(yaml.safe_dump(launch))
    (tmp_path / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": launch}))
    assert read_skyrl_metrics(str(tmp_path)) == {}
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "IRIS_TASK_ID"):
        environment.pop(key, None)
    script = r"""
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from unittest.mock import patch
from cloud.iris.export_hf_checkpoint import ExportJobSpec, checkpoint_export_launch_config
from cloud.iris.launch import LaunchOutcome, execute_launch
from cloud.iris.launch_config import load_launch_config
from cloud.iris.skyrl_entrypoint import _start_attempt_record
from marinskyrl.checkpoint_paths import policy_export_path
from cloud.iris.artifacts import CHECKPOINT_MARKER_FILENAME
from skyrl_train.hf_export_schema import HFExportRequest
from skyrl_train.utils.tracking import Tracking
from finestore.rl.run_records import (
    AttemptRecord, CheckpointReference, LaunchResult, LaunchState,
    read_attempt, read_attempt_metrics, run_records_root, update_attempt,
)
from finestore.store import DataStore

path = Path(sys.argv[1])
root = path.parent
invalid = OmegaConf.load(path)
invalid.iris.max_retries = 1
invalid_path = root / "retry-enabled.yaml"
invalid_path.write_text(OmegaConf.to_yaml(invalid))
try:
    load_launch_config(invalid_path)
except ValueError as error:
    assert "records backend requires iris.max_retries=0" in str(error), error
else:
    raise AssertionError("retry-enabled records launch was accepted")
import cloud.iris.launch, skyrl_train, taskcompendium, finestore.rl.run_records
for module in (cloud.iris.launch, skyrl_train, taskcompendium, finestore.rl.run_records):
    assert Path(module.__file__).resolve().is_relative_to(Path.cwd()), module.__file__

def history_config(attempt_id, export_root):
    return OmegaConf.create({"trainer": {"export_path": str(export_root)},
        "runtime": {"run_id": "history", "attempt_id": attempt_id}})

ancestor_root = root / "ancestor-exports"
original = Tracking("tests-project", "history", ("records",), history_config("original", ancestor_root))
original.start(resumed_step=None)
for step in (0, 1, 2, 3, 8):
    original.log({"loss": step / 10}, step=step)
original.log({"ancestor_only": 77}, step=0)
assert read_attempt_metrics(str(ancestor_root), "original")[8]["loss"] == 0.8
parent = CheckpointReference(export_root=str(ancestor_root), attempt_id="original", step=3)
with Tracking("tests-project", "history", ("records",), history_config("middle", root / "exports")) as middle:
    middle.start(resumed_step=3, checkpoint=parent)
    for step in (4, 5, 9):
        middle.log({"loss": step / 10}, step=step)
    middle.log({"eval/all/avg_score": 0.4}, step=5)
with Tracking("tests-project", "history", ("records",), history_config("sibling", root / "exports")) as sibling:
    sibling.start(resumed_step=3, checkpoint=parent)
    sibling.log({"loss": 9.9}, step=4)
try:
    with Tracking("tests-project", "history", ("records",), history_config("middle", root / "exports")) as replay:
        replay.start(resumed_step=3, checkpoint=parent)
except ValueError as error:
    assert "fresh attempt_id" in str(error), error
else:
    raise AssertionError("replayed execution reused its old metric rows")
with Tracking("tests-project", "history", ("records",), history_config("rewind", root / "exports")) as rewind:
    rewind.start(resumed_step=1, checkpoint=parent.model_copy(update={"step": 1}))
    rewind.log({"loss": 2.5}, step=2)
with Tracking("tests-project", "history", ("records",), history_config("rewind-middle", root / "exports")) as rewind:
    rewind.start(resumed_step=2,
        checkpoint=CheckpointReference(export_root=str(root / "exports"), attempt_id="middle", step=2))
    rewind.log({"loss": 9.3}, step=4)
with Tracking("tests-project", "history", ("records",), history_config("stage", root / "exports")) as stage:
    stage.start(resumed_step=0,
        checkpoint=CheckpointReference(export_root=str(root / "exports"), attempt_id="middle", step=5))
    stage.log({"loss": 9}, step=0)
    stage.log({"loss": 8}, step=1)
with Tracking("tests-project", "history", ("records",), history_config("missing-parent", root / "exports")) as missing:
    missing.start(resumed_step=2,
        checkpoint=CheckpointReference(export_root=str(ancestor_root), attempt_id="unrecorded", step=2))
    missing.log({"loss": 3}, step=3)
with Tracking("tests-project", "history", ("records",), history_config("main-resume", root / "exports")) as main:
    main.start(resumed_step=8)
    assert read_attempt(str(root / "exports"), "main-resume").metrics_started_at is not None
    for invalid in ({"step": 8}, {"attempt_id": "foreign"}, {"metric_attempt_ids": {}}, {"label": "text"}):
        try:
            main.log(invalid, step=9)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid numeric metric was accepted: {invalid}")
    main.log({"loss": 0.9}, step=9)
unknown = CheckpointReference(export_root=str(root / "absent-exports"), attempt_id="unknown", step=2)
with Tracking("tests-project", "history", ("records",), history_config("missing-store", root / "exports")) as child:
    child.start(resumed_step=2, checkpoint=unknown)
    child.log({"loss": 0.3}, step=3)
with Tracking("tests-project", "history", ("records",), history_config("grandchild", root / "exports")) as child:
    child.start(resumed_step=3,
        checkpoint=CheckpointReference(export_root=str(root / "exports") + "/", attempt_id="missing-store", step=3))
    child.log({"loss": 0.4}, step=4)
for first in ("metrics", "result"):
    race_root = str(root / ("race-" + first))
    attempt_id = "race-" + first
    with DataStore.open(run_records_root(race_root)) as seed:
        update_attempt(seed, AttemptRecord(run_id="history", attempt_id=attempt_id,
            started_at="2026-11-11T01:00:00+00:00", vllm_version="first"))
    with DataStore.open(run_records_root(race_root)) as left, DataStore.open(run_records_root(race_root)) as right:
        metrics = AttemptRecord(run_id="history", attempt_id=attempt_id,
            metrics_started_at="2026-11-11T01:01:00+00:00", resumed_step=3, parent=parent)
        result = AttemptRecord(run_id="history", attempt_id=attempt_id,
            result=LaunchResult(state=LaunchState.FAILED, failure="interleaved terminal"))
        updates = ((left, metrics), (right, result))
        for store, update in updates if first == "metrics" else reversed(updates):
            update_attempt(store, update)
    stored = read_attempt(race_root, attempt_id)
    assert stored.started_at == "2026-11-11T01:00:00+00:00" and stored.vllm_version == "first"
    assert stored.parent == parent and stored.resumed_step == 3
    assert stored.metrics_started_at == "2026-11-11T01:01:00+00:00"
    assert stored.result.state == LaunchState.FAILED and stored.result.failure == "interleaved terminal"
    with DataStore.open(run_records_root(race_root)) as clear:
        update_attempt(clear, AttemptRecord(run_id="history", attempt_id=attempt_id, vllm_version=None))
    cleared = read_attempt(race_root, attempt_id)
    assert cleared.vllm_version is None and cleared.parent == parent and cleared.result == stored.result
cycle_root = str(root / "cycle-exports")
with DataStore.open(run_records_root(cycle_root)) as cyclic:
    for attempt_id, parent_id in (("left", "right"), ("right", "left")):
        update_attempt(cyclic, AttemptRecord(run_id="history", attempt_id=attempt_id, resumed_step=1,
            parent=CheckpointReference(export_root=cycle_root + "/", attempt_id=parent_id, step=1)))
try:
    read_attempt_metrics(cycle_root, "left")
except ValueError as error:
    assert "cycle" in str(error), error
else:
    raise AssertionError("Checkpoint alias cycle was accepted")
try:
    with Tracking("tests-project", "history", ("records",), history_config("failed-direct", root / "exports")) as failed:
        failed.start(resumed_step=None)
        failed.log({"loss": 0.9}, step=1)
        raise RuntimeError("fixture workload failed")
except RuntimeError as error:
    assert str(error) == "fixture workload failed", error

class LocalBackend:
    def launch(self, config_path):
        config = load_launch_config(config_path)
        resolved = Path(config.artifacts.resolved_config_uri)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(OmegaConf.to_yaml(config))
        if config.run.mode == "checkpoint_export":
            policy = Path(policy_export_path(str(config.artifacts.export_root), 8))
            policy.mkdir(parents=True, exist_ok=True)
            for name in ("config.json", "model.safetensors", "tokenizer.json"):
                (policy / name).write_text("{}")
        elif config.run.attempt_id != "earlier-failed-attempt":
            with open_dict(config.skyrl):
                config.skyrl.runtime = {"run_id": config.run.id, "attempt_id": config.run.attempt_id,
                    "iris_job_id": "/local/training", "coordinator_iris_job_id": None}
            with patch("cloud.iris.skyrl_entrypoint.version", return_value="fixture-vllm"):
                _start_attempt_record(config)
                interrupted = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
                interrupted.run.attempt_id = "interrupted-attempt"
                interrupted.skyrl.runtime.attempt_id = "interrupted-attempt"
                interrupted.skyrl.runtime.iris_job_id = "/local/interrupted"
                interrupted.run.coordinator_iris_job_id = "/local/interrupted-coordinator"
                interrupted.skyrl.runtime.coordinator_iris_job_id = "/local/interrupted-coordinator"
                _start_attempt_record(interrupted)
                foreign = OmegaConf.to_container(interrupted, resolve=True)
                foreign["run"]["id"] = "unrelated-native-run"
                foreign["run"]["attempt_id"] = "foreign-native"
                with DataStore.open(run_records_root(str(config.artifacts.export_root))) as store:
                    update_attempt(store, AttemptRecord(run_id="unrelated-native-run", attempt_id="foreign-native",
                        config=foreign, started_at="2026-11-11T01:00:00+00:00"))
            startup = read_attempt(str(config.artifacts.export_root), str(config.run.attempt_id))
            assert startup is not None and startup.vllm_version == "fixture-vllm"
            (root / "startup-observation.json").write_text(startup.model_dump_json())
            with Tracking("tests-project", "record-boundary", ("records",), config.skyrl) as tracker:
                tracker.start(resumed_step=5,
                    checkpoint=CheckpointReference(export_root=str(root / "exports"), attempt_id="middle", step=5))
                immediate = read_attempt(str(config.artifacts.export_root), str(config.run.attempt_id))
                assert immediate.metrics_started_at is not None and immediate.parent.attempt_id == "middle"
                tracker.log({"startup/resume": 0.05, "eval/all/avg_score": 0.6}, step=5)
                tracker.log({"loss": 1.25}, step=6)
                tracker.log({"loss": 0.25, "entropy": 0.75}, step=6)
            checkpoints = Path(config.artifacts.checkpoint_root)
            checkpoints.mkdir(parents=True, exist_ok=True)
            (checkpoints / CHECKPOINT_MARKER_FILENAME).write_text("8\n")
        failed = config.run.attempt_id == "earlier-failed-attempt"
        return LaunchOutcome("/local/training", "failed" if failed else "succeeded", int(failed))

    def export_terminal_policy(self, config_path):
        config = load_launch_config(config_path)
        request = HFExportRequest(step=8, checkpoint_base_path=str(config.artifacts.checkpoint_root),
            checkpoint_path=str(Path(config.artifacts.checkpoint_root) / "global_step_8"),
            export_path=str(config.artifacts.export_root), model_path=str(config.skyrl.trainer.policy.model.path),
            model_source_uri=str(config.inputs.model.uri), model_source_identity=str(config.inputs.model.identity),
            num_nodes=1, gpus_per_node=4)
        spec = ExportJobSpec(request=request, cluster="local", priority="batch", gpu_variant="H100",
            job_name=None, timeout=60, no_wait=False)
        export = checkpoint_export_launch_config(config, request, spec)
        export_path = root / "export.yaml"
        export_path.write_text(OmegaConf.to_yaml(export))
        result = execute_launch(export_path, backend=self)
        assert result.state == "succeeded", result

    def validate(self, config_path):
        raise AssertionError("the test exercises wait-mode execution")

config = OmegaConf.load(path)
current = config.run.attempt_id
config.run.attempt_id = "earlier-failed-attempt"
config.run.export_hf = False
path.write_text(OmegaConf.to_yaml(config))
assert execute_launch(path, backend=LocalBackend()).state == "failed"
config.run.attempt_id = current
config.run.export_hf = True
path.write_text(OmegaConf.to_yaml(config))
result = execute_launch(path, backend=LocalBackend())
assert result.state == "succeeded", result
original.log({"loss": 12}, step=12)
original.close(exit_code=0)
receipt = json.loads(Path(config.artifacts.terminal_manifest_uri).read_text())
(root / "resolved-launch.yaml").write_text(OmegaConf.to_yaml({"config": receipt["config"]}))
"""
    result = subprocess.run(
        skyrl_cpu_command("python", "-c", script, str(launch_path)),
        env=environment,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.setattr("marin.rl.skyrl.connect_controller", lambda **kwargs: pytest.fail("unexpected Iris lookup"))
    receipt = read_stored_skyrl_receipt(str(tmp_path))
    assert receipt is not None and receipt.native.result.model is not None
    assert receipt.native.result.model.global_step == 8
    assert read_skyrl_launch_receipt(str(tmp_path)) == receipt
    current_id, attempts = read_stored_skyrl_attempts(str(tmp_path))
    assert current_id == config.attempt_id
    assert set(attempts) == {config.attempt_id, "earlier-failed-attempt", "interrupted-attempt"}
    assert attempts["earlier-failed-attempt"].native.result.failure == "Iris job reached failed"
    interrupted = attempts["interrupted-attempt"]
    assert interrupted.native.result.state == LaunchState.SUBMITTED
    assert interrupted.native.result.iris_job_id == "/local/interrupted"
    assert interrupted.native.config["run"]["coordinator_iris_job_id"] == "/local/interrupted-coordinator"
    assert interrupted.native.config["skyrl"]["runtime"]["iris_job_id"] == "/local/interrupted"
    assert interrupted.metrics_started_at is None and interrupted.started_at is not None
    with DataStore.open(run_records_root(str(tmp_path / "exports"))) as store:
        update_attempt(store, AttemptRecord(run_id=receipt.run_id, attempt_id=receipt.attempt_id, result=None))
    assert read_attempt(str(tmp_path / "exports"), receipt.attempt_id).result is None
    fallback = read_stored_skyrl_receipt(str(tmp_path))
    assert fallback.native.result.state == LaunchState.SUCCEEDED
    assert fallback.started_at == receipt.started_at and fallback.vllm_version == receipt.vllm_version
    assert fallback.native.config["skyrl"]["runtime"]["iris_job_id"] == "/local/training"
    with DataStore.open(run_records_root(str(tmp_path / "exports"))) as store:
        update_attempt(store, AttemptRecord(run_id=receipt.run_id, attempt_id=receipt.attempt_id, result=receipt.result))
    tracking = attempt_tracking(receipt)
    assert tracking["attempt_id"] == config.attempt_id
    assert tracking["wandb_url"] == f"https://wandb.ai/tests-team/tests-project/runs/{config.attempt_id}"
    assert "vllm_version" in tracking["engine"]
    metrics = read_skyrl_metrics(str(tmp_path))
    assert list(metrics) == list(range(7))
    assert [row["loss"] for row in metrics.values()] == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.25]
    assert metrics[6] == {
        "step": 6,
        "attempt_id": config.attempt_id,
        "loss": 0.25,
        "entropy": 0.75,
        "metric_attempt_ids": {"loss": config.attempt_id, "entropy": config.attempt_id},
    }
    assert metrics[5]["attempt_id"] == config.attempt_id
    assert metrics[5]["metric_attempt_ids"] == {
        "loss": "middle",
        "startup/resume": config.attempt_id,
        "eval/all/avg_score": config.attempt_id,
    }
    assert metrics[5]["loss"] == 0.5 and metrics[5]["eval/all/avg_score"] == 0.6
    assert receipt.parent is not None and receipt.parent.attempt_id == "middle"
    assert receipt.metrics_started_at is not None and receipt.started_at is not None
    startup = AttemptRecord.model_validate_json((tmp_path / "startup-observation.json").read_text())
    assert receipt.started_at == startup.started_at and receipt.vllm_version == startup.vllm_version == "fixture-vllm"
    export_root = str(tmp_path / "exports")
    rewind = read_attempt_metrics(export_root, "rewind")
    assert list(rewind) == [0, 1, 2] and rewind[2]["loss"] == 2.5
    middle_rewind = read_attempt_metrics(export_root, "rewind-middle")
    assert list(middle_rewind) == [0, 1, 2, 4] and middle_rewind[4]["loss"] == 9.3
    stage = read_attempt_metrics(export_root, "stage")
    assert list(stage) == [0, 1] and [row["loss"] for row in stage.values()] == [9, 8]
    assert "ancestor_only" not in stage[0]
    assert read_attempt(export_root, "stage").parent.step == 5
    assert read_attempt_metrics(export_root, "failed-direct")[1]["loss"] == 0.9
    assert read_attempt_metrics(export_root, "missing-parent")[3]["loss"] == 3
    assert read_attempt_metrics(export_root, "main-resume")[9]["loss"] == 0.9
    assert read_attempt(export_root, "main-resume").resumed_step == 8
    assert read_attempt(export_root, "main-resume").parent is None
    assert read_attempt_metrics(export_root, "missing-store")[3]["loss"] == 0.3
    grandchild = read_attempt_metrics(export_root, "grandchild")
    assert list(grandchild) == [3, 4] and grandchild[3]["loss"] == 0.3 and grandchild[4]["loss"] == 0.4
    damaged_root = str(tmp_path / "absent-exports")
    with DataStore.open(run_records_root(damaged_root)) as store:
        update_attempt(store, AttemptRecord(run_id="history", attempt_id="unknown"))
    manifest = Path(ReadView(run_records_root(damaged_root)).token.manifest_path)
    contents = manifest.read_bytes()
    manifest.unlink()
    try:
        with pytest.raises(FileNotFoundError):
            read_attempt_metrics(export_root, "missing-store")
    finally:
        manifest.write_bytes(contents)
    (tmp_path / "terminal.json").unlink()
    assert read_stored_skyrl_receipt(str(tmp_path)) == receipt
    assert read_stored_skyrl_attempts(str(tmp_path))[0] == config.attempt_id
    pending = AttemptRecord(
        run_id=receipt.run_id,
        attempt_id=receipt.attempt_id,
        result=receipt.native.result.model_copy(update={"state": LaunchState.SUBMITTED}),
    )
    with DataStore.open(run_records_root(export_root)) as store:
        update_attempt(store, pending)

    @contextmanager
    def controller(**_kwargs):
        yield SimpleNamespace(url="http://controller.test", credentials=None)

    @contextmanager
    def rpc(*_args):
        job = job_pb2.JobStatus(
            job_id=f"/tests/{receipt.native.config['iris']['job_name']}", state=job_pb2.JOB_STATE_FAILED
        )
        yield SimpleNamespace(list_jobs=lambda _query: controller_pb2.Controller.ListJobsResponse(jobs=[job]))

    with monkeypatch.context() as network:
        network.setattr("marin.rl.skyrl.connect_controller", controller)
        network.setattr("marin.rl.skyrl.rpc_client", rpc)
        refreshed = read_skyrl_launch_receipt(str(tmp_path))
    assert refreshed is not None and refreshed.native.result.state == "failed"
    assert refreshed.started_at == startup.started_at and refreshed.vllm_version == "fixture-vllm"
    with DataStore.open(run_records_root(export_root)) as store:
        update_attempt(store, AttemptRecord(run_id=receipt.run_id, attempt_id=receipt.attempt_id, result=receipt.result))
    (tmp_path / "resolved-launch.yaml").write_text(
        yaml.safe_dump({"config": attempts["earlier-failed-attempt"].native.config})
    )
    (tmp_path / "terminal.json").write_text(receipt.native.model_dump_json())
    assert read_stored_skyrl_receipt(str(tmp_path)) == attempts["earlier-failed-attempt"]
    (tmp_path / "resolved-launch.yaml").unlink()
    legacy = receipt.native.model_dump(mode="json")
    legacy["config"]["artifacts"]["export_root"] = str(tmp_path / "legacy-exports")
    (tmp_path / "terminal.json").write_text(json.dumps(legacy))
    terminal_only = read_stored_skyrl_receipt(str(tmp_path))
    assert terminal_only.native.model_dump(mode="json") == legacy
    assert terminal_only.started_at is None and terminal_only.vllm_version is None
    (tmp_path / "terminal.json").unlink()
    assert read_stored_skyrl_attempts(str(tmp_path)) == (None, {})
    assert read_skyrl_launch_receipt(str(tmp_path)) is None


def test_run_skyrl_succeeds_without_hf_export(monkeypatch: pytest.MonkeyPatch) -> None:
    step = skyrl_step(_spec(), _execution())
    config = step.build_config(
        StepContext.for_run(
            output_path="s3://test/run",
            prefix="s3://test",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )
    response = {
        "run_id": config.run_id,
        "attempt_id": config.attempt_id,
        "state": "succeeded",
        "iris_job_id": "01KNOEXPORT",
        "iris_job_state": "succeeded",
        "failure": None,
        "model": None,
    }

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda _command, **kwargs: _FakeLauncherProcess(
            response=json.dumps(response),
            returncode=0,
            stdout=kwargs["stdout"],
        ),
    )
    monkeypatch.setattr("marin.rl.skyrl.record_rollout_run", lambda _record: None)

    run = run_skyrl(config)

    assert run.hf_model_uri is None
    assert run.global_step is None
    assert run.iris_job_id == "01KNOEXPORT"


def test_tasktrove_data_source_resolves_exact_file_and_verifier(tmp_path: Path) -> None:
    release = ArtifactStep.adopt("tasktrove/clean", "2026.09.10", str(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps({"verify_tool_ref": "verifyit@abc123"}))
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
        relative_path="tasks.parquet",
    )
    context = StepContext.for_run(
        output_path=str(tmp_path / "output"),
        prefix=str(tmp_path),
        runtime_args={},
        deps=(release,),
    )

    resolved = source.resolve(context)

    assert resolved.uri == str(tmp_path / "tasks.parquet")
    assert resolved.relative_path == "tasks.parquet"
    assert resolved.verifier_ref == "verifyit@abc123"
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

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **_kwargs: _FakeLauncherProcess(response="", logs="entrypoint must be a registered name\n"),
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


def test_megatron_microbatch_mismatch_fails_before_launch() -> None:
    # Reject incompatible microbatch sizes before allocating the GPU workers.
    base = _spec()
    plan = dataclasses.replace(base.topology.role_plan, micro_train_batch_size_per_gpu=2)
    recipe = yaml.safe_load(_config_yaml(strategy="megatron"))
    recipe["trainer"]["micro_forward_batch_size_per_gpu"] = 1
    with pytest.raises(ValueError, match="equal micro_forward_batch_size_per_gpu"):
        dataclasses.replace(
            base,
            topology=dataclasses.replace(base.topology, role_plan=plan),
            config_yaml=yaml.safe_dump(recipe),
        )
    # Matching sizes still validate: the gate rejects the mismatch, not Megatron itself.
    recipe["trainer"]["micro_forward_batch_size_per_gpu"] = 2
    dataclasses.replace(
        base,
        topology=dataclasses.replace(base.topology, role_plan=plan),
        config_yaml=yaml.safe_dump(recipe),
    )


def test_megatron_geometry_rejects_incomplete_expert_groups() -> None:
    geometry = MegatronGeometry(1, 2, 1, 8, 1)
    with pytest.raises(ValueError, match=r"Megatron policy has 8 GPUs.*TP=1, PP=2, CP=1, EP=8.*ETP\*EP\*PP"):
        validate_megatron_geometry(geometry, role="policy", num_gpus=8)


def test_megatron_geometry_uses_separate_dense_context_and_expert_grids() -> None:
    # This 192-GPU layout has dense DP=2 and EP=8: CP ranks also join expert groups.
    geometry = MegatronGeometry(1, 24, 4, 8, 1)
    validate_megatron_geometry(geometry, role="policy", num_gpus=192)

    # Increasing ETP requires a larger expert grid even though the dense grid still fits.
    with pytest.raises(ValueError):
        validate_megatron_geometry(
            dataclasses.replace(geometry, expert_tensor_parallel_size=2), role="policy", num_gpus=192
        )


@pytest.mark.parametrize("role", ["policy", "ref"])
@pytest.mark.parametrize("explicit_null", [False, True], ids=["omitted", "null"])
def test_skyrl_spec_inherits_expert_tensor_parallelism_from_role_tp(role: str, explicit_null: bool) -> None:
    recipe = yaml.safe_load(_config_yaml())
    geometry: dict[str, int | None] = {"tensor_model_parallel_size": 2, "expert_model_parallel_size": 2}
    if explicit_null:
        geometry["expert_tensor_parallel_size"] = None
    recipe["trainer"][role] = {"megatron_config": geometry}
    base = _spec()
    dataclasses.replace(base, config_yaml=yaml.safe_dump(recipe))

    geometry["expert_model_parallel_size"] = 4
    with pytest.raises(ValueError):
        dataclasses.replace(base, config_yaml=yaml.safe_dump(recipe))

    geometry["expert_tensor_parallel_size"] = 1
    dataclasses.replace(base, config_yaml=yaml.safe_dump(recipe))


def test_skyrl_spec_rejects_unsupported_training_strategy() -> None:
    # Unsupported curriculum strategies must fail before the launcher allocates workers.
    with pytest.raises(ValueError, match=r"trainer\.strategy=megatron"):
        dataclasses.replace(_spec(), config_yaml=_config_yaml(strategy="fsdp2"))


def test_finished_policy_resolves_terminal_export_as_model_input(tmp_path):
    root = tmp_path / "finished"
    root.mkdir()
    result = SkyRLRun(
        path=str(root),
        hf_model_uri=str(root / "policy"),
        global_step=100,
        tokenizer_uri="tests/tokenizer",
        tokenizer_revision="tokenizer-commit",
        checkpoint_root=str(root / "checkpoints"),
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(root / "terminal.json"),
        iris_job_id="/test/run",
    )
    record = ArtifactRecord(result_type=result_type_name(SkyRLRun), result=result.result_payload())
    (root / RECORD_FILENAME).write_text(record.model_dump_json())
    source = ArtifactStep.adopt("tests/finished", "2026.10.26", str(root), kind=SkyRLRun)
    model = SkyRLPolicyModel(source)
    context = StepContext.for_run(str(tmp_path / "evaluation"), str(tmp_path), deps=model.deps())
    resolved = model.resolve(context)
    assert resolved.uri == str(root / "policy")
    assert (resolved.tokenizer_uri, resolved.tokenizer_revision) == ("tests/tokenizer", "tokenizer-commit")
    evaluation = ModelConfig(name="finished", location=SKYRL_POLICY_LOCATION, tokenizer="declared/tokenizer")
    evaluated = resolve_skyrl_model(context, source, evaluation)
    assert (evaluated.location, evaluated.tokenizer, evaluated.tokenizer_revision) == (
        str(root / "policy"),
        "tests/tokenizer",
        "tokenizer-commit",
    )
    fingerprint = StepContext.for_fingerprint({}, model.deps())
    policy_identity = model.resolve(fingerprint)
    evaluation_identity = resolve_skyrl_model(fingerprint, source, evaluation)
    assert (evaluation_identity.location, evaluation_identity.tokenizer, evaluation_identity.tokenizer_revision) == (
        policy_identity.uri,
        policy_identity.tokenizer_uri,
        policy_identity.tokenizer_revision,
    )
    assert evaluated.identity == evaluation_identity.identity == f"tests/finished@2026.10.26:{source.fingerprint()}"
    assert evaluation_identity.location == f"tests/finished@2026.10.26:{source.fingerprint()}/policy"
    other = ArtifactStep.adopt("tests/other-policy", "2026.10.27", str(root), kind=SkyRLRun)
    other_model = resolve_skyrl_model(StepContext.for_fingerprint({}, (other,)), other, evaluation)
    assert other_model.identity == f"tests/other-policy@2026.10.27:{other.fingerprint()}"
    assert other_model.location == f"tests/other-policy@2026.10.27:{other.fingerprint()}/policy"
