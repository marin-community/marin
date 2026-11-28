# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
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
from marin.execution.artifact import (
    RECORD_FILENAME,
    Artifact,
    ArtifactRecord,
    FingerprintMismatchError,
    result_type_name,
    write_record,
)
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity, materialized_config
from marin.execution.step_runner import StepRunner
from marin.execution.step_status import StatusFile
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
    read_skyrl_attempt,
    reject_forced_socket_transport,
    run_skyrl,
    skyrl_cpu_command,
    skyrl_step,
    skyrl_temporary_run_path,
    validate_megatron_geometry,
)
from marin.rl.skyrl import _run_launcher as run_launcher_for_test
from marin.rl.skyrl_records import (
    attempt_tracking,
    read_stored_skyrl_attempt,
    read_stored_skyrl_attempts,
    recorded_launch,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import cluster_config
from rigging.filesystem.step_status import STATUS_SUCCESS
from skyrlrecipe.locality import RoleTopologyMode

from experiments.post_training.skyrl_test_support import read_skyrl_metrics

_PROCESS_FIXTURE = str(Path(__file__).with_name("skyrl_process_fixture.py"))


def _model_step() -> ArtifactStep[LevanterCheckpoint]:
    return ArtifactStep.adopt(
        "tests/iceball-sft",
        "2026.08.01",
        "s3://test/iceball-sft",
        kind=LevanterCheckpoint,
    )


def test_launcher_credentials_are_scoped_to_its_subprocess(monkeypatch):
    key = "MICRO210_TEST_JUDGE_KEY"
    monkeypatch.setenv(key, "parent")
    completed = run_launcher_for_test(
        [sys.executable, _PROCESS_FIXTURE, "environment-value", key],
        env_vars={key: "per-evaluation"},
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "per-evaluation"
    assert os.environ[key] == "parent"


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
        ArtifactStep.adopt("tests/model", "2026.08.01", str(root), kind=Artifact),
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
    image_python = tmp_path / "image-python"
    managed_root = tmp_path / "managed-python"
    operating_system = "macos" if sys.platform == "darwin" else "linux"
    architecture = "aarch64" if platform.machine() == "arm64" else platform.machine()
    libc = "none" if sys.platform == "darwin" else "gnu"
    managed_python = managed_root / f"cpython-{platform.python_version()}-{operating_system}-{architecture}-{libc}"
    for interpreter in (image_python, managed_python):
        (interpreter / "bin").mkdir(parents=True)
        shutil.copy2(
            Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}",
            interpreter / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}",
        )
        (interpreter / "lib").symlink_to(Path(sys.base_prefix) / "lib", target_is_directory=True)
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", str(managed_root))
    monkeypatch.setenv("UV_PYTHON_PREFERENCE", "managed")
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
    (project / ".python-version").write_text(f"{python_version[0]}.{python_version[1]}\n")
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
    shutil.copyfile(Path(__file__).with_name("skyrl_uv_dispatch.sh"), dispatcher)
    dispatcher.chmod(0o755)
    monkeypatch.setenv("MARIN_TEST_UV_EXECUTABLE", real_uv)
    monkeypatch.setenv("PATH", os.pathsep.join((str(binaries), str(image_python / "bin"), os.environ["PATH"])))
    monkeypatch.setattr("marin.rl.skyrl.__file__", str(workspace / "lib/marin/src/marin/rl/skyrl.py"))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "incompatible-parent-python"))
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, "/parent/environment")
    monkeypatch.chdir(workspace)
    result = run_launcher_for_test(skyrl_cpu_command("python", _PROCESS_FIXTURE, "environment"))
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "python": python_version,
        "base_prefix": str(image_python.resolve()),
        "cwd": str(workspace),
        "environment": {
            "PYTHONPATH": None,
            "PYTHONHOME": None,
            "UV_PROJECT_ENVIRONMENT": None,
            "VIRTUAL_ENV": str(project / ".venv"),
        },
    }


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


@pytest.mark.parametrize(
    ("trainer_gpus", "rollout_gpus", "expected_nodes"),
    [
        (32, 16, 12),
        (32, 32, 16),
        (32, 64, 24),
        (48, 16, 16),
        (48, 32, 20),
        (48, 64, 28),
        (64, 16, 20),
        (64, 32, 24),
        (64, 64, 32),
        (48, 12, 15),
        (72, 16, 22),
    ],
)
def test_cross_host_ep_launch_reserves_policy_and_rollout_nodes(trainer_gpus, rollout_gpus, expected_nodes):
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        policy_num_nodes=trainer_gpus // 4,
        num_inference_engines=1,
        inference_engine_data_parallel_size=rollout_gpus,
        inference_engine_expert_parallel_size=rollout_gpus,
    )
    recipe = yaml.safe_load(_config_yaml())
    recipe["trainer"]["policy"] = {
        "megatron_config": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 4,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
        }
    }
    spec = dataclasses.replace(
        _spec(),
        config_yaml=yaml.safe_dump(recipe),
        topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=4, gpu_variant="GB200", role_plan=plan),
    )
    step = skyrl_step(spec, _execution())
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    allocation = yaml.safe_load(config.launch_config_yaml)["iris"]["allocation"]

    assert allocation["num_nodes"] == expected_nodes
    assert allocation["num_nodes"] * allocation["gpus_per_node"] == trainer_gpus + rollout_gpus


def test_nvlink_launch_requests_extra_hosts_when_each_group_rounds_up():
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        num_inference_engines=7,
        inference_engine_data_parallel_size=18,
        inference_engine_expert_parallel_size=18,
    )
    spec = dataclasses.replace(
        _spec(),
        topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=4, gpu_variant="GB200", role_plan=plan),
    )
    execution = dataclasses.replace(_execution(), role_topology=RoleTopologyMode.NVLINK_DOMAIN)
    step = skyrl_step(spec, execution)
    config = step.build_config(
        StepContext.for_run(
            output_path="s3://test/run", prefix="s3://test", runtime_args=step.runtime_args, deps=step.deps
        )
    )
    allocation = yaml.safe_load(config.launch_config_yaml)["iris"]["allocation"]
    assert allocation["num_nodes"] == 34


def test_nvlink_launch_rejects_tensor_parallel_inference_before_submission():
    plan = dataclasses.replace(
        _role_plan(), colocate_all=False, num_inference_engines=1, inference_engine_tensor_parallel_size=2
    )
    spec = dataclasses.replace(
        _spec(),
        topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=4, gpu_variant="GB200", role_plan=plan),
    )
    step = skyrl_step(spec, dataclasses.replace(_execution(), role_topology=RoleTopologyMode.NVLINK_DOMAIN))
    with pytest.raises(ValueError, match="requires vLLM TP1/PP1 EP=DP"):
        step.build_config(
            StepContext.for_run(
                output_path="s3://test/run", prefix="s3://test", runtime_args=step.runtime_args, deps=step.deps
            )
        )


@pytest.mark.parametrize(
    ("gpus_per_node", "engines", "dp", "rollout_nodes"),
    [(4, 4, 3, 4), (8, 5, 3, 3), (4, 3, 10, 8), (8, 2, 12, 3)],
)
def test_rollout_pool_respects_host_local_engines_and_cross_host_workers(gpus_per_node, engines, dp, rollout_nodes):
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        policy_num_gpus_per_node=gpus_per_node,
        num_inference_engines=engines,
        inference_engine_data_parallel_size=dp,
        inference_engine_expert_parallel_size=dp,
    )
    assert plan.num_nodes == 1 + rollout_nodes


@pytest.mark.parametrize(("tp", "pp", "dp", "ep"), [(2, 1, 4, 4), (1, 2, 4, 4), (1, 1, 8, 4)])
def test_unsupported_cross_host_layouts_still_require_one_node(tp, pp, dp, ep):
    plan = dataclasses.replace(
        _role_plan(),
        colocate_all=False,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=tp,
        inference_engine_pipeline_parallel_size=pp,
        inference_engine_data_parallel_size=dp,
        inference_engine_expert_parallel_size=ep,
    )
    with pytest.raises(ValueError, match="each SkyRL inference engine must fit on one node"):
        SkyRLTopology(num_nodes=3, gpus_per_node=4, gpu_variant="GB200", role_plan=plan)


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


def _generation_plan() -> SkyRLGenerationPlan:
    return SkyRLGenerationPlan(
        gpus_per_node=4,
        num_inference_engines=4,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=1,
        n_samples_per_prompt=4,
    )


@pytest.mark.parametrize("failure", ["extra-node", "node-width", "entrypoint"])
def test_generation_plan_rejects_geometry_conflicts(failure):
    spec, plan = _spec(), _generation_plan()
    topology = dataclasses.replace(spec.topology, num_nodes=plan.num_nodes, role_plan=plan)
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
                role_plan=dataclasses.replace(plan, gpus_per_node=2, num_inference_engines=2),
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
    if entrypoint is not None:
        plan = dataclasses.replace(plan, entrypoint=entrypoint)

    return dataclasses.replace(
        _spec(),
        topology=dataclasses.replace(_spec().topology, role_plan=plan),
        config_yaml=yaml.safe_dump(recipe),
    )


def test_expert_block_colocation_fails_before_job_submission() -> None:
    with pytest.raises(ValueError, match="colocated expert_block requires entrypoint=standard"):
        _colocated_expert_block_spec("taskcompendium")


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
    task_set = dataclasses.replace(_data_step(), name="tests/heldout-task_set", version="2026.11.05")
    source = ArtifactDataSource(task_set, relative_path="validation.parquet")
    profiled_spec = dataclasses.replace(
        spec, config_yaml=yaml.safe_dump(recipe), evaluation_data=(("quality", (source,)),)
    )
    profiled = skyrl_step(profiled_spec, _execution())
    assert [(dep.name, dep.version) for dep in profiled.deps] == [
        ("tests/iceball-sft", "2026.08.01"),
        ("tests/iceball-gsm8k", "2026.08.01"),
        ("tests/heldout-task_set", "2026.11.05"),
    ]
    changed_task_set = dataclasses.replace(source, step=dataclasses.replace(task_set, version="2026.11.05.1"))
    assert (
        profiled.fingerprint()
        != skyrl_step(
            dataclasses.replace(profiled_spec, evaluation_data=(("quality", (changed_task_set,)),)), _execution()
        ).fingerprint()
    )

    conflicting_source = dataclasses.replace(
        spec.train_data[0],
        step=dataclasses.replace(spec.train_data[0].step, version="2026.11.17"),
        relative_path="validation.parquet",
    )
    with pytest.raises(ValueError, match="distinct data artifacts share a staging directory"):
        skyrl_step(dataclasses.replace(spec, validation_data=(conflicting_source,)), _execution()).fingerprint()


@pytest.mark.parametrize(
    ("cluster", "explicit_hca", "expected_hca"),
    [
        ("cw-us-east-02a", None, "ibp"),
        ("cw-us-east-02a", "custom_hca", "custom_hca"),
        ("cw-us-east-08a", None, None),
    ],
)
def test_skyrl_child_request_selects_east02a_ib_rails(
    cluster: str, explicit_hca: str | None, expected_hca: str | None
) -> None:
    spec = _spec()
    if explicit_hca is not None:
        recipe = yaml.safe_load(spec.config_yaml)
        recipe["extra_env"] = {"NCCL_IB_HCA": explicit_hca}
        spec = dataclasses.replace(spec, config_yaml=yaml.safe_dump(recipe))

    step = skyrl_step(spec, _execution(cluster))
    launch = yaml.safe_load(materialized_config(step, prefix="s3://test").launch_config_yaml)

    assert launch["runtime"]["task_env"].get("NCCL_IB_HCA") == expected_hca


def test_starting_checkpoint_binds_identity_and_rejects_cached_output_reuse(tmp_path: Path, monkeypatch) -> None:
    spec = _spec()
    checkpoints = [
        ArtifactStep.adopt("tests/native", f"2026.11.13.{step}", f"s3://checkpoints/global_step_{step}")
        for step in (2, 5)
    ]
    runs = [
        skyrl_step(dataclasses.replace(spec, starting_checkpoint=checkpoint), _execution()) for checkpoint in checkpoints
    ]
    for run, checkpoint in zip(runs, checkpoints, strict=True):
        config = materialized_config(run, prefix="s3://marin-us-east-08a")
        launch = yaml.safe_load(config.launch_config_yaml)
        assert checkpoint in run.deps
        assert launch["inputs"]["starting_checkpoint"] == {
            "uri": checkpoint.adopt_source,
            "identity": artifact_identity(checkpoint),
        }
        assert not checkpoint.adopt_source.startswith(config.output.checkpoint_root)
    assert runs[0].fingerprint() != runs[1].fingerprint()
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    output = runs[0].path(str(tmp_path))
    write_record(ArtifactRecord(output_path=output, fingerprint=runs[0].fingerprint()))
    StatusFile(output, worker_id="test").write_status(STATUS_SUCCESS)
    StepRunner().run([runs[0].lower()])
    with pytest.raises(FingerprintMismatchError):
        StepRunner().run([runs[1].lower()])


@pytest.mark.parametrize(
    ("checkpoint_ttl_days", "temporary_ttl_days", "resolved_checkpoint_ttl_days"),
    [(None, 14, 14), (None, 3, 3), (1, 14, 1)],
)
def test_skyrl_step_routes_disposable_state_to_ttl_storage(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_ttl_days: int | None,
    temporary_ttl_days: int,
    resolved_checkpoint_ttl_days: int,
) -> None:
    monkeypatch.setattr(
        "marin.rl.skyrl.skyrl_temporary_run_path",
        lambda _output_path, *, ttl_days: f"s3://temp/ttl={ttl_days}d/skyrl/users/alice/run",
    )
    recipe = yaml.safe_load(_spec().config_yaml)
    recipe["trainer"]["callbacks"] = [{"type": "best_checkpoint", "initial_model_uri": "wrong-model"}]
    spec = dataclasses.replace(
        _spec(),
        name="users/alice/tests/iceball-rl",
        version="dev",
        config_yaml=yaml.safe_dump(recipe),
        retention=SkyRLRetentionPolicy(
            checkpoint_ttl_days=checkpoint_ttl_days, temporary_storage_ttl_days=temporary_ttl_days
        ),
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
    checkpoint_root = f"s3://temp/ttl={resolved_checkpoint_ttl_days}d/skyrl/users/alice/run/checkpoints"
    assert config.output.checkpoint_root == checkpoint_root
    assert config.output.export_root == f"{output_path}/exports"
    launch = yaml.safe_load(config.launch_config_yaml)
    assert launch["run"]["coordinator_iris_job_id"] == "/tests/coordinator"
    assert launch["skyrl"]["trainer"]["callbacks"][0]["initial_model_uri"] == config.model.uri
    assert launch["artifacts"] == {
        "checkpoint_root": checkpoint_root,
        "export_root": f"{output_path}/exports",
        "attempts_root": f"s3://temp/ttl={temporary_ttl_days}d/skyrl/users/alice/run/attempts",
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

    def fake_launcher(command, **_kwargs) -> subprocess.CompletedProcess[str]:
        config_path = command[command.index("--config") + 1]
        launch_configs.append(yaml.safe_load(Path(config_path).read_text()))
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr("marin.rl.skyrl._run_launcher", fake_launcher)
    catalog_rows = []
    monkeypatch.setattr("marin.rl.skyrl.record_rollout_run", catalog_rows.append)

    model = run_skyrl(config, env_vars={})

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
    launch["run"].update(
        coordinator_iris_job_id="/local/coordinator",
        index={
            "run_id": "indexed-native-writers",
            "model": "snowball",
            "preset": "train-short",
            "preset_version": "fixture",
            "user": "test",
            "branch": "main",
            "source_revision": "test-commit",
            "dirty": False,
            "wandb_project": "tests-project",
        },
    )
    launch_path = tmp_path / "launch.yaml"
    launch_path.write_text(yaml.safe_dump(launch))
    (tmp_path / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": launch}))
    assert read_skyrl_metrics(str(tmp_path)) == {}
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "IRIS_TASK_ID"):
        environment.pop(key, None)
    result = subprocess.run(
        skyrl_cpu_command("python", "-I", "-m", "cloud.iris.tests.native_writer_roundtrip", str(launch_path)),
        env=environment,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.setattr("marin.rl.skyrl.connect_controller", lambda **kwargs: pytest.fail("unexpected Iris lookup"))
    receipt = read_stored_skyrl_attempt(str(tmp_path))
    assert receipt is not None and receipt.native.result.model is not None
    assert receipt.native.result.model.global_step == 8
    assert read_skyrl_attempt(str(tmp_path)) == receipt
    current_id, attempts = read_stored_skyrl_attempts(str(tmp_path))
    assert current_id == config.attempt_id
    assert set(attempts) == {config.attempt_id, "earlier-failed-attempt", "interrupted-attempt"}
    assert attempts["earlier-failed-attempt"].native.result.state == LaunchState.FAILED
    interrupted = attempts["interrupted-attempt"]
    assert interrupted.native.result.state == LaunchState.SUBMITTED
    assert interrupted.native.result.iris_job_id == "/local/interrupted"
    assert interrupted.native.config["run"]["coordinator_iris_job_id"] == "/local/interrupted-coordinator"
    assert interrupted.native.config["skyrl"]["runtime"]["iris_job_id"] == "/local/interrupted"
    assert interrupted.metrics_started_at is None and interrupted.started_at is not None
    with DataStore.open(run_records_root(str(tmp_path / "exports"))) as store:
        update_attempt(store, AttemptRecord(run_id=receipt.run_id, attempt_id=receipt.attempt_id, result=None))
    assert read_attempt(str(tmp_path / "exports"), receipt.attempt_id).result is None
    fallback = read_stored_skyrl_attempt(str(tmp_path))
    assert fallback.native.result.state == LaunchState.SUCCEEDED
    assert fallback.started_at == receipt.started_at and fallback.vllm_version == receipt.vllm_version
    assert fallback.native.config["skyrl"]["runtime"]["iris_job_id"] == "/local/training"
    with DataStore.open(run_records_root(str(tmp_path / "exports"))) as store:
        update_attempt(store, AttemptRecord(run_id=receipt.run_id, attempt_id=receipt.attempt_id, result=receipt.result))
    stored = receipt.model_copy(deep=True)
    stored.config["future_runtime"] = {"version": 2}
    stored.config["skyrl"]["generator"]["future_engine_option"] = True
    tracking = attempt_tracking(stored, recorded_launch(stored))
    for field, value in (("model_dtype", 7), ("backend", None)):
        malformed = stored.model_copy(deep=True)
        if value is None:
            del malformed.config["skyrl"]["generator"][field]
        else:
            malformed.config["skyrl"]["generator"][field] = value
        with pytest.raises(ValueError, match=f"{receipt.attempt_id}.*{field}") as error:
            recorded_launch(malformed)
        assert "input_value" not in "".join(traceback.format_exception(error.value))
    assert tracking["attempt_id"] == config.attempt_id
    assert tracking["wandb_url"] == f"https://wandb.ai/tests-team/tests-project/runs/{config.attempt_id}"
    assert tracking["engine"]["vllm_version"] == "fixture-vllm"
    assert tracking["engine"]["backend"] == "vllm"
    assert tracking["engine"]["batch_invariant"] is False
    metrics = read_skyrl_metrics(str(tmp_path))
    assert list(metrics) == [3, 5, 6]
    assert [row["loss"] for row in metrics.values()] == [0.3, 0.5, 0.25]
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
    assert read_attempt_metrics(export_root, "failed-direct")[1]["loss"] == 0.9
    (tmp_path / "terminal.json").unlink()
    assert read_stored_skyrl_attempt(str(tmp_path)) == receipt
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
        refreshed = read_skyrl_attempt(str(tmp_path))
    assert refreshed is not None and refreshed.native.result.state == "failed"
    assert refreshed.started_at == startup.started_at and refreshed.vllm_version == "fixture-vllm"
    with DataStore.open(run_records_root(export_root)) as store:
        update_attempt(store, AttemptRecord(run_id=receipt.run_id, attempt_id=receipt.attempt_id, result=receipt.result))
    (tmp_path / "resolved-launch.yaml").write_text(
        yaml.safe_dump({"config": attempts["earlier-failed-attempt"].native.config})
    )
    (tmp_path / "terminal.json").write_text(receipt.native.model_dump_json())
    assert read_stored_skyrl_attempt(str(tmp_path)) == attempts["earlier-failed-attempt"]
    earlier = attempts["earlier-failed-attempt"]
    (tmp_path / "terminal.json").write_text(earlier.native.model_dump_json())
    with monkeypatch.context() as missing_attempt:
        missing_attempt.setattr("marin.rl.skyrl_records.read_attempt", lambda *_args: None)
        recovered = read_stored_skyrl_attempt(str(tmp_path))
    assert recovered is not None and recovered.native == earlier.native
    assert recovered.index_run_id == earlier.index_run_id == "indexed-native-writers"
    (tmp_path / "resolved-launch.yaml").unlink()
    (tmp_path / "terminal.json").unlink()
    assert read_stored_skyrl_attempts(str(tmp_path)) == (
        None,
        {"earlier-failed-attempt": attempts["earlier-failed-attempt"]},
    )
    assert read_skyrl_attempt(str(tmp_path)) is None


@pytest.mark.parametrize("selection", [None, "best", "latest"], ids=["disabled", "none-eligible", "latest-required"])
def test_run_skyrl_handles_a_missing_hf_export(monkeypatch: pytest.MonkeyPatch, selection: str | None) -> None:
    spec = _spec()
    if selection is not None:
        recipe = yaml.safe_load(spec.config_yaml)
        recipe["trainer"]["hf_export_checkpoint"] = selection
        spec = dataclasses.replace(spec, config_yaml=yaml.safe_dump(recipe))
    step = skyrl_step(spec, _execution(), export_hf=selection is not None)
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
        "marin.rl.skyrl._run_launcher",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, json.dumps(response), ""),
    )
    recorded = []
    monkeypatch.setattr("marin.rl.skyrl.record_rollout_run", recorded.append)

    if selection == "latest":
        with pytest.raises(ValueError, match="requires a model"):
            run_skyrl(config, env_vars={})
        assert recorded[0].status == "failed"
        return

    run = run_skyrl(config, env_vars={})

    assert recorded[0].status == "succeeded"
    assert run.hf_model_uri is None
    assert run.global_step is None
    assert run.iris_job_id == "01KNOEXPORT"
    retained = SkyRLRun.model_validate_json(run.model_dump_json())
    assert (retained.run_id, retained.attempt_id) == (config.run_id, config.attempt_id)


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
        "marin.rl.skyrl._run_launcher",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, "", "entrypoint must be a registered name\n"),
    )

    with pytest.raises(RuntimeError, match="entrypoint must be a registered name"):
        run_skyrl(config, env_vars={})

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
    monkeypatch.setattr(sys, "stderr", MarkFirstWrite())

    completed = run_launcher_for_test([sys.executable, _PROCESS_FIXTURE, "stream", str(observed)])

    assert completed.stdout == "saw"
    assert "launcher line" in completed.stderr


def test_launcher_survives_undecodable_bytes_on_stderr() -> None:
    """Native CUDA and NCCL layers emit non-UTF-8 bytes; strict decoding would wedge the run."""
    completed = run_launcher_for_test([sys.executable, _PROCESS_FIXTURE, "invalid-stderr"])

    assert completed.returncode == 4
    assert "bad" in completed.stderr


def test_launcher_retains_stderr_tail_for_high_descriptors(tmp_path, capsys) -> None:
    soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if 0 <= soft_limit <= 1032:
        pytest.skip("File descriptor limit too low to exercise a high-numbered pipe")
    diagnostics = "old" + "x" * 200000 + "last"
    source = tmp_path / "diagnostics"
    source.write_text(diagnostics)
    with ExitStack() as files:
        while files.enter_context(open(os.devnull, "rb")).fileno() < 1024:
            pass
        completed = run_launcher_for_test(["cp", str(source), "/dev/stderr"])
    assert completed.returncode == 0
    assert completed.stderr == "x" * (64 * 1024 - 4) + "last"
    assert capsys.readouterr().err == diagnostics


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


@pytest.mark.parametrize("tokenizer", ["tests/tokenizer", None])
def test_finished_policy_resolves_terminal_export_as_model_input(tmp_path, tokenizer):
    root = tmp_path / "finished"
    root.mkdir()
    result = SkyRLRun(
        path=str(root),
        hf_model_uri=str(root / "policy"),
        global_step=100,
        tokenizer_uri=tokenizer,
        tokenizer_revision="tokenizer-commit" if tokenizer else None,
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
    assert (resolved.tokenizer_uri, resolved.tokenizer_revision) == (
        tokenizer,
        "tokenizer-commit" if tokenizer else None,
    )
    identity = f"tests/finished@2026.10.26:{source.fingerprint()}"
    assert resolved.identity == identity
    fingerprint = StepContext.for_fingerprint({}, model.deps())
    policy_identity = model.resolve(fingerprint)
    assert (policy_identity.uri, policy_identity.identity) == (f"{identity}/policy", identity)


def test_reject_forced_socket_transport_blocks_multi_node_overrides() -> None:
    # Multi-node jobs must keep NCCL on InfiniBand; the audit in #11619 measured
    # 12-24x lower socket bandwidth and bootstrap hangs for partial overrides.
    with pytest.raises(ValueError, match=r"NCCL_NET=Socket"):
        reject_forced_socket_transport({"NCCL_NET": "Socket"}, num_nodes=32)
    with pytest.raises(ValueError, match=r"NCCL_IB_DISABLE=1"):
        reject_forced_socket_transport({"NCCL_IB_DISABLE": "1"}, num_nodes=5)
    with pytest.raises(ValueError, match=r"NCCL_NET=Socket, NCCL_IB_DISABLE=1"):
        reject_forced_socket_transport({"NCCL_NET": "Socket", "NCCL_IB_DISABLE": "1"}, num_nodes=2)


def test_reject_forced_socket_transport_allows_non_forcing_env() -> None:
    # AF_INET alone does not force sockets, single-node jobs have no
    # cross-node transport to protect, and an empty env passes.
    reject_forced_socket_transport({"NCCL_SOCKET_FAMILY": "AF_INET"}, num_nodes=32)
    reject_forced_socket_transport({"NCCL_NET": "Socket"}, num_nodes=1)
    reject_forced_socket_transport({}, num_nodes=32)


def test_ttl_run_path_preserves_a_run_already_under_the_temporary_root(monkeypatch):
    monkeypatch.setattr("marin.rl.skyrl.marin_temp_bucket", lambda **kwargs: "s3://regional/tmp/ttl=14d")
    temporary = skyrl_temporary_run_path("s3://regional/marin/run", ttl_days=14)
    assert skyrl_temporary_run_path(temporary, ttl_days=14) == temporary
    sibling = skyrl_temporary_run_path("s3://regional/tmp/ttl=14d-other/run", ttl_days=14)
    assert sibling == "s3://regional/tmp/ttl=14d/skyrl/tmp/ttl=14d-other/run"
