# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Micro210 preparation and Snowball evaluation using the existing Mini runtime."""

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from iris.rpc import job_pb2
from marin.evaluation.model_config import ModelConfig, load_model_config
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, artifact_identity
from marin.execution.remote import remote
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    IrisSkyRLExecution,
    ResolvedModelLocator,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from rigging.config_discovery import find_project_root
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.long_loop.archive_selection import trace_retention
from experiments.post_training.long_loop.eval_runtime.judge import JudgedEvaluation, run_judged
from experiments.post_training.long_loop.eval_runtime.profile import DATASET, REVISION
from experiments.post_training.long_loop.eval_runtime.recipe import recipe, validate_environment_preflight
from experiments.post_training.long_loop.eval_runtime.repair_release import repaired_table
from experiments.post_training.long_loop.eval_runtime.selection import selected_rollouts
from experiments.post_training.long_loop.execution_policy import DEFAULT_TASK_TIMEOUT_SECONDS
from experiments.post_training.long_loop.rl_selection import validate_full_selection
from experiments.post_training.long_loop.serving import check_context_limit, snowball_context_override
from experiments.post_training.long_loop.sft import read_json
from vendor.skyrl.marinskyrl.resource_locator import validate_replayable_model_reference

WHEELS_REVISION = "d5dd315d58f0f47514dd0a2208e90e2a48541328"


def prepare_micro(
    *, output: str, dataset: str, revision: str, wheels_revision: str, tasks_per_benchmark: int = 1
) -> None:
    path = hf_hub_download(dataset, "data/validation.parquet", repo_type="dataset", revision=revision)
    table = pq.read_table(path)
    if table.num_rows != 210:
        raise ValueError("Pinned Micro210 must contain 210 unique tasks")
    expanded_path = hf_hub_download(dataset, "data/validation-rollouts.parquet", repo_type="dataset", revision=revision)
    expanded, receipt = selected_rollouts(table, pq.read_table(expanded_path), tasks_per_benchmark)
    smoke = repaired_table(expanded, wheels_revision)
    receipt.update(dataset=dataset, revision=revision, wheels_revision=wheels_revision)
    receipt["unique_source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt["expanded_source_sha256"] = hashlib.sha256(Path(expanded_path).read_bytes()).hexdigest()
    receipt_fs, receipt_key = filesystem_for(prefix_join(output, "selection.json"))
    receipt_fs.pipe_file(receipt_key, json.dumps(receipt, indent=2).encode())
    fs, key = filesystem_for(prefix_join(output, "rollouts.parquet"))
    with fs.open(key, "wb") as target:
        pq.write_table(smoke, target)
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--with-editable",
            "./vendor/harbor",
            "--with",
            "daytona==0.200.2",
            "--prerelease",
            "allow",
            "python",
            "-m",
            "experiments.post_training.long_loop.eval_runtime.environment_preflight",
            *(["--smoke"] if tasks_per_benchmark == 1 else []),
            "--credential-env",
            "DAYTONA_EVAL_API_KEY",
            "--data-uri",
            output,
            "--output-prefix",
            output,
            "--report-name",
            "environment.json",
        ],
        check=True,
    )
    validate_environment_preflight(read_json(prefix_join(output, "environment.json")), output)


def validate_eval_policy(model: ResolvedModelLocator, source_uri: str) -> None:
    validate_replayable_model_reference(model.local_path, source_uri, model.identity)


@dataclass(frozen=True)
class TerminalPolicy:
    step: ArtifactStep[SkyRLRun]
    expected_step: int | None = 3

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ResolvedModelLocator:
        if ctx.is_fingerprint:
            locator = ResolvedModelLocator(
                uri=prefix_join(artifact_identity(self.step), "terminal-export"),
                identity=artifact_identity(self.step),
                local_path="/tmp/marinskyrl/models/long-loop-terminal",
                tokenizer_uri="<terminal-tokenizer>",
                tokenizer_revision="<terminal-tokenizer-revision>",
            )
            validate_eval_policy(locator, prefix_join(self.step.path(), "terminal-export"))
            return locator
        result = ctx.resolved(self.step)
        if result.hf_model_uri is None or result.global_step is None:
            raise ValueError("Evaluation requires a committed RL export")
        if self.expected_step is not None and result.global_step != self.expected_step:
            raise ValueError("RL export does not match the required checkpoint step")
        if self.expected_step is None:
            validate_full_selection(result)
        return ResolvedModelLocator(
            uri=result.hf_model_uri,
            identity=artifact_identity(self.step),
            local_path="/tmp/marinskyrl/models/long-loop-terminal",
            tokenizer_uri=result.tokenizer_uri,
            tokenizer_revision=result.tokenizer_revision,
        )


def evaluation_step(
    version: str,
    trained: ArtifactStep[SkyRLRun],
    data: ArtifactStep,
    *,
    replicas: int = 1,
    expected_step: int | None = 3,
) -> ArtifactStep:
    config, plan = recipe("runtime://minimax-judge", "DAYTONA_EVAL_API_KEY")
    plan = replace(
        plan,
        policy_num_nodes=replicas,
        policy_num_gpus_per_node=8,
        num_inference_engines=replicas,
        inference_engine_tensor_parallel_size=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    )
    _materialize_role_plan_config(config, plan)
    config["trajectory_runner"]["max_concurrent_tasks"] = 8 * replicas
    for role in ("policy", "ref"):
        config["trainer"][role]["megatron_config"].update(tensor_model_parallel_size=1, expert_model_parallel_size=8)
    config["generator"]["engine_init_kwargs"].update(
        tool_call_parser="hermes",
        hf_overrides=snowball_context_override(config["context_budget"]["request_window_tokens"]),
    )
    config["generator"]["sampling_params"].update(skip_special_tokens=False, stop_token_ids=[128001, 128009])
    config["generator"]["eval_sampling_params"].update(skip_special_tokens=False, stop_token_ids=[128001, 128009])
    config["generator"]["trajectory_retention"] = trace_retention(
        prefix_join(prefix_join("s3://marin-us-east-02a/marin/long-loop/traces", version), "eval"),
        version,
        fraction=1.0,
        phase="eval",
    )
    root = find_project_root()
    assert root is not None, "Run from the Marin worktree"
    judge = load_model_config(
        root / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/"
        "olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-131k.yaml"
    )
    # Context bound from the exact judge revision's text_config, not its serving window.
    if (judge.location, judge.revision) != ("MiniMaxAI/MiniMax-M3-MXFP8", "c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"):
        raise ValueError("Revalidate the judge context bound when changing its pinned revision")
    check_context_limit(judge.location, 1048576, judge.serve.max_model_len)
    cluster = yaml.safe_load((root / IRIS_HUB_CLUSTER_CONFIG).read_text())

    return configured_evaluation_step(
        version,
        trained,
        data,
        name="long-loop/micro-smoke" if expected_step == 3 else "long-loop/micro-full",
        config=config,
        role_plan=plan,
        execution=IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=32,
            memory="512GB",
            disk="256GB",
            priority="production",
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=DEFAULT_TASK_TIMEOUT_SECONDS // 3600,
            coordinator_name=f"long-loop-plumbing-{version}",
            job_timeout_seconds=DEFAULT_TASK_TIMEOUT_SECONDS,
            required_env_vars=("DAYTONA_EVAL_API_KEY", "DAYTONA_TARGET", "TOGETHER_API_KEY", "MARIN_PREFIX"),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        judge=judge,
        judge_gpu_count=8,
        seed=20261030,
        expected_step=expected_step,
        gpu_variant="H100",
        capability_origin=cluster["dashboard_url"],
        data_relative_path="rollouts.parquet",
    )


def configured_evaluation_step(
    version: str,
    trained: ArtifactStep[SkyRLRun],
    data: ArtifactStep,
    *,
    name: str,
    config: dict,
    role_plan: SkyRLRolePlan,
    execution: IrisSkyRLExecution,
    judge: ModelConfig,
    judge_gpu_count: int,
    seed: int,
    expected_step: int | None,
    gpu_variant: str,
    capability_origin: str,
    data_relative_path: str,
) -> ArtifactStep:
    """Bind an explicit evaluation recipe to pinned data and the committed model."""
    config = deepcopy(config)
    materialized = deepcopy(config)
    _materialize_role_plan_config(materialized, role_plan)
    if materialized != config:
        raise ValueError("Evaluation recipe and explicit role plan disagree")
    retention = config["generator"]["trajectory_retention"]
    if not (
        retention["enabled"]
        and retention["required"]
        and retention["eval_sample_fraction"] == 1.0
        and "eval" in retention["phases"]
    ):
        raise ValueError("Evaluation recipe must retain all evaluation traces")
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(config),
            model=TerminalPolicy(trained, expected_step),
            train_data=(),
            validation_data=(ArtifactDataSource(data, relative_path=data_relative_path),),
            topology=SkyRLTopology(
                num_nodes=role_plan.num_nodes,
                gpus_per_node=role_plan.policy_num_gpus_per_node,
                gpu_variant=gpu_variant,
                role_plan=role_plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=seed,
        ),
        execution,
        export_hf=False,
    )

    def build_config(ctx: StepContext) -> JudgedEvaluation:
        run = step.build_config(ctx)
        launch = yaml.safe_load(run.launch_config_yaml)
        model = launch["inputs"]["model"]
        launch["skyrl"]["generator"]["trajectory_retention"].update(
            model_path=model["uri"],
            model_source_identity=model["identity"],
            model_checkpoint_step=expected_step if ctx.is_fingerprint else ctx.resolved(trained).global_step,
        )
        run = replace(run, launch_config_yaml=yaml.safe_dump(launch))
        if not ctx.is_fingerprint:
            validate_replayable_model_reference(model["local_path"], model["uri"], model["identity"])
        if not ctx.is_fingerprint:
            manifest = read_json(prefix_join(ctx.artifact_path(data), "environment.json"))
            validate_environment_preflight(manifest, ctx.artifact_path(data))
        return JudgedEvaluation(run, judge, capability_origin, judge_gpu_count)

    return replace(step, run=run_judged, build_config=build_config)


def micro_inputs(version: str, *, tasks_per_benchmark: int = 1) -> ArtifactStep:
    return apply(
        "long-loop/micro-inputs" if tasks_per_benchmark == 1 else "long-loop/micro-full-inputs",
        remote(
            prepare_micro,
            name=f"long-loop-micro-inputs-{version}",
            env_vars={
                name: os.environ[name]
                for name in ("DAYTONA_EVAL_API_KEY", "DAYTONA_TARGET", "TOGETHER_API_KEY")
                if name in os.environ
            },
            resources=ResourceConfig.with_gpu(
                "H100",
                count=8,
                cpu=32,
                ram="128GB",
                disk="512GB",
                target_cluster=None,
                container_profile="CONTAINER_PROFILE_PRIVILEGED",
            ),
        ),
        version=version,
        output=OUT,
        dataset=DATASET,
        revision=REVISION,
        wheels_revision=WHEELS_REVISION,
        tasks_per_benchmark=tasks_per_benchmark,
    )
