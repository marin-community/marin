# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Micro210 preparation and Snowball evaluation using the existing Mini runtime."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
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
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.eval_runtime.judge import JudgedEvaluation, run_judged
from experiments.post_training.long_loop.eval_runtime.repair_release import repaired_table
from experiments.post_training.long_loop.eval_runtime.selection import selected_rollouts
from experiments.post_training.long_loop.rl_selection import validate_full_selection
from experiments.post_training.long_loop.stage_config import validate_stage_config
from vendor.skyrl.marinskyrl.resource_locator import validate_replayable_model_reference

WHEELS_REVISION = "d5dd315d58f0f47514dd0a2208e90e2a48541328"


def prepare_micro(*, output: str, dataset: str, revision: str, wheels_revision: str, tasks_per_benchmark: int) -> None:
    path = hf_hub_download(dataset, "data/validation.parquet", repo_type="dataset", revision=revision)
    table = pq.read_table(path)
    if table.num_rows != 210:
        raise ValueError("Pinned Micro210 must contain 210 unique tasks")
    expanded_path = hf_hub_download(dataset, "data/validation-rollouts.parquet", repo_type="dataset", revision=revision)
    expanded, receipt = selected_rollouts(table, pq.read_table(expanded_path), tasks_per_benchmark)
    prepared = repaired_table(expanded, wheels_revision)
    receipt.update(dataset=dataset, revision=revision, wheels_revision=wheels_revision)
    receipt["unique_source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt["expanded_source_sha256"] = hashlib.sha256(Path(expanded_path).read_bytes()).hexdigest()
    (StoragePath(output) / "selection.json").write_bytes(json.dumps(receipt, indent=2).encode())
    fs, key = filesystem_for(prefix_join(output, "rollouts.parquet"))
    with fs.open(key, "wb") as target:
        pq.write_table(prepared, target)
    environment = {
        "dataset": dataset,
        "revision": revision,
        "wheels_revision": wheels_revision,
        "data_uri": output,
        "provisioning": "performed by actual evaluation trials",
        "setup_preflight_executed": False,
        "tasks": [
            {
                "task_id": row["task_id"],
                "repetition_index": row["repetition_index"],
                "task_spec_sha256": hashlib.sha256(row["task_spec"].encode()).hexdigest(),
                "task_rollout_spec_sha256": hashlib.sha256(row["task_rollout_spec"].encode()).hexdigest(),
            }
            for row in prepared.to_pylist()
        ],
    }
    environment_fs, environment_key = filesystem_for(prefix_join(output, "environment.json"))
    environment_fs.pipe_file(environment_key, json.dumps(environment, indent=2).encode())


def validate_eval_policy(model: ResolvedModelLocator, source_uri: str) -> None:
    validate_replayable_model_reference(model.local_path, source_uri, model.identity)


@dataclass(frozen=True)
class TerminalPolicy:
    step: ArtifactStep[SkyRLRun]
    expected_step: int | None

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


def _checkpoint_step(policy: TerminalPolicy | ArtifactHfModel, ctx: StepContext) -> int | None:
    """The RL step the traces belong to; an adopted export has none."""
    if isinstance(policy, ArtifactHfModel):
        return None
    return policy.expected_step if ctx.is_fingerprint else ctx.resolved(policy.step).global_step


def configured_evaluation_step(
    version: str,
    policy: TerminalPolicy | ArtifactHfModel,
    data: ArtifactStep,
    *,
    name: str,
    config: dict,
    role_plan: SkyRLRolePlan,
    execution: IrisSkyRLExecution,
    judge: ModelConfig,
    judge_gpu_count: int,
    seed: int,
    gpu_variant: str,
    capability_origin: str,
    data_relative_path: str,
    ray_timeouts: dict[str, int],
) -> ArtifactStep:
    """Bind an explicit evaluation recipe to pinned data and the committed model.

    ``policy`` is the RL terminal export, or an adopted HF export evaluated as-is.
    """
    validate_stage_config(config, "evaluation")
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
            model=policy,
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
        launch["ray"].update(ray_timeouts)
        model = launch["inputs"]["model"]
        launch["skyrl"]["generator"]["trajectory_retention"].update(
            model_path=model["uri"],
            model_source_identity=model["identity"],
            model_checkpoint_step=_checkpoint_step(policy, ctx),
        )
        run = replace(run, launch_config_yaml=yaml.safe_dump(launch))
        if not ctx.is_fingerprint:
            validate_replayable_model_reference(model["local_path"], model["uri"], model["identity"])
        return JudgedEvaluation(run, judge, capability_origin, judge_gpu_count)

    return replace(step, run=run_judged, build_config=build_config)
