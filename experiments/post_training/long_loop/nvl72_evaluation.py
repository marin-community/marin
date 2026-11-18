# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate pinned HF policies through the shared NVL72 long-loop entrypoint."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import draccus
import yaml
from marin.evaluation.model_config import ModelConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.eval_runtime.judge import JudgedEvaluation, run_judged
from experiments.post_training.long_loop.nvl72 import EXECUTION, RECIPE_PATH, SEED

RECIPE = Path(__file__).with_name("specs") / "evaluation.yaml"


@dataclass(frozen=True)
class NVL72EvaluationSpec:
    format: str
    marin_revision: str
    output_prefix: str
    name: str
    model_uri: str
    model_step: int
    model_manifest_sha256: str
    model_config_sha256: str
    tokenizer_uri: str
    tokenizer_revision: str
    data_uri: str
    data_sha256: str
    input_version: str
    job_timeout_hours: int
    context_tokens: int
    max_new_tokens: int


def evaluation_from_spec(spec: NVL72EvaluationSpec, config_output: Path) -> ArtifactStep[SkyRLRun]:
    """Bind an adopted policy and micro dataset to a generation-only launch."""
    if spec.format != "nvl72_eval":
        raise ValueError("Expected nvl72_eval format")
    for name, expected in (
        ("export-manifest.json", spec.model_manifest_sha256),
        ("config.json", spec.model_config_sha256),
    ):
        observed = hashlib.sha256((StoragePath(spec.model_uri) / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Pinned model {name} differs from its SHA-256")
    producer = json.loads((StoragePath(spec.model_uri) / "export-manifest.json").read_bytes())
    if not producer["request"]["config"]["checkpoint"].endswith(f"/step-{spec.model_step}"):
        raise ValueError("Evaluation checkpoint step differs from the pinned export producer")
    plan = SkyRLRolePlan(
        colocate_all=True,
        policy_num_nodes=4,
        policy_num_gpus_per_node=4,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=16,
        inference_engine_expert_parallel_size=16,
        train_batch_size=1,
        policy_mini_batch_size=1,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
    )
    model_config = json.loads((StoragePath(spec.model_uri) / "config.json").read_bytes())
    if (
        model_config["num_experts"] % plan.inference_engine_expert_parallel_size
        or spec.context_tokens > model_config["max_position_embeddings"]
    ):
        raise ValueError("The model does not support the inference expert parallelism or selected context")
    data = StoragePath(spec.data_uri) / "rollouts.parquet"
    with data.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != spec.data_sha256:
            raise ValueError("Pinned micro evaluation data differs from its SHA-256")
    recipe = yaml.safe_load(RECIPE.read_text())
    config = recipe["config"]
    _materialize_role_plan_config(config, plan)
    config["context_budget"].update(
        request_window_tokens=spec.context_tokens, max_new_tokens_per_turn=spec.max_new_tokens
    )
    serving_recipe = yaml.safe_load(RECIPE_PATH.read_text())
    serving = serving_recipe["generator"]["engine_init_kwargs"]
    for setting in ("compilation_config", "model_loader_extra_config"):
        config["generator"]["engine_init_kwargs"][setting] = serving[setting]
    # Match the graph-enabled Mochi serving profile qualified in #11353 and #11430.
    config["generator"].update(
        vllm_attention_backend="FLASHINFER",
        max_num_seqs=128,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        vllm_v1_disable_multiproc=False,
    )
    config["generator"]["engine_init_kwargs"]["moe_backend"] = serving["moe_backend"]
    config["extra_env"]["PYTORCH_CUDA_ALLOC_CONF"] = serving_recipe["extra_env"]["PYTORCH_CUDA_ALLOC_CONF"]
    config["generator"]["enforce_eager"] = False
    for role in ("policy", "ref"):
        config["trainer"][role]["megatron_config"]["expert_model_parallel_size"] = 4
    agent = config["terminal_bench"]["installed_agent"]["agent"]["kwargs"]
    agent["model_info"].update(max_input_tokens=spec.context_tokens, max_output_tokens=spec.max_new_tokens)
    agent["compaction_reserve_tokens"] = spec.max_new_tokens
    agent["max_turns"] = 128
    config["context_budget"]["max_turns"] = 128
    name = user_owned_name(spec.name)
    version = resolve_version(name, None)
    model = ArtifactHfModel(
        ArtifactStep.adopt(f"{name}/model", spec.input_version, source=spec.model_uri, kind=LevanterCheckpoint),
        tokenizer_uri=spec.tokenizer_uri,
        tokenizer_revision=spec.tokenizer_revision,
        relative_path="",
    )
    dataset = ArtifactStep.adopt(
        f"{name}/data/{spec.data_sha256}", spec.input_version, source=spec.data_uri, kind=Artifact
    )
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(config),
            model=model,
            train_data=(),
            validation_data=(ArtifactDataSource(dataset, relative_path="rollouts.parquet"),),
            topology=SkyRLTopology(
                num_nodes=plan.num_nodes,
                gpus_per_node=plan.policy_num_gpus_per_node,
                gpu_variant="GB200",
                role_plan=plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=-1, temporary_storage_ttl_days=14),
            seed=SEED,
        ),
        replace(
            EXECUTION,
            coordinator_name=f"mochi-{spec.name}-{version}",
            job_timeout_seconds=spec.job_timeout_hours * 3600,
            role_topology=None,
        ),
        export_hf=False,
    )
    judge = draccus.decode(ModelConfig, recipe["judge"])
    judge = replace(
        judge,
        resource_hint=replace(judge.resource_hint, gpu={"GB200": 4}),
        serve=replace(judge.serve, tensor_parallel_size=4, pipeline_parallel_size=1),
    )
    original = step.build_config

    def build_config(ctx: StepContext) -> JudgedEvaluation:
        run = original(ctx)
        launch = yaml.safe_load(run.launch_config_yaml)
        locator = launch["inputs"]["model"]
        launch["skyrl"]["generator"]["trajectory_retention"].update(
            model_path=locator["uri"], model_source_identity=locator["identity"], model_checkpoint_step=spec.model_step
        )
        run = replace(run, launch_config_yaml=yaml.safe_dump(launch))
        return JudgedEvaluation(run, judge, "https://iris.oa.dev", 4, "GB200", "cw-us-east-08a")

    step = replace(step, build_config=build_config, run=run_judged)
    step = replace(step, expected_fingerprint=step.fingerprint())
    config_output.write_text(materialized_config(step, spec.output_prefix).skyrl.launch_config_yaml)
    return step
