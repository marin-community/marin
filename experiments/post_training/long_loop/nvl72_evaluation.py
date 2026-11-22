# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate pinned HF policies through the shared NVL72 long-loop entrypoint."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import draccus
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.experiment.namespacing import user_owned_name

from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    SkyRLGenerationPlan,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.evaluation.judge_config import JudgeConfig
from experiments.evaluation.micro210 import PREPARED_INPUTS, judge_runtime_config
from experiments.post_training.long_loop.eval_runtime.judge import JudgedEvaluation, run_judged
from experiments.post_training.long_loop.nvl72 import EXECUTION, SEED

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
    model_config = json.loads((StoragePath(spec.model_uri) / "config.json").read_bytes())
    if model_config["num_experts"] % 12 or spec.context_tokens > model_config["max_position_embeddings"]:
        raise ValueError("The model does not support EP12 or the selected context")
    if spec.data_uri != PREPARED_INPUTS:
        raise ValueError("NVL72 evaluation requires the current Micro210 judge-bound prepared inputs")
    data = StoragePath(spec.data_uri) / "rollouts.parquet"
    if file_sha256(data) != spec.data_sha256:
        raise ValueError("Pinned micro evaluation data differs from its SHA-256")
    recipe = yaml.safe_load(RECIPE.read_text())
    config = judge_runtime_config(recipe["config"], draccus.decode(JudgeConfig, recipe["judge"]))
    # The generation-only launch derives its entrypoint from the plan, never the recipe.
    config.pop("entrypoint", None)
    plan = SkyRLGenerationPlan(
        gpus_per_node=4,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=12,
        inference_engine_expert_parallel_size=12,
        n_samples_per_prompt=1,
    )
    _materialize_role_plan_config(config, plan)
    config["context_budget"].update(
        request_window_tokens=spec.context_tokens, max_new_tokens_per_turn=spec.max_new_tokens
    )
    config["generator"]["engine_init_kwargs"]["compilation_config"] = {"mode": 3, "cudagraph_mode": "PIECEWISE"}
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
                gpus_per_node=plan.gpus_per_node,
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
    judge = draccus.decode(JudgeConfig, recipe["judge"])
    original = step.build_config

    def build_config(ctx: StepContext) -> JudgedEvaluation:
        run = original(ctx)
        launch = yaml.safe_load(run.launch_config_yaml)
        locator = launch["inputs"]["model"]
        launch["skyrl"]["generator"]["trajectory_retention"].update(
            model_path=locator["uri"], model_source_identity=locator["identity"], model_checkpoint_step=spec.model_step
        )
        run = replace(run, launch_config_yaml=yaml.safe_dump(launch))
        return JudgedEvaluation(run, judge)

    step = replace(step, build_config=build_config, run=run_judged)
    step = replace(step, expected_fingerprint=step.fingerprint())
    config_output.write_text(materialized_config(step, spec.output_prefix).skyrl.launch_config_yaml)
    return step
