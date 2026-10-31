# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate released Mini840 rollouts through native Harbor environments without training."""

import json
from dataclasses import replace
from pathlib import Path

import click
import yaml
from iris.rpc import job_pb2
from marin.evaluation.model_config import load_model_config
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy_runtime.judge import JudgedEvaluation, run_judged, run_with_existing_judge
from experiments.evaluation.mini_policy_runtime.profile import (
    MAX_OUTPUT_TOKENS,
    MODEL,
    MODEL_REVISION,
    SEED,
    TOTAL_CONTEXT_LENGTH,
    counterpart_environment,
    pi_agent_config,
)


def recipe(
    environment_overrides: dict, agent_mcp_servers: dict, verifier_env: dict, judge_base_url: str, credential_env: str
) -> tuple[dict, SkyRLRolePlan]:
    plan = SkyRLRolePlan(
        colocate_all=True,
        policy_num_nodes=1,
        policy_num_gpus_per_node=2,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=2,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=1,
        train_batch_size=1,
        policy_mini_batch_size=1,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
    )
    geometry = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
    }
    config = {
        "entrypoint": "generate",
        "config_groups": {"terminal_bench_config": "terminal_bench"},
        "context_budget": {
            "request_window_tokens": TOTAL_CONTEXT_LENGTH,
            "max_new_tokens_per_turn": MAX_OUTPUT_TOKENS,
            "max_turns": 1024,
        },
        "environment": {
            "task_sessions": {
                "local": {},
                "remote_judge": {"base_url": judge_base_url, "api_key_env": "JUDGE_API_KEY"},
            }
        },
        "trainer": {
            "strategy": "megatron",
            "evaluation_runner": "task",
            "eval_interval": 1,
            "eval_batch_size": 32,
            "resume_mode": "none",
            "seed": SEED,
            "algorithm": {"use_kl_loss": False},
            "policy": {"megatron_config": geometry},
            "ref": {"megatron_config": geometry},
            "logger": "console",
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "max_num_seqs": 8,
            "engine_init_kwargs": {"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_coder"},
            "run_engines_locally": True,
            "enable_http_endpoint": False,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": False,
            "chat_template": {"source": "name", "name_or_path": None},
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling_params": {"temperature": 0.6, "top_p": 0.95, "logprobs": 0},
            "eval_sampling_params": {"temperature": 0.6, "top_p": 0.95},
            "eval_n_samples_per_prompt": 1,
        },
        "terminal_bench": {
            "harbor": {
                "environment_type": "daytona",
                "daytona_api_key_env": credential_env,
                "eval_daytona_api_key_env": credential_env,
                "verifier_override_timeout_sec": 300,
                "override_timeout_sec": 1800,
                "trial_attempt_timeout_sec": 2460,
                "n_concurrent_trials": 8,
                "max_retries": 0,
            },
            "installed_agent": {
                "agent": pi_agent_config(),
                "setup_timeout": 360,
                "run_timeout": 1800,
                "cleanup_timeout": 60,
                "service_startup_timeout": 60,
                "service_shutdown_timeout": 60,
                "environment_snapshots": {},
                "environment_overrides": environment_overrides,
                "agent_mcp_servers": agent_mcp_servers,
                "verifier_env": {
                    source: {
                        "values": {name: value for name, value in environment.items() if not value.startswith("${")},
                        "variables": {
                            name: value[2:-1]
                            for name, value in environment.items()
                            if value.startswith("${") and value.endswith("}")
                        },
                    }
                    for source, environment in verifier_env.items()
                },
                "trial_reward_key": "reward",
            },
        },
        "trajectory_runner": {"max_concurrent_tasks": 8},
        "data": {"kind": "parquet", "shuffle": False, "train_data": [], "val_data": []},
        "extra_env": {**counterpart_environment(), "JUDGE_BASE_URL": judge_base_url},
    }
    _materialize_role_plan_config(config, plan)
    return config, plan


def validate_environment_preflight(manifest: dict, data_uri: str) -> None:
    """Reject incomplete or mismatched environment checks before constructing a GPU artifact."""
    if manifest.get("scope") == "diagnostic":
        raise click.ClickException("A diagnostic preflight does not cover an evaluation dataset.")
    required = set(manifest["required_source_keys"])
    missing = required - set(manifest["validated_source_keys"])
    if manifest["failures"] or missing or len(required) != manifest["expected_source_keys"]:
        raise click.ClickException(f"Environment manifest is incomplete ({len(missing)} missing keys).")
    if data_uri != manifest["data_uri"]:
        raise click.ClickException("Environment manifest belongs to a different dataset scope.")


@click.command(help=__doc__)
@click.option("--model-uri", required=True, help="Complete, revision-pinned RNO2A weight mirror.")
@click.option("--data-uri", required=True, help="Preflight-validated rollouts.parquet on RNO2A storage.")
@click.option("--environment-manifest", required=True, help="Immutable CPU environment-preflight manifest URI.")
@click.option("--priority", type=click.Choice(["batch", "production"]), default="batch")
@click.option("--judge-base-url", help="Existing MiniMax endpoint; otherwise co-host the pinned 131k judge.")
@rl_build_options
def main(
    model_uri: str, data_uri: str, environment_manifest: str, judge_base_url: str | None, priority: str
) -> ArtifactStep:
    # StoragePath.open does not route named CoreWeave buckets to their credentials.
    fs, path = filesystem_for(environment_manifest)
    with fs.open(path, "r") as file:
        manifest = json.load(file)
    validate_environment_preflight(manifest, data_uri)
    config, plan = recipe(
        manifest["environment_overrides"],
        manifest["agent_mcp_servers"],
        manifest["verifier_env"],
        judge_base_url or "runtime://minimax-judge",
        manifest["credential_env"],
    )
    data = ArtifactStep.adopt("documents/mini840-" + fingerprint_hash(data_uri), "2026.10.31", data_uri)
    model = ArtifactStep.adopt(
        "checkpoints/mini840-qwen36", "2026.10.31", model_uri, config={"revision": MODEL_REVISION}
    )
    step = skyrl_step(
        SkyRLSpec(
            name="checkpoints/mini840-e2e",
            version=resolve_version("checkpoints/mini840-e2e", None),
            config_yaml=yaml.safe_dump(config),
            model=ArtifactHfModel(model, relative_path="", tokenizer_uri=MODEL, tokenizer_revision=MODEL_REVISION),
            train_data=(),
            validation_data=(ArtifactDataSource(data, relative_path="rollouts.parquet"),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=2, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=32,
            memory="256GB",
            disk="256GB",
            priority=priority,
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=25,
            job_timeout_seconds=86400,
            required_env_vars=(manifest["credential_env"], "TOGETHER_API_KEY")
            + (("JUDGE_API_KEY",) if judge_base_url else ()),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=False,
    )
    if judge_base_url:
        return replace(step, run=run_with_existing_judge)
    root = Path(__file__).resolve().parents[3]
    judge = load_model_config(
        root / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/olympiadbench_judge/"
        "MiniMaxAI-MiniMax-M3-MXFP8-131k.yaml"
    )
    cluster = yaml.safe_load((root / IRIS_HUB_CLUSTER_CONFIG).read_text())
    return replace(
        step,
        run=run_judged,
        build_config=lambda ctx: JudgedEvaluation(step.build_config(ctx), judge, cluster["dashboard_url"]),
    )


if __name__ == "__main__":
    main()
