# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train on BFCL complement with four native harnesses after DPO recovery."""

from dataclasses import replace
from pathlib import Path

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactHfModel, SkyRLRolePlan, SkyRLRun, SkyRLTopology, skyrl_step
from marin.training.training import LevanterCheckpoint

from experiments.post_training.bfcl_rl.collect import (
    COLLECTION_EXECUTION,
    MODELS,
    collection_recipe,
    collection_spec,
)

MAX_RETAINED_BYTES_PER_STEP = 16 * 1024**3

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=12,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=4,
    inference_engine_expert_parallel_size=4,
    train_batch_size=512,
    policy_mini_batch_size=512,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=16,
)


def recovered_model(recovery_version: str, policy_export_version: str | None) -> ArtifactHfModel:
    """Bind the recovery artifact's latest HF export, failing if none exists."""
    name = user_owned_name("inputs/bfcl-rl-recovered-policy")
    producer = "bfcl-rl-recovery-dpo-full" if policy_export_version is None else "bfcl-rl-recovery-policy-export"
    producer_name = user_owned_name(f"models/{producer}")
    producer_version = recovery_version if policy_export_version is None else policy_export_version
    source = MODELS["student"]
    step = ArtifactStep.adopt(
        name,
        producer_version,
        f"{producer_name}/{producer_version}",
        kind=LevanterCheckpoint,
        config={
            "starting_model": source.model,
            "starting_revision": source.revision,
            "recovery_version": recovery_version,
        },
    )
    return ArtifactHfModel(step, source.model, source.revision)


def rl_recipe(images: tuple[str, str, str], num_train_steps: int, reader_concurrency: int, model_loading: str) -> str:
    """Adapt the copied v125 recipe to Harbor without changing optimizer settings."""
    recipe = yaml.safe_load(Path(__file__).with_name("v125_async.yaml").read_text())["skyrl"]
    recipe["entrypoint"] = "terminal_bench"
    recipe["config_groups"] = {"terminal_bench_config": "terminal_bench"}
    collection = yaml.safe_load(collection_recipe(images))
    recipe["context_budget"] = collection["context_budget"]
    recipe["terminal_bench"] = collection["terminal_bench"]
    harbor = recipe["terminal_bench"]["harbor"]
    harbor.update(name="opencode", version="1.18.2")
    harbor.pop("thinking_format")
    harbor["agent_profiles"] = [
        {"name": "opencode", "version": "1.18.2", "collect_rollout_details": True},
        {"name": "claude-code", "version": "2.1.284", "collect_rollout_details": True},
        {"name": "codex", "version": "0.118.0", "collect_rollout_details": True},
        {"name": "mini-swe-agent", "version": "2.1.0", "collect_rollout_details": True},
    ]

    trainer = recipe["trainer"]
    old_async = trainer.pop("fully_async")
    trainer["rollout_buffer"] = {
        "max_staleness_steps": old_async["max_staleness_steps"],
        "max_in_flight": old_async["num_parallel_generation_workers"],
        "batch_policy": "full_batch",
    }
    algorithm = trainer["algorithm"]
    if not algorithm.pop("use_tis") or algorithm.pop("tis_imp_ratio_cap") != 2.0:
        raise ValueError("The v125 recipe must use TIS with ratio cap 2.0")
    algorithm["off_policy_correction"] = "tis"
    algorithm["tito_full"] = True
    trainer.update(
        max_steps=num_train_steps,
        eval_before_train=False,
        eval_interval=-1,
        hf_save_interval=2,
        resume_mode="none",
        project_name="bfcl-rl",
    )
    for key in ("run_name", "export_path", "ckpt_path"):
        trainer.pop(key, None)
    trainer["policy"].pop("model")
    trainer["policy"].pop("fsdp_config")
    trainer["ref"].pop("fsdp_config")

    generator = recipe["generator"]
    for key in ("async_engine", "batched", "chat_template"):
        generator.pop(key)
    generator["trajectory_retention"] = {
        "enabled": True,
        "required": True,
        "phases": ["train"],
        "sample_count_per_step": 0,
        "sample_fraction": 1.0,
        "max_bytes_per_step": MAX_RETAINED_BYTES_PER_STEP,
        "max_bytes_per_run": MAX_RETAINED_BYTES_PER_STEP * num_train_steps,
    }
    generator["engine_init_kwargs"].pop("served_model_name")
    generator["engine_init_kwargs"]["model_loader_extra_config"] = {"concurrency": reader_concurrency}
    generator["model_loading"] = model_loading
    recipe["data"] = {"kind": "tasks", "train_data": [], "val_data": [], "shuffle": False}
    recipe.pop("environment")
    recipe.pop("terminal_bench_config")
    recipe["trajectory_runner"] = {"rollout_workers": {"num_workers": 12, "cpus_per_worker": 4}}
    return yaml.safe_dump(recipe, sort_keys=False)


def rl_step(
    recovery_version: str,
    policy_export_version: str | None,
    num_train_steps: int,
    images: tuple[str, str, str],
    reader_concurrency: int,
    model_loading: str,
) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name("models/bfcl-rl-multi-harness")
    spec = replace(
        collection_spec("student", None, images),
        name=name,
        version=resolve_version(name, None),
        config_yaml=rl_recipe(images, num_train_steps, reader_concurrency, model_loading),
        model=recovered_model(recovery_version, policy_export_version),
        topology=SkyRLTopology(num_nodes=10, gpus_per_node=8, gpu_variant="H100", role_plan=ROLE_PLAN),
    )
    return skyrl_step(spec, COLLECTION_EXECUTION, export_hf=True)


@click.command(help=__doc__)
@click.option("--recovery-version", required=True)
@click.option("--policy-export-version", default=None, help="Use an explicit saved-policy CPU export producer.")
@click.option("--num-train-steps", type=click.IntRange(min=2), required=True)
@click.option(
    "--reader-concurrency",
    type=click.IntRange(min=1),
    required=True,
    help="Concurrent S3 checkpoint readers per rollout-engine process.",
)
@click.option("--python-image", required=True)
@click.option("--model-loading", type=click.Choice(["stream", "stage_local"]), required=True)
@click.option("--java-image", required=True)
@click.option("--javascript-image", required=True)
@rl_build_options
def main(
    recovery_version: str,
    policy_export_version: str | None,
    num_train_steps: int,
    reader_concurrency: int,
    model_loading: str,
    python_image: str,
    java_image: str,
    javascript_image: str,
) -> ArtifactStep[SkyRLRun]:
    return rl_step(
        recovery_version,
        policy_export_version,
        num_train_steps,
        (python_image, java_image, javascript_image),
        reader_concurrency,
        model_loading,
    )


if __name__ == "__main__":
    main()
