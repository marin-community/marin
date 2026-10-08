# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the audited unseen-task DPO pairs with native Pi parity validation."""

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, MODELS, collection_recipe


class RunScale(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


@dataclass(frozen=True)
class DPOParallelism:
    tensor: int
    pipeline: int
    context: int
    expert: int
    first_stage_layers: int
    last_stage_layers: int


POLICY_PARALLELISM = DPOParallelism(1, 4, 1, 8, 6, 6)
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=16,
    policy_mini_batch_size=16,
    micro_train_batch_size_per_gpu=2,
    n_samples_per_prompt=2,
)
MODEL_VERSION = "2026.10.08.304"
MODEL_NAME = "models/bfcl-rl-recovery-dpo-native"
MODEL_URI = "s3://marin-us-east-02a/marin/users/benfeuer/models/bfcl-rl-recovery-dpo-native/2026.10.08.304"
INPUT_NAME = "data/bfcl-rl-final-dpo-inputs"
POLICY_FILE = "final_dpo_pi_policy.yaml"


def final_dpo_recipe(scale: RunScale) -> str:
    """Bind the frozen Pi conditions to static masked DPO and its separate evaluator."""
    policy = yaml.safe_load(Path(__file__).with_name(POLICY_FILE).read_text())
    recipe = yaml.safe_load(collection_recipe())
    recipe["entrypoint"] = "standard"
    recipe["config_groups"]["algorithm_recipe"] = "dpo"
    recipe["context_budget"]["serving_window_tokens"] = 73728
    harbor = recipe["terminal_bench"]["harbor"]
    for field in ("override_cpus", "override_memory_mb", "override_storage_mb"):
        harbor.pop(field)
    (agent,) = policy["agents"]
    harbor.update(
        name=agent["name"],
        version=agent["kwargs"]["version"],
        override_setup_timeout_sec=agent["override_setup_timeout_sec"],
        override_timeout_sec=agent["override_timeout_sec"],
        eval_timeout_override_sec=agent["override_timeout_sec"],
        verifier_override_timeout_sec=policy["verifier"]["override_timeout_sec"],
        environment_type=policy["environment"]["type"],
        n_concurrent_trials=policy["n_concurrent_trials"],
        **policy["environment"]["kwargs"],
        **policy["retry"],
    )
    parallel = POLICY_PARALLELISM
    megatron = {
        "tensor_model_parallel_size": parallel.tensor,
        "pipeline_model_parallel_size": parallel.pipeline,
        "context_parallel_size": parallel.context,
        "expert_model_parallel_size": parallel.expert,
        "expert_tensor_parallel_size": 1,
        "transformer_config_kwargs": {
            "num_layers_in_first_pipeline_stage": parallel.first_stage_layers,
            "num_layers_in_last_pipeline_stage": parallel.last_stage_layers,
        },
    }
    updates = 4 if scale is RunScale.SMOKE else 1712 // ROLE_PLAN.train_batch_size
    recipe["data"] = {
        "kind": "parquet",
        "train_data": [],
        "val_data": [],
        "preference_pair_format": "tokenized",
        "shuffle": False,
    }
    recipe["environment"] = {"env_class": "preference_pair"}
    trainer = recipe["trainer"]
    trainer.update(
        epochs=1,
        max_steps=updates,
        update_epochs_per_batch=1,
        use_sample_packing=False,
        evaluation_runner="harbor",
        eval_before_train=True,
        eval_interval=4,
        eval_batch_size=123,
        eval_num_prompts=None,
        resume_mode="none",
        gradient_checkpointing=True,
        flash_attn=True,
        offload_optimizer_during_rollouts=True,
        ckpt_interval=4,
        hf_save_interval=4 if scale is RunScale.SMOKE else -1,
        enable_db_registration=False,
        logger="wandb",
        project_name="bfcl-rl",
        policy={
            "sequence_parallel_size": 1,
            "megatron_config": megatron,
            "optimizer_config": {
                "optimizer": "AdamW",
                "lr": 4e-6,
                "adam_betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "max_grad_norm": 0.5,
                "num_warmup_steps": 0,
                "scheduler": "constant_with_warmup",
            },
        },
        ref={"sequence_parallel_size": 1, "megatron_config": megatron},
        rollout_buffer={"max_staleness_steps": 0, "max_in_flight": 16, "batch_policy": "full_batch"},
        step_phase_budgets={"evaluation": 21600, "policy_training": 7200, "checkpoint_work": 7200},
        callbacks=[
            {"type": "logging"},
            {"type": "checkpoint", "save_steps": 4, "save_on_train_end": True},
            {"type": "evaluation", "eval_steps": 4, "eval_before_train": True, "eval_on_train_end": True},
            {
                "type": "best_checkpoint",
                "initial_model_uri": f"{MODEL_URI}/hf/step-1",
                "expected_tasks": 123,
                "minimum_scored_tasks": 111,
            },
        ],
    )
    trainer["algorithm"] = {"use_kl_loss": False, "dpo": {"beta": 0.1, "label_smoothing": 0.0}}
    if scale is RunScale.SMOKE:
        trainer["callbacks"].append({"type": "hf_model_save", "save_steps": 4, "save_on_train_end": True})
    generator = recipe["generator"]
    generator.update(
        model_loading="stream",
        use_conversation_multi_turn=True,
        eval_n_samples_per_prompt=1,
        eval_sampling_params={"temperature": 1.0, "top_p": 1.0, "logprobs": None},
        sampling_params={"temperature": 1.0, "top_p": 1.0, "logprobs": None},
    )
    generator["trajectory_retention"].update(
        phases=["eval"], max_bytes_per_run=512 * 1024**3, max_bytes_per_step=16 * 1024**3
    )
    recipe["trajectory_runner"] = {"rollout_workers": {"num_workers": 1, "cpus_per_worker": 4}}
    return yaml.safe_dump(recipe, sort_keys=False)


def final_dpo_step(input_version: str, scale: RunScale) -> ArtifactStep:
    inputs_name = user_owned_name(INPUT_NAME)
    inputs = ArtifactStep.adopt(inputs_name + "-input", input_version, f"{inputs_name}/{input_version}", kind=Artifact)
    model_name = user_owned_name(MODEL_NAME)
    checkpoint = ArtifactStep.adopt(model_name + "-input", MODEL_VERSION, MODEL_URI, kind=LevanterCheckpoint)
    tokenizer = MODELS["student"]
    data_file = "full-batches.parquet"
    name = user_owned_name(f"models/bfcl-rl-final-native-dpo-{scale.value}")
    spec = SkyRLSpec(
        name=name,
        version=resolve_version(name, None),
        config_yaml=final_dpo_recipe(scale),
        runtime=SkyRLRuntime(SkyRLRuntimeProfile.MEGATRON),
        model=ArtifactHfModel(checkpoint, tokenizer.model, tokenizer.revision, relative_path="hf/step-1"),
        train_data=(ArtifactDataSource(inputs, relative_path=data_file),),
        validation_data=(ArtifactDataSource(inputs, relative_path="bfcl_parity"),),
        topology=SkyRLTopology(num_nodes=5, gpus_per_node=8, gpu_variant="H100", role_plan=ROLE_PLAN),
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=14),
        seed=42,
    )
    execution = replace(
        COLLECTION_EXECUTION, wandb_entity="nyu-dice-lab", coordinator_timeout_hours=72, job_timeout_seconds=72 * 3600
    )
    return skyrl_step(spec, execution, export_hf=scale is RunScale.SMOKE)


@click.command(help=__doc__)
@click.option("--input-version", required=True)
@click.option("--scale", type=click.Choice([value.value for value in RunScale]), required=True)
@rl_build_options
def main(input_version: str, scale: str) -> ArtifactStep:
    return final_dpo_step(input_version, RunScale(scale))


if __name__ == "__main__":
    main()
