# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment-owned asynchronous GS8 RL recipe and artifact binding."""

from dataclasses import replace
from pathlib import Path

import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.long_loop.archive_selection import trace_retention, validation_selection
from experiments.post_training.task_curation.inventory import TRAIN_FILENAME, VALIDATION_FILENAME

SELECTION_PREFIX = "rl-data/task-selection"
DEFAULT_SELECTION_URI = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04"
SMOKE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=64,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)


def smoke_recipe() -> dict:
    """Load the local recipe; final evaluation belongs to the separate Eval-org stage."""
    config = yaml.safe_load(Path(__file__).with_name("rl-smoke.yaml").read_text())
    _materialize_role_plan_config(config, SMOKE_PLAN)
    return config


def smoke_rl_step(
    *,
    version: str,
    model: ArtifactHfModel,
    selection: ArtifactStep[Artifact],
) -> ArtifactStep[SkyRLRun]:
    """Consume the preceding verified BF16 handoff, never a fixed released SFT model."""
    config = smoke_recipe()
    config["trainer"]["dump_training_order"] = True
    config["generator"]["trajectory_retention"] = trace_retention(
        prefix_join(prefix_join("s3://marin-us-east-02a/marin/long-loop/traces", version), "rl"),
        version,
        fraction=0.1,
        phase="train",
    )
    return configured_rl_step(
        version=version,
        model=model,
        train_data=(ArtifactDataSource(selection, relative_path=TRAIN_FILENAME),),
        validation_data=(ArtifactDataSource(selection, relative_path=VALIDATION_FILENAME),),
        config=config,
        plan=SMOKE_PLAN,
        name="long-loop-plumbing-rl",
        retained=2,
        seed=20261020,
        execution=IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=32,
            memory="1800GB",
            disk="2TB",
            priority="production",
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=36,
            coordinator_name=f"long-loop-plumbing-{version}",
            job_timeout_seconds=36 * 60 * 60,
            wandb_entity="marin-community",
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            required_env_vars=(
                "DAYTONA_RL_API_KEY",
                "DAYTONA_EVAL_API_KEY",
                "DAYTONA_TARGET",
                "TOGETHER_API_KEY",
                "MARIN_PREFIX",
            ),
        ),
    )


def configured_rl_step(
    *,
    version: str,
    model: ArtifactHfModel,
    train_data: tuple[ArtifactDataSource, ...],
    validation_data: tuple[ArtifactDataSource, ...],
    config: dict,
    plan: SkyRLRolePlan,
    execution: IrisSkyRLExecution,
    name: str,
    retained: int,
    seed: int,
) -> ArtifactStep[SkyRLRun]:
    """Bind an explicit recipe to admitted data; admission belongs to the run-spec stage.

    Unlike the historical smoke wrapper, this binding supplies no data selection,
    training geometry, credential configuration, or trace-retention defaults.
    """
    buffer = config.get("trainer", {}).get("rollout_buffer", {})
    quotas = buffer.get("consumed_route_quotas", {})
    if quotas and set(buffer.get("source_routes", {}).values()) != set(quotas):
        raise ValueError("RL route admission is incomplete: supply verified source manifests and classifications")
    step = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/{name}"),
            version=version,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            topology=SkyRLTopology(
                num_nodes=plan.num_nodes,
                gpus_per_node=plan.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=retained),
            seed=seed,
        ),
        execution,
        export_hf=True,
    )

    original_builder = step.build_config

    def bind_runtime(context: StepContext) -> SkyRLRunConfig:
        launch = original_builder(context)
        document = yaml.safe_load(launch.launch_config_yaml)
        recipe = document["skyrl"]
        for callback in recipe["trainer"].get("callbacks", []):
            if callback["type"] == "best_checkpoint":
                callback["initial_model_uri"] = document["inputs"]["model"]["uri"]
        traces = prefix_join(document["artifacts"]["attempts_root"], "traces")
        retention = recipe["generator"].get("trajectory_retention")
        if retention is not None:
            retention["output_path"] = prefix_join(traces, "rl")
            retention["run_id"] = document["run"]["id"]
        installed = recipe.get("terminal_bench", {}).get("installed_agent")
        if installed is not None:
            installed["output_path"] = prefix_join(traces, "installed-agent")
        return replace(launch, launch_config_yaml=yaml.safe_dump(document, sort_keys=False))

    return replace(step, build_config=bind_runtime)


FULL_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=16,
    policy_num_gpus_per_node=8,
    num_inference_engines=16,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=64,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)


def full_recipe(*, initial_model: str, validation_tasks: int, source_routes: dict[str, str]) -> dict:
    """Construct the target geometry; live qualification remains mandatory."""
    config = yaml.safe_load(Path(__file__).with_name("rl-smoke.yaml").read_text())
    _materialize_role_plan_config(config, FULL_PLAN)
    config["context_budget"] = {
        "request_window_tokens": 102_400,
        "max_new_tokens_per_turn": 8_192,
        "max_turns": 128,
    }
    trainer = config["trainer"]
    trainer.update(
        max_steps=10_000,
        step_wise_training=False,
        dump_training_order=True,
        ckpt_interval=1,
        max_ckpts_to_keep=-1,
        long_context_qualification={
            "enabled": True,
            "minimum_tokens": 65537,
            "maximum_tokens": 102400,
            "maximum_steps": 5,
        },
    )
    if set(source_routes.values()) != {"agentic", "nonagentic"}:
        raise ValueError("Full RL requires admitted agentic and nonagentic source provenance")
    trainer["rollout_buffer"].update(
        consumed_route_quotas={"agentic": 32, "nonagentic": 32}, source_routes=source_routes
    )
    selection = validation_selection(initial_model=initial_model, task_count=validation_tasks)
    trainer.update(selection)
    trainer["callbacks"] = [
        {"type": "checkpoint", "save_steps": 1, "save_on_train_end": True},
        {"type": "evaluation", "eval_steps": 1, "eval_before_train": False, "eval_on_train_end": True},
        *selection["callbacks"],
        {"type": "hf_model_save", "save_steps": -1, "save_on_train_end": True},
        {"type": "wall_time_budget", "seconds": 8 * 3600, "reserve": 1800, "minimum_steps": 5, "groups_per_step": 64},
    ]
    for role in ("policy", "ref"):
        trainer[role]["megatron_config"]["context_parallel_size"] = 4
    sessions = config["environment"]["task_sessions"]
    sessions["daytona"]["eval_api_key_env"] = "DAYTONA_EVAL_API_KEY"
    sessions["task_spec"] = {
        "session": {
            "max_turns": 128,
            "attempt_timeout": 1800,
            "total_turn_timeout": 1800,
            "model_turn_timeout": 1800,
        }
    }
    config["generator"]["max_num_seqs"] = 64
    config["generator"]["engine_init_kwargs"]["hf_overrides"] = {
        "max_position_embeddings": 262144,
        "max_seq_len": 262144,
    }
    return config
