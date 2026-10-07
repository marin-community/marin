# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate whole-rollout and step-wise SkyRL training with cat-count and two-turn math tasks."""

from dataclasses import dataclass, replace

import click
import yaml
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from rigging.filesystem.storage_path import prefix_join
from zephyr.writers import write_parquet_file

from experiments.post_training.cat_count_canary.data import TRAIN_FILENAME, VALIDATION_FILENAME, cat_count_record
from experiments.post_training.cat_count_canary.launcher import (
    CLUSTER,
    CPUS_PER_NODE,
    DEFAULT_MODEL,
    GPU_VARIANT,
    GPUS_PER_NODE,
    MODELS,
    QWEN_SOURCE,
    SEED,
    role_plan,
    run_canary,
    training_config,
)

BATCH_SIZE = 8
GROUP_SIZE = 4
MICRO_BATCH_SIZE = 4
VALIDATION_ROWS = BATCH_SIZE
RETENTION_BYTES_PER_STEP = 16 * 1024 * 1024
RETENTION_BYTES_PER_RUN = 128 * 1024 * 1024


@dataclass(frozen=True)
class ValidationDataConfig:
    output_path: str
    rows: int


def validation_record(index: int) -> dict:
    if index % 2 == 0:
        count = 1 + index % 13
        return {
            **cat_count_record(count, "validation", index),
            "reward_spec": {"method": "rule", "ground_truth": str(count)},
        }
    left, right = 3 + index % 11, 2 + index % 7
    return {
        "data_source": "task_session_math",
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"Compute {left} times {right}. Explain your calculation first. "
                    "Do not give a final answer or use #### until I ask for the final answer."
                ),
            }
        ],
        "env_class": "gsm8k_multi_turn",
        "reward_spec": {"method": "rule", "ground_truth": str(left * right)},
        "extra_info": {"index": index, "split": "validation"},
    }


def write_validation_data(config: ValidationDataConfig) -> None:
    for filename, count in ((TRAIN_FILENAME, config.rows), (VALIDATION_FILENAME, VALIDATION_ROWS)):
        write_parquet_file(
            (validation_record(index) for index in range(count)),
            prefix_join(config.output_path, filename),
        )


def build_run(*, mode: str, version: str, steps: int) -> ArtifactStep[SkyRLRun]:
    config = training_config(
        preset="dry",
        lane="sync",
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        micro_train_batch_size=MICRO_BATCH_SIZE,
    )
    config["context_budget"] = {
        "request_window_tokens": 1024,
        "max_new_tokens_per_turn": 128,
        "max_turns": 2,
    }
    trainer = config["trainer"]
    trainer.update(
        {
            "max_steps": steps,
            "update_epochs_per_batch": 1,
            "step_wise_training": mode == "step",
            "eval_before_train": False,
            "eval_interval": -1,
            "ckpt_interval": steps,
            "hf_save_interval": steps,
            "resume_mode": "none",
            "project_name": "marin-task-session-validation",
        }
    )
    # Keep checkpoint and token evidence. Omit evaluation and database registration.
    trainer["callbacks"] = [
        {"type": "checkpoint", "save_steps": steps},
        {"type": "hf_model_save", "save_steps": steps},
        {"type": "inference_stats", "log_every_steps": 1, "log_to_console": True, "log_to_tracker": True},
    ]
    config["generator"]["trajectory_retention"] = {
        "sample_count_per_step": 8,
        "max_bytes_per_step": RETENTION_BYTES_PER_STEP,
        "max_bytes_per_run": RETENTION_BYTES_PER_RUN,
        "required": True,
    }
    rows = BATCH_SIZE * steps
    data_name = user_owned_name(f"documents/task-session-validation/{fingerprint_hash(repr(rows))}")
    data = ArtifactStep(
        name=data_name,
        version=resolve_version(data_name, version),
        artifact_type=Artifact,
        run=remote(write_validation_data, resources=ResourceConfig.with_cpu(cpu=2, ram="8g", disk="8g")),
        build_config=lambda ctx: ValidationDataConfig(ctx.output_path, rows),
    )
    model = MODELS[DEFAULT_MODEL].step
    download = model.build_config(StepContext.for_fingerprint(model.runtime_args, model.deps))
    adopted_model = ArtifactStep.adopt(model.name, model.version, QWEN_SOURCE, kind=model.artifact_type)
    run = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/task-session-validation/{mode}"),
            version=version,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=adopted_model,
                relative_path="",
                tokenizer_uri=download.hf_dataset_id,
                tokenizer_revision=download.revision,
            ),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=2,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=role_plan(
                    batch_size=BATCH_SIZE, group_size=GROUP_SIZE, micro_train_batch_size=MICRO_BATCH_SIZE
                ),
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=CPUS_PER_NODE,
            memory="512GB",
            disk="1TB",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=2,
            wandb_entity=None,
            job_timeout_seconds=3600,
        ),
        export_hf=True,
    )
    return replace(run, run=run_canary)


@click.command(help=__doc__)
@click.option("--mode", type=click.Choice(("whole", "step")), required=True)
@click.option("--steps", type=click.IntRange(min=1), default=10, show_default=True)
@rl_build_options
def main(mode: str, steps: int) -> ArtifactStep[SkyRLRun]:
    return build_run(
        mode=mode, version=resolve_version(f"checkpoints/task-session-validation/{mode}", None), steps=steps
    )


if __name__ == "__main__":
    main()
