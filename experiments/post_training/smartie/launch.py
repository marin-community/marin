# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Smartie training from a pinned initializer and an admitted mixed-task dataset."""

import json
from copy import deepcopy
from pathlib import Path

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

MODEL_REPOSITORY = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
MODEL_REVISION = "50fe2f488aee7730bbccc721024b35229075602e"
CLUSTER = "cw-rno2a"
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=8,
    policy_num_gpus_per_node=8,
    num_inference_engines=12,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=4,
    inference_engine_expert_parallel_size=4,
    train_batch_size=64,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)


def training_config(
    max_steps: int,
    resume_path: str | None = None,
    *,
    initial_model_uri: str,
    learning_rate: float,
    benchmark_selection: Path | None = None,
) -> dict:
    """Render the objective and runtime settings; SkyRLSpec supplies role geometry."""
    with Path(__file__).with_name("recipe.yaml").open() as stream:
        config = yaml.safe_load(stream)
    trainer = config["trainer"]
    trainer["policy"]["optimizer_config"]["lr"] = learning_rate
    trainer.update(
        max_steps=max_steps,
        flash_attn=False,
        offload_optimizer_during_rollouts=True,
        resume_mode="from_path" if resume_path is not None else "none",
        resume_path=resume_path,
        restore_dataloader_state=True,
        reset_global_step_on_resume=False,
        evaluation_runner="training",
        tracker_commit_each_step=True,
        callbacks=[
            {"type": "checkpoint", "save_steps": 2},
            {"type": "evaluation", "eval_steps": 2, "eval_before_train": True, "eval_on_train_end": True},
            {
                "type": "best_checkpoint",
                "initial_model_uri": initial_model_uri,
                "expected_tasks": 1000,
                "minimum_scored_tasks": 950,
            },
        ],
    )
    geometry = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 16,
        "expert_tensor_parallel_size": 1,
        "logprob_chunk_size": 256,
        "transformer_config_kwargs": {},
    }
    trainer["ref"] = {"megatron_config": deepcopy(geometry)}
    geometry["ddp_config"] = {
        "overlap_grad_reduce": False,
        "overlap_param_gather": False,
        "grad_reduce_in_fp32": False,
    }
    geometry["optimizer_checkpoint_sharding_type"] = "dp_reshardable"
    trainer["policy"]["megatron_config"] = geometry
    trainer["placement"] = {"policy_strict_spread_pg": True}
    if benchmark_selection is not None:
        selection = json.loads(benchmark_selection.read_text())
        cohorts = []
        for name in ("bfcl_parity", "swebench-verified", "openthoughts-tblite"):
            tasks = [
                {**{key: task[key] for key in ("git_url", "git_commit_id", "path")}, "source": name}
                for task in selection["tasks"]
                if task["benchmark"] == name
            ]
            if len(tasks) != 30:
                raise ValueError(f"Expected 30 fixed validation tasks for {name}")
            cohorts.append(
                {
                    "name": name,
                    "reward_key": "reward",
                    "job": {
                        "tasks": tasks,
                        "n_attempts": 1,
                        "n_concurrent_trials": 30,
                        "environment": {"type": "daytona", "delete": True},
                        "retry": {
                            "max_retries": 1,
                            "include_exceptions": [
                                "APIConnectionError",
                                "APITimeoutError",
                                "SandboxBuildFailedError",
                                "ContainerRegistryThrottledError",
                                "DaytonaProcessError",
                            ],
                        },
                        "agents": [
                            {
                                "name": "terminus-2",
                                "override_timeout_sec": 600,
                                "override_setup_timeout_sec": 360,
                                "kwargs": {
                                    "temperature": 1.0,
                                    "max_turns": 32,
                                    "model_info": {"max_input_tokens": 32768, "max_output_tokens": 8192},
                                },
                            }
                        ],
                        "verifier": {"override_timeout_sec": 300},
                        "trial_attempt_timeout_sec": 1800,
                    },
                }
            )
        trainer["callbacks"].insert(
            2,
            {
                "type": "harbor_evaluation",
                "cohorts": cohorts,
                "timeout": 10800,
                "minimum_coverage": 0.9,
            },
        )
        config["generator"].update(enable_http_endpoint=True, http_endpoint_host="127.0.0.1", http_endpoint_port=8000)
    config["generator"].update(
        weight_sync_backend="nccl",
        weight_sync_transport="broadcast",
        weight_sync_pause={"mode": "keep", "clear_cache": True},
        chat_template=None,
    )
    config["generator"]["engine_init_kwargs"]["moe_backend"] = "triton"
    config["data"].update(kind="parquet", train_data=[], val_data=[], shuffle=True)
    config["extra_env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NCCL_DEBUG": "WARN"}
    return config


def build_run(
    *,
    model_uri: str,
    data_uri: str,
    data_version: str,
    max_steps: int,
    learning_rate: float,
    resume_path: str | None = None,
    benchmark_selection: Path | None = None,
) -> ArtifactStep:
    """Bind regional model bytes and native train/validation tasks to a 112-H100 run."""
    name = user_owned_name("models/snowball-smartie")
    model = ArtifactStep.adopt(
        user_owned_name("models/smartie-initializer"),
        "2026.10.02.1",
        model_uri,
        kind=LevanterCheckpoint,
        config={"hf_repository": MODEL_REPOSITORY, "hf_revision": MODEL_REVISION},
    )
    data = ArtifactStep.adopt(user_owned_name("datasets/smartie-mixed"), data_version, data_uri)
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(
                training_config(
                    max_steps,
                    resume_path,
                    initial_model_uri=model_uri,
                    learning_rate=learning_rate,
                    benchmark_selection=benchmark_selection,
                ),
                sort_keys=False,
            ),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=model, tokenizer_uri=MODEL_REPOSITORY, tokenizer_revision=MODEL_REVISION, relative_path=""
            ),
            train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
            validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
            topology=SkyRLTopology(num_nodes=14, gpus_per_node=8, gpu_variant="H100", role_plan=ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=5),
            seed=42,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=48,
            memory="1800GB",
            disk="21745Gi",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=72,
            wandb_entity=None,
            required_env_vars=("DAYTONA_RL_API_KEY",),
        ),
        export_hf=False,
    )


@click.command(help=__doc__)
@click.option("--model-uri", required=True, help="Regional HF-format bytes verified against the pinned initializer.")
@click.option("--data-uri", required=True, help="Admitted mixed native-task directory, including validation.parquet.")
@click.option("--data-version", required=True, help="Immutable admitted dataset version.")
@click.option("--max-steps", type=click.IntRange(min=20), default=20, show_default=True)
@click.option("--learning-rate", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--resume-path", help="Complete native training checkpoint; max-steps is the total update target.")
@click.option(
    "--benchmark-selection",
    type=click.Path(path_type=Path, exists=True),
    default=Path(__file__).with_name("benchmark_selection.json"),
    show_default=True,
)
@rl_build_options
def main(
    model_uri: str,
    data_uri: str,
    data_version: str,
    max_steps: int,
    learning_rate: float,
    resume_path: str | None,
    benchmark_selection: Path,
) -> ArtifactStep:
    return build_run(
        model_uri=model_uri,
        data_uri=data_uri,
        data_version=data_version,
        max_steps=max_steps,
        learning_rate=learning_rate,
        resume_path=resume_path,
        benchmark_selection=benchmark_selection,
    )


if __name__ == "__main__":
    main()
