# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect pinned teacher or student BFCL complement trajectories without optimizer updates."""

import re
from dataclasses import dataclass

import click
import yaml
from marin.execution.artifact import Artifact
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

from experiments.post_training.bfcl_rl.data import DATASET_COMMIT

DATA_URI = f"s3://marin-us-east-02a/marin/users/benfeuer/bfcl-rl/data/bfcl-complement-{DATASET_COMMIT}"
SMOKE_TASKS = ("bfcl-irrelevance-0", "bfcl-simple-java-0", "bfcl-simple-javascript-0")
DOCKERFILE_HASHES = (
    "66449a5ed97b0f0feaa6e484e7128dd869a31305bab1a26333ddeda59170e965",
    "b607af0e6fdd7f7519fcdad1b7ee02cf684300f3fd867c2990e7947a1c59067e",
    "f78a98ce9c108903392ad7bb44f5ae2f680bb3b5e187911bddd43d1503c814f5",
)


@dataclass(frozen=True)
class ModelSource:
    model: str
    revision: str
    uri: str
    version: str


MODELS = {
    "teacher": ModelSource(
        "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21",
        "b8c07f7df1df65525abbfdbcd1572318ba11c42f",
        "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21",
        "2026.09.21",
    ),
    "student": ModelSource(
        "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02",
        "50fe2f488aee7730bbccc721024b35229075602e",
        "s3://marin-us-east-02a/marin/users/benfeuer/checkpoints/antidoom-rlvr1-async/2026.10.02.6/exports/global_step_12/policy",
        "2026.10.02.12",
    ),
}

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=True,
    policy_num_nodes=1,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=8,
    policy_mini_batch_size=8,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=1,
)


def collection_recipe(images: tuple[str, str, str]) -> str:
    """Render policy-matched Pi collection with required, complete token retention."""
    if any(re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) is None for image in images):
        raise ValueError("BFCL task images must be immutable registry digests")
    return yaml.safe_dump(
        {
            "entrypoint": "terminal_bench_generate",
            "config_groups": {"terminal_bench_config": "terminal_bench"},
            "context_budget": {
                "request_window_tokens": 40960,
                "max_new_tokens_per_turn": 8192,
                "max_turns": 999999,
            },
            "terminal_bench": {
                "harbor": {
                    "name": "pi",
                    "version": "0.87.0",
                    "thinking_format": "chat-template",
                    "override_setup_timeout_sec": 360,
                    "override_timeout_sec": 1800,
                    "eval_timeout_override_sec": 1800,
                    "verifier_override_timeout_sec": 300,
                    "import_path": "marinskyrl.iris_harbor_environment:IrisEnvironment",
                    "container_profile": "gvisor",
                    "prebuilt_images": dict(zip(DOCKERFILE_HASHES, images, strict=True)),
                    "override_cpus": 1,
                    "override_memory_mb": 2048,
                    "override_storage_mb": 10240,
                    "auto_snapshot": False,
                    "n_concurrent_trials": 32,
                    "max_retries": 0,
                    "collect_rollout_details": True,
                    "enable_reward_shaping": False,
                    "enable_error_classification": True,
                    "mask_exceptions": [
                        "IrisSandboxError",
                        "EnvironmentStartTimeoutError",
                        "NetworkError",
                        "ConnectionError",
                        "RewardFileNotFoundError",
                        "RewardFileEmptyError",
                        "VerifierTimeoutError",
                        "VerifierRuntimeError",
                    ],
                    "default_error_treatment": "zero",
                },
                "model_info": {},
                "archiving": {"enabled": False},
                "trace_upload": {"enabled": False},
            },
            "trainer": {
                "strategy": "megatron",
                "max_steps": 1,
                "eval_before_train": False,
                "eval_interval": -1,
                "resume_mode": "none",
                "enable_db_registration": False,
                "logger": "console",
                "project_name": "bfcl-rl",
                "algorithm": {"use_kl_loss": False, "off_policy_correction": "tis", "tito_full": True},
            },
            "generator": {
                "backend": "vllm",
                "model_dtype": "bfloat16",
                "vllm_attention_backend": "FLASH_ATTN",
                "run_engines_locally": True,
                "enable_http_endpoint": True,
                "gpu_memory_utilization": 0.8,
                "max_num_batched_tokens": 7168,
                "max_num_seqs": 32,
                "sampling_params": {"temperature": 1.0, "top_p": 1.0, "logprobs": 0},
                "eval_sampling_params": {"temperature": 1.0, "top_p": 1.0, "logprobs": 0},
                "engine_init_kwargs": {
                    "tool_call_parser": "hermes",
                    "model_loader_extra_config": {"distributed": True},
                },
                "trajectory_retention": {
                    "enabled": True,
                    "phases": ["eval"],
                    "sample_count_per_step": 0,
                    "sample_fraction": 1.0,
                    "required": True,
                    "max_bytes_per_step": 17179869184,
                    "max_bytes_per_run": 17179869184,
                },
            },
            "data": {"kind": "tasks", "train_data": [], "val_data": []},
            "trajectory_runner": {"rollout_workers": {"num_workers": 1, "cpus_per_worker": 4}},
        },
        sort_keys=False,
    )


def collection_step(model: str, task: str | None, images: tuple[str, str, str]) -> ArtifactStep:
    source = MODELS[model]
    model_step = ArtifactStep.adopt(
        user_owned_name(f"models/bfcl-rl-{model}"),
        source.version,
        source.uri,
        kind=LevanterCheckpoint,
        config={"model": source.model, "revision": source.revision},
    )
    data_step = ArtifactStep.adopt(
        user_owned_name("data/bfcl-rl-complement"),
        "2026.10.03",
        DATA_URI,
        kind=Artifact,
        config={"dataset_commit": DATASET_COMMIT, "train_tasks": 3518, "parity_tasks": 123},
    )
    name = user_owned_name(f"rollouts/bfcl-rl-recovery-{model}-{task or 'full'}")
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            config_yaml=collection_recipe(images),
            model=ArtifactHfModel(model_step, source.model, source.revision, relative_path=""),
            train_data=(
                ArtifactDataSource(data_step, relative_path=f"bfcl_complement/{task}" if task else "bfcl_complement"),
            ),
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=ROLE_PLAN.policy_num_nodes,
                gpus_per_node=ROLE_PLAN.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1, temporary_storage_ttl_days=14),
            seed=42,
        ),
        IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            cpu=48,
            memory="1611Gi",
            disk="21745Gi",
            priority="interactive",
            max_retries=0,
            coordinator_timeout_hours=24,
            job_timeout_seconds=24 * 60 * 60,
        ),
    )


@click.command(help=__doc__)
@click.option("--model", type=click.Choice(tuple(MODELS)), required=True)
@click.option("--task", type=click.Choice(SMOKE_TASKS), default=None, help="Unchanged complement task for a smoke.")
@click.option("--python-image", required=True)
@click.option("--java-image", required=True)
@click.option("--javascript-image", required=True)
@rl_build_options
def main(model: str, task: str | None, python_image: str, java_image: str, javascript_image: str) -> ArtifactStep:
    return collection_step(model, task, (python_image, java_image, javascript_image))


if __name__ == "__main__":
    main()
