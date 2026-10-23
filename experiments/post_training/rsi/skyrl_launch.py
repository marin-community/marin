# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the ``train`` box of the Baby RSI loop as a SkyRL TaskCompendium run that reads a live task feed.

The loop appends task rows under ``--feed-root``. SkyRL polls the feed, trains full-parameter GRPO with Megatron,
and serves the live policy on an OpenAI-compatible HTTP endpoint, so the loop's Taskforge gate solves with the
current weights. The policy and the vLLM engines have their own nodes and staleness is 1: the engines keep serving
while the policy trains. The run ends with an HF export of its terminal checkpoint, which the loop's evaluation
serves with vLLM.

Plan or run one arm::

    uv run python -m experiments.post_training.rsi.skyrl_launch --version 2026.10.23 \\
        --model qwen3-4b --arm control --feed-root s3://marin-us-east-02a/<prefix>/control/feed --export-every 120
    uv run python -m experiments.post_training.rsi.skyrl_launch --version 2026.10.23 \\
        --model qwen3-4b --arm rsi --feed-root s3://marin-us-east-02a/<prefix>/rsi/feed --export-every 120 --run
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactHfModel,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.models import qwen3_4b_instruct_2507

EXPERIMENT_NAME = "baby-rsi-skyrl"
# Marin-Snowball-67B-A2B-v0.1: the HF export of step 14 of Smartie RL run 2026.10.15.32, which its model card names,
# adopted in place from US-EAST-02A (about 134 GB, referenced, not copied).
SNOWBALL_V0_1 = ArtifactStep.adopt(
    "models/marin-snowball-67b-a2b-v0.1",
    "2026.10.18.1",
    "s3://marin-us-east-02a/marin/users/benfeuer/models/smartie-run32-step14/2026.10.18.1/hf",
    kind=LevanterCheckpoint,
)
# The feed, the Snowball weights and the loop's other jobs are in US-EAST-02A.
CLUSTER = "cw-us-east-02a"
GPU_VARIANT = "H100"
# Whole nodes only: a privileged task sees every GPU of its node, and SkyRL does not remap CUDA_VISIBLE_DEVICES.
GPUS_PER_NODE = 8
ARMS = ("control", "rsi")
CHECKPOINT_EVERY = 10
FEED_POLL_SECONDS = 10
# The loop can stop writing to the feed for a long time (synthesis, a preempted loop job); the run waits.
ADMISSION_STALL_TIMEOUT = 7200
# Fixed so that the loop's gate can find the endpoint; each node runs one task, but privileged tasks share the
# host network.
HTTP_ENDPOINT_PORT = 32571
# Bubblewrap graders run on the trainer node: four rollout workers of 8 CPUs, plus the grader machines.
ROLLOUT_WORKERS = 4
CPUS_PER_ROLLOUT_WORKER = 8
ATTEMPTS_PER_WORKER = 64
GRADER_MACHINES = 32
TASK_CPUS = 96
# A synthesized task gives the solver a shell tool, and the engines refuse a chat request with tools (HTTP 400)
# unless they parse tool calls. Qwen3 and Snowball write Hermes <tool_call> tags.
TOOL_CALLS = MappingProxyType({"enable_auto_tools": True, "tool_parser": "hermes"})
# The rollout limits of the Baby RSI policy (run.py POLICY_LIMITS) and the shell tool of its shell harness.
SESSION = MappingProxyType(
    {
        "max_turns": 20,
        "model_turn_timeout": 600,
        "tool_turn_timeout": 120,
        "total_turn_timeout": None,
        "attempt_timeout": 1800,
        "verifier_timeout": 360,
        "cleanup_timeout": 60,
        "command_timeout": 30.0,
        "max_total_tokens": None,
        "shell_tool": {"name": "shell", "command_parameter": "command"},
    }
)


@dataclass(frozen=True)
class PolicyProfile:
    """The model, its placement, its context budget and its engine settings."""

    step: ArtifactStep[LevanterCheckpoint]
    tokenizer_uri: str
    tokenizer_revision: str
    policy_nodes: int
    megatron: MegatronGeometry
    num_engines: int
    engine_tensor_parallel: int
    engine_data_parallel: int
    engine_expert_parallel: int
    request_window_tokens: int
    max_new_tokens_per_turn: int
    host_memory: str
    disk: str
    max_num_seqs: int
    generator: Mapping[str, object]

    def role_plan(self, train_batch: int, samples_per_prompt: int) -> SkyRLRolePlan:
        # One update per step: the mini-batch is the batch. Sequences are up to the request window long, so one
        # sequence per GPU per micro-step.
        return SkyRLRolePlan(
            colocate_all=False,
            policy_num_nodes=self.policy_nodes,
            policy_num_gpus_per_node=GPUS_PER_NODE,
            num_inference_engines=self.num_engines,
            inference_engine_tensor_parallel_size=self.engine_tensor_parallel,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=self.engine_data_parallel,
            inference_engine_expert_parallel_size=self.engine_expert_parallel,
            train_batch_size=train_batch,
            policy_mini_batch_size=train_batch,
            micro_train_batch_size_per_gpu=1,
            n_samples_per_prompt=samples_per_prompt,
        )

    @property
    def data_parallel_size(self) -> int:
        geometry = self.megatron
        model_parallel = (
            geometry.tensor_model_parallel_size * geometry.pipeline_model_parallel_size * geometry.context_parallel_size
        )
        return self.policy_nodes * GPUS_PER_NODE // model_parallel


PROFILES: Mapping[str, PolicyProfile] = MappingProxyType(
    {
        # One policy node (Megatron TP1, DP8) and one node of eight TP1 vLLM engines.
        "qwen3-4b": PolicyProfile(
            step=qwen3_4b_instruct_2507,
            tokenizer_uri="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            policy_nodes=1,
            megatron=MegatronGeometry(1, 1, 1, 1, 1),
            num_engines=GPUS_PER_NODE,
            engine_tensor_parallel=1,
            engine_data_parallel=1,
            engine_expert_parallel=1,
            request_window_tokens=16384,
            max_new_tokens_per_turn=4096,
            host_memory="512GB",
            disk="1TB",
            max_num_seqs=128,
            generator=MappingProxyType({"gpu_memory_utilization": 0.8, "engine_init_kwargs": dict(TOOL_CALLS)}),
        ),
        # The 40-GPU geometry of experiments/post_training/async_rl.py: four policy nodes (Megatron TP1, PP2, EP8,
        # so DP16) and one node-sized engine that shards the 256 experts over its eight ranks (DP8, EP8).
        "snowball": PolicyProfile(
            step=SNOWBALL_V0_1,
            tokenizer_uri="open-athena/Marin-Snowball-67B-A2B-v0.1",
            tokenizer_revision="adc0078a19a4383bc998fcdb43867b3360baf5f3",
            policy_nodes=4,
            megatron=MegatronGeometry(1, 2, 1, 8, 1),
            num_engines=1,
            engine_tensor_parallel=1,
            engine_data_parallel=GPUS_PER_NODE,
            engine_expert_parallel=GPUS_PER_NODE,
            request_window_tokens=32768,
            max_new_tokens_per_turn=8192,
            host_memory="1800GB",
            disk="2TB",
            max_num_seqs=256,
            generator=MappingProxyType(
                {
                    "gpu_memory_utilization": 0.75,
                    # Each expert matrix goes from the Megatron rank that holds it to the vLLM ranks that serve it.
                    "weight_sync_transport": "expert_block",
                    "expert_block_sync": {"timeout_seconds": 600, "verify": False},
                    "engine_init_kwargs": {"moe_backend": "triton", **TOOL_CALLS},
                }
            ),
        ),
    }
)


def training_config(
    profile: PolicyProfile,
    *,
    feed_root: str,
    steps: int,
    train_batch: int,
    samples_per_prompt: int,
    lr: float,
    seed: int,
    export_every: int,
    request_window_tokens: int,
) -> dict:
    """The SkyRL recipe: GRPO with the rollout-anchored clipped loss on the feed, at staleness 1."""
    plan = profile.role_plan(train_batch, samples_per_prompt)
    geometry = asdict(profile.megatron)
    config = {
        "entrypoint": "taskcompendium",
        "context_budget": {
            "request_window_tokens": request_window_tokens,
            "max_new_tokens_per_turn": profile.max_new_tokens_per_turn,
            "max_turns": SESSION["max_turns"],
        },
        "environment": {
            "task_sessions": {
                # Script graders run in bubblewrap with the worker's Python environment.
                "local": {"python_environment": "host", "max_machines": GRADER_MACHINES},
            },
        },
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "gradient_checkpointing": True,
            "offload_optimizer_during_rollouts": False,
            # The feed grows during the run, so max_steps decides the length, not the epoch count.
            "epochs": 1000,
            "max_steps": steps,
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
            "eval_interval": -1,
            "eval_before_train": False,
            "ckpt_interval": CHECKPOINT_EVERY,
            # An HF export every export_every steps, which the loop's evaluation serves with vLLM.
            "hf_save_interval": export_every,
            "resume_mode": "latest",
            "restore_dataloader_state": True,
            "seed": seed,
            # The step metrics go to the job log; the loop records each group itself.
            "logger": "console",
            "project_name": EXPERIMENT_NAME,
            "tracker_commit_each_step": True,
            "algorithm": {
                "advantage_estimator": "grpo",
                "loss_reduction": "token_mean",
                # A clipped ratio against the vLLM sampling log probabilities. It is
                # anchored at the rollout, so a group sampled one step earlier needs no other correction.
                "policy_loss_type": "behavior_clip",
                "off_policy_correction": "none",
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
                "use_kl_loss": False,
                "use_kl_in_reward": False,
                "dynamic_sampling": {"type": None},
                "group_admission": {"stall_timeout": ADMISSION_STALL_TIMEOUT},
            },
            "policy": {
                "optimizer_config": {"optimizer": "AdamW", "lr": lr, "weight_decay": 0.0, "max_grad_norm": 1.0},
                "megatron_config": {**geometry, "optimizer_checkpoint_sharding_type": "dp_reshardable"},
            },
            "ref": {"megatron_config": geometry},
            "rollout_buffer": {
                "max_staleness_steps": 1,
                # Each step trains on the groups leased for it, so slow shell tasks keep their share of a batch.
                "batch_policy": "full_batch",
                "max_in_flight": 2 * train_batch,
                "object_store_root": None,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "weight_sync_pause": {"mode": "keep", "clear_cache": True},
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "max_num_seqs": profile.max_num_seqs,
            "enable_http_endpoint": True,
            "http_endpoint_host": "0.0.0.0",
            "http_endpoint_port": HTTP_ENDPOINT_PORT,
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "logprobs": 0},
            **dict(profile.generator),
        },
        "data": {
            "kind": "tasks",
            "shuffle": False,
            "train_data": [],
            "val_data": [],
            "feed": {
                "root": feed_root,
                "poll_seconds": FEED_POLL_SECONDS,
                "min_rows": train_batch,
                "session": dict(SESSION),
            },
        },
        "trajectory_runner": {
            "max_concurrent_tasks": ATTEMPTS_PER_WORKER,
            "rollout_workers": {"num_workers": ROLLOUT_WORKERS, "cpus_per_worker": CPUS_PER_ROLLOUT_WORKER},
        },
        # A privileged pod also sees the node's Ethernet RDMA functions; NCCL must use the InfiniBand HCAs only, or
        # the first cross-node collective of a multi-node policy fails with IBV_WC_RETRY_EXC_ERR.
        "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NCCL_IB_HCA": "ibp"},
    }
    _materialize_role_plan_config(config, plan)
    return config


def build_run(
    *,
    model: str,
    arm: str,
    feed_root: str,
    steps: int,
    train_batch: int,
    samples_per_prompt: int,
    lr: float,
    seed: int,
    export_every: int,
    request_window_tokens: int | None,
) -> ArtifactStep[SkyRLRun]:
    profile = PROFILES[model]
    if export_every % CHECKPOINT_EVERY:
        raise click.BadParameter(f"must be a multiple of {CHECKPOINT_EVERY}", param_hint="--export-every")
    if not feed_root.startswith("s3://"):
        raise click.BadParameter(f"the feed must be an s3:// prefix, got {feed_root!r}", param_hint="--feed-root")
    if steps <= 0 or train_batch <= 0 or samples_per_prompt <= 0:
        raise click.BadParameter("--steps, --train-batch and --samples-per-prompt must be positive")
    if train_batch * samples_per_prompt % profile.data_parallel_size:
        raise click.BadParameter(
            f"{train_batch} prompts x {samples_per_prompt} samples must divide over "
            f"{profile.data_parallel_size} data-parallel ranks"
        )
    plan = profile.role_plan(train_batch, samples_per_prompt)
    config = training_config(
        profile,
        feed_root=feed_root.rstrip("/"),
        steps=steps,
        train_batch=train_batch,
        samples_per_prompt=samples_per_prompt,
        lr=lr,
        seed=seed,
        export_every=export_every,
        request_window_tokens=request_window_tokens or profile.request_window_tokens,
    )
    base_name = f"checkpoints/{EXPERIMENT_NAME}/{model}-{arm}-seed{seed}"
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=resolve_version(base_name, None),
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=ArtifactHfModel(
                step=profile.step,
                tokenizer_uri=profile.tokenizer_uri,
                tokenizer_revision=profile.tokenizer_revision,
                relative_path="",
            ),
            # The feed replaces the training data; the loop evaluates the terminal export itself.
            train_data=(),
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=plan.num_nodes,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=seed,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=TASK_CPUS,
            memory=profile.host_memory,
            disk=profile.disk,
            priority="interactive",
            max_retries=2,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=72,
            wandb_entity=None,
            required_env_vars=("HF_TOKEN",),
            # Bubblewrap graders need namespaces.
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--model", type=click.Choice(tuple(PROFILES)), required=True)
@click.option("--arm", type=click.Choice(ARMS), required=True, help="Names the run; the feed decides the arm.")
@click.option("--feed-root", required=True, help="The s3:// prefix of the task feed that the loop writes.")
@click.option("--steps", type=int, default=150, show_default=True)
@click.option("--train-batch", type=int, default=32, show_default=True, help="Prompts per step.")
@click.option("--samples-per-prompt", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1.0e-6, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--export-every", type=int, required=True, help=f"Steps between HF exports, a multiple of {CHECKPOINT_EVERY}."
)
@click.option(
    "--request-window-tokens",
    type=int,
    help="The tokens of a whole multi-turn trajectory, in place of the profile's; a long window can exhaust GPU memory.",
)
@rl_build_options
def main(
    model: str,
    arm: str,
    feed_root: str,
    steps: int,
    train_batch: int,
    samples_per_prompt: int,
    lr: float,
    seed: int,
    export_every: int,
    request_window_tokens: int | None,
) -> ArtifactStep[SkyRLRun]:
    return build_run(
        model=model,
        arm=arm,
        feed_root=feed_root,
        steps=steps,
        train_batch=train_batch,
        samples_per_prompt=samples_per_prompt,
        lr=lr,
        seed=seed,
        export_every=export_every,
        request_window_tokens=request_window_tokens,
    )


if __name__ == "__main__":
    main()
