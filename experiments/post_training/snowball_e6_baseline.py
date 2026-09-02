# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball E6 at its specified geometry, as the performance baseline.

This is the #7786 E6 arm as written -- AdamW, 8 policy nodes + 2 inference engines, 256 prompts per
step, on ``cw-rno2a`` -- expressed as a Marin artifact graph. It exists to be *measured*, not to
learn: it is the control that every `rl-perf` arm is compared against, and the run whose
``policy_train`` phase the telemetry work instruments.

It is derived from ``snowball_e6_muonh.py``, which shares its structure but is a different run in
two respects that both matter here. That file swaps AdamW for MuonH (marin#8060 objective 2), and it
runs a deliberately reduced smoke geometry -- 5 nodes, 32 prompts -- to stay schedulable. **Neither
deviation is carried over.** The delta table lives in
``notes/workstreams/rl-perf/N1-MATERIALIZED-REQUEST.md`` §4.

The values here come from penfever's #7786 gist ``snowball_e6_rno2a_rlvrmath.yaml``, which is the
specification of record; the campaign issue's body was edited away when it closed on 2026-08-27, so
the gist is what survived. Where the gist and this file disagree, the disagreement is commented.

Defaults are set for a **timing run**: two steps, no checkpoint, no resume, no retry. Pass
``--max-steps 20`` to run it as the science configuration instead.

Print the plan (this is the dry run -- there is no ``--dry-run`` flag)::

    python -m experiments.post_training.snowball_e6_baseline --version dev --wandb-entity dogml

Run it::

    python -m experiments.post_training.snowball_e6_baseline --version 2026.09.02 \
        --wandb-entity dogml --run

``--wandb-entity`` is not optional in practice. Iris forwards only ``WANDB_API_KEY`` and ``HF_TOKEN``
from the submitting process (``EnvironmentSpec.to_proto``), so ``WANDB_ENTITY`` does not survive the
hop into the nested job -- and sourcing ``~/Documents/secrets.env``, which is itself load-bearing for
the API key, sets it to something other than ``dogml``. Unset, the run silently lands in a different
entity from the baseline and the campaign cannot be joined.
"""

from __future__ import annotations

from dataclasses import replace

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    ExternalDataSource,
    ExternalModel,
    IrisSkyRLExecution,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)

E6_MODEL_NAME = "snowball-e6-baseline"

# Identical to snowball_e6_muonh.py, and deliberately so: holding the checkpoint and the dataset
# fixed is what makes any of these runs comparable to the #7786 campaign. See that file for why the
# model is named by its S3 mirror rather than its Hub id (MarinSkyRL's ModelSource rejects a
# non-object-store source at the Marin request boundary).
SNOWBALL_SFT_MIRROR = "s3://marin-us-east-02a/models/marin-community--grug-67b-a2b-sft-s2-thinking-step630"
SNOWBALL_SFT_REPO = "marin-community/grug-67b-a2b-sft-s2-thinking-step630"
SNOWBALL_SFT_REVISION = "6808fe5c"

E6_DATA_PREFIX = "s3://marin-us-east-02a/iris/rl-data/snowball-67b-a2b-rlvrmath-7498"
E6_DATA_IDENTITY = "snowball-67b-a2b-rlvrmath-7498:rlvrmath-bd2a9355+math500-6e4ed1a2"
E6_TRAIN_FILENAME = "train.parquet"
E6_VALIDATION_FILENAME = "validation.parquet"

# ---------------------------------------------------------------------------------------------
# GEOMETRY -- E6 as specified. Every value below is the gist's, with one deliberate exception.
#
#   8 policy nodes x 8 H100          = 64 training GPUs
#   2 inference engines, TP1 x DP8   = 16 rollout GPUs, one engine per node
#                                    = 10 nodes, 80 H100, on cw-rno2a
#
# The engine arithmetic -- adding the two counts to get the node total -- is only correct because an
# engine's GPU count is TP * PP * DP (ray_wrapped_inference_engine.py:357) and TP1 x DP8 is exactly
# one node. Change the engine's DP and this stops being true.
#
# The exception is cpu: the gist's launch.sh asks 128 and that is UNSCHEDULABLE, permanently and not
# under load. iris converts the ask with int(cpu * 1000) (cluster/types.py:663) and requests it as
# millicores (backends/k8s/tasks.py:828); an H100 node advertises 127960m allocatable, so 128000m is
# excluded from every node in the fleet. This gated a run for 24h on 2026-08-20 until it died on an
# execution timeout, reported only as `excluded: resource "cpu": 64`.
# ---------------------------------------------------------------------------------------------
E6_CLUSTER = "cw-rno2a"
E6_CLUSTER_CONFIG = f"lib/iris/config/{E6_CLUSTER}.yaml"
E6_GPU_VARIANT = "H100"
E6_GPUS_PER_NODE = 8
E6_POLICY_NODES = 8
E6_INFERENCE_ENGINES = 2
E6_NUM_NODES = E6_POLICY_NODES + E6_INFERENCE_ENGINES
E6_CPU_PER_NODE = 96
E6_WANDB_PROJECT = "snowball_67b_a2b_rl_7786"

# 8192 = 1664 input + 6528 generation. E4 showed 12k and 16k OOM on the FP32-einsum path.
#
# 1664, not the 1600 the gist's own launch.sh header claims: generate_configs.py raised the input
# allowance on 2026-08-02 after 1600 killed the run at step 16, and the YAML shipped 1664. The
# trainer's guard measures the *templated* prompt (max 1,604), not the raw one (1,587), and a single
# over-long row leaves response_end_idx unbound and aborts the whole job.
E6_CONTEXT_WINDOW = 8192
E6_MAX_NEW_TOKENS = 6528

# 256 prompts x 16 samples = 4,096 sequences per step = 64 per policy GPU, and at
# micro_train_batch_size_per_gpu=1 that is 64 gradient-accumulation micro-steps per GPU per step.
# That micro-step count is the thing the telemetry work is trying to see inside.
#
# policy_mini_batch_size == train_batch_size means one optimizer update per step, with no PPO inner
# minibatching -- so "step" and "update" are the same unit here, which they are not in general.
E6_TRAIN_BATCH_SIZE = 256

E6_ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=E6_POLICY_NODES,
    policy_num_gpus_per_node=E6_GPUS_PER_NODE,
    num_inference_engines=E6_INFERENCE_ENGINES,
    inference_engine_tensor_parallel_size=1,
    train_batch_size=E6_TRAIN_BATCH_SIZE,
    policy_mini_batch_size=E6_TRAIN_BATCH_SIZE,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=16,
)


def _rl_config(*, max_steps: int) -> str:
    """Render the SkyRL config. ``ckpt_interval`` is derived so a timing run writes no checkpoint.

    A Grug checkpoint is ~539 GB. At the default two steps we want none written at all, so the
    interval is set past the end of the run rather than to the gist's 2.
    """
    ckpt_interval = max_steps + 1

    # Notes on the values that are NOT simply the gist's, and on the ones that look like oversights
    # but are not:
    #
    #   optimizer            -- ABSENT, and that is the point. The gist never states an optimizer;
    #                           AdamW is inherited (ppo_base_config.yaml:91), as are betas
    #                           [0.9, 0.999] and weight_decay 1e-2. snowball_e6_muonh.py sets
    #                           weight_decay: 0, which is correct there only because build_grug_muonh
    #                           forces zero decay on every group regardless. Under AdamW that value
    #                           is real, and wrong. Omitting the whole optimizer block is what
    #                           reproduces E6.
    #                           Consequence worth knowing: parameter STORAGE dtype is selected by
    #                           optimizer name -- fsdp_strategy.py:72 returns fp32 for MuonH and
    #                           bf16 otherwise -- so this run's persistent parameter memory is half
    #                           the MuonH file's, and none of that file's memory arithmetic carries.
    #
    #   resume_mode: null    -- NOT the base default (latest). A retry that resumes on different
    #                           hardware appends to what looks like one attempt, which silently
    #                           destroys a timing measurement. ResumeMode._missing_ maps None to
    #                           ResumeMode.NONE (trainer_utils.py:46-49), so `null` and `"none"` are
    #                           equivalent; null is used because the override form is `=null`.
    #
    #   flash_attn: false    -- eager attention, via attn_backend "auto". This is E6's behaviour and
    #   use_sample_packing   -- packing off; Grug rejects it, and the base config defaults it TRUE.
    #   use_grouped_mm       -- ABSENT, inheriting false: the eager 256-expert path.
    #                           These three are the slow defaults this workstream exists to
    #                           interrogate. They stay ON in the baseline. Each becomes an arm as a
    #                           single override string; none of them is a bug in this file.
    #
    #   use_kl_loss: false   -- load-bearing: the base default is true. With use_kl_in_reward also
    #                           false, use_ref_model is false (trainer.py:848) and NO reference model
    #                           is built. colocate_policy_ref below is therefore inert -- it is
    #                           carried from the gist for fidelity, not because it does anything.
    #
    #   expert_model_parallel_size: 1
    #                        -- the TRAINER's expert parallelism, inherited as 1. Stated so it is not
    #                           confused with the generator's EP=8 below; they are separate axes, and
    #                           "TP1 + DP8 + EP8" in the E6 notes refers to the generator.
    #
    #   grug_query_bias_update_mode
    #                        -- required, no default at the call site; the validator raises if absent.
    return f"""\
entrypoint: standard

context_budget:
  request_window_tokens: {E6_CONTEXT_WINDOW}
  max_new_tokens_per_turn: {E6_MAX_NEW_TOKENS}
  max_turns: 1

model_num_attention_heads: 20

environment:
  env_class: aime

trainer:
  strategy: fsdp2
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: false
    use_entropy_loss: false
    eps_clip_low: 0.2
    eps_clip_high: 0.2

  epochs: 1
  max_steps: {max_steps}
  ckpt_interval: {ckpt_interval}
  resume_mode: null

  train_batch_size: {E6_ROLE_PLAN.train_batch_size}
  policy_mini_batch_size: {E6_ROLE_PLAN.policy_mini_batch_size}
  eval_batch_size: 256
  micro_forward_batch_size_per_gpu: 1
  micro_train_batch_size_per_gpu: {E6_ROLE_PLAN.micro_train_batch_size_per_gpu}

  use_sample_packing: false
  flash_attn: false
  attn_backend: "auto"
  gradient_checkpointing: true
  gradient_checkpointing_use_reentrant: false

  eval_before_train: false
  eval_interval: 0

  project_name: {E6_WANDB_PROJECT}
  logger: wandb

  policy:
    grug_query_bias_update_mode: "frozen"
    optimizer_config:
      lr: 1.0e-5
      max_grad_norm: 0.5
    fsdp_config:
      cpu_offload: false
      reshard_after_forward: true
      expert_model_parallel_size: 1

  placement:
    colocate_all: false
    colocate_policy_ref: true
    policy_num_nodes: {E6_ROLE_PLAN.policy_num_nodes}
    policy_num_gpus_per_node: {E6_ROLE_PLAN.policy_num_gpus_per_node}
    policy_strict_spread_pg: true
    policy_per_gpu_bundles: true

generator:
  backend: vllm
  model_dtype: bfloat16
  inference_engine_tensor_parallel_size: {E6_ROLE_PLAN.inference_engine_tensor_parallel_size}
  inference_engine_data_parallel_size: 8
  inference_engine_expert_parallel_size: 8
  num_inference_engines: {E6_ROLE_PLAN.num_inference_engines}
  n_samples_per_prompt: {E6_ROLE_PLAN.n_samples_per_prompt}
  gpu_memory_utilization: 0.85
  enforce_eager: false
  max_num_batched_tokens: 16384
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true

extra_env:
  HF_HUB_OFFLINE: "1"
  NCCL_SOCKET_IFNAME: "^ibs,ibp,lo,docker,veth,cilium,lxc"

data:
  kind: parquet
  train_data: []
  val_data: []
"""


def build_workflow(
    *,
    version: str | None = None,
    wandb_entity: str | None = None,
    max_steps: int = 2,
) -> ArtifactStep[SkyRLModel]:
    """Compose the E6 baseline as one inspectable artifact step."""
    base_name = f"checkpoints/{E6_MODEL_NAME}"
    data = ExternalDataSource(uri=E6_DATA_PREFIX, identity=E6_DATA_IDENTITY)
    train_data = replace(data, relative_path=E6_TRAIN_FILENAME)
    validation_data = replace(data, relative_path=E6_VALIDATION_FILENAME)
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            config_yaml=_rl_config(max_steps=max_steps),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.FSDP),
            model=ExternalModel(
                uri=SNOWBALL_SFT_MIRROR,
                identity=f"{SNOWBALL_SFT_REPO}@{SNOWBALL_SFT_REVISION}",
                tokenizer_uri=SNOWBALL_SFT_REPO,
                tokenizer_revision=SNOWBALL_SFT_REVISION,
            ),
            train_data=(train_data,),
            validation_data=(validation_data,),
            topology=SkyRLTopology(
                num_nodes=E6_NUM_NODES,
                gpus_per_node=E6_GPUS_PER_NODE,
                gpu_variant=E6_GPU_VARIANT,
                role_plan=E6_ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=17,
        ),
        IrisSkyRLExecution(
            cluster=E6_CLUSTER,
            cluster_config=E6_CLUSTER_CONFIG,
            cpu=E6_CPU_PER_NODE,
            memory="700GB",
            disk="10800Gi",
            # E6's own launch.sh used interactive, and on a contended pool a batch job sits behind
            # every interactive task. This is the documented exception to the house `batch` default;
            # matching the workflow's own precedent is what keeps the comparison honest.
            priority="interactive",
            # Zero, not the gist's default. A retry is a second attempt writing into the same
            # W&B run and the same attempt directory, on possibly different hardware -- which is
            # indistinguishable from one slow attempt when you are reading a step time. One attempt,
            # or a clean failure.
            max_retries=0,
            wandb_entity=wandb_entity,
        ),
    )


@click.command(help=__doc__)
@click.option(
    "--wandb-entity",
    default=None,
    help="W&B team for the run. Pass `dogml`. Unset lands in whatever team the API key defaults to; "
    "WANDB_ENTITY is not forwarded into the nested job.",
)
@click.option(
    "--max-steps",
    default=2,
    show_default=True,
    help="Training steps. The default is a timing run: one warm-up plus one measured step, with "
    "ckpt_interval derived past the end so no ~539 GB checkpoint is written. Pass 20 for E6's "
    "science configuration.",
)
@build_options
def main(wandb_entity: str | None, max_steps: int) -> ArtifactStep[SkyRLModel]:
    return build_workflow(wandb_entity=wandb_entity, max_steps=max_steps)


if __name__ == "__main__":
    main()
