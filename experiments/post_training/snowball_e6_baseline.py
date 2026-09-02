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

Defaults are set for a **timing run**: four steps, no checkpoint, no Hub upload, no resume. Pass
``--max-steps 20 --ckpt-interval 2`` to run it as the science configuration instead.

One property of this run is a deliberate choice rather than a default, and it must be stated
wherever a number from it is quoted:

**The optimizer implementation is selected, not inherited.** At the current MarinSkyRL pin every
``AdamW`` is routed through ``build_adamw`` (``distributed/fsdp_strategy.py``), which returns a
custom stochastic-rounding BF16 optimizer for ``bf16_update_mode`` of ``stochastic`` (the default)
or ``kahan``, and plain ``torch.optim.AdamW`` for ``nearest``. E6 ran ``torch.optim.AdamW``, because
``build_adamw`` did not exist at its pin.

``--bf16-update-mode`` therefore chooses what is being measured:

* ``stochastic`` (default) -- the stack as it runs today. This is what we are trying to make
  faster, and what an arm result has to generalise to.
* ``nearest`` -- E6's optimizer implementation, for a comparison against run ``nk0ehfrv``.

Neither is free: they differ numerically, so runs under different modes are not paired
observations. **Hold the mode fixed across every arm and record it in the decision record.**

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


def _rl_config(*, max_steps: int, ckpt_interval: int, bf16_update_mode: str) -> str:
    """Render the SkyRL config.

    ``ckpt_interval`` must be **0** for a timing run, and 0 is not merely "an interval past the end".
    Setting it past the end does NOT stop a checkpoint: any positive interval installs
    ``CheckpointCallback`` (callbacks/builtin.py), whose ``save_on_train_end`` defaults to True, so a
    ~539 GB checkpoint is written when training ends no matter how large the interval is. 0 skips
    the callback entirely.

    0 also closes the Hugging Face upload path, which is the more dangerous one.
    ``config/callbacks.py`` enables the Hub callback only when ``hf_hub_repo_id`` is truthy **and**
    ``hf_save_interval > 0``; ``hf_save_interval`` defaults to ``${trainer.ckpt_interval}``. That
    matters because the launcher auto-defaults ``hf_hub_repo_id`` to **``laion/<job_name>``** when it
    is unset (cloud/iris/rl_config_translation.py) -- an org we do not own -- and the publisher
    explicitly turns HF_HUB_OFFLINE back off before uploading (skyrl_train/hf_publisher.py), so the
    ``HF_HUB_OFFLINE=1`` set in ``extra_env`` below does NOT protect against it.
    """

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
    #   resume_mode: null    -- INERT HERE, and kept only so the intent reads in one place. The
    #                           launcher appends `++trainer.resume_mode=latest` itself, which beats
    #                           the YAML body; the setting that actually takes is the spec-level
    #                           override in build_workflow below. ResumeMode._missing_ maps None to
    #                           ResumeMode.NONE (trainer_utils.py), so null and "none" are equivalent.
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
      bf16_update_mode: {bf16_update_mode}
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
    wandb_entity: str = "dogml",
    max_steps: int = 4,
    ckpt_interval: int = 0,
    bf16_update_mode: str = "stochastic",
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
            config_yaml=_rl_config(max_steps=max_steps, ckpt_interval=ckpt_interval, bf16_update_mode=bf16_update_mode),
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
            # The launcher hardcodes `++trainer.resume_mode=latest` (cloud/iris/iris_backend.py),
            # which beats anything in the YAML body -- so `resume_mode: null` there is inert. Spec
            # overrides are appended AFTER the launcher's own, and later Hydra overrides win, so this
            # is the only place the setting actually takes. A resumed timing run appends a second
            # attempt, on possibly different hardware, to what reads as one run.
            overrides=("++trainer.resume_mode=none",),
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
            # Both budgets, because they are separate axes and only bounding the first leaves the
            # other at iris's default of 1000. A retry is a second attempt writing into the same
            # W&B run and the same attempt directory, on possibly different hardware -- which is
            # indistinguishable from one slow attempt when you are reading a step time. And with
            # resume disabled a preempted relaunch restarts from step 0, so an unbounded preemption
            # budget can spend the 80-GPU gang's cost repeatedly and never finish. One attempt, or a
            # clean failure.
            max_retries=0,
            max_retries_preemption=0,
            wandb_entity=wandb_entity,
        ),
    )


@click.command(help=__doc__)
@click.option(
    "--wandb-entity",
    default="dogml",
    show_default=True,
    help="W&B team for the run. Defaults rather than being optional: the launcher drops the flag "
    "when it is falsey, and WANDB_ENTITY is not forwarded into the nested job, so an unset value "
    "silently lands the run in whatever team the API key defaults to -- unjoinable with the campaign.",
)
@click.option(
    "--max-steps",
    default=4,
    show_default=True,
    help="Training steps. 4 is the timing-run minimum: AdamW allocates its moment tensors lazily on "
    "the first optimizer step, so step 1 is discarded and steps 2-4 give a dispersion estimate. Two "
    "steps yield a single observation with no stability check. Pass 20 for the science configuration.",
)
@click.option(
    "--bf16-update-mode",
    type=click.Choice(["stochastic", "nearest", "kahan"]),
    default="stochastic",
    show_default=True,
    help="Which AdamW implementation runs. `stochastic` is the stack's own default and measures "
    "today's code; `nearest` falls through to torch.optim.AdamW, which is what E6 ran. Stated "
    "explicitly because inheriting it silently attributes an optimizer change to the model.",
)
@click.option(
    "--ckpt-interval",
    default=0,
    show_default=True,
    help="0 disables checkpointing AND the Hugging Face upload path. Any positive value writes a "
    "~539 GB checkpoint at train end regardless of how large it is. See _rl_config.",
)
@build_options
def main(wandb_entity: str, max_steps: int, ckpt_interval: int, bf16_update_mode: str) -> ArtifactStep[SkyRLModel]:
    return build_workflow(
        wandb_entity=wandb_entity,
        max_steps=max_steps,
        ckpt_interval=ckpt_interval,
        bf16_update_mode=bf16_update_mode,
    )


if __name__ == "__main__":
    main()
