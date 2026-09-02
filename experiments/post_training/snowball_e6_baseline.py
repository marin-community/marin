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

# ── SMOKE GEOMETRY ───────────────────────────────────────────────────────────────────────────
# --smoke reduces the geometry so the SAME model, dataset, environment and code path produce real
# telemetry cheaply. It is the right smoke test precisely because nothing about the path changes:
# iceball would have exercised a different model on a dependency chain hardcoded to GB200.
#
#   constant                here (smoke)   E6      why
#   policy nodes            2              8       memory floor is ~3 nodes under MuonH's fp32
#                                                  parameter storage; under AdamW storage is BF16, so
#                                                  2 is the first count with margin at 80 GB/GPU.
#   inference engines       1              2       an engine is TP1 x DP8 = one node.
#   train_batch_size        32             256     floor is policy_dp_size (utils.py). 8x cheaper.
#   policy_mini_batch_size  32             256     must divide train_batch_size.
#
# ⚠️ The node reduction alone changes no RL math -- train_batch_size counts prompts globally, so the
# effective batch is identical and only grad-accumulation depth changes. The BATCH reduction DOES
# change semantics (32 GRPO groups per update, not 256) and must never be carried into a science run.
# ⚠️ Timings from a smoke run are NOT comparable to the 4209.7 s baseline. It answers "do rows
# arrive, and is the decomposition sane", nothing else.
SMOKE_POLICY_NODES = 2
SMOKE_INFERENCE_ENGINES = 1
SMOKE_BATCH = 32


def _role_plan(*, policy_nodes: int, inference_engines: int, batch: int) -> SkyRLRolePlan:
    return SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=policy_nodes,
        policy_num_gpus_per_node=E6_GPUS_PER_NODE,
        num_inference_engines=inference_engines,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=batch,
        policy_mini_batch_size=batch,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=16,
    )


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


def _rl_config(*, role_plan: SkyRLRolePlan, max_steps: int, ckpt_interval: int) -> str:
    """Render the SkyRL config.

    ``ckpt_interval`` defaults to ``max_steps + 1``: no periodic checkpoint, but one terminal
    checkpoint at train end. That terminal write is **required**, not incidental. On a successful
    launch ``cloud/iris/job.py`` unconditionally runs ``export_terminal_policy``, which reads the
    checkpoint marker and raises ``"Successful Iris job did not commit a checkpoint marker"``
    (``cloud/iris/artifacts.py``) when there is none -- so ``ckpt_interval: 0`` trains all four steps
    on 80 GPUs and then reports a failed artifact with no terminal manifest.

    The Hugging Face upload is closed at its own lever instead, ``trainer.hf_save_interval``, as an
    override in ``build_workflow``. Conflating the two is easy and wrong: ``hf_save_interval`` only
    *defaults* to ``${trainer.ckpt_interval}``.
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

  train_batch_size: {role_plan.train_batch_size}
  policy_mini_batch_size: {role_plan.policy_mini_batch_size}
  eval_batch_size: 256
  micro_forward_batch_size_per_gpu: 1
  micro_train_batch_size_per_gpu: {role_plan.micro_train_batch_size_per_gpu}

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
    policy_num_nodes: {role_plan.policy_num_nodes}
    policy_num_gpus_per_node: {role_plan.policy_num_gpus_per_node}
    policy_strict_spread_pg: true
    policy_per_gpu_bundles: true

generator:
  backend: vllm
  model_dtype: bfloat16
  inference_engine_tensor_parallel_size: {role_plan.inference_engine_tensor_parallel_size}
  inference_engine_data_parallel_size: 8
  inference_engine_expert_parallel_size: 8
  num_inference_engines: {role_plan.num_inference_engines}
  n_samples_per_prompt: {role_plan.n_samples_per_prompt}
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
    ckpt_interval: int | None = None,
    bf16_update_mode: str = "stochastic",
    policy_train_spans: bool = True,
    spans_synchronize: bool = True,
    smoke: bool = False,
    debug_distributed: bool = False,
    label: str = "",
) -> ArtifactStep[SkyRLModel]:
    """Compose the E6 baseline as one inspectable artifact step."""
    role_plan = (
        _role_plan(
            policy_nodes=SMOKE_POLICY_NODES,
            inference_engines=SMOKE_INFERENCE_ENGINES,
            batch=SMOKE_BATCH,
        )
        if smoke
        else E6_ROLE_PLAN
    )
    num_nodes = role_plan.policy_num_nodes + role_plan.num_inference_engines
    # The artifact name IS the telemetry identity, so it has to separate every variant we intend to
    # compare. run_id is f"{step_name}-{version}" (marin/rl/skyrl.py) and is passed through to
    # SKYRL_RUN_ID, which becomes the `run_id` resource attribute on every row. Override strings
    # change the FINGERPRINT but not the name -- so without this, a spans-on and a spans-off run at
    # the same version publish into the same run_id and their rows are indistinguishable.
    variant = "-smoke" if smoke else ""
    if not policy_train_spans:
        variant += "-nospans"
    if policy_train_spans and not spans_synchronize:
        variant += "-nosync"
    if debug_distributed:
        variant += "-nccldbg"
    if bf16_update_mode != "stochastic":
        variant += f"-{bf16_update_mode}"
    if label:
        variant += f"-{label}"
    base_name = f"checkpoints/{E6_MODEL_NAME}{variant}"
    data = ExternalDataSource(uri=E6_DATA_PREFIX, identity=E6_DATA_IDENTITY)
    train_data = replace(data, relative_path=E6_TRAIN_FILENAME)
    validation_data = replace(data, relative_path=E6_VALIDATION_FILENAME)
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            config_yaml=_rl_config(
                role_plan=role_plan,
                max_steps=max_steps,
                ckpt_interval=max_steps + 1 if ckpt_interval is None else ckpt_interval,
            ),
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
                num_nodes=num_nodes,
                gpus_per_node=E6_GPUS_PER_NODE,
                gpu_variant=E6_GPU_VARIANT,
                role_plan=role_plan,
            ),
            # The launcher hardcodes `++trainer.resume_mode=latest` (cloud/iris/iris_backend.py),
            # which beats anything in the YAML body -- so `resume_mode: null` there is inert. Spec
            # overrides are appended AFTER the launcher's own, and later Hydra overrides win, so this
            # is the only place the setting actually takes. A resumed timing run appends a second
            # attempt, on possibly different hardware, to what reads as one run.
            overrides=(
                "++trainer.resume_mode=none",
                # `++`, and an override rather than a config key, because BOTH matter. The key is
                # absent from optimizer_config's schema at this pin -- fsdp_strategy reads it with
                # .get(), but the schema never declares it -- and rl_config_translation emits config
                # body keys with NO prefix unless they match _OPTIONAL_HYDRA_PATTERNS. A plain
                # override of an undeclared key fails Hydra's struct check, which would abort the run
                # after the 80-GPU gang had already started.
                f"++trainer.policy.optimizer_config.bf16_update_mode={bf16_update_mode}",
                # The Hugging Face upload gate, and it is NOT ckpt_interval. The Hub callback
                # needs `trainer.hf_hub_repo_id` truthy AND `trainer.hf_save_interval > 0`
                # (skyrl_train/config/callbacks.py); hf_save_interval merely DEFAULTS to
                # ${trainer.ckpt_interval}. The launcher force-sets hf_hub_repo_id to
                # laion/<job_name> when unset -- an org we do not own -- so this interval is
                # the only lever we control. The separate terminal export never uploads:
                # TerminalPolicyExport carries no repo id, so checkpoint_export.hf_hub_repo_id
                # resolves to null and hub_publisher returns None.
                "++trainer.hf_save_interval=0",
                # Worker-side decomposition of policy_train. `++` because these keys are
                # declared in the pinned MarinSkyRL but not in every base config revision, and
                # an undeclared key fails Hydra's struct check after the gang has started.
                f"++trainer.policy_train_spans={str(policy_train_spans).lower()}",
                f"++trainer.policy_train_spans_synchronize={str(spans_synchronize).lower()}",
                # NCCL_DEBUG=INFO + NCCL_DEBUG_SUBSYS=...NET,GRAPH + TORCH_NCCL_ENABLE_TIMING=1 and
                # the flight-recorder artifact dirs, all from one switch. This is what answers H10:
                # whether the collectives ran on InfiniBand or fell back to sockets. Exposure is
                # ~34-42 TB/rank/step, so a 2x fabric inefficiency is ~1,900 s -- half of
                # policy_train. ⚠️ It is also a timing contaminant: enable it in BOTH cells of a
                # comparison, or in a separate diagnostic attempt.
                *(("++trainer.debug_mode=distributed",) if debug_distributed else ()),
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
    "--label",
    default="",
    help="Extra suffix on the artifact name, and therefore on run_id. The flags that change "
    "behaviour already encode themselves; use this to separate two runs that are otherwise "
    "identical (a repeat, or a different cluster).",
)
@click.option(
    "--smoke",
    is_flag=True,
    default=False,
    help="Reduced geometry (2 policy nodes + 1 engine, batch 32) on the SAME model, data and code "
    "path. For proving telemetry arrives and the decomposition is sane -- its timings are NOT "
    "comparable to the 4209.7 s baseline, and its batch change alters RL semantics.",
)
@click.option(
    "--debug-distributed",
    is_flag=True,
    default=False,
    help="trainer.debug_mode=distributed: NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS with NET and GRAPH, "
    "TORCH_NCCL_ENABLE_TIMING=1 and the flight-recorder dirs. This is what answers whether the "
    "collectives ran on InfiniBand or fell back to sockets. It is a timing contaminant -- use it in "
    "both cells of a comparison, or in a separate diagnostic attempt.",
)
@click.option(
    "--policy-train-spans/--no-policy-train-spans",
    default=True,
    show_default=True,
    help="Decompose policy_train from inside the policy worker. On by default: policy_train is 90.4% "
    "of the step and a run without this produces the same single opaque number we already have.",
)
@click.option(
    "--spans-synchronize/--no-spans-synchronize",
    default=True,
    show_default=True,
    help="Synchronise CUDA at span boundaries. On attributes time to the right span; off measures "
    "end-to-end without serialising the pipeline. A spans-on run is for ATTRIBUTION -- do not use it "
    "as an unbiased whole-step baseline against a spans-off run.",
)
@click.option(
    "--ckpt-interval",
    type=int,
    default=None,
    help="Checkpoint interval. Default is max_steps + 1: no periodic checkpoint, one terminal "
    "checkpoint. Do not set 0 -- the launcher's unconditional terminal export then fails on the "
    "missing marker and the whole 80-GPU run reports a failed artifact. See _rl_config.",
)
@build_options
def main(
    wandb_entity: str,
    max_steps: int,
    ckpt_interval: int,
    bf16_update_mode: str,
    policy_train_spans: bool,
    spans_synchronize: bool,
    smoke: bool,
    debug_distributed: bool,
    label: str,
) -> ArtifactStep[SkyRLModel]:
    return build_workflow(
        wandb_entity=wandb_entity,
        max_steps=max_steps,
        ckpt_interval=ckpt_interval,
        bf16_update_mode=bf16_update_mode,
        policy_train_spans=policy_train_spans,
        spans_synchronize=spans_synchronize,
        smoke=smoke,
        debug_distributed=debug_distributed,
        label=label,
    )


if __name__ == "__main__":
    main()
