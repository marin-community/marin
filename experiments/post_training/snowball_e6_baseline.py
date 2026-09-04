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
    SkyRLTelemetryRun,
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
# ── PR488-MATCHED GEOMETRY ───────────────────────────────────────────────────────────────────
# Russell Power's PR #488 published a Megatron-vs-FSDP2 table at a specific shape, and the whole
# value of matching it is that his numbers become a DIRECT comparison for ours rather than an
# analogy. His stated setup: 1024 prompt + 8192 response tokens, 64 prompts x 8 samples, four
# PP2xEP8 policy nodes and four vLLM nodes, no KL loss.
#   phase           Megatron     FSDP2 (his)
#   step            190-196 s    634 s
#   policy_train     26-27 s     457 s
#   generate        134-139 s    131 s
#   fwd_logprobs      6-7 s       32 s
#   sync_weights     15-17 s      11.3 s
# ⚠️ We cannot match PP2xEP8 -- that is Megatron's parallelism, and our path is FSDP2 with
# expert_model_parallel_size 1. So this matches the WORKLOAD (tokens, prompts, samples) and the
# NODE COUNT (4 policy + 4 inference), not the sharding. His FSDP2 column is the honest comparand.
PR488_POLICY_NODES = 4
PR488_INFERENCE_ENGINES = 4
PR488_BATCH = 64
PR488_N_SAMPLES = 8
PR488_CONTEXT_WINDOW = 1024 + 8192
PR488_MAX_NEW_TOKENS = 8192

SMOKE_POLICY_NODES = 2
SMOKE_INFERENCE_ENGINES = 1
SMOKE_BATCH = 32


def _role_plan(*, policy_nodes: int, inference_engines: int, batch: int, n_samples: int = 16) -> SkyRLRolePlan:
    return SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=policy_nodes,
        policy_num_gpus_per_node=E6_GPUS_PER_NODE,
        num_inference_engines=inference_engines,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=batch,
        policy_mini_batch_size=batch,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=n_samples,
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


def _rl_config(
    *,
    role_plan: SkyRLRolePlan,
    max_steps: int,
    ckpt_interval: int,
    debug_distributed: bool,
    reshard_after_forward: bool = True,
    flash_attn: bool = True,
    pr488_geometry: bool = False,
) -> str:
    """Render the SkyRL config.

    ``ckpt_interval`` defaults to ``max_steps + 1``: no periodic checkpoint, but one terminal
    checkpoint at train end. That terminal write is **required**, not incidental. On a successful
    launch ``cloud/iris/job.py`` runs ``export_terminal_policy``, which reads the checkpoint marker
    and raises ``"Successful Iris job did not commit a checkpoint marker"``
    (``cloud/iris/artifacts.py``) when there is none -- so a bare ``ckpt_interval: 0`` trains all
    four steps on 80 GPUs and then reports a failed artifact with no terminal manifest.
    ``--telemetry-only`` is how a run asks for both halves at once: it sets ``ckpt_interval`` to 0
    *and* tells the launcher to expect no model, so the artifact succeeds without one.

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
    #   flash_attn: false    -- eager attention, via attn_backend "auto". This is E6's behaviour, and
    #                           A12's arm. It is expensive twice over: eager materialises an fp32
    #                           [b,heads,L,L] score tensor (3.42 GiB at L=6775, the allocation behind
    #                           every OOM here) AND computes the full L^2 before masking a 2048
    #                           sliding window, discarding ~3.3x of that on 24 of 26 layers.
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
    # In the config BODY, not a ++ override. The launcher's build_debug_launch_env reads
    # args.debug_mode or the RL config YAML; a Hydra override reaches the driver and NOT the
    # launcher, so SKYRL_DEBUG_ARTIFACT_DIR is never set and sync_debug_artifacts early-returns.
    # The workers still write NCCL logs to pod-local /tmp, the artifact-sync line still prints a
    # destination, and nothing reaches S3 -- the logs die with the pods.
    debug_mode_line = "  debug_mode: distributed\n" if debug_distributed else ""

    return f"""\
entrypoint: standard

context_budget:
  request_window_tokens: {PR488_CONTEXT_WINDOW if pr488_geometry else E6_CONTEXT_WINDOW}
  max_new_tokens_per_turn: {PR488_MAX_NEW_TOKENS if pr488_geometry else E6_MAX_NEW_TOKENS}
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
{debug_mode_line}
  train_batch_size: {role_plan.train_batch_size}
  policy_mini_batch_size: {role_plan.policy_mini_batch_size}
  eval_batch_size: 256
  micro_forward_batch_size_per_gpu: 1
  micro_train_batch_size_per_gpu: {role_plan.micro_train_batch_size_per_gpu}

  use_sample_packing: false
  # A12's arm. false is E6's behaviour and it is expensive twice over: eager attention
  # materialises an fp32 [b, num_heads, L, L] score tensor -- 3.42 GiB at L=6775, the exact
  # allocation behind every OOM in this workstream (F6, F12) -- and it computes the full L^2
  # before masking a 2048-token sliding window, so ~3.3x of that work is discarded on 24 of 26
  # layers. After use_grouped_mm removed the expert cost, policy_forward (44.04 s) and
  # fwd_logprobs (43.60 s) CONVERGED to within 1%, which says both are now bound by the same
  # non-MoE term. This is the arm that tests whether that term is attention.
  # ⚠️ Grug supports it -- tests/gpu/test_grug_flash_attention.py is an H100 correctness and
  # memory gate -- but use_sample_packing must stay false regardless, because packing needs
  # flash-attn's varlen kernel AND Grug rejects packing separately.
  flash_attn: {str(flash_attn).lower()}
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
      # O2's arm. true re-gathers the full 67B parameter set for the backward instead of holding
      # it resident, so each micro-step pays two all-gathers and one reduce-scatter to compute on
      # 2B active parameters. That cost is exactly as independent of token content as the eager
      # expert loop is, and F3 records that our spans cannot separate the two -- FSDP2 issues these
      # collectives INSIDE forward and backward. Setting it false trades memory for traffic and is
      # the cheapest discriminator we have. ⚠️ F6 says memory is already tight; an OOM here is
      # itself the answer, not a failed run.
      reshard_after_forward: {str(reshard_after_forward).lower()}
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
  # Step 2 of both smoke runs died with `CUDA out of memory. Tried to allocate 3.58 GiB. GPU 0 has
  # 79.18 GiB of which 3.17 GiB is free` -- a fragmentation failure, not a capacity one: step 1 runs
  # on a clean allocator and step 2 cannot find a contiguous block. Four production configs in
  # MarinSkyRL's cloud/iris/configs already set this for the same reason, and it is on the launcher's
  # env passthrough list (cloud/iris/env_vars.py:268). Eager attention materialises a fp32
  # [1,20,8192,8192] score tensor at ~5.4 GiB per copy, which is why the headroom is this thin.
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
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
    # OFF by default since 2026-09-03. The spans themselves are ~free (the spans-OFF control was the
    # SLOWEST arm at both steps), but synchronising serialises the CUDA pipeline the run is trying to
    # overlap and cost +2.13% at step 2 -- a systematic penalty on every headline total this recipe
    # produces. It buys attribution: unattributable residual 2.56% -> 0.11%. Turn it on to ATTRIBUTE,
    # leave it off to MEASURE, and never compare the two -- the rows carry `_launch` vs `_wall` in
    # clock_domain precisely so a consumer cannot mix them by accident.
    spans_synchronize: bool = False,
    smoke: bool = False,
    debug_distributed: bool = False,
    collective_diagnostics: bool = False,
    telemetry_only: bool = False,
    # Every default here must match its click option -- otherwise a programmatic caller silently gets
    # different behaviour from the CLI, which is exactly the skew the merge gate caught on
    # bf16_grad_reduce and then caught AGAIN here when grouped_mm was turned off in one place only.
    # A test pins the two together now (tests/post_training/test_snowball_e6_baseline_defaults.py).
    grouped_mm: bool = True,
    reshard_after_forward: bool = True,
    flash_attn: bool = True,
    pr488_geometry: bool = False,
    bf16_grad_reduce: bool = True,
    log_ratio_probe: bool = False,
    label: str = "",
) -> ArtifactStep[SkyRLModel] | ArtifactStep[SkyRLTelemetryRun]:
    """Compose the E6 baseline as one inspectable artifact step."""
    if pr488_geometry:
        role_plan = _role_plan(
            policy_nodes=PR488_POLICY_NODES,
            inference_engines=PR488_INFERENCE_ENGINES,
            batch=PR488_BATCH,
            n_samples=PR488_N_SAMPLES,
        )
    elif smoke:
        role_plan = _role_plan(
            policy_nodes=SMOKE_POLICY_NODES,
            inference_engines=SMOKE_INFERENCE_ENGINES,
            batch=SMOKE_BATCH,
        )
    else:
        role_plan = E6_ROLE_PLAN
    num_nodes = role_plan.policy_num_nodes + role_plan.num_inference_engines
    # The artifact name IS the telemetry identity, so it has to separate every variant we intend to
    # compare. run_id is f"{step_name}-{version}" (marin/rl/skyrl.py) and is passed through to
    # SKYRL_RUN_ID, which becomes the `run_id` resource attribute on every row. Override strings
    # change the FINGERPRINT but not the name -- so without this, a spans-on and a spans-off run at
    # the same version publish into the same run_id and their rows are indistinguishable.
    variant = "-pr488" if pr488_geometry else ("-smoke" if smoke else "")
    if not policy_train_spans:
        variant += "-nospans"
    if policy_train_spans and not spans_synchronize:
        variant += "-nosync"
    if debug_distributed:
        variant += "-nccldbg"
    if collective_diagnostics:
        variant += "-colldiag"
    if telemetry_only:
        variant += "-telonly"
    if not grouped_mm:
        # Every measured throughput flag is ON by default now, so it is the ABSENCE that names a run.
        # NB: 'measured', not 'maximal' -- these three were each measured on their own arm; nothing
        # here establishes a global optimum, and the PR must not claim one.
        variant += "-eagermoe"
    if not reshard_after_forward:
        variant += "-noreshard"
    if not flash_attn:
        variant += "-eagerattn"
    if log_ratio_probe:
        variant += "-ratioprobe"
    if not bf16_grad_reduce:
        # A9 is the default now, so it is its ABSENCE that distinguishes a run.
        variant += "-fp32grad"
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
                # 0 skips CheckpointCallback entirely -- no ~539 GB terminal write, and no export
                # job after it. Every telemetry row was published per-step inside ppo_train, well
                # before either would have happened.
                ckpt_interval=(0 if telemetry_only else (max_steps + 1 if ckpt_interval is None else ckpt_interval)),
                debug_distributed=debug_distributed,
                reshard_after_forward=reshard_after_forward,
                flash_attn=flash_attn,
                pr488_geometry=pr488_geometry,
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
                # F2's headline arm, and a single flag: the baseline leaves use_grouped_mm absent and
                # so inherits false, which is the eager 256-expert Python loop. True routes Grug MoE
                # blocks through torch._grouped_mm instead (model_wrapper.py _enable_native_grug_grouping
                # -> enable_grug_grouped_mm), which needs no expert parallelism -- the EP coupling runs
                # the other way, validate_grug_expert_parallel_runtime rejects EP>1 WITHOUT grouped_mm,
                # never the reverse.
                # NO `++` here, deliberately, unlike bf16_update_mode above: this key IS declared in
                # ppo_base_config.yaml, so a plain override fails closed if it is ever renamed, while
                # `++` would silently add a dead key and the arm would read as a null result.
                *(("trainer.policy.fsdp_config.use_grouped_mm=true",) if grouped_mm else ()),
                # A9. FSDP2's MixedPrecisionPolicy takes reduce_dtype from fsdp_config, defaulting
                # to fp32 (fsdp_strategy.py:357-362, and the fsdp2 branch builds the policy from
                # those same vars at :400-402). Every gradient reduce-scatter therefore moves 2x the
                # bytes it needs to. Russell's PR488 config sets grad_reduce_in_fp32: false
                # explicitly, overriding megatron's own `true` default -- so this is his setting,
                # not an invention.
                # `++` is REQUIRED and is not belt-and-braces here, unlike use_grouped_mm:
                # `mixed_precision` is absent from ppo_base_config.yaml entirely, so a plain
                # override fails Hydra's struct check and would abort the run after the gang started.
                *(("++trainer.policy.fsdp_config.mixed_precision.reduce_dtype=bf16",) if bf16_grad_reduce else ()),
                # F25 diagnostic. Repeats the old-logprob forward on the same micro-batch and logs
                # the delta, which separates nondeterminism from a deterministic eval/train seam.
                *(("++trainer.log_ratio_repeat_probe=true",) if log_ratio_probe else ()),
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
                # Per-micro-batch, per-rank NCCL sequence numbers and mesh coordinates, at exactly
                # the seams the spans use. What it adds over a wall-clock span is WHICH process group
                # a straggler is behind on -- FSDP _ALLGATHER_BASE versus EP ALLTOALL_BASE. It reads
                # existing counters and issues no collectives of its own.
                *(("++trainer.collective_phase_diagnostics=true",) if collective_diagnostics else ()),
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=17,
            telemetry_only=telemetry_only,
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
    "--collective-diagnostics",
    is_flag=True,
    default=False,
    help="trainer.collective_phase_diagnostics: per-micro-batch, per-rank NCCL sequence numbers and "
    "mesh coordinates at the same seams the spans use. Tells you which process group a straggler is "
    "behind on, which wall-clock spans cannot. Reads existing counters, issues no collectives.",
)
@click.option(
    "--telemetry-only",
    is_flag=True,
    default=False,
    help="Write NO checkpoint and run NO export job. Sets ckpt_interval=0, which skips the "
    "~539 GB terminal checkpoint and the separate 64-GPU HF export that follows it. Telemetry is "
    "published from inside ppo_train every step, so all the data lands before either would have "
    "happened. The artifact succeeds as a SkyRLTelemetryRun, which carries no policy and so cannot "
    "be fed to an evaluation step -- read the rows in finelog and W&B, keyed by run_id.",
)
@click.option(
    "--grouped-mm/--no-grouped-mm",
    default=True,
    show_default=True,
    help="Route Grug MoE blocks through torch._grouped_mm instead of the eager 256-expert Python "
    "loop. This is the workstream's headline arm -- 22x on policy_train at PR488 geometry, and F2 "
    "measures the eager path at ~0.6% MFU against Megatron's 10%. Needs no expert parallelism; the "
    "EP constraint runs the other way. Encodes itself into the artifact name as -eagermoe when OFF, "
    "so its rows are separable by run_id. "
    "It broke the exact PPO ratio invariant until 2026-09-04 and was default-OFF for that reason; "
    "the grouped combine now reduces each token's rows in a fixed order (MarinSkyRL "
    "atqamar/grouped-mm-fix), and the E6 smoke on the real checkpoint reports log_ratio_abs_max 0 "
    "and exact-unit 1 on every step with the flag ON. See notes/workstreams/rl-perf/grouped-mm-fix/ "
    "for the mechanism and the evidence.",
)
@click.option(
    "--reshard-after-forward/--no-reshard-after-forward",
    default=True,
    show_default=True,
    help="FSDP2 reshard_after_forward. On (the baseline) re-gathers all 67B parameters for the "
    "backward; off keeps them resident. This is O2's discriminator: FSDP parameter traffic is as "
    "token-independent as the eager expert loop, and our spans cannot separate them because the "
    "collectives are issued inside forward and backward. If --no-reshard-after-forward moves the "
    "clock, the cost is traffic and use_grouped_mm will buy nothing. Trades memory for traffic; an "
    "OOM is an informative result, not a failed run.",
)
@click.option(
    "--flash-attn/--no-flash-attn",
    default=True,
    show_default=True,
    help="trainer.flash_attn. False is E6's behaviour: eager attention materialises an fp32 "
    "[b,heads,L,L] score tensor (3.42 GiB at L=6775 -- the allocation behind every OOM here) and "
    "computes the full L^2 before masking a 2048 sliding window. After use_grouped_mm removed the "
    "expert cost, policy_forward and fwd_logprobs converged to within 1%, so both are now bound by "
    "the same non-MoE term; this arm tests whether that term is attention. Grug supports it "
    "(tests/gpu/test_grug_flash_attention.py). use_sample_packing stays false either way.",
)
@click.option(
    "--pr488-geometry",
    is_flag=True,
    default=False,
    help="⚠️ TEMPORARY -- REMOVE BEFORE OPENING A PR. This exists only to produce the three-way "
    "comparison table for the PR body; it is a benchmarking shape, not a configuration anyone "
    "should train with, and leaving it in ships a permanent flag for a one-off measurement. "
    "Match Russell Power's PR #488 comparison shape so his published table is a DIRECT "
    "comparand rather than an analogy: 1024+8192 tokens, 64 prompts x 8 samples, 4 policy + 4 "
    "inference nodes. His numbers -- step 190-196s Megatron vs 634s FSDP2, policy_train 26-27 vs "
    "457, generate 134-139 vs 131, fwd_logprobs 6-7 vs 32. We cannot match PP2xEP8 (that is "
    "Megatron parallelism; we run FSDP2 at EP1), so this matches workload and node count, not "
    "sharding -- his FSDP2 column is the honest comparand.",
)
@click.option(
    "--bf16-grad-reduce/--no-bf16-grad-reduce",
    default=True,
    show_default=True,
    help="A9. Set fsdp_config.mixed_precision.reduce_dtype=bf16. ON BY DEFAULT since 2026-09-03: "
    "measured -10.3% on policy_ppo_train at steady state (83.05 -> 74.49 s, smoke geometry), with "
    "policy_backward -15.5% against policy_forward -1.7%. That split is the mechanism check -- "
    "reduce_dtype touches only the gradient reduce-scatter and there are no gradients in forward, "
    "so a single-number speedup could be noise but the asymmetry cannot be. FSDP2 defaults it to "
    "fp32 (fsdp_strategy.py:357-362), so every reduce-scatter moved twice the bytes it needed. "
    "Russell Power's PR488 sets grad_reduce_in_fp32: false explicitly, overriding megatron's own "
    "true default -- so this is a MATCHED setting in the three-way comparison, not an advantage we "
    "hold alone. It reduces peak memory rather than raising it. The one condition on reverting: if "
    "A10 (deferred gradient sync) ever becomes feasible at a smaller geometry, re-examine, because "
    "reduce_dtype then starts governing the accumulator rather than only the wire.",
)
@click.option(
    "--log-ratio-probe",
    is_flag=True,
    default=False,
    help="F25 diagnostic. Repeat the old-logprob forward on the same micro-batch and log the max "
    "delta. Repeats disagreeing means the forward is NONDETERMINISTIC; repeats agreeing while eval "
    "still differs from train means a deterministic eval/train SEAM. Those need opposite fixes, and "
    "no offline probe separates them -- two came back bitwise clean, the second with four layers at "
    "production width, real weights and production tokens-per-expert. One extra forward per "
    "micro-batch, so a diagnostic run rather than a default.",
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
    default=False,
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
    "checkpoint. Do not set 0 here -- the terminal export then fails on the missing marker and the "
    "whole 80-GPU run reports a failed artifact. Pass --telemetry-only instead, which also tells "
    "the launcher to expect no model. See _rl_config.",
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
    collective_diagnostics: bool,
    telemetry_only: bool,
    grouped_mm: bool,
    reshard_after_forward: bool,
    flash_attn: bool,
    pr488_geometry: bool,
    bf16_grad_reduce: bool,
    log_ratio_probe: bool,
    label: str,
) -> ArtifactStep[SkyRLModel] | ArtifactStep[SkyRLTelemetryRun]:
    return build_workflow(
        wandb_entity=wandb_entity,
        max_steps=max_steps,
        ckpt_interval=ckpt_interval,
        bf16_update_mode=bf16_update_mode,
        policy_train_spans=policy_train_spans,
        spans_synchronize=spans_synchronize,
        smoke=smoke,
        debug_distributed=debug_distributed,
        collective_diagnostics=collective_diagnostics,
        telemetry_only=telemetry_only,
        grouped_mm=grouped_mm,
        reshard_after_forward=reshard_after_forward,
        flash_attn=flash_attn,
        pr488_geometry=pr488_geometry,
        bf16_grad_reduce=bf16_grad_reduce,
        log_ratio_probe=log_ratio_probe,
        label=label,
    )


if __name__ == "__main__":
    main()
