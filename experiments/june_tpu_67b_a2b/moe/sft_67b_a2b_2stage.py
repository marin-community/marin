# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-stage chat SFT of the June TPU 67B-A2B Grug MoE (step-42150 cooldown checkpoint).

Stage 1 (``wildchat``): math-weak plain chat -- establishes the chat template / format, no
thinking traces -- initialised (weights-only) from the step-42150 base checkpoint.
Stage 2 (``thinking``): the larger Llama-Nemotron science-reasoning canonical-think dataset --
builds the reasoning region -- chained (weights-only) from Stage 1's output checkpoint.

Order is load-bearing (chat format first, reasoning second). Each stage is 1 epoch, sequence
packing on, completions-masked (assistant span only), chat template = the ported Delphi v0 jinja.

The model architecture is the EXACT cooldown ``_model`` (this is why the launcher lives in the
vendored ``june_tpu_67b_a2b`` tree -- the live ``experiments/grug/moe`` tree's Transformer pytree
is incompatible with the checkpoint). Weights-only init + optimizer/step reset is marin #650 (see
``sft_launch.run_grug_moe_sft_trial`` -> ``train.init_weights_only_from_checkpoint``).

LAUNCH-GATED NUMBERS (finalise before a real launch; see the experiment POLICY/STATE):
  * ``_JOB{1,2}_STEPS`` -- packed 1-epoch step counts = total_tokens / seq_len / batch, read from
    each tokenized cache's shard ledger AFTER a dry-run cache build. Placeholders below.
  * ``_REVISION_*`` -- pin each HF dataset to a 7-char commit for a content-stable fingerprint.
  * ``_SFT_MUON_LR`` / ``_SFT_ADAM_LR`` -- no corpus-stated Grug SFT LR exists; first-pass values,
    validate in the smoke and confirm with the operator.
  * Slice geometry (``_SLICE`` / ``_EXPERT_PARALLEL`` / ``_REPLICA_AXIS`` / ``_BATCH`` / ``_SEQ`` /
    ``_PER_DEVICE_PARALLELISM``) -- 67B full-FT on v6e-32 at long context is memory-tight; confirm
    feasibility (or escalate to v6e-128) before launch.
  * Stage 2 needs the Delphi think/tool tokens as SINGLE ids in the tokenizer; ``marin_tokenizer``
    must be verified/prepared for those (Stage 1 plain chat is fine as-is).

Submit (per stage, us-east5, preemptible; MARIN_PREFIX must be gs://marin-us-east5)::

    cd ~/Documents/marin && source secrets.env
    uv run iris --cluster=marin job run --job-name grug-67b-sft-coord --region us-east5 \\
      --cpu 1 --memory 2G --extra cpu --priority interactive --max-retries 10 --no-wait \\
      -e MARIN_PREFIX gs://marin-us-east5 -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage
"""

import dataclasses
import math
import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import (
    ChatDatasetSpec,
    GrugMoeSFTConfig,
    build_grug_chat_data_config,
    run_grug_moe_sft_trial,
)
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer
from experiments.sft_launcher.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE

# --- Model: the EXACT cooldown architecture (arch parity is required for the weights load) -------
_DIM: int = 2560
_QK_MULT: float = 1.3 * (0.1 * math.log(65_536 / 8_192) + 1.0)  # 1.5703, as trained (YaRN mscale)
_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=65_536)

# --- Slice geometry (LAUNCH-GATED — v6e-32 committed; validate memory or escalate to v6e-128) ----
_SLICE: str = "v6e-32"
_EXPERT_PARALLEL: int = 8  # shard the 256 experts across 8 chips
_REPLICA_AXIS: int = 1  # FSDP across the slice (params too large to replicate)
_SEQ: int = 32_768  # SFT packed length. 64k is the cooled window; 32k for v6e-32 memory headroom.
_BATCH: int = 32  # batch_shards = replica*data*expert = 1*4*8 = 32 -> per_device_parallelism = 1
_PER_DEVICE_PARALLELISM: int = 1

_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=_QK_MULT,
    max_seq_len=_SEQ,  # training seq len = model.max_seq_len; RoPE is position-computed (no param change)
)

# --- Optimizer: FRESH SFT schedule (weights-only init resets it). First-pass LRs — see header. ---
_SFT_MUON_LR: float = 2e-4
_SFT_ADAM_LR: float = 5e-5
_optimizer = GrugMoeMuonHConfig(
    learning_rate=_SFT_MUON_LR,
    adam_lr=_SFT_ADAM_LR,
    beta1=0.9,
    beta2=0.95,
    max_grad_norm=1.0,
    rmsnorm_to_adam=True,  # matches use_array_stacked_blocks=True (as in the cooldown optimizer)
    weight_decay=0.0,
    min_lr_ratio=0.1,
    warmup=0.03,
    lr_schedule="cosine",
)

# --- Datasets (both already OpenAI role/content; no ShareGPT remap) ------------------------------
_REVISION_WILDCHAT: str = "46a5bb5"  # nyu-dice-lab/wildchat50m-rewild-sft-385700 HEAD (2026-07-15)
_REVISION_THINKING: str = "bae881d"  # laion/llama-nemotron-science-reasoning-on-canonical-think-full HEAD
_JOB1_DATASET = ChatDatasetSpec(
    slug="wildchat_386k",
    hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
    revision=_REVISION_WILDCHAT,
    messages_field="conversation",
)
_JOB2_DATASET = ChatDatasetSpec(
    slug="nemotron_science_think",
    hf_dataset_id="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
    revision=_REVISION_THINKING,
    messages_field="messages",
)

# --- Packed 1-epoch step counts (LAUNCH-GATED — recompute from each cache's shard ledger) --------
# steps = total_tokens / seq_len / batch. Placeholders until the dry-run cache build reports tokens.
_JOB1_STEPS: int = 1000
_JOB2_STEPS: int = 2000

# Base checkpoint, co-located with the v6e pool (POLICY: MARIN_PREFIX=gs://marin-us-east5; pre-stage
# step-42150 into this bucket via mirror:// before launch to avoid the CrossRegionGuardedFS block).
_BASE_CKPT: str = (
    "gs://marin-us-east5/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3/"
    "checkpoints/step-42150/"
)

_JOB1_RUN_ID: str = "grug_67b_a2b_sft_s1_wildchat"
_JOB2_RUN_ID: str = "grug_67b_a2b_sft_s2_thinking"


def _tracker(run_id: str, stage: str) -> WandbConfig:
    return WandbConfig(
        project="marin_moe_sft",
        tags=["moe", "june_tpu", "67b_a2b", "sft", stage, f"seq{_SEQ}", _SLICE],
        group="grug-67b-a2b-sft",
        name=None,
    )


def _sft_config(ctx: StepContext, *, run_id: str, dataset: ChatDatasetSpec, init_from: str, steps: int, stage: str):
    data = build_grug_chat_data_config(
        datasets=[dataset],
        tokenizer=marin_tokenizer,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        mixture_block_size=2048,
    )
    return GrugMoeSFTConfig(
        model=_model,
        data=data,
        output_path=ctx.output_path,
        run_id=run_id,
        resources=ctx.runtime_arg("train_resources"),
        steps=steps,
        batch_size=_BATCH,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=_tracker(run_id, stage),
        optimizer=_optimizer,
        init_from_path=init_from,
        expert_parallel=_EXPERT_PARALLEL,
        per_device_parallelism=_PER_DEVICE_PARALLELISM,
        save_interval_minutes=30,
        checkpoint_keep=[{"every": 1000}],
        grug_trainer=GrugTrainerConfig(
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
            replica_axis_size=_REPLICA_AXIS,
        ),
        eval=None,
    )


def build_job1(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Stage 1 — wildchat plain chat, weights-only init from the step-42150 base."""

    def build_config(ctx: StepContext) -> GrugMoeSFTConfig:
        return _sft_config(
            ctx, run_id=_JOB1_RUN_ID, dataset=_JOB1_DATASET, init_from=_BASE_CKPT, steps=_JOB1_STEPS, stage="s1_chat"
        )

    return ArtifactStep(
        name=user_namespaced_name(f"grug/{_JOB1_RUN_ID}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_sft_trial,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": ResourceConfig.with_tpu(_SLICE, preemptible=True)},
    )


def build_job2(job1: ArtifactStep[LevanterCheckpoint], *, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Stage 2 — thinking dataset, weights-only init CHAINED from Stage 1's output checkpoint."""

    def build_config(ctx: StepContext) -> GrugMoeSFTConfig:
        init_from = os.path.join(ctx.artifact_path(job1), "checkpoints")
        return _sft_config(
            ctx, run_id=_JOB2_RUN_ID, dataset=_JOB2_DATASET, init_from=init_from, steps=_JOB2_STEPS, stage="s2_think"
        )

    return ArtifactStep(
        name=user_namespaced_name(f"grug/{_JOB2_RUN_ID}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_sft_trial,
        build_config=build_config,
        deps=(job1,),
        runtime_args={"train_resources": ResourceConfig.with_tpu(_SLICE, preemptible=True)},
    )


if __name__ == "__main__":
    job1 = build_job1()
    job2 = build_job2(job1)
    StepRunner().run([job2.lower()])
