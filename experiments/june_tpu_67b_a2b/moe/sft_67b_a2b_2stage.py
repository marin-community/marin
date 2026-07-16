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
  * GPU mesh geometry (``_NODES`` / ``_EXPERT_PARALLEL`` / ``_REPLICA_AXIS`` / ``_BATCH`` / ``_SEQ`` /
    ``_PER_DEVICE_PARALLELISM``) -- 67B full-FT on H100x8 nodes at long context is memory-tight;
    confirm feasibility (drop ``_SEQ`` or raise ``_NODES``) before launch.
  * Stage 2 needs the Delphi think/tool tokens as SINGLE ids in the tokenizer; ``marin_tokenizer``
    must be verified/prepared for those (Stage 1 plain chat is fine as-is).

Compute = CoreWeave ``cw-us-east-02a`` H100 GPU cluster (8x H100-80GB + InfiniBand per node), the
FSDP + ring-EP JAX/XLA path (mirrors ``experiments/grug/moe/launch_cw_scale.py``). The base
checkpoint is read in-cluster from the CW ``s3://marin-us-east-02a`` (LOTA) mirror -- no cross-region
port needed. The coordinator MUST run in-cluster (the Mac can't reach cwlota.com).

Submit (per stage, cw-us-east-02a, preemptible; MARIN_PREFIX must be s3://marin-us-east-02a/marin;
AWS creds auto-injected in-pod via the iris-task-env secret -- do NOT forward AWS_*)::

    cd ~/Documents/marin && source secrets.env   # or "$DC_AGENT_SECRET_ENV"
    export KUBECONFIG=~/.kube/coreweave-iris-gpu
    uv run iris --cluster=cw-us-east-02a job run --job-name grug-67b-sft-smoke-coord \\
      --cpu 1 --memory 2G --extra cpu --priority interactive --max-retries 10 --no-wait \\
      -e MARIN_PREFIX s3://marin-us-east-02a/marin -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage smoke
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
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
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

# --- GPU mesh geometry (COMPUTE PIVOT 2026-07-16: CoreWeave cw-us-east-02a H100x8 nodes) ----------
# Each node = 8x H100-80GB + InfiniBand. Params are FSDP-sharded over the cross-node ``data`` axis;
# the 256 routed experts are sharded 8-way over the intra-node NVLink ``expert`` axis (ring-EP).
# Batch is sharded over (replica, data, expert); batch_shards = replica*data*expert = 1*N*8 = 8N
# where N=_NODES (data absorbs the remaining 8*N/expert = N devices). _BATCH must be a multiple of 8N.
_NODES: int = 4  # full-run gang size (H100x8 nodes) -> 32 GPUs; 8N=32 -> _BATCH multiple of 32
_SMOKE_NODES: int = 2  # smoke gang (2 nodes) -> 16 GPUs; 8N=16 -> _SMOKE_BATCH multiple of 16
_EXPERT_PARALLEL: int = 8  # shard the 256 experts across the 8 intra-node GPUs (ring-EP over NVLink)
_REPLICA_AXIS: int = 1  # pure FSDP (one model copy sharded over all N*8 GPUs; no cross-node replicate)
_SEQ: int = 8_192  # full-run SFT packed length (64k cooled window; 8k = near-zero padding + tput on H100)
_SMOKE_SEQ: int = 4_096  # shorter packed length for the 2-node smoke (memory headroom + fast steps)
_BATCH: int = 64  # full: multiple of 8N=32 (per_device_parallelism auto-derives to 2)
_SMOKE_BATCH: int = 16  # smoke: multiple of 8N=16 (per_device auto-derives to 1). Was 32: OOM'd on first
# jit_train_step (24.36GiB step tile atop ~61GiB persistent state > 80GB H100). Halving batch halves the
# activation tile (~24->~12GiB -> 61+12=73<80) while KEEPING seq=4096 for the thinking-content goal.
_PER_DEVICE_PARALLELISM: int = -1  # auto: Levanter derives batch/(batch_shards) given the mesh

_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=_QK_MULT,
    max_seq_len=_SEQ,  # training seq len = model.max_seq_len; RoPE is position-computed (no param change)
    # H100 GPU attention backend. gpu_fa4_cute (NOT gpu_fa4_thd) because sliding_window=2048 is a SHORT
    # window; thd only handles full-causal windows (canary_ferry.py maps thd -> window=2*seq to fake it).
    attention_implementation="gpu_fa4_cute",
)
_smoke_model = dataclasses.replace(_model, max_seq_len=_SMOKE_SEQ)  # 2-node smoke: 4k packed length

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

# Base checkpoint, read in-cluster from the CoreWeave s3://marin-us-east-02a (LOTA) mirror. No
# cross-region port needed on CW (contrast the TPU-era mirror:// pre-stage). AWS creds are injected
# in-pod via the iris-task-env secret; the tensorstore S3 reader lists step-42150 directly.
_BASE_CKPT: str = (
    "s3://marin-us-east-02a/marin/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3/"
    "checkpoints/step-42150/"
)

_JOB1_RUN_ID: str = "grug_67b_a2b_sft_s1_wildchat"
_JOB2_RUN_ID: str = "grug_67b_a2b_sft_s2_thinking"


def _tracker(run_id: str, stage: str, seq: int) -> WandbConfig:
    return WandbConfig(
        project="marin_moe_sft",
        tags=["moe", "june_tpu", "67b_a2b", "sft", stage, f"seq{seq}", "cw-h100"],
        group="grug-67b-a2b-sft",
        name=None,
    )


def _sft_config(
    ctx: StepContext,
    *,
    run_id: str,
    dataset: ChatDatasetSpec,
    init_from: str,
    steps: int,
    stage: str,
    model: GrugModelConfig = _model,
    batch_size: int = _BATCH,
):
    data = build_grug_chat_data_config(
        datasets=[dataset],
        tokenizer=marin_tokenizer,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        mixture_block_size=2048,
    )
    return GrugMoeSFTConfig(
        model=model,
        data=data,
        output_path=ctx.output_path,
        run_id=run_id,
        resources=ctx.runtime_arg("train_resources"),
        steps=steps,
        batch_size=batch_size,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=_tracker(run_id, stage, model.max_seq_len),
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
        runtime_args={
            "train_resources": ResourceConfig.with_gpu(
                "H100", count=8, cpu=32, ram="512g", disk="256g", replicas=_NODES, preemptible=True
            )
        },
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
        runtime_args={
            "train_resources": ResourceConfig.with_gpu(
                "H100", count=8, cpu=32, ram="512g", disk="256g", replicas=_NODES, preemptible=True
            )
        },
    )


_SMOKE_STEPS: int = 40


def build_smoke(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Stage-5 smoke: the real 67B on 2x H100x8 nodes (cw-us-east-02a), seq 4k, batch 32, wildchat,
    weights-only init from step-42150, 40 steps + a mid-run native checkpoint save. Validates native
    S3 ckpt load -> chat+packing -> weights-only init (step starts at 0) -> loss finite/downward ->
    save, before committing to the 1-epoch Job1. (HF export is validated separately via the grug->HF
    converter on the saved checkpoint.)"""

    def build_config(ctx: StepContext) -> GrugMoeSFTConfig:
        cfg = _sft_config(
            ctx,
            run_id="grug_67b_a2b_sft_smoke",
            dataset=_JOB1_DATASET,
            init_from=_BASE_CKPT,
            steps=_SMOKE_STEPS,
            stage="smoke",
            model=_smoke_model,  # seq 4k
            batch_size=_SMOKE_BATCH,  # 32 = multiple of 8*_SMOKE_NODES(16)
        )
        # Save a native checkpoint mid-smoke so the save path (and resume-on-preempt) is exercised.
        return dataclasses.replace(cfg, save_interval_minutes=5, checkpoint_keep=[{"every": 20}])

    return ArtifactStep(
        name=user_namespaced_name("grug/grug_67b_a2b_sft_smoke", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_sft_trial,
        build_config=build_config,
        deps=(),
        runtime_args={
            "train_resources": ResourceConfig.with_gpu(
                "H100", count=8, cpu=32, ram="512g", disk="256g", replicas=_SMOKE_NODES, preemptible=True
            )
        },
    )


if __name__ == "__main__":
    import sys

    # Usage: python -m ...sft_67b_a2b_2stage [smoke|job1|2stage] [version]
    # The optional version (default "dev") namespaces the executor output dir, so a re-run under a
    # fresh version gets a distinct output path (avoids racing a prior coordinator's StepRunner on
    # the same step-status file).
    which = sys.argv[1] if len(sys.argv) > 1 else "2stage"
    version = sys.argv[2] if len(sys.argv) > 2 else "dev"
    if which == "smoke":
        StepRunner().run([build_smoke(version=version).lower()])
    elif which == "job1":
        StepRunner().run([build_job1(version=version).lower()])
    elif which == "2stage":
        job1 = build_job1(version=version)
        job2 = build_job2(job1, version=version)
        StepRunner().run([job2.lower()])
    else:
        raise SystemExit(f"unknown target {which!r}; use one of: smoke | job1 | 2stage")
