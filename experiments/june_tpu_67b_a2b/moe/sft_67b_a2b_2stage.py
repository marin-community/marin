# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-stage chat SFT of the June TPU 67B-A2B Grug MoE (step-42150 cooldown checkpoint).

An experiment on the general ``experiments.sft`` launcher: each stage composes the shared
``sft_step`` (dataset transforms + chat template + spec + CLI) with a ``GrugModel`` model source
(native weights-only init + the ring-EP ``run_grug`` backend). Model and data are independent
inputs — swap ``_JOB{1,2}_DATASET`` for a different mixture, or the ``GrugModel`` for another
checkpoint, without touching the launcher.

Stage 1 (``wildchat``): math-weak plain chat -- establishes the chat template / format, no
thinking traces -- initialised (weights-only) from the step-42150 base checkpoint.
Stage 2 (``thinking``): the larger Llama-Nemotron science-reasoning canonical-think dataset --
builds the reasoning region -- chained (weights-only) from Stage 1's output checkpoint (Stage 1's
``ArtifactStep`` is the Stage 2 ``GrugModel.init_from``, so the chain is a real graph dependency).

The order matters (chat format first, reasoning second). Each stage is 1 epoch, sequence
packing on, completions-masked (assistant span only), chat template = the shared Delphi v0 jinja.

The model architecture is the exact cooldown ``_model`` (this is why the launcher lives in the
vendored ``june_tpu_67b_a2b`` tree -- the live ``experiments/grug/moe`` tree's Transformer pytree
is incompatible with the checkpoint). Weights-only init + optimizer/step reset is marin #650 (see
``sft_launch.run_grug_moe_sft_trial`` -> ``train.init_weights_only_from_checkpoint``).

Launch-gated numbers (finalise before a real launch; see the experiment POLICY/STATE):
  * Step counts are no longer hand-set: Job1/Job2 are ``num_train_epochs=1`` and ``sft_step`` resolves
    the packed-epoch step count from the ``chat_tokenize`` cache's token total at run time (marin #7244).
  * ``_REVISION_*`` -- pin each HF dataset to a 7-char commit for a content-stable fingerprint.
  * GPU mesh geometry (``_NODES`` / ``_EXPERT_PARALLEL`` / ``_REPLICA_AXIS`` / ``_BATCH`` / ``_SEQ`` /
    ``_PER_DEVICE_PARALLELISM``) -- 67B full-FT on H100x8 nodes at long context is memory-tight;
    confirm feasibility (drop ``_SEQ`` or raise ``_NODES``) before launch.
  * Stage 2 needs the Delphi think/tool tokens as single ids in the tokenizer; ``marin_tokenizer``
    must be verified/prepared for those (Stage 1 plain chat is fine as-is).

Compute = CoreWeave ``cw-us-east-02a`` H100 GPU cluster (8x H100-80GB + InfiniBand per node), the
FSDP + ring-EP JAX/XLA path (mirrors ``experiments/grug/moe/launch_cw_scale.py``). The base
checkpoint is read in-cluster from the CW ``s3://marin-us-east-02a`` (LOTA) mirror -- no cross-region
port needed. The coordinator must run in-cluster (the Mac can't reach cwlota.com).

Submit (per stage, cw-us-east-02a, preemptible; MARIN_PREFIX must be s3://marin-us-east-02a/marin;
AWS creds auto-injected in-pod via the iris-task-env secret -- do not forward AWS_*)::

    cd ~/Documents/marin && source secrets.env   # or "$DC_AGENT_SECRET_ENV"
    export KUBECONFIG=~/.kube/coreweave-iris-gpu
    uv run iris --cluster=cw-us-east-02a job run --job-name grug-67b-sft-smoke-coord \\
      --cpu 1 --memory 2G --extra cpu --priority interactive --max-retries 10 --no-wait \\
      -e MARIN_PREFIX s3://marin-us-east-02a/marin -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage --stage smoke --version dev --run

Drop ``--run`` to print the lowered plan (and each artifact's resolved ``name@version``) without
launching; ``--version`` is required (pass ``--version dev`` to iterate).
"""

import dataclasses
import math

import click
from fray.cluster import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugModel
from experiments.marin_tokenizer import marin_tokenizer
from experiments.sft.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE
from experiments.sft.launcher import DatasetSpec, SFTSpec, sft_step

_WANDB_PROJECT = "marin_moe_sft"

# --- Model: the exact cooldown architecture (arch parity is required for the weights load) -------
_DIM: int = 2560
_QK_MULT: float = 1.3 * (0.1 * math.log(65_536 / 8_192) + 1.0)  # 1.5703, as trained (YaRN mscale)
_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=65_536)

# --- GPU mesh geometry (compute pivot 2026-07-16: CoreWeave cw-us-east-02a H100x8 nodes) ----------
# Each node = 8x H100-80GB + InfiniBand. Params are FSDP-sharded over the cross-node ``data`` axis;
# the 256 routed experts are sharded 8-way over the intra-node NVLink ``expert`` axis (ring-EP).
# Batch is sharded over (replica, data, expert); batch_shards = replica*data*expert = 1*N*8 = 8N
# where N=_NODES (data absorbs the remaining 8*N/expert = N devices). _BATCH must be a multiple of 8N.
# Geometry (2026-07-16 decision): AdamH (not Muon) + tensor/model parallelism. Muon's Newton-Schulz
# workspace is a ~21GiB replicated per-device floor that never shards (num_layers=26 is coprime to any
# expert-inclusive batch_shards), so it OOMs on H100 at any node count -> switched to AdamH (no NS).
# Path B (2026-07-16 operator decision): pure data-parallel, model_axis=1. Tensor/model parallelism is
# architecturally impossible for this model (model_axis>1 must divide num_kv_heads=5 [prime -> {1,5}] and
# vocab_size=128256 [vocab-parallel embed/lm_head], and 128256 % 5 != 0 -> no valid width >1). Since bs=8
# would require TP (data=1), we run data-parallel with bs=8N instead. Mesh = (replica=1, data=N, expert=8,
# model=1) on _NODES*8 GPUs; batch_shards = replica*data*expert = 8N; _BATCH = 8N -> per_device = 1.
# The seq32k per-seq activation (~43 GiB, pd=1) doesn't shard across DP nodes, so N=8 (bs=64) is the
# smallest gang that fits (~68 GiB/dev, AdamH fp32 + cut-CE); N<=4 OOMs. See GEOMETRY_ANALYSIS.md.
_NODES: int = 8  # full-run gang: 8x H100x8 = 64 GPUs (data-parallel)
_SMOKE_NODES: int = 8  # smoke at the same target geometry (the real HBM test at ~68 GiB/dev prediction)
_EXPERT_PARALLEL: int = 8  # shard the 256 experts across the 8 intra-node GPUs (ring-EP over NVLink)
_MODEL_PARALLEL: int = 1  # no tensor parallelism (architecturally impossible; see header)
_REPLICA_AXIS: int = 1  # pure FSDP (one model copy sharded over all 64 GPUs; no cross-node replicate)
_SEQ: int = 32_768  # full-run SFT packed length (operator target ctx_len=32k)
_SMOKE_SEQ: int = 32_768  # smoke at the target seq len
_BATCH: int = 64  # global batch = 8N = batch_shards (replica*data*expert = 1*8*8); per_device -> 1
_SMOKE_BATCH: int = 64  # smoke at the target global batch
_PER_DEVICE_PARALLELISM: int = -1  # auto: Levanter derives batch/(batch_shards); grug real pd = 64/64 = 1

# GrugModel.build_train_config pins model.max_seq_len to each stage's seq_len, so _model carries the
# base cooldown arch and the run's seq_len is set on the spec (max_seq_len below is the default stage).
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=_QK_MULT,
    max_seq_len=_SEQ,  # training seq len = model.max_seq_len; RoPE is position-computed (no param change)
    # H100 GPU attention backend. gpu_fa4_cute (not gpu_fa4_thd) because sliding_window=2048 is a short
    # window; thd only handles full-causal windows (canary_ferry.py maps thd -> window=2*seq to fake it).
    attention_implementation="gpu_fa4_cute",
    # Blocked-vocab (cut) cross-entropy: avoids materializing the [tokens, vocab] logits tile at seq32k
    # (~15.7 GiB/dev on the default full-logits GPU path).
    ce_implementation="batched_xla",
)

# --- Optimizer: AdamH (not Muon). Muon's Newton-Schulz workspace is a ~21GiB replicated per-device
# floor that never shards -> OOMs on H100 (marin #6693). AdamH (grug_moe_adamh_v2) has no NS
# workspace (elementwise m/v moments). Fresh SFT schedule (weights-only init resets it). First-pass LRs. ---
_SFT_ADAMH_LR: float = 5e-5  # adamh group (attn/dense matrices) + expert group (expert_lr=None -> this)
_SFT_ADAM_LR: float = 5e-5  # adam group (norms / router / embeddings)
_optimizer = GrugMoeAdamHConfig(
    learning_rate=_SFT_ADAMH_LR,
    adam_lr=_SFT_ADAM_LR,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    max_grad_norm=1.0,
    weight_decay=0.0,
    min_lr_ratio=0.1,
    warmup=0.03,
    lr_schedule="cosine",
)

# --- Datasets (both already OpenAI role/content; multi_turn_adapter canonicalizes columns) --------
_REVISION_WILDCHAT: str = "46a5bb5"  # nyu-dice-lab/wildchat50m-rewild-sft-385700 HEAD (2026-07-15)
_REVISION_THINKING: str = "bae881d"  # laion/llama-nemotron-science-reasoning-on-canonical-think-full HEAD
_JOB1_DATASET = DatasetSpec(
    slug="wildchat_386k",
    hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
    revision=_REVISION_WILDCHAT,
    adapter_kwargs=dict(conversation_column="conversation"),  # role/content, user/assistant defaults
    weight=1.0,
)
_JOB2_DATASET = DatasetSpec(
    slug="nemotron_science_think",
    hf_dataset_id="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
    revision=_REVISION_THINKING,
    adapter_kwargs=dict(),  # multi_turn_adapter defaults: messages / role / user / assistant
    weight=1.0,
)

# Job1/Job2 are one packed epoch (num_train_epochs=1); sft_step resolves the concrete step count from
# the chat cache's token total at run time (marin #7244), so there is no hand-calibrated count. For
# reference, the wildchat_386k chat cache is ~538.9M tokens -> 256.96 -> 257 steps at seq32768/bs64.
_SMOKE_STEPS: int = 8  # validation: clear the first jit_train_step at the target geometry and bank a few steps

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
_SMOKE_RUN_ID: str = "grug_67b_a2b_sft_smoke"


def _gpu_resources(nodes: int) -> ResourceConfig:
    return ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="512g", disk="256g", replicas=nodes, preemptible=True)


def _grug_source(
    init_from: str | ArtifactStep,
    *,
    stage: str,
    seq: int,
    save_interval_minutes: int = 30,
    checkpoint_keep: list[dict] | None = None,
) -> GrugModel:
    """The Grug 67B model source for one stage (native weights-only init from ``init_from``)."""
    return GrugModel(
        model=_model,
        tokenizer_path=marin_tokenizer,
        init_from=init_from,
        expert_parallel=_EXPERT_PARALLEL,
        model_axis=_MODEL_PARALLEL,
        replica_axis=_REPLICA_AXIS,
        per_device_parallelism=_PER_DEVICE_PARALLELISM,
        z_loss_weight=1e-4,
        ema_beta=None,
        log_every=1,
        save_interval_minutes=save_interval_minutes,
        checkpoint_keep=checkpoint_keep if checkpoint_keep is not None else [{"every": 1000}],
        wandb_tags=["moe", "june_tpu", "67b_a2b", "sft", stage, f"seq{seq}", "cw-h100"],
        wandb_group="grug-67b-a2b-sft",
    )


def _spec(
    *,
    name: str,
    version: str,
    dataset: DatasetSpec,
    model_source: GrugModel,
    steps: int | None = None,
    epochs: int | None = None,
    seq: int = _SEQ,
    batch: int = _BATCH,
) -> SFTSpec:
    return SFTSpec(
        name=name,
        version=version,
        model=model_source,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        datasets=[dataset],
        optimizer=_optimizer,
        seq_len=seq,
        batch_size=batch,
        num_train_steps=steps,
        num_train_epochs=epochs,
        wandb_project=_WANDB_PROJECT,
    )


def build_job1(version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Stage 1 -- wildchat plain chat, weights-only init from the step-42150 base (one packed epoch)."""
    step_name = f"grug/{_JOB1_RUN_ID}"
    version = resolve_version(step_name, version)
    spec = _spec(
        name=user_namespaced_name(step_name, version),
        version=version,
        dataset=_JOB1_DATASET,
        model_source=_grug_source(_BASE_CKPT, stage="s1_chat", seq=_SEQ),
        epochs=1,
    )
    return sft_step(spec, _gpu_resources(_NODES))


def build_job2(job1: ArtifactStep[LevanterCheckpoint], version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Stage 2 -- thinking dataset, weights-only init chained from Stage 1 (one packed epoch)."""
    step_name = f"grug/{_JOB2_RUN_ID}"
    version = resolve_version(step_name, version)
    spec = _spec(
        name=user_namespaced_name(step_name, version),
        version=version,
        dataset=_JOB2_DATASET,
        model_source=_grug_source(job1, stage="s2_think", seq=_SEQ),
        epochs=1,
    )
    return sft_step(spec, _gpu_resources(_NODES))


def build_smoke(version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Stage-5 smoke: the real 67B at the target Job1 geometry -- 8x H100x8 nodes (cw-us-east-02a),
    AdamH, expert=8, model=1 (data-parallel), replica=1, bs=64, seq=32768, per_device=1, few steps +
    a mid-run native checkpoint save. Validates native S3 ckpt load -> chat+packing -> weights-only
    init (step starts at 0) -> first jit_train_step with no OOM (AdamH has no ~21GiB Muon-NS floor)
    -> loss finite -> save, before committing to the 1-epoch Job1."""
    step_name = f"grug/{_SMOKE_RUN_ID}"
    version = resolve_version(step_name, version)
    spec = _spec(
        name=user_namespaced_name(step_name, version),
        version=version,
        dataset=_JOB1_DATASET,
        # Save a native checkpoint mid-smoke so the save path (and resume-on-preempt) is exercised.
        model_source=_grug_source(
            _BASE_CKPT, stage="smoke", seq=_SMOKE_SEQ, save_interval_minutes=5, checkpoint_keep=[{"every": 20}]
        ),
        steps=_SMOKE_STEPS,  # explicit few-step count, not a full epoch
        seq=_SMOKE_SEQ,
        batch=_SMOKE_BATCH,
    )
    return sft_step(spec, _gpu_resources(_SMOKE_NODES))


@click.command()
@click.option(
    "--stage",
    type=click.Choice(["smoke", "job1", "2stage"]),
    default="2stage",
    show_default=True,
    help="Which stage(s) to build: smoke | job1 | 2stage (Stage 1 -> Stage 2 chained).",
)
@build_options
def main(stage: str) -> ArtifactStep[LevanterCheckpoint]:
    if stage == "smoke":
        return build_smoke()
    if stage == "job1":
        return build_job1()
    return build_job2(build_job1())


if __name__ == "__main__":
    main()
