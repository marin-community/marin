# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""General SFT launcher — a parameterized ``sft_step`` on marin's ``ArtifactStep`` flow.

The chat template is an INPUT (``SFTSpec.chat_template``), not a hardcoded default: this
launcher trains ANY chat-SFT recipe. Delphi is one config (``configs/delphi_1e22.py``).

WHY A CUSTOM STEP (not ``marin.experiment.train.train_lm``)
----------------------------------------------------------
Two gaps in the high-level helpers, both on marin HEAD 981b1a56f6:

  1. ``marin.experiment.data.tokenized`` hardcodes ``TextLmDatasetFormat`` — it cannot emit
     a chat ``ChatLmDatasetFormat`` component (template + completions-only masking).
  2. ``marin.experiment.train.train_lm`` inits from a checkpoint handle
     (``init_from`` -> ``initialize_from_checkpoint_path``), not ``initialize_from_hf`` +
     ``use_hf_model_config`` (the SFT-of-an-HF-checkpoint path).

So this step assembles a full ``TrainLmConfig`` (chat components carrying ``spec.chat_template``
+ ``initialize_from_hf``) as the identity-bearing literal and dispatches training via
``remote(run_levanter_train_lm, resources=...)`` — generalizing the validated scaffold from the
Delphi Levanter-SFT parity experiment (single-node smoke job 49041380).

The dataset side uses the NATIVE ``transform_dataset_step`` + ``multi_turn_adapter``
(``experiments/datasets/instruction.py``) to canonicalize each source (ShareGPT ``from``/``value``
or OpenAI ``role``/``content``) into an OpenAI-messages cache the chat tokenizer reads.

  identity/graph = ArtifactStep (cluster-agnostic);
  execution/mesh = cluster-specific (``remote()`` -> Fray -> ``initialize_iris_jax`` on Iris/CW/TPU;
  ``srun`` -> ``LevanterSlurmCluster`` on SLURM). On Iris/TPU the single-driver ``remote()`` dispatch
  makes the ``StepRunner`` per-step lock a no-op (no ``DELPHI_DIRECT_RUN`` bypass needed).

RUN (post container/dataset staging; CPU coordinator dispatches the training sub-job):

    uv run iris --cluster=marin job run --job-name sft-coord --region us-east5 \
      --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \
      -e MARIN_PREFIX gs://marin-us-east5 -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
      -- python -m experiments.sft_launcher.marin_sft_launcher

  CoreWeave H100: add ``--target-cluster cw-us-east-02a`` + ``-e MARIN_PREFIX s3://marin-us-east-02a/marin``
  and set the config's ``resources`` to ``ResourceConfig.with_gpu("H100", count=8, regions=[ANY_REGION])``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import draccus
from fray.types import GpuConfig, ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    run_levanter_train_lm,
)

from experiments.datasets.instruction import (
    InstructionDatasetConfig,
    multi_turn_adapter,
    transform_dataset_step,
)

# ---------------------------------------------------------------------------------------------
# Spec dataclasses — every identity-bearing SFT decision, none hardcoded to a model family.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """One instruction source in the SFT mixture.

    ``adapter_kwargs`` are forwarded to ``multi_turn_adapter(**adapter_kwargs)`` so a source's
    ShareGPT/OpenAI schema (column + role/content tag conventions) is declared per-source. ``weight``
    is the (un-normalized) mixture weight; the training data config normalizes across the mix.
    """

    slug: str  # a short mixture key, e.g. "magpie" / "warmup"
    hf_dataset_id: str  # e.g. "Magpie-Align/Magpie-Llama-3.3-Pro-500K-Filtered"
    revision: str  # 7-char commit pin (fingerprint stability)
    adapter_kwargs: Mapping[str, object]  # -> multi_turn_adapter(**adapter_kwargs)
    weight: float


@dataclass(frozen=True)
class SFTSpec:
    """A full chat-SFT run. The chat template is a parameter — Delphi is just one instance."""

    name: str  # artifact name, e.g. "checkpoints/delphi-1e22-magpie-warmup-levanter-sft"
    version: str  # CALVER "2026.07.15"; a "-dev" suffix opts out of the cache (always rebuild)
    model_ref: str  # HF id / prepared-ckpt dir -> initialize_from_hf + reference_checkpoint
    tokenizer_path: str  # tokenizer dir (Delphi: the PREPARED single-id think/tool tokenizer)
    chat_template: str  # THE PLUGGABLE PARAMETER — any jinja; MUST carry a {% generation %} block
    datasets: Sequence[DatasetSpec]  # the instruction mixture (+ any warmup source)
    resources: ResourceConfig  # TPU with_tpu(...) | CoreWeave with_gpu("H100", regions=[ANY_REGION])
    seq_len: int = 4096
    pack: bool = True  # ChatLmDatasetFormat packs by default (num_train_steps must count PACKED examples)
    lr: float = 1e-5
    batch_size: int = 16
    num_train_steps: int = 5307  # PACKED 1-epoch: total_tokens/seq_len / weight / batch
    beta2: float = 0.98
    warmup_ratio: float = 0.1
    eos_token_ids: Sequence[int] = (128001, 128009)  # Delphi: <|end_of_text|> + <|eot_id|>
    wandb_project: str = "marin-sft-launcher"


# ---------------------------------------------------------------------------------------------
# TrainLmConfig assembly — generalized from the validated parity build_train_config.
# ---------------------------------------------------------------------------------------------


def _chat_format(spec: SFTSpec) -> dict:
    """ChatLmDatasetFormat (registry 'chat') carrying the spec's template + completions-only mask."""
    fmt = {
        "type": "chat",
        "messages_field": "messages",
        "chat_template": spec.chat_template,
        "mask_user_turns": True,  # supervise only the assistant {% generation %} span
    }
    if not spec.pack:
        fmt["pack"] = "pad"  # default (unset) packs; "pad" forces unpacked (parity-neutral, slower)
    return fmt


def _data_block(spec: SFTSpec, dep_paths: Sequence[str]) -> dict:
    """LmDataConfig dict: one cache-backed chat component per dataset, weighted by spec.datasets."""
    components: dict[str, dict] = {}
    weights: dict[str, float] = {}
    for d, cache_dir in zip(spec.datasets, dep_paths, strict=True):
        components[d.slug] = {
            "source": {"type": "url", "train_urls": [os.path.join(cache_dir, "**/*.jsonl.gz")]},
            "cache_dir": cache_dir,
            "format": _chat_format(spec),
            "split": "train",
        }
        weights[d.slug] = d.weight
    return {
        "tokenizer": spec.tokenizer_path,
        "chat_template": spec.chat_template,  # data-level default; component format overrides anyway
        "enforce_eos": True,
        "auto_build_caches": True,
        "components": components,
        "train_weights": weights,
        "stop_strategy": "restart",
        "mixture_block_size": 2048,
    }


def _optimizer_block(spec: SFTSpec) -> dict:
    return {
        "type": "adam",
        "learning_rate": spec.lr,
        "beta1": 0.9,
        "beta2": spec.beta2,
        "epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "lr_schedule": "cosine",
        "warmup": spec.warmup_ratio,
        "min_lr_ratio": 0.0,
    }


def build_sft_train_config(spec: SFTSpec, dep_paths: Sequence[str]) -> TrainLmConfig:
    """Assemble the identity-bearing TrainLmConfig for ``spec`` (full-FT, chat, init-from-HF).

    ``dep_paths`` are the resolved output paths of the ``transform_dataset_step`` deps, aligned
    with ``spec.datasets`` (pulled from the ``StepContext`` at build time so the components read
    region-aware artifact paths rather than hardcoded URLs).
    """
    # cuda_async is a GPU-only PJRT allocator (the resume-OOM defrag fix, marin #7115). It is NOT
    # inert on TPU: passing `allocator:...` to PJRT_Client_Create ABORTS the TPU backend
    # ("Unexpected option name passed to PJRT_Client_Create: allocator" -> JAX falls back to CPU ->
    # "No accelerator found"). So set the PJRT client-create options ONLY on GPU.
    jax_config = {
        "jax_threefry_partitionable": True,
        "jax_softmax_custom_jvp": True,
    }
    if isinstance(spec.resources.device, GpuConfig):
        jax_config["jax_pjrt_client_create_options"] = "allocator:cuda_async"
    cfg = {
        "data": _data_block(spec, dep_paths),
        "model": {
            "type": "qwen3",
            "reference_checkpoint": spec.model_ref,
            "max_seq_len": spec.seq_len,
            "tie_word_embeddings": False,
            "use_bias": False,
            # use_hf_model_config=True re-derives the arch from the checkpoint; these fall to the
            # Llama defaults (both True) -> the memory-safe remat-on / scanned path. Kept as intent.
            "gradient_checkpointing": True,
            "scan_layers": True,
            "attn_backend": "nvte",  # NVTE flash on GPU; re-derived arch keeps flash default on both
        },
        "optimizer": _optimizer_block(spec),
        "trainer": {
            "train_batch_size": spec.batch_size,
            "num_train_steps": spec.num_train_steps,
            "steps_per_eval": 500,
            # Re-lists the two DEFAULT_JAX_CONFIG keys (supplying jax_config REPLACES them); the
            # cuda_async allocator (marin #7115 resume-OOM defrag) is added GPU-ONLY above — it
            # aborts the TPU PJRT backend if set on TPU.
            "jax_config": jax_config,
            "mp": "p=f32,c=bfloat16",
            "per_device_parallelism": -1,  # auto microbatch; raise on OOM (math-equivalent)
            "tracker": {
                "type": "wandb",
                "project": spec.wandb_project,
                "tags": ["sft", "levanter", spec.name.split("/")[-1]],
            },
            "checkpointer": {
                "base_path": "checkpoints/",
                "save_interval": "30m",
                "keep": [{"every": 5000}],
                "append_run_id_to_base_path": True,
            },
        },
        "train_seq_len": spec.seq_len,
        "initialize_from_hf": spec.model_ref,
        "use_hf_model_config": True,
        # Qwen (and others) pad the embedding vocab past the tokenizer's for TPU efficiency
        # (Qwen3: model 151936 vs tokenizer 151669). Without this, the Vocab axis is built from
        # len(tokenizer) while the checkpoint embedding is larger -> a pytree Vocab-size mismatch at
        # train_step trace. Pad the tokenizer up to the model's vocab (no-op when they already match,
        # e.g. the Delphi prepared tokenizer).
        "pad_tokenizer_to_match_model": True,
        "hf_save_steps": spec.num_train_steps,  # one HF export at the end (round-trip)
        "hf_generation_eos_token_ids": list(spec.eos_token_ids),
        "z_loss_weight": None,
    }
    return draccus.decode(TrainLmConfig, cfg)


# ---------------------------------------------------------------------------------------------
# The ArtifactStep — the general launcher.
# ---------------------------------------------------------------------------------------------


def _dataset_deps(spec: SFTSpec) -> tuple[ArtifactStep, ...]:
    """One native ShareGPT/OpenAI -> canonical transform per source (schema from adapter_kwargs)."""
    return tuple(
        transform_dataset_step(
            InstructionDatasetConfig(
                hf_dataset_id=d.hf_dataset_id,
                revision=d.revision,
                adapter=multi_turn_adapter(**dict(d.adapter_kwargs)),
                metadata_columns=[],
                name=d.slug,
            )
        )
        for d in spec.datasets
    )


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    """The ArtifactStep's ``run``: dispatch the config as its own Fray training job.

    Identical idiom to ``marin.experiment.train._train_job`` — ``remote(...)`` submits
    ``run_levanter_train_lm`` and blocks; the SPMD launch is the Fray backend's job, not the
    (locked) StepRunner step's.
    """
    remote(run_levanter_train_lm, resources=pod_config.resources)(pod_config)


def sft_step(spec: SFTSpec) -> ArtifactStep[LevanterCheckpoint]:
    """The chat-SFT run as a lazy ``ArtifactStep[LevanterCheckpoint]``.

    ``build_config(ctx)`` is pure over the ``StepContext``: the identity-bearing TrainLmConfig is a
    literal over ``spec`` (with dep cache paths pulled via ``ctx.artifact_path``); only
    ``output_path`` and ``resources`` come from ``ctx`` -> excluded from the fingerprint.
    """
    deps = _dataset_deps(spec)

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        dep_paths = [ctx.artifact_path(dep) for dep in deps]
        train_config = build_sft_train_config(spec, dep_paths)
        return TrainLmOnPodConfig(
            train_config=train_config,
            resources=ctx.runtime_arg("train_resources"),
            output_path=ctx.output_path,
            auto_build_caches=True,
        )

    return ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=LevanterCheckpoint,
        run=_train_job,
        build_config=build_config,
        deps=deps,
        runtime_args={"train_resources": spec.resources},
    )


if __name__ == "__main__":
    # Delphi is ONE example config; swap the import to launch a different recipe.
    from experiments.sft_launcher.configs.delphi_1e22 import SPEC

    StepRunner().run([lower(sft_step(SPEC))])
