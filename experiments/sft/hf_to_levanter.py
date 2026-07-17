# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize an HF->Levanter checkpoint conversion as a reproducible marin ``ArtifactStep``.

``HfModel`` inits an SFT run straight from an HF checkpoint (``initialize_from_hf`` +
``use_hf_model_config``): Levanter re-derives the architecture from the checkpoint and converts the
weights in-process, on every launch. :func:`hf_to_levanter` turns that conversion into a first-class,
cacheable graph node instead — it converts the HF checkpoint to a native Levanter checkpoint once and
returns a handle a :class:`~experiments.sft.launcher.LevanterCheckpointModel` inits from (weights only,
fresh optimizer), so a large base (e.g. a 67B) is converted once rather than per run and every
``sft_step`` trains from a native Levanter checkpoint.

Two things the inline path does implicitly have to be made explicit so native init is *numerically
identical* to ``initialize_from_hf``:

  1. **The architecture is carried forward.** Native init builds the model pytree from a concrete
     ``LmConfig`` (it does not re-derive it the way ``use_hf_model_config`` does). :func:`hf_to_levanter`
     resolves that ``LmConfig`` from the pinned HF ``config.json`` at graph-construction time, so it is
     an identity-bearing input available before the run and the downstream builds the matching pytree.
  2. **The tokenizer is padded to the model vocab.** ``train_lm`` builds the ``Vocab`` axis from
     ``len(tokenizer)``; models like Qwen3 pad their embedding past the tokenizer (151936 vs 151669).
     The conversion emits a tokenizer padded to the model vocab (appending never-emitted dummy tokens,
     so real text tokenizes identically) and the downstream resolves its tokenizer to it, so
     ``Vocab == checkpoint embedding axis`` with no change to how ``Vocab`` is computed.

The weights are converted at the run's *compute* dtype (bf16 under marin's ``p=f32,c=bfloat16``),
matching how ``initialize_from_hf`` loads them, so the bf16->f32 round-trip into the f32 master params
is the same on both paths. The checkpoint holds only the model (no optimizer state), so the artifact is
the model's size, not ~3x it; native init loads it strictly (every model leaf must be present, see
``TrainLmConfig.initialize_model_from_checkpoint_path``), so a malformed conversion fails loudly rather
than silently re-initializing weights.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass

import draccus
import jax.numpy as jnp
from fray.types import ResourceConfig
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, load_tokenizer
from levanter.models.lm_model import LmConfig
from levanter.utils.jax_utils import use_cpu_device
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint
from marin.utils import get_directory_friendly_name
from rigging.filesystem import StoragePath, prefix_join

logger = logging.getLogger(__name__)

# Runtime-arg key for where the conversion job runs (excluded from the fingerprint).
_CONVERT_RESOURCES = "convert_resources"
# The subdir of the artifact holding the model checkpoint (a `model/` subtree + metadata.json), so
# `latest_checkpoint_path` discovers it and `load_checkpoint(..., subpath="model")` reads the weights.
_MODEL_CHECKPOINT_SUBDIR = "converted"
# Compute in bf16 by default, matching marin's compute precision (`_MARIN_PRECISION`) so the weights
# round-trip through the same dtype as the inline `initialize_from_hf` path.
DEFAULT_COMPUTE_DTYPE = "bfloat16"
# The conversion loads one model into host memory and streams it back out; no accelerator is needed.
# Sized for models up to ~10B in bf16 — pass a larger host for bigger bases (peak host memory exceeds
# the raw bf16 size because of HF shard staging and serialization buffers, so do not size to the model
# bytes exactly).
DEFAULT_CONVERT_RESOURCES = ResourceConfig.with_cpu(cpu=8, ram="64g", disk="64g")


@dataclass(frozen=True)
class HfToLevanterConfig:
    """Identity-bearing inputs for one HF->Levanter conversion.

    ``model_config_json`` is the draccus-encoded ``LmConfig`` resolved from the pinned HF checkpoint at
    graph-construction time; the worker decodes it (keyed by ``model_type``) so the model it loads and
    saves is exactly the architecture the downstream builds its pytree from. ``resources`` and
    ``output_path`` are pulled from the step context at run time and never enter the artifact identity.
    """

    hf_id: str  # HF repo id of the base checkpoint
    hf_revision: str  # commit pin for the base checkpoint (fingerprint stability)
    model_type: str  # Levanter model-registry key, e.g. "qwen3"
    tokenizer: str  # tokenizer repo id/dir the converter binds to
    tokenizer_revision: str  # commit pin for the tokenizer
    compute_dtype: str  # dtype the weights are converted in (matches the run's compute precision)
    model_config_json: dict  # draccus-encoded resolved LmConfig
    output_path: str
    resources: ResourceConfig


def resolve_lm_config(
    model_type: str, hf_id: str, hf_revision: str, tokenizer: str, tokenizer_revision: str
) -> LmConfig:
    """Resolve the concrete ``LmConfig`` for ``hf_id@hf_revision`` — the arch native init needs.

    Reads the pinned HF ``config.json`` through the model class's ``HFCheckpointConverter`` (the same
    derivation ``use_hf_model_config`` performs). Called at graph-construction time so the architecture
    is an identity-bearing input, not something reconstructed only on the worker.
    """
    config_cls = LmConfig.get_choice_class(model_type)
    converter = (
        config_cls()
        .hf_checkpoint_converter()
        .replaced(
            reference_checkpoint=RepoRef(hf_id, hf_revision),
            tokenizer=load_tokenizer(tokenizer, revision=tokenizer_revision),
        )
    )
    return converter.config_from_hf_config(converter.default_hf_config)


def run_hf_to_levanter(config: HfToLevanterConfig) -> None:
    """Convert the HF checkpoint to a native Levanter model checkpoint and emit a padded tokenizer.

    Runs on a worker; reads the base checkpoint from the Hub, so it needs ``HF_TOKEN`` in the
    environment for gated repos (propagated the same way as the training job).
    """
    model_config = draccus.decode(LmConfig.get_choice_class(config.model_type), config.model_config_json)
    ref = RepoRef(config.hf_id, config.hf_revision)
    tokenizer = load_tokenizer(config.tokenizer, revision=config.tokenizer_revision)
    converter = model_config.hf_checkpoint_converter().replaced(reference_checkpoint=ref, tokenizer=tokenizer)

    logger.info(
        "Loading HF checkpoint %s@%s as %s in %s",
        config.hf_id,
        config.hf_revision,
        config.model_type,
        config.compute_dtype,
    )
    with use_cpu_device():
        model = converter.load_pretrained(
            model_config.model_type,
            ref=ref,
            config=model_config,
            dtype=getattr(jnp, config.compute_dtype),
            resize_vocab_to_match_tokenizer=False,  # keep the model's own (padded) vocab
        )

    checkpoint_path = prefix_join(config.output_path, _MODEL_CHECKPOINT_SUBDIR)
    StoragePath(checkpoint_path).mkdirs()
    manager = GlobalAsyncCheckpointManager()
    # Save just the model, under the "model" key so it lands at the same `model/` subpath a full
    # TrainerState uses — the downstream loads it with subpath="model".
    save_checkpoint(
        tree={"model": model},
        step=0,
        checkpoint_path=checkpoint_path,
        manager=manager,
        is_temporary=False,
    )
    manager.wait_until_finished()  # block until the async commit lands before the artifact record is written
    logger.info("Saved Levanter model checkpoint to %s", checkpoint_path)

    # Emit a tokenizer padded to the model vocab, so the downstream builds Vocab == the checkpoint's
    # embedding axis. Padding appends never-emitted dummy tokens, so real text tokenizes identically.
    padded_tokenizer = converter.with_tokenizer_padded_to_match_model().tokenizer
    with tempfile.TemporaryDirectory() as tokenizer_dir:
        padded_tokenizer.save_pretrained(tokenizer_dir)
        for name in os.listdir(tokenizer_dir):
            if name.startswith("."):
                continue
            StoragePath(prefix_join(config.output_path, name)).upload_from(os.path.join(tokenizer_dir, name))
    logger.info("Emitted padded tokenizer (vocab %d) to %s", len(padded_tokenizer), config.output_path)


def _convert_job(config: HfToLevanterConfig) -> None:
    """The step's ``run``: dispatch the conversion as its own Fray job."""
    remote(run_hf_to_levanter, resources=config.resources)(config)


@dataclass(frozen=True)
class HfToLevanterCheckpoint:
    """A materialized HF->Levanter conversion: the checkpoint step plus the resolved architecture.

    ``step`` produces the native model checkpoint + padded tokenizer at its output directory;
    ``model`` is the ``LmConfig`` the checkpoint was saved with (so the downstream builds the matching
    pytree without re-deriving it). Pass one to
    :class:`~experiments.sft.launcher.LevanterCheckpointModel` as ``init_from``.
    """

    step: ArtifactStep[LevanterCheckpoint]
    model: LmConfig

    @property
    def model_checkpoint_path(self) -> str:
        """The concrete model-checkpoint directory, relative resolution deferred to the launcher."""
        return _MODEL_CHECKPOINT_SUBDIR


def hf_to_levanter(
    hf_id: str,
    *,
    model_type: str,
    hf_revision: str,
    version: str,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    compute_dtype: str = DEFAULT_COMPUTE_DTYPE,
    resources: ResourceConfig = DEFAULT_CONVERT_RESOURCES,
) -> HfToLevanterCheckpoint:
    """Convert ``hf_id@hf_revision`` to a native Levanter model checkpoint as a lazy ``ArtifactStep``.

    ``tokenizer`` defaults to ``hf_id`` (the common case); ``tokenizer_revision`` defaults to
    ``hf_revision``. The returned handle carries the resolved ``LmConfig`` (read from the pinned HF
    ``config.json`` now, so it is identity-bearing) and the conversion step; feed it to a
    ``LevanterCheckpointModel`` as ``init_from``.
    """
    tokenizer = tokenizer or hf_id
    tokenizer_revision = tokenizer_revision or hf_revision
    model_config = resolve_lm_config(model_type, hf_id, hf_revision, tokenizer, tokenizer_revision)
    model_config_json = draccus.encode(model_config)
    name = f"checkpoints/hf-to-levanter/{get_directory_friendly_name(hf_id)}-{model_type}"

    def build_config(ctx: StepContext) -> HfToLevanterConfig:
        return HfToLevanterConfig(
            hf_id=hf_id,
            hf_revision=hf_revision,
            model_type=model_type,
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            compute_dtype=compute_dtype,
            model_config_json=model_config_json,
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_CONVERT_RESOURCES),
        )

    step: ArtifactStep[LevanterCheckpoint] = ArtifactStep(
        name=name,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=_convert_job,
        build_config=build_config,
        runtime_args={_CONVERT_RESOURCES: resources},
    )
    return HfToLevanterCheckpoint(step=step, model=model_config)
