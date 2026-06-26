# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a Levanter LM model from a local checkpoint or a HuggingFace repo.

A loaded model is sharded according to ``axis_mapping`` and ready to run under an active
device mesh. There are two cases, one per loader:

- :func:`load_levanter_checkpoint` — a path containing ``step-*`` checkpoint dirs (or a
  single resolved checkpoint dir): the latest checkpoint is discovered, loaded on CPU to keep
  peak device memory low, then sharded.
- :func:`load_hf_checkpoint` — a HuggingFace repo id, ``hf://`` URL, or object-store path: the
  config's HF converter loads and converts the weights. A bare repo id is first mirrored once
  to a region-local TTL cache under a distributed lock, so repeated loads of the same model
  read the snapshot from nearby storage instead of re-downloading from HuggingFace.
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis
from haliax.partitioning import ResourceMapping

from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.model_cache import resolve_cached_model_path
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.jax_utils import use_cpu_device

# Sub-path under the region-local TTL bucket mirrored HuggingFace snapshots are written to.
_HF_CACHE_PREFIX = "hf-models"
# Days a mirrored HuggingFace snapshot is kept before the bucket lifecycle deletes it.
_DEFAULT_HF_CACHE_TTL_DAYS = 30


def load_levanter_checkpoint(
    model_config: LmConfig,
    checkpoint: str,
    *,
    Vocab: Axis,
    axis_mapping: ResourceMapping,
    key: jax.Array,
) -> LmHeadModel:
    """Load and shard an ``LmHeadModel`` from a local Levanter checkpoint.

    Must be called under an active device mesh (e.g. ``trainer.use_device_mesh()``). The
    latest checkpoint under *checkpoint* is discovered (an already-resolved ``step-N`` dir is
    returned unchanged), loaded on CPU to keep peak device memory low, then sharded.

    Args:
        model_config: Model configuration used to build the model template.
        checkpoint: Path to a run directory holding ``step-*`` checkpoints, or a single
            resolved checkpoint dir.
        Vocab: Vocabulary axis used to build the model template.
        axis_mapping: Resource mapping the model is sharded with.
        key: PRNG key for building the model template.
    """
    with use_cpu_device():
        model = eqx.filter_eval_shape(model_config.build, Vocab, key=key)
        model = load_checkpoint(model, latest_checkpoint_path(checkpoint), subpath="model")
    return hax.shard_with_axis_mapping(model, axis_mapping)


def load_hf_checkpoint(
    model_config: LmConfig,
    checkpoint: str,
    *,
    axis_mapping: ResourceMapping,
    tokenizer,
    compute_dtype: jnp.dtype,
    cache_ttl_days: int = _DEFAULT_HF_CACHE_TTL_DAYS,
) -> LmHeadModel:
    """Load and shard an ``LmHeadModel`` from a HuggingFace checkpoint.

    Must be called under an active device mesh (e.g. ``trainer.use_device_mesh()``). A bare
    HuggingFace repo id is mirrored once to a region-local TTL cache under a distributed lock
    before loading; ``hf://`` URLs and object-store paths are loaded in place.

    Args:
        model_config: Model configuration; must expose ``hf_checkpoint_converter``.
        checkpoint: HuggingFace repo id, ``hf://`` URL, or object-store path holding an HF
            snapshot.
        axis_mapping: Resource mapping the model is sharded with.
        tokenizer: Tokenizer the HuggingFace converter is bound to.
        compute_dtype: dtype the HuggingFace weights are loaded in.
        cache_ttl_days: TTL for the region-local mirror of a HuggingFace repo id. ``<= 0``
            disables mirroring (the repo loads directly from HuggingFace). Ignored for
            ``hf://`` and object-store paths.
    """
    if not hasattr(model_config, "hf_checkpoint_converter"):
        raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")

    resolved = resolve_cached_model_path(checkpoint, cache_ttl_days=cache_ttl_days, cache_prefix=_HF_CACHE_PREFIX)
    ref = RepoRef.from_string(resolved)
    converter: HFCheckpointConverter = model_config.hf_checkpoint_converter().replaced(
        reference_checkpoint=ref, tokenizer=tokenizer
    )
    model = converter.load_pretrained(model_config.model_type, ref=ref, axis_mapping=axis_mapping, dtype=compute_dtype)
    # `load_pretrained` is typed as the broad `ModelWithHfSerializationMixin`, but loading a
    # config's `model_type` always yields an `LmHeadModel` at runtime.
    return cast(LmHeadModel, model)
