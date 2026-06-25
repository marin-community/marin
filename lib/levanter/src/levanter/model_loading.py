# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a Levanter LM model from a local checkpoint or a HuggingFace repo.

A loaded model is sharded according to ``axis_mapping`` and ready to run under an
active device mesh. The two cases are:

- **Local Levanter checkpoint** (a path containing ``step-*`` checkpoint dirs, or
  a single resolved checkpoint dir): the latest checkpoint is discovered, loaded
  on CPU to keep peak device memory low, then sharded.
- **HuggingFace checkpoint** (a repo id, ``hf://`` URL, or object-store path): the
  config's HF converter loads and converts the weights. A bare HuggingFace repo
  id is first mirrored once to a region-local TTL cache under a distributed lock
  (via ``cache_hf_model``), so repeated loads of the same model read the snapshot
  from nearby storage instead of re-downloading from HuggingFace. Paths that
  already live in an object store or on local disk are loaded in place.
"""

from __future__ import annotations

import logging
import re
from typing import cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis
from haliax.partitioning import ResourceMapping
from rigging.filesystem import marin_temp_bucket

from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, _is_hf_model_id
from levanter.model_cache import cache_hf_model
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)

# Sub-path under the region-local TTL bucket for mirrored eval-model snapshots.
_HF_MODEL_CACHE_PREFIX = "eval-models"
# Written last after a snapshot mirror so a half-uploaded cache never reads as a hit.
_CACHE_COMPLETE_MARKER = ".eval_model_cache_complete"
# Days a mirrored HuggingFace snapshot is kept before the bucket lifecycle deletes it.
_DEFAULT_HF_CACHE_TTL_DAYS = 30


def load_lm_model_from_checkpoint(
    model_config: LmConfig,
    checkpoint: str,
    *,
    is_hf: bool,
    Vocab: Axis,
    axis_mapping: ResourceMapping,
    tokenizer,
    compute_dtype: jnp.dtype,
    key: jax.Array,
    hf_cache_ttl_days: int | None = _DEFAULT_HF_CACHE_TTL_DAYS,
) -> LmHeadModel:
    """Load and shard an ``LmHeadModel`` from a local or HuggingFace checkpoint.

    Must be called under an active device mesh (e.g. ``trainer.use_device_mesh()``).

    Args:
        model_config: Model configuration. For HuggingFace checkpoints it must
            expose ``hf_checkpoint_converter``.
        checkpoint: Local checkpoint path, HuggingFace repo id, ``hf://`` URL, or
            object-store path holding an HF snapshot.
        is_hf: Whether ``checkpoint`` is a HuggingFace checkpoint rather than a
            local Levanter checkpoint.
        Vocab: Vocabulary axis used to build the model template (local case).
        axis_mapping: Resource mapping the model is sharded with.
        tokenizer: Tokenizer the HuggingFace converter is bound to (HF case).
        compute_dtype: dtype the HuggingFace weights are loaded in (HF case).
        key: PRNG key for building the model template.
        hf_cache_ttl_days: TTL for the region-local mirror of a HuggingFace repo
            id. ``None`` disables mirroring (the repo loads directly from
            HuggingFace). Ignored for non-HuggingFace and object-store paths.

    Returns:
        The loaded, sharded model.
    """
    if not is_hf:
        return _load_levanter_checkpoint(model_config, checkpoint, Vocab=Vocab, axis_mapping=axis_mapping, key=key)

    if not hasattr(model_config, "hf_checkpoint_converter"):
        raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")

    ref = _resolve_hf_ref(checkpoint, hf_cache_ttl_days)
    converter: HFCheckpointConverter = model_config.hf_checkpoint_converter().replaced(
        reference_checkpoint=ref, tokenizer=tokenizer
    )
    model = converter.load_pretrained(model_config.model_type, ref=ref, axis_mapping=axis_mapping, dtype=compute_dtype)
    # `load_pretrained` is typed as the broad `ModelWithHfSerializationMixin`, but loading a
    # config's `model_type` always yields an `LmHeadModel` at runtime.
    return cast(LmHeadModel, model)


def _load_levanter_checkpoint(
    model_config: LmConfig,
    checkpoint: str,
    *,
    Vocab: Axis,
    axis_mapping: ResourceMapping,
    key: jax.Array,
) -> LmHeadModel:
    with use_cpu_device():
        model = eqx.filter_eval_shape(model_config.build, Vocab, key=key)
        model = load_checkpoint(model, latest_checkpoint_path(checkpoint), subpath="model")
    return hax.shard_with_axis_mapping(model, axis_mapping)


def _resolve_hf_ref(checkpoint: str, cache_ttl_days: int | None) -> RepoRef:
    """Resolve *checkpoint* to a ``RepoRef``, mirroring a bare HF repo id to a TTL cache."""
    cache_path = _hf_cache_path(checkpoint, cache_ttl_days)
    if cache_path is None:
        return RepoRef.from_string(checkpoint)

    ref = RepoRef.from_string(checkpoint)
    mirrored = cache_hf_model(
        cache_path, ref.model_name_or_path, revision=ref.revision, complete_marker=_CACHE_COMPLETE_MARKER
    )
    return RepoRef(mirrored, None)


def _hf_cache_path(checkpoint: str, cache_ttl_days: int | None) -> str | None:
    """Return the TTL cache prefix to mirror *checkpoint* into, or ``None`` to load it in place.

    ``None`` when caching is disabled or when ``checkpoint`` already names an
    object-store or on-disk snapshot (which needs no mirroring). The cache prefix
    is keyed on the full ``checkpoint`` string so distinct repos and revisions do
    not collide.
    """
    if cache_ttl_days is None:
        return None
    if not _is_hf_model_id(RepoRef.from_string(checkpoint).model_name_or_path):
        return None
    return marin_temp_bucket(cache_ttl_days, f"{_HF_MODEL_CACHE_PREFIX}/{_cache_slug(checkpoint)}")


def _cache_slug(checkpoint: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint.strip("/"))
