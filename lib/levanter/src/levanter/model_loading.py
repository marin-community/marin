# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a Levanter LM model from a local checkpoint or a HuggingFace repo.

A loaded model is sharded according to ``axis_mapping`` and ready to run under an active
device mesh. There are two cases, one per loader:

- :func:`load_levanter_checkpoint` — a path containing ``step-*`` checkpoint dirs (or a
  single resolved checkpoint dir). Named models load on CPU before sharding; native models
  restore directly into their explicit mesh placement after checking checkpoint metadata.
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

from levanter.checkpoint import discover_latest_checkpoint, load_checkpoint, resolve_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig, RepoRef
from levanter.model_cache import resolve_cached_model_path
from levanter.models.lm_model import LmConfig, LmHeadModel, ModelPlacement
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
    placement: ModelPlacement | None = None,
) -> LmHeadModel:
    """Load and shard an ``LmHeadModel`` from a local Levanter checkpoint.

    Must be called under an active device mesh (e.g. ``trainer.use_device_mesh()``). When
    *checkpoint* is a run directory, the latest ``step-*`` checkpoint under it is loaded; when
    it already names a concrete checkpoint dir it is loaded directly (so checkpoints without a
    ``metadata.json`` discovery marker still load). Named weights are loaded on CPU before
    sharding. Native metadata supplies the complete vocabulary and validates the configured
    architecture. Metadata-free native checkpoints trust the caller's original model
    configuration and complete Vocab; architecture and activation identity cannot be
    verified from same-shaped weights alone. Present metadata is validated even when
    its step or timestamp cannot be used for discovery; unreadable metadata raises.

    Args:
        model_config: Model configuration used to build the model template.
        checkpoint: Path to a run directory holding ``step-*`` checkpoints, or a concrete
            checkpoint dir.
        Vocab: Vocabulary axis used to build the model template.
        axis_mapping: Resource mapping the model is sharded with.
        key: PRNG key for building the model template.
        placement: Framework mappings for a config with explicit parameter placement.
    """
    if model_config.owns_parameter_placement:
        if placement is None:
            raise ValueError("An explicitly placed checkpoint requires model placement")
        resolved = resolve_checkpoint(checkpoint)
        source_vocab = model_config.checkpoint_vocab_size(resolved.metadata)
        if source_vocab is not None:
            if Vocab.size > source_vocab:
                raise ValueError("Tokenizer vocabulary exceeds the checkpoint vocabulary")
            Vocab = Vocab.resize(source_vocab)
        mesh = jax.sharding.get_mesh()
        model = eqx.filter_eval_shape(model_config.build_for_mesh, Vocab, key=key, placement=placement)
        return load_checkpoint(model, resolved.path, subpath="model", mesh=mesh)

    resolved = discover_latest_checkpoint(checkpoint) or checkpoint
    with use_cpu_device():
        model = eqx.filter_eval_shape(model_config.build, Vocab, key=key)
        model = load_checkpoint(model, resolved, subpath="model")
    return hax.shard_with_axis_mapping(model, axis_mapping)


def load_hf_checkpoint(
    model_config: LmConfig,
    checkpoint: str,
    *,
    axis_mapping: ResourceMapping,
    tokenizer,
    compute_dtype: jnp.dtype | None,
    placement: ModelPlacement | None = None,
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
        compute_dtype: Named-model load dtype. Native masters are loaded in FP32 and cast
            to computation precision by the evaluation caller.
        placement: Framework mappings for a config with explicit parameter placement.
        cache_ttl_days: TTL for the region-local mirror of a HuggingFace repo id. ``<= 0``
            disables mirroring (the repo loads directly from HuggingFace). Ignored for
            ``hf://`` and object-store paths.
    """
    if not isinstance(model_config, HFCompatConfig):
        raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")

    resolved = resolve_cached_model_path(checkpoint, cache_ttl_days=cache_ttl_days, cache_prefix=_HF_CACHE_PREFIX)
    ref = RepoRef.from_string(resolved)
    converter = model_config.hf_checkpoint_converter_for_source(ref, tokenizer)
    if model_config.owns_parameter_placement:
        if placement is None:
            raise ValueError("An explicitly placed HF checkpoint requires model placement")
        hf_config = converter.default_hf_config
        Vocab = model_config.hf_checkpoint_vocab(hf_config, converter.Vocab)
        template = eqx.filter_eval_shape(
            model_config.build_for_mesh, Vocab, key=jax.random.PRNGKey(0), placement=placement
        )
        model = converter.load_pretrained(
            model_config.model_type,
            ref=ref,
            config=model_config,
            axis_mapping=axis_mapping,
            dtype=jnp.float32,
            model_template=template,
        )
    else:
        model = converter.load_pretrained(
            model_config.model_type, ref=ref, axis_mapping=axis_mapping, dtype=compute_dtype
        )
    # `load_pretrained` is typed as the broad `ModelWithHfSerializationMixin`, but loading a
    # config's `model_type` always yields an `LmHeadModel` at runtime.
    return cast(LmHeadModel, model)
