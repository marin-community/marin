# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batched Harrier inference through Levanter's Qwen3 model on JAX accelerators."""

from dataclasses import dataclass, replace
from functools import cache
from typing import Literal, cast

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis, NamedArray
from haliax.partitioning import set_mesh
from jax.sharding import Mesh
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.layers.attention import AttentionMask
from levanter.model_cache import resolve_cached_model_path
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3 import Qwen3Config as HfQwen3Config

BATCH_AXIS_NAME = "batch"
DATA_AXIS_NAME = "data"
POSITION_AXIS_NAME = "position"


@dataclass(frozen=True)
class HarrierJaxConfig:
    model: str
    revision: str
    tokenizer_revision: str
    dimensions: int
    max_tokens: int
    compute_dtype: str
    platform: Literal["cpu", "gpu", "tpu"]
    batch_size_per_device: int = 32
    batch_token_limit_per_device: int = 16_384
    cache_ttl_days: int = 30

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.batch_size_per_device <= 0 or self.batch_token_limit_per_device <= 0:
            raise ValueError("Harrier token and batch limits must be positive")


def _embed_batch(model: Qwen3LMHeadModel, input_ids: NamedArray, lengths: NamedArray) -> NamedArray:
    batch = input_ids.resolve_axis(BATCH_AXIS_NAME)
    position = input_ids.resolve_axis(POSITION_AXIS_NAME)
    hidden = model.activations(input_ids, attn_mask=AttentionMask.causal())
    pooled = hax.take(hidden, position, lengths - 1).rearrange((batch, model.config.Embed))
    vectors = pooled.array.astype(jnp.float32)
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return hax.named(vectors / jnp.maximum(norms, 1e-12), (batch, model.config.Embed))


_compiled_embed_batch = hax.named_jit(_embed_batch, axis_resources={BATCH_AXIS_NAME: DATA_AXIS_NAME})


class HarrierJaxEmbedder:
    """Return normalized Harrier embeddings for input text."""

    def __init__(self, config: HarrierJaxConfig) -> None:
        self.config = config
        devices = jax.local_devices()
        if not devices or devices[0].platform != config.platform:
            raise ValueError(f"Harrier requires JAX {config.platform} devices")
        self.mesh = Mesh(np.array(devices), (DATA_AXIS_NAME,))
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(config.model, revision=config.tokenizer_revision, padding_side="right"),
        )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Harrier tokenizer has no padding token")

        checkpoint = resolve_cached_model_path(
            f"{config.model}@{config.revision}", cache_ttl_days=config.cache_ttl_days, cache_prefix="hf-models"
        )
        converter = HFCheckpointConverter(
            Qwen3Config,
            reference_checkpoint=RepoRef.from_string(checkpoint),
            tokenizer=self.tokenizer,
            HfConfigClass=HfQwen3Config,
        )
        hf_config = converter.default_hf_config
        if config.max_tokens > hf_config.max_position_embeddings:
            raise ValueError("Harrier max_tokens exceeds the checkpoint context length")
        if config.dimensions != hf_config.hidden_size:
            raise ValueError("Harrier embedding dimension differs from the checkpoint hidden size")
        model_config = replace(Qwen3Config.from_hf_config(hf_config), gradient_checkpointing=False)
        with set_mesh(self.mesh):
            self.model = cast(
                Qwen3LMHeadModel,
                converter.load_pretrained(
                    Qwen3LMHeadModel,
                    config=model_config,
                    axis_mapping={BATCH_AXIS_NAME: DATA_AXIS_NAME},
                    dtype=jnp.dtype(config.compute_dtype),
                    state_dict_prefix="model",
                ),
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32, L2-normalized vectors in input order."""
        config = self.config
        output = np.empty((len(texts), config.dimensions), dtype=np.float32)
        if not texts:
            return output

        token_ids = self.tokenizer(texts, padding=False, truncation=False)["input_ids"]
        lengths = [len(ids) for ids in token_ids]
        if min(lengths) == 0 or max(lengths) > config.max_tokens:
            raise ValueError(f"Harrier input must contain 1 to {config.max_tokens} tokens")

        buckets: dict[int, list[int]] = {}
        for index, length in enumerate(lengths):
            sequence_size = max(128, 1 << (length - 1).bit_length())
            buckets.setdefault(sequence_size, []).append(index)

        device_count = len(self.mesh.devices.flat)
        pad_token_id = self.tokenizer.pad_token_id
        assert pad_token_id is not None
        with set_mesh(self.mesh):
            for sequence_size, indices in sorted(buckets.items()):
                per_device = min(
                    config.batch_size_per_device, max(1, config.batch_token_limit_per_device // sequence_size)
                )
                batch_size = per_device * device_count
                batch = Axis(BATCH_AXIS_NAME, batch_size)
                position = Axis(POSITION_AXIS_NAME, sequence_size)
                for start in range(0, len(indices), batch_size):
                    selected = indices[start : start + batch_size]
                    padded_ids = np.full((batch_size, sequence_size), pad_token_id, dtype=np.int32)
                    padded_lengths = np.ones(batch_size, dtype=np.int32)
                    for row, index in enumerate(selected):
                        padded_ids[row, : lengths[index]] = token_ids[index]
                        padded_lengths[row] = lengths[index]
                    ids = hax.shard_with_axis_mapping(
                        hax.named(jnp.asarray(padded_ids), (batch, position)),
                        {BATCH_AXIS_NAME: DATA_AXIS_NAME},
                        mesh=self.mesh,
                    )
                    token_lengths = hax.shard_with_axis_mapping(
                        hax.named(jnp.asarray(padded_lengths), (batch,)),
                        {BATCH_AXIS_NAME: DATA_AXIS_NAME},
                        mesh=self.mesh,
                    )
                    vectors = _compiled_embed_batch(self.model, ids, token_lengths)
                    output[selected] = np.asarray(vectors.array)[: len(selected)]
        return output


@cache
def load_harrier_jax(config: HarrierJaxConfig) -> HarrierJaxEmbedder:
    """Keep one loaded model per process and inference configuration."""
    return HarrierJaxEmbedder(config)
