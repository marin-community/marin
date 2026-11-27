# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Snowball serving settings for the 64K TPU AgentX workload."""

import jax
import jax.numpy as jnp
from haliax.partitioning import ResourceMapping
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.snowball_loading import load_snowball_serving_model
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from marin.inference.config import SnowballAgentxGeometry

AGENTX_CONTEXT = 65536
AGENTX_MAX_CONCURRENCY = 128
AGENTX_BATCH_TIMEOUT = 0.001


def require_agentx_devices(geometry: SnowballAgentxGeometry) -> None:
    """Check the geometry before allocating the model and compact KV pools."""
    devices = jax.devices()
    if len(devices) != geometry.expert_shards or any(device.device_kind != geometry.device_kind for device in devices):
        raise ValueError(f"Native Snowball AgentX requires one {geometry.tpu_type}, got {devices}")


def agentx_engine_config(
    geometry: SnowballAgentxGeometry, concurrency: int, *, max_rounds: int | None = None
) -> InferenceEngineConfig:
    """Build the paged generation configuration for a native TPU profile."""
    if not 1 <= concurrency <= AGENTX_MAX_CONCURRENCY:
        raise ValueError(f"AgentX supports 1 to {AGENTX_MAX_CONCURRENCY} concurrent sequences")
    return InferenceEngineConfig(
        max_seq_len=AGENTX_CONTEXT,
        max_seqs=concurrency,
        max_pages=geometry.max_pages,
        window_cache_max_pages=geometry.window_pages,
        window_cache_retained_spans=geometry.retained_spans,
        max_retained_prompts=geometry.retained_prompts,
        max_queued_tokens=max(geometry.max_tokens_per_round, concurrency),
        max_seqs_in_prefill=min(4, concurrency),
        max_prefill_size=AGENTX_CONTEXT,
        max_rounds=geometry.max_rounds if max_rounds is None else max_rounds,
        max_tokens_per_round=geometry.max_tokens_per_round,
        prefill_bucket_sizes=geometry.prefill_buckets,
        decode_bucket_sizes=tuple(size for size in geometry.decode_buckets if size < concurrency),
        compute_dtype=jnp.bfloat16,
        logprob_mode=LogprobMode.POLICY,
    )


def load_agentx_model(
    converter: HFCheckpointConverter,
    ref: RepoRef,
    config: SnowballConfig,
    axis_mapping: ResourceMapping,
    geometry: SnowballAgentxGeometry,
) -> SnowballLMHeadModel:
    """Load BF16 experts in the layout selected by the AgentX hardware profile."""
    separate_tiles = geometry.separate_expert_tiles
    if separate_tiles is not None and config.router_combine_fp32:
        raise ValueError("The separate expert projection profile requires BF16 expert combination")
    model = load_snowball_serving_model(
        converter, ref, config, dtype=jnp.bfloat16, axis_mapping=axis_mapping, pack_expert_weights=separate_tiles is None
    )
    model = model.with_paged_attention("tpu_v3").with_split_decode_cache().with_expert_sharded_kv()
    if separate_tiles is not None:
        model = model.with_separate_serving_experts(separate_tiles.for_rows)
    return model
