# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Snowball serving settings for the 64K v5p AgentX workload."""

import jax
import jax.numpy as jnp
from haliax.partitioning import ResourceMapping
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.snowball_loading import load_snowball_serving_model
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from marin.inference.config import SNOWBALL_AGENTX_EXPERT_SHARDS

AGENTX_CONTEXT = 65536
AGENTX_MAX_CONCURRENCY = 128
AGENTX_BATCH_TIMEOUT = 0.001
_PREFILL_BUCKETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 49152)
_DECODE_BUCKETS = (4, 8, 16)


def require_agentx_devices() -> None:
    """Check the geometry before allocating the model and compact KV pools."""
    devices = jax.devices()
    if len(devices) != SNOWBALL_AGENTX_EXPERT_SHARDS or any(device.device_kind != "TPU v5" for device in devices):
        raise ValueError(f"Native Snowball AgentX requires one v5p-8, got {devices}")


def agentx_engine_config(concurrency: int, *, max_rounds: int = 6) -> InferenceEngineConfig:
    """Build the paged generation configuration used by the v5p AgentX replay."""
    if not 1 <= concurrency <= AGENTX_MAX_CONCURRENCY:
        raise ValueError(f"AgentX supports 1 to {AGENTX_MAX_CONCURRENCY} concurrent sequences")
    return InferenceEngineConfig(
        max_seq_len=AGENTX_CONTEXT,
        max_seqs=concurrency,
        max_pages=49152,
        window_cache_max_pages=5120,
        window_cache_retained_spans=2,
        max_retained_prompts=512,
        max_queued_tokens=max(32, concurrency),
        max_seqs_in_prefill=min(4, concurrency),
        max_prefill_size=AGENTX_CONTEXT,
        max_rounds=max_rounds,
        max_tokens_per_round=32,
        prefill_bucket_sizes=_PREFILL_BUCKETS,
        decode_bucket_sizes=tuple(size for size in _DECODE_BUCKETS if size < concurrency),
        compute_dtype=jnp.bfloat16,
        logprob_mode=LogprobMode.POLICY,
    )


def load_agentx_model(
    converter: HFCheckpointConverter,
    ref: RepoRef,
    config: SnowballConfig,
    axis_mapping: ResourceMapping,
) -> SnowballLMHeadModel:
    """Load packed BF16 experts and configure the v5p paged attention path."""
    model = load_snowball_serving_model(
        converter, ref, config, dtype=jnp.bfloat16, axis_mapping=axis_mapping, pack_expert_weights=True
    )
    model = model.with_paged_attention("tpu_v3").with_split_decode_cache()
    return model.with_expert_sharded_kv()
