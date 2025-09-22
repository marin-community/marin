# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from:
# https://github.com/stanford-crfm/levanter/blob/main/src/levanter/models/attention.py
import functools

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as PS
from scalax.sharding import MeshShardingHelper

# Copied from
# https://github.com/google/maxtext/blob/db31dd4b0b686bca4cd7cf940917ec372faa183a/MaxText/layers/attentions.py#L179


def _tpu_splash_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    attention_mask: jnp.ndarray,
    dropout: float = 0.0,
    *,
    attention_dtype: jnp.dtype | None = None,
    block_size: int | None = None,
) -> jnp.ndarray | None:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask

    # Splash attention requires BHSD format
    # We need to reshape the input to match this format
    if dropout != 0.0:
        raise NotImplementedError("Splash attention does not support dropout")

    # if attention_dtype is not None and attention_dtype != jnp.float32:
    #     warnings.warn("Splash attention only supports float32. Switching to float32.")

    attention_dtype = jnp.float32

    B, Sq, Hq, D = query.shape
    Bk, Sk, Hk, Dk = key.shape

    # pre-divide q_ by sqrt(d) to match the reference implementation
    query = query / jnp.sqrt(D)

    # number
    if Sk % 128 != 0:
        raise NotImplementedError(f"Splash attention requires KPos to be a multiple of 128, got {Sk}")

    if block_size is not None and block_size % 128 != 0:
        raise NotImplementedError(f"Splash attention requires block_size to be a multiple of 128, got {block_size}")

    # TODO: must Dk == Dv?
    if key.shape != value.shape:
        raise ValueError("k and v must have the same axes")

    # TODO: this isn't really necessary on TPU?
    if B != Bk:
        raise ValueError(f"Batch axes must be the same for q, k, and v: {B} != {Bk}")

    if D != Dk:
        raise ValueError(f"Embedding axes must be the same for q, k, and v: {D} != {Dk}")

    # MaxText uses a block size of 512
    block_size = block_size or 512

    # copied from MaxText
    @functools.partial(
        shard_map,
        mesh=MeshShardingHelper.get_global_mesh(),
        in_specs=(
            PS(("replica", "fsdp"), "tensor", None, None),
            PS(("replica", "fsdp"), "tensor", None, None),
            PS(("replica", "fsdp"), "tensor", None, None),
            PS(("replica", "fsdp"), None),
        ),
        out_specs=PS(("replica", "fsdp"), "tensor", None, None),
        check_rep=False,
    )
    def wrap_flash_attention(q, k, v, attention_mask):
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=min(block_size, Sq),
            block_kv_compute=min(block_size, Sk),
            block_kv=min(block_size, Sk),
            block_q_dkv=min(block_size, Sq),
            block_kv_dkv=min(block_size, Sk),
            block_kv_dkv_compute=min(block_size, Sq),
            block_q_dq=min(block_size, Sq),
            block_kv_dq=min(block_size, Sq),
        )

        segment_ids = splash_attention_kernel.SegmentIds(
            q=attention_mask[:, :Sq],
            kv=attention_mask,
        )

        kernel_mask = splash_attention_mask.MultiHeadMask(
            [splash_attention_mask.CausalMask((Sq, Sq)) for _ in range(Hq)],
        )

        # copied from MaxText
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=kernel_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
        )

        q = q.astype(attention_dtype)
        k = k.astype(attention_dtype)
        v = v.astype(attention_dtype)
        return jax.vmap(splash_kernel)(q, k, v, segment_ids=segment_ids)

    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    attn_output = wrap_flash_attention(query, key, value, attention_mask)
    attn_output = attn_output.transpose(0, 2, 1, 3)
    return attn_output


# Copied from  https://github.com/Sea-Snell/llama3_train/blob/fixed_fast_inference/llama_train/paged.py


def _tpu_paged_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    lengths: jnp.ndarray,
    *,
    attention_dtype: jnp.dtype | None = None,
    page_size: int = 512,
    pages_per_compute_block: int = 1,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
    use_int8: bool = False,
) -> jnp.ndarray | None:
    # NOTE: ALL SEQUENCES MUST BE RIGHT PADDED!
    from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
    from jax.experimental.pallas.ops.tpu.paged_attention.quantization_utils import QuantizedTensor

    B, Sq, Hq, D = query.shape
    if use_int8:
        Bk, Sk, Hk, Dk = key.weight.shape
    else:
        Bk, Sk, Hk, Dk = key.shape

    # pre-divide q_ by sqrt(d) to match the reference implementation
    query = query / jnp.sqrt(D)

    # TODO: must Dk == Dv?
    if use_int8:
        if key.weight.shape != value.weight.shape:
            raise ValueError("k and v must have the same axes")
        if key.scales.shape != value.scales.shape:
            raise ValueError("k and v scales must have the same axes")
    else:
        if key.shape != value.shape:
            raise ValueError("k and v must have the same axes")

    # TODO: this isn't really necessary on TPU?
    if B != Bk:
        raise ValueError(f"Batch axes must be the same for q, k, and v: {B} != {Bk}")

    if D != Dk:
        raise ValueError(f"Embedding axes must be the same for q, k, and v: {D} != {Dk}")

    if Sq != 1:
        raise NotImplementedError("Paged attention only supports query sequence length 1")

    assert Sk % page_size == 0, "Key sequence length must be a multiple of page size"

    pages_per_seq = Sk // page_size

    # copied from MaxText
    @functools.partial(
        shard_map,
        mesh=MeshShardingHelper.get_global_mesh(),
        in_specs=(
            PS(("replica", "fsdp"), None, "tensor", None),
            PS(("replica", "fsdp"), None, "tensor", None),
            PS(("replica", "fsdp"), None, "tensor", None),
            PS(("replica", "fsdp")),
        ),
        out_specs=PS(("replica", "fsdp"), None, "tensor", None),
        check_rep=False,
    )
    def wrap_paged_attention(
        q,
        k,
        v,
        lengths,
    ):
        if use_int8:
            local_B, _, local_Hk, local_Dk = k.weight.shape
            k_flat = QuantizedTensor(
                k.weight.reshape(local_B * pages_per_seq, page_size, local_Hk, local_Dk).transpose(2, 0, 1, 3),
                k.scales.reshape(local_B * pages_per_seq, page_size, local_Hk, 1).transpose(2, 0, 1, 3),
            )
            v_flat = QuantizedTensor(
                v.weight.reshape(local_B * pages_per_seq, page_size, local_Hk, local_Dk).transpose(2, 0, 1, 3),
                v.scales.reshape(local_B * pages_per_seq, page_size, local_Hk, 1).transpose(2, 0, 1, 3),
            )
        else:
            local_B, _, local_Hk, local_Dk = k.shape
            k_flat = k.reshape(local_B * pages_per_seq, page_size, local_Hk, local_Dk).transpose(2, 0, 1, 3)
            v_flat = v.reshape(local_B * pages_per_seq, page_size, local_Hk, local_Dk).transpose(2, 0, 1, 3)

        q_flat = q[:, 0, :, :]

        page_indices = jnp.arange(local_B * pages_per_seq).reshape(local_B, pages_per_seq)

        return jnp.expand_dims(
            paged_attention(
                q_flat,
                k_flat,
                v_flat,
                lengths,
                page_indices,
                pages_per_compute_block=pages_per_compute_block,
                megacore_mode=megacore_mode,
                inline_seq_dim=inline_seq_dim,
            ),
            axis=1,
        )

    attn_output = wrap_paged_attention(query, key, value, lengths)
    return attn_output
