# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.layers.attention import AttentionBackend
from levanter.models.llama import LlamaConfig


def llama_test_config(
    attention_backend: AttentionBackend = AttentionBackend.VANILLA,
    num_kv_heads: int = 4,
    seq_len: int = 64,
) -> LlamaConfig:
    return LlamaConfig(
        max_seq_len=seq_len,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
        attn_backend=attention_backend,
        flash_attention_block_size=8 if attention_backend == AttentionBackend.DEFAULT else None,
    )
