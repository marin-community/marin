# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Initial model surface for JPEG tokenizer trials.

The first implementation pass reuses the grug base transformer directly so the
project can validate data and tokenization choices before forking the model
surface into a fully independent template.
"""

from experiments.grug.base.model import GrugModelConfig, Transformer, debug_mesh_and_token_pspec

JpegLmConfig = GrugModelConfig
JpegTransformer = Transformer

JPEG_TOKENIZER_V0_MODEL = JpegLmConfig(
    vocab_size=4096,
    hidden_dim=512,
    intermediate_dim=1792,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
)

JPEG_TOKENIZER_V1_LARGE_MODEL = JpegLmConfig(
    vocab_size=4096,
    hidden_dim=768,
    intermediate_dim=2688,
    num_layers=8,
    num_heads=12,
    num_kv_heads=12,
    max_seq_len=4096,
    head_dim=None,
)

__all__ = [
    "JPEG_TOKENIZER_V0_MODEL",
    "JPEG_TOKENIZER_V1_LARGE_MODEL",
    "JpegLmConfig",
    "JpegTransformer",
    "debug_mesh_and_token_pspec",
]
