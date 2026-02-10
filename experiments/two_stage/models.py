# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig

from experiments.llama import llama_1_4b, llama_1_9b, llama_150m, llama_300m, llama_600m

llama_150m_4096_config = dataclasses.replace(llama_150m, max_seq_len=4096)
llama_300m_4096_config = dataclasses.replace(llama_300m, max_seq_len=4096)
llama_600m_4096_config = dataclasses.replace(llama_600m, max_seq_len=4096)
llama_1_4b_4096_config = dataclasses.replace(llama_1_4b, max_seq_len=4096)
llama_1_9b_4096_config = dataclasses.replace(llama_1_9b, max_seq_len=4096)

llama_8b_config = LlamaConfig(
    max_seq_len=4096,  # Seq len set to reproduce Tulu SFT
    hidden_dim=4096,
    intermediate_dim=14336,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,
    use_bias=False,
    use_layer_norm_weight=True,
    initializer_range=0.02,
    rope=Llama3RotaryEmbeddingsConfig(
        # Using Llama3 defaults from the code
        theta=500000,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ),
)

model_dict = {
    "150m4k": llama_150m_4096_config,
    "300m4k": llama_300m_4096_config,
    "600m4k": llama_600m_4096_config,
    "1_4b4k": llama_1_4b_4096_config,
    "1_9b4k": llama_1_9b_4096_config,
    "l8b": llama_8b_config,
}
