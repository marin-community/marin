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

"""
Specifies the model configs
"""

import dataclasses

from levanter.models.llama import LlamaConfig
from levanter.models.rotary import Llama3RotaryEmbeddingsConfig

from levanter.models.mixtral import MixtralConfig

from experiments.llama import llama_1_4b, llama_1_9b, llama_150m, llama_300m, llama_600m


llama_1_5b_config = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=1536,
    intermediate_dim=5376,
    num_heads=16,
    num_kv_heads=8,
    num_layers=36,
)

llama_3_2b_config = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=48,
)

llama_3_2b_config_all_norm = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=48,
    hybrid_norm=True,
    use_qk_norm=True,
)

llama_3_2b_config_qk_norm = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=48,
    hybrid_norm=False,
    use_qk_norm=True,
)

SEQ_LEN = 4096

llama_150m_4096_config = dataclasses.replace(llama_150m, max_seq_len=SEQ_LEN)
llama_300m_4096_config = dataclasses.replace(llama_300m, max_seq_len=SEQ_LEN)
llama_600m_4096_config = dataclasses.replace(llama_600m, max_seq_len=SEQ_LEN)
llama_1_4b_4096_config = dataclasses.replace(llama_1_4b, max_seq_len=SEQ_LEN)
llama_1_9b_4096_config = dataclasses.replace(llama_1_9b, max_seq_len=SEQ_LEN)
llama_1_5b_4096_config = dataclasses.replace(llama_1_5b_config, max_seq_len=SEQ_LEN)
llama_3_2b_4096_config = dataclasses.replace(llama_3_2b_config, max_seq_len=SEQ_LEN)
llama_3_2b_4096_config_all_norm = dataclasses.replace(llama_3_2b_config_all_norm, max_seq_len=SEQ_LEN)
llama_3_2b_4096_config_qk_norm = dataclasses.replace(llama_3_2b_config_qk_norm, max_seq_len=SEQ_LEN)

llama_3b_config = LlamaConfig(
    max_seq_len=SEQ_LEN,  # Seq len set to reproduce Tulu SFT
    hidden_dim=3072,
    intermediate_dim=8192,
    num_layers=28,
    num_heads=24,
    num_kv_heads=8,
    use_bias=False,
    use_layer_norm_weight=True,
    initializer_range=0.02,
    rope=Llama3RotaryEmbeddingsConfig(
        # Using Llama3 defaults from the code
        theta=500000,
        factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ),
    tie_word_embeddings=True,
)

llama_8b_config = LlamaConfig(
    max_seq_len=SEQ_LEN,  # Seq len set to reproduce Tulu SFT
    hidden_dim=4096,
    intermediate_dim=14336,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,
    use_bias=False,
    use_layer_norm_weight=True,
    initializer_range=0.02,
    # use_flash_attention=True,
    flash_attention_block_size=512,
    rope=Llama3RotaryEmbeddingsConfig(
        # Using Llama3 defaults from the code
        theta=500000,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ),
)

olmoe_8x_1_4b = MixtralConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=1024,
    num_heads=16,
    num_kv_heads=16,
    num_layers=16,
    n_routed_experts=64,
    num_experts_per_tok=8,
)

moe_8x_300m = MixtralConfig(
    max_seq_len=4096,
    hidden_dim=768,
    intermediate_dim=1344,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    # gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=8,
    num_experts_per_tok=2,
    # Disable MoE auxiliary loss logging to prevent JAX tracer leaks
    lbl_coef=None,  # Disables load balancing loss logging
    rzl_coef=None,  # Disables router z-loss logging
)

moe_debug = MixtralConfig(
    max_seq_len=1024,
    hidden_dim=128,
    intermediate_dim=128,
    num_layers=4,
    num_heads=4,
    num_kv_heads=4,
    # gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=8,
    num_experts_per_tok=2,
    # Disable MoE auxiliary loss logging to prevent JAX tracer leaks
    lbl_coef=None,  # Disables load balancing loss logging
    rzl_coef=None,  # Disables router z-loss logging
)

model_dict = {
    "150m4k": llama_150m_4096_config,
    "300m4k": llama_300m_4096_config,
    "600m4k": llama_600m_4096_config,
    "1_4b4k": llama_1_4b_4096_config,
    "1_9b4k": llama_1_9b_4096_config,
    "l3b": llama_3b_config,
    "l8b": llama_8b_config,
    "olmoe": olmoe_8x_1_4b,
    "300moe": moe_8x_300m,
    "moedebug": moe_debug,
    "1_5b4k": llama_1_5b_4096_config,
    "3_2b4k": llama_3_2b_4096_config,
    "3_2b4k_alln": llama_3_2b_4096_config_all_norm,
    "3_2b4k_qkn": llama_3_2b_4096_config_qk_norm,
}
