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
Specifies a sequence of Llama 3 models from small to large.
"""

from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config, QwenConfig
from levanter.utils.activation import ActivationFunctionEnum

# default head_dim = hidden_dim // num_heads = 64, mismatch with "Qwen/Qwen3-0.6B"
qwen3_0_6b = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3072,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# qwen3_0_6b_hd128: head_dim=128, identical to "Qwen/Qwen3-0.6B"
qwen3_0_6b_hd128 = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3072,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
    head_dim=128,
)

qwen3_1_7b = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=6144,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# default head_dim = hidden_dim // num_heads = 80, mismatch with "Qwen/Qwen3-4B"
qwen3_4b = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=2560,
    intermediate_dim=9728,
    num_heads=32,
    num_kv_heads=8,
    num_layers=36,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# qwen3_4b_hd128: head_dim=128, identical to "Qwen/Qwen3-4B"
qwen3_4b_hd128 = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=2560,
    intermediate_dim=9728,
    num_heads=32,
    num_kv_heads=8,
    num_layers=36,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
    head_dim=128,
)

qwen3_8b_tokenizer = "Qwen/Qwen3-8B"
qwen3_8b = Qwen3Config(
    # Matching defaults in https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
    max_seq_len=32768,
    hidden_dim=4096,
    intermediate_dim=12288,
    num_heads=32,
    num_kv_heads=8,
    num_layers=36,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,
    tie_word_embeddings=False,
    reference_checkpoint="Qwen/Qwen3-8B",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)

# same as olmo 32b
qwen3_32b = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)

qwen2_5_1_5b_tokenizer = "Qwen/Qwen2.5-1.5B"
qwen2_5_1_5b = QwenConfig(
    # Matching defaults in https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json
    max_seq_len=32768,
    hidden_dim=1536,
    intermediate_dim=8960,
    num_heads=12,
    num_kv_heads=2,
    num_layers=28,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,
    tie_word_embeddings=True,  # NOTE: This is different than 7B
    reference_checkpoint="Qwen/Qwen2.5-1.5B",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)

qwen2_5_1_5b_instruct_tokenizer = "Qwen/Qwen2.5-1.5B-Instruct"
qwen2_5_1_5b_instruct = QwenConfig(
    # Matching defaults in https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
    max_seq_len=32768,
    hidden_dim=1536,
    intermediate_dim=8960,
    num_heads=12,
    num_kv_heads=2,
    num_layers=28,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,
    tie_word_embeddings=True,  # NOTE: This is different than 7B
    reference_checkpoint="Qwen/Qwen2.5-1.5B-Instruct",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)

qwen2_5_3b_tokenizer = "Qwen/Qwen2.5-3B"
qwen2_5_3b = QwenConfig(
    # Matching defaults in https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json
    max_seq_len=32768,
    hidden_dim=2048,
    intermediate_dim=11008,
    num_heads=16,
    num_kv_heads=2,
    num_layers=36,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,
    tie_word_embeddings=True,  # NOTE: This is different than 7B
    reference_checkpoint="Qwen/Qwen2.5-3B",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)

qwen2_5_3b_instruct_tokenizer = "Qwen/Qwen2.5-3B-Instruct"
qwen2_5_3b_instruct = QwenConfig(
    # Matching defaults in https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json
    max_seq_len=32768,
    hidden_dim=2048,
    intermediate_dim=11008,
    num_heads=16,
    num_kv_heads=2,
    num_layers=36,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,
    tie_word_embeddings=True,  # NOTE: This is different than 7B
    reference_checkpoint="Qwen/Qwen2.5-3B-Instruct",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)

qwen2_5_7b_tokenizer = "Qwen/Qwen2.5-7B"
qwen2_5_7b = QwenConfig(
    # Matching defaults in https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json
    max_seq_len=131072,
    hidden_dim=3584,
    intermediate_dim=18944,
    num_heads=28,
    num_kv_heads=4,
    num_layers=28,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,
    tie_word_embeddings=False,
    reference_checkpoint="Qwen/Qwen2.5-7B",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)

qwen2_5_7b_instruct_tokenizer = "Qwen/Qwen2.5-7B-Instruct"
qwen2_5_7b_instruct = QwenConfig(
    # Matching defaults in https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
    max_seq_len=131072,
    hidden_dim=3584,
    intermediate_dim=18944,
    num_heads=28,
    num_kv_heads=4,
    num_layers=28,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,
    tie_word_embeddings=False,
    reference_checkpoint="Qwen/Qwen2.5-7B-Instruct",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)

qwen2_5_32b_tokenizer = "Qwen/Qwen2.5-32B"
qwen2_5_32b = QwenConfig(
    # Matching defaults in https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json
    max_seq_len=131072,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    tie_word_embeddings=False,
    reference_checkpoint="Qwen/Qwen2.5-32B",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)

qwen2_5_32b_instruct_tokenizer = "Qwen/Qwen2.5-32B-Instruct"
qwen2_5_32b_instruct = QwenConfig(
    # Matching defaults in https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/config.json
    max_seq_len=131072,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,  # NOTE: This is changed from the base 32B version
    tie_word_embeddings=False,
    reference_checkpoint="Qwen/Qwen2.5-32B-Instruct",
    rope=DefaultRotaryEmbeddingsConfig(
        theta=1000000.0,
        factor=1.0
    ),
)


marin_32b = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)
