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

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig, DefaultRotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config, QwenConfig

qwen3_0_6b = Qwen3Config(
    seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3072,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

qwen3_1_7b = Qwen3Config(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=6144,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

qwen3_4b = Qwen3Config(
    seq_len=4096,
    hidden_dim=2560,
    intermediate_dim=9728,
    num_heads=32,
    num_kv_heads=8,
    num_layers=36,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

qwen3_8b = Qwen3Config(
    seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=12288,
    num_heads=32,
    num_kv_heads=8,
    num_layers=36,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# same as olmo 32b
qwen3_32b = Qwen3Config(
    seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# Qwen2.5-7B-Instruct
qwen2_5_7b_instruct = QwenConfig(
    seq_len=4096,
    hidden_dim=3584,
    intermediate_dim=18944,
    num_heads=28,
    num_kv_heads=4,
    num_layers=28,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000),
)

# seems not supported by levanter yet
qwen2_5_32b = QwenConfig(
    seq_len=131072,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)

marin_32b = Qwen3Config(
    seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)
