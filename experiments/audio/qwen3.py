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
Specifies a sequence of small Qwen 3 models (below 600M parameters)
"""

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

# Official Qwen3 (>=600M parameters) uses head_dim=128
# This scripts follows experiments/qwen3.py,
# which uses Llama-like config: hidden_dim//num_heads => head_dim

# These models are based on experiments/llama.py, but with some values changed
# to match the target size

# 305M (based on llama_300m, adjusted for ~300M params)
qwen3_300m = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=768,
    intermediate_dim=2688,
    num_heads=12,
    num_kv_heads=12,
    num_layers=22,  # Increased from 12 to reach ~300M
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# Qwen3 ~150M (based on llama_150m, adjusted for ~150M params)
qwen3_150m = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=512,
    intermediate_dim=1792,
    num_heads=8,
    num_kv_heads=8,
    num_layers=20,  # Increased from 6 to reach ~150M
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# Qwen3 ~135M (based on llama_150m, adjusted for ~135M params)
qwen3_135m = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=512,
    intermediate_dim=1792,
    num_heads=8,
    num_kv_heads=8,
    num_layers=15,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)


# Qwen3 ~75M (based on llama_75m, adjusted for ~75M params)
qwen3_75m = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=336,
    intermediate_dim=1024,
    num_heads=4,
    num_kv_heads=4,
    num_layers=16,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# Qwen3 ~50M (based on llama_50m, adjusted for ~50M params)
qwen3_50m = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=256,  # Increased from 192
    intermediate_dim=896,  # Increased from 448
    num_heads=4,  # Increased from 2
    num_kv_heads=4,  # Increased from 2
    num_layers=12,  # Increased from 4 to reach ~50M
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# Qwen3 ~30M (based on llama_30m, adjusted for ~30M params)
qwen3_30m = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=192,
    intermediate_dim=448,
    num_heads=2,
    num_kv_heads=2,
    num_layers=8,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)
