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
Specifies a sequence of Qwen 3 models from small to large.
"""

import dataclasses
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig, DefaultRotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

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

# match https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
qwen3_06b = Qwen3Config(
    seq_len=4096,  # speedrun-friendly; HF uses 40960
    hidden_dim=1024,
    intermediate_dim=3072,
    num_layers=28,
    num_heads=16,
    num_kv_heads=8,
    head_dim=128,  # override otherwise 1024 // 16
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
    tie_word_embeddings=True,
    layer_norm_epsilon=1e-6,
)


@dataclasses.dataclass(frozen=True)
class Qwen3NoQKNormConfig(Qwen3Config):
    def attention_config(self):
        cfg = super().attention_config()
        return dataclasses.replace(cfg, qk_norm=None)


qwen3_06b_no_qk_norm = Qwen3NoQKNormConfig(**dataclasses.asdict(qwen3_06b))
