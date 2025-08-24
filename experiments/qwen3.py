"""
Specifies a sequence of Llama 3 models from small to large.
"""

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
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
