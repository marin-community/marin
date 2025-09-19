"""
Specifies a sequence of Llama 3 models from small to large.
"""

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

qwen3_0_6b = Qwen3Config(
    seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3072,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
)

qwen3_1_7b = Qwen3Config(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=6144,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
)

qwen3_4b = Qwen3Config(
    seq_len=4096,
    hidden_dim=2560,
    intermediate_dim=9728,
    num_heads=32,
    num_kv_heads=8,
    num_layers=36,
    rope=Llama3RotaryEmbeddingsConfig(),
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
