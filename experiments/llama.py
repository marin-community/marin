"""
Specifies a sequence of Llama 3 models from small to large.
"""

from levanter.models.llama import LlamaConfig

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"

llama_150m = LlamaConfig(
    seq_len=4096,
    hidden_dim=512,
    intermediate_dim=1792,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
)

llama_300m = LlamaConfig(
    seq_len=4096,
    hidden_dim=768,
    intermediate_dim=2688,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
)

llama_600m = LlamaConfig(
    seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_layers=24,
    num_heads=16,
    num_kv_heads=8,
)

llama_1_4b = LlamaConfig(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_layers=16,
    num_heads=16,
    num_kv_heads=8,
)

llama_1_9b = LlamaConfig(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_layers=24,
    num_heads=16,
    num_kv_heads=8,
)

llama_3_5b = LlamaConfig(
    seq_len=4096,
    hidden_dim=2560,
    intermediate_dim=8960,
    num_layers=32,
    num_heads=20,
    num_kv_heads=10,
)

llama_8b = LlamaConfig(
    seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,
)

# For scaling laws
scaling_llamas = [llama_150m, llama_300m, llama_600m, llama_1_4b, llama_1_9b, llama_3_5b, llama_8b]
