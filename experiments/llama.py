"""
Specifies a sequence of Llama 3 models from small to large.
"""

from levanter.models.llama import LlamaConfig

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"

llama_150m = LlamaConfig(
    seq_len=4096,
    hidden_dim=512,
    intermediate_dim=1792,
    num_heads=8,
    num_kv_heads=8,
    num_layers=6,
)

llama_300m = LlamaConfig(
    seq_len=4096,
    hidden_dim=768,
    intermediate_dim=2688,
    num_heads=12,
    num_kv_heads=12,
    num_layers=12,
)

llama_600m = LlamaConfig(
    seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
)

llama_1_4b = LlamaConfig(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=16,
)

llama_1_9b = LlamaConfig(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
)

llama_3_5b = LlamaConfig(
    seq_len=4096,
    hidden_dim=2560,
    intermediate_dim=8960,
    num_heads=20,
    num_kv_heads=10,
    num_layers=32,
)

llama_8b = LlamaConfig(
    seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
)


def compute_num_parameters(config: LlamaConfig) -> int:
    head_size = config.hidden_dim // config.num_heads
    q_params = config.num_heads * head_size * config.hidden_dim
    k_params = config.num_kv_heads * head_size * config.hidden_dim
    v_params = config.num_kv_heads * head_size * config.hidden_dim
    o_params = config.num_heads * head_size * config.hidden_dim
    attention_params = q_params + k_params + v_params + o_params

    layer_norm_params = 2 * config.hidden_dim

    gate_params = config.hidden_dim * config.intermediate_dim
    up_params = config.hidden_dim * config.intermediate_dim
    down_params = config.intermediate_dim * config.hidden_dim
    mlp_params = gate_params + up_params + down_params

    nonembedding_params = config.num_layers * (attention_params + mlp_params + layer_norm_params)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llama3_tokenizer)
    embedding_params = 2 * tokenizer.vocab_size * config.hidden_dim

    return nonembedding_params + embedding_params


# For scaling laws
scaling_llamas = [llama_150m, llama_300m, llama_600m, llama_1_4b, llama_1_9b, llama_3_5b, llama_8b]


if __name__ == "__main__":
    for llama in scaling_llamas:
        print(f"{compute_num_parameters(llama):,}")
