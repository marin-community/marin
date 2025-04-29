"""
Specifies a sequence of Llama 3 models from small to large.
"""

from levanter.models.llama import LlamaConfig
from levanter.models.rotary import Llama3RotaryEmbeddingsConfig

from experiments.simple_train_config import SimpleTrainConfig

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"
llama3_tokenizer_vocab_size = 128_256
llama3_instruct_tokenizer = "meta-llama/Meta-Llama-3.1-8B-Instruct"


llama_150m = LlamaConfig(
    seq_len=1024,
    hidden_dim=512,
    intermediate_dim=1792,
    num_heads=8,
    num_kv_heads=8,
    num_layers=6,
)

llama_300m = LlamaConfig(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=2688,
    num_heads=12,
    num_kv_heads=12,
    num_layers=12,
)

llama_600m = LlamaConfig(
    seq_len=1024,
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
    rope=Llama3RotaryEmbeddingsConfig(),
)


llama_8b_old_rotary = LlamaConfig(
    seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    # Levanter defaults to Llama2 rotary
    # rope=Llama3RotaryEmbeddingsConfig(),
)


llama_13b = LlamaConfig(
    seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=13824,
    num_heads=40,
    num_kv_heads=8,
    num_layers=40,
    rope=Llama3RotaryEmbeddingsConfig(),
)


# With Llama 3 tokenizer, this is 24B
llama_24b = LlamaConfig(
    seq_len=4096,
    hidden_dim=6144,
    intermediate_dim=16384,
    num_heads=48,
    num_kv_heads=16,
    num_layers=56,
    rope=Llama3RotaryEmbeddingsConfig(),
)


# same as olmo 32b
llama_32b = LlamaConfig(
    seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)


llama_56b = LlamaConfig(
    seq_len=4096,
    hidden_dim=8192,
    intermediate_dim=28672,
    num_heads=64,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)


llama_70b = LlamaConfig(
    seq_len=4096,
    hidden_dim=8192,
    intermediate_dim=28672,
    num_heads=64,
    num_kv_heads=8,
    num_layers=80,
    rope=Llama3RotaryEmbeddingsConfig(),
)


llama_150m_train_config = SimpleTrainConfig(
    tpu_type="v4-32",
    train_batch_size=512,
    num_train_steps=20000,  # 1024 * 1024 * 20000 = 20B tokens
    learning_rate=3e-3,
    weight_decay=0.1,
)
# (18B is way overtrained, but...)

llama_300m_train_config = SimpleTrainConfig(
    tpu_type="v4-64",
    train_batch_size=1024,
    num_train_steps=18000,  # 1024 * 1024 * 18000 = 18B tokens
    learning_rate=3e-3,
    weight_decay=0.1,
)
# (18B is way overtrained, but...)

llama_1_4b_train_config = SimpleTrainConfig(
    tpu_type="v4-128",
    train_batch_size=1024,
    num_train_steps=10000,  # 4096 * 1024 * 10000 = 42B tokens
    learning_rate=3e-4,
    weight_decay=0.1,
)

llama_8b_train_config = SimpleTrainConfig(
    tpu_type="v4-512",
    train_batch_size=1024,
    num_train_steps=40000,  # 4096 * 1024 * 40000 = 167B tokens
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
    learning_rate=2e-3,
    weight_decay=0.05,
)


def compute_num_parameters(config: LlamaConfig, vocab_size: int) -> int:

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
    embedding_params = 2 * vocab_size * config.hidden_dim

    return nonembedding_params + embedding_params


# For scaling laws
scaling_llamas = [llama_150m, llama_300m, llama_600m, llama_1_4b, llama_1_9b, llama_3_5b, llama_8b]


if __name__ == "__main__":
    for llama in scaling_llamas:
        print(f"{compute_num_parameters(llama, llama3_tokenizer_vocab_size) :,}")
