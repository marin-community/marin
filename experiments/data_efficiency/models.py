import dataclasses

from levanter.models.llama import LlamaConfig
from levanter.models.rotary import Llama3RotaryEmbeddingsConfig

from levanter.models.mixtral import MixtralConfig

from experiments.llama import llama_1_4b, llama_1_9b, llama_150m, llama_300m, llama_600m

SEQ_LEN = 4096

llama_150m_4096_config = dataclasses.replace(llama_150m, seq_len=SEQ_LEN)
llama_300m_4096_config = dataclasses.replace(llama_300m, seq_len=SEQ_LEN)
llama_600m_4096_config = dataclasses.replace(llama_600m, seq_len=SEQ_LEN)
llama_1_4b_4096_config = dataclasses.replace(llama_1_4b, seq_len=SEQ_LEN)
llama_1_9b_4096_config = dataclasses.replace(llama_1_9b, seq_len=SEQ_LEN)

llama_3b_config = LlamaConfig(
    seq_len=SEQ_LEN,  # Seq len set to reproduce Tulu SFT
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
    seq_len=SEQ_LEN,  # Seq len set to reproduce Tulu SFT
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
    seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=1024,
    num_heads=16,
    num_kv_heads=16,
    num_layers=16,
    n_routed_experts=64,
    num_experts_per_tok=8,
)

moe_8x_300m = MixtralConfig(
    seq_len=4096,
    hidden_dim=768,
    intermediate_dim=768,
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
    seq_len=1024,
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
}
