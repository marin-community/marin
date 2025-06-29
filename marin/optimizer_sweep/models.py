from levanter.models.attention import AttentionBackend
from levanter.models.llama import LlamaConfig


def calculate_param(config):
    return config.num_layers * (
        config.hidden_dim * config.intermediate_dim * 3 + config.hidden_dim * config.hidden_dim * 4
    )


def calculate_chinchilla(config):
    # 1x Chinchilla rule of thumb = 20 tokens/parameter
    return calculate_param(config) * 20


# ----- Model definitions -----

# 1) ~130M params: embed_dim=512 => intermediate=2048 => 8 heads
# use jax flash attention for better numerical stability
llama_130m = LlamaConfig(
    seq_len=4096,
    hidden_dim=512,
    intermediate_dim=2048,
    num_heads=8,
    num_kv_heads=8,  # following the same ratio as the original code
    num_layers=32,
    attn_backend=AttentionBackend.JAX_FLASH,
)

llama_130m_new = LlamaConfig(
    seq_len=4096,
    hidden_dim=512,
    intermediate_dim=2048,
    num_heads=8,
    num_kv_heads=8,  # following the same ratio as the original code
    num_layers=32,
    upcast_attn=True,
)

llama_130m_old = LlamaConfig(
    seq_len=4096,
    hidden_dim=512,
    intermediate_dim=2048,
    num_heads=8,
    num_kv_heads=8,  # following the same ratio as the original code
    num_layers=32,
    attn_backend=AttentionBackend.SPLASH,
)

# 2) ~300M params: embed_dim=768 => intermediate=3072 => 12 heads
llama_300m = LlamaConfig(
    seq_len=4096,
    hidden_dim=768,
    intermediate_dim=3072,
    num_heads=12,
    num_kv_heads=12,
    num_layers=32,
)

# 3) ~520M params: embed_dim=1024 => intermediate=4096 => 16 heads
llama_520m = LlamaConfig(
    seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=4096,
    num_heads=16,
    num_kv_heads=16,
    num_layers=32,
)

# 4) ~1.2B params: embed_dim=1536 => intermediate=6144 => 24 heads
llama_1_2b = LlamaConfig(
    seq_len=4096,
    hidden_dim=1536,
    intermediate_dim=6144,
    num_heads=24,
    num_kv_heads=24,
    num_layers=32,
)

# ----- Registry of models -----

map_tag_to_model = {
    "130m": llama_130m,
    "300m": llama_300m,
    "520m": llama_520m,
    "1.2b": llama_1_2b,
    "130m_new": llama_130m_new,
    "130m_old": llama_130m_old,
}


# ----- Example usage -----

if __name__ == "__main__":
    for tag, model_cfg in map_tag_to_model.items():
        params = calculate_param(model_cfg)
        chinchilla_tokens = calculate_chinchilla(model_cfg)
        print(f"{tag}: ~{params/1e6:.2f}M params, 1x Chinchilla = ~{chinchilla_tokens/1e9:.1f}B tokens")
