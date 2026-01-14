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
Proxy/swarm model training configurations based on:

1. OLMo 3 (arXiv:2512.13961) - Constrained data mixing swarm procedure
   - 30M parameter OLMo3 architecture models
   - 3B tokens per proxy (5× Chinchilla)

2. RegMix (arXiv:2407.01492) - Regression-based mixture optimization
   - 1M non-embedding parameter models
   - 1B tokens per proxy

These configurations are designed for quick proxy runs to evaluate data mixtures
before training larger models.
"""

import logging
import os

from levanter.models.llama import LlamaConfig
from levanter.models.olmo3 import Olmo3Config
from levanter.optim import MuonHConfig

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

AUTHOR = Author(
    name="Calvin Xu",
    affiliation="Stanford University",
    url="https://github.com/calvinxu",
)


# =============================================================================
# OLMo3 30M Proxy Configuration (from OLMo 3 technical report)
# =============================================================================
#
# OLMo3 architecture features:
# - Post-norm: LayerNorm after attention/MLP output, before residual
# - QK-norm: LayerNorm on Q and K after projection
# - Mixed attention: 3/4 layers use sliding window, every 4th uses full attention
# - SwiGLU activation
#
# Target: ~30M parameters
# The paper doesn't specify exact layer/width, so we design a config that:
# - Follows OLMo3 architecture patterns
# - Achieves approximately 30M total parameters
#
# With Llama3 tokenizer (vocab=128,256), embeddings dominate small models.
# Using hidden_dim=224, intermediate_dim=560, num_layers=4:
#   - Non-embedding params: ~2.3M
#   - Embedding params: ~28.7M
#   - Total: ~31M (close to 30M target)

olmo3_30m_proxy = Olmo3Config(
    max_seq_len=2048,
    hidden_dim=224,
    intermediate_dim=560,  # ~2.5x hidden_dim
    num_layers=4,
    num_heads=4,  # head_size = 56
    num_kv_heads=4,
    sliding_window=2048,  # Same as seq_len for small models
    tie_word_embeddings=True,  # Saves parameters, common for smaller models
    gradient_checkpointing=True,
    scan_layers=True,
)


def compute_olmo3_params(config: Olmo3Config, vocab_size: int) -> dict:
    """
    Compute parameter count for OLMo3 config.

    OLMo3 has:
    - Post-norm architecture (norm after attention/MLP, before residual)
    - QK normalization (per-head normalization of Q and K)
    - SwiGLU MLP (gate, up, down projections)
    """
    head_size = config.head_dim if config.head_dim else config.hidden_dim // config.num_heads

    # Attention parameters
    q_proj = config.hidden_dim * head_size * config.num_heads
    k_proj = config.hidden_dim * head_size * config.num_kv_heads
    v_proj = config.hidden_dim * head_size * config.num_kv_heads
    o_proj = head_size * config.num_heads * config.hidden_dim
    attn_params = q_proj + k_proj + v_proj + o_proj

    # MLP parameters (SwiGLU: gate, up, down)
    mlp_params = 3 * config.hidden_dim * config.intermediate_dim

    # Layer norm parameters (RMSNorm has only weight, no bias)
    # Post-attention norm + Post-feedforward norm
    layer_norm_params = 2 * config.hidden_dim

    # QK norm parameters (applied per-head)
    # Q norm: (num_kv_heads, num_q_per_kv, head_size) = num_heads * head_size
    # K norm: (num_kv_heads, head_size) = num_kv_heads * head_size
    q_norm_params = config.num_heads * head_size
    k_norm_params = config.num_kv_heads * head_size
    qk_norm_params = q_norm_params + k_norm_params

    # Per layer total
    per_layer = attn_params + mlp_params + layer_norm_params + qk_norm_params

    # Transformer total
    transformer_params = config.num_layers * per_layer + config.hidden_dim  # final norm

    # Embeddings
    embedding_params = vocab_size * config.hidden_dim

    # LM head (0 if tied)
    lm_head_params = 0 if config.tie_word_embeddings else embedding_params

    total = transformer_params + embedding_params + lm_head_params

    return {
        "per_layer": per_layer,
        "transformer": transformer_params,
        "embeddings": embedding_params,
        "lm_head": lm_head_params,
        "total": total,
        "non_embedding": transformer_params + lm_head_params,
    }


# =============================================================================
# RegMix 1M Proxy Configuration (from RegMix paper)
# =============================================================================
#
# RegMix uses a simpler architecture for their proxy models.
# From the paper (Table 9):
#   - nlayers: 2
#   - nheads: 8
#   - dembedding: 256 (hidden_dim)
#   - dmodel: 512 (they use this for intermediate/FFN dim)
#   - vocab: 50,432 (GPTNeoX tokenizer)
#
# They explicitly count parameters EXCLUDING embeddings.
# Target: 1M non-embedding parameters
#
# We'll use LlamaConfig (standard decoder-only transformer with SwiGLU)
# to approximate their setup.
#
# Note: RegMix used a GPT-like architecture, but we use Llama-style
# for consistency with the rest of the codebase. The key is matching
# the non-embedding parameter count.

# GPTNeoX tokenizer vocab size (used in RegMix)
REGMIX_VOCAB_SIZE = 50_432

# RegMix paper Table 9 - 1M proxy config:
#   - nlayers: 2
#   - nheads: 8
#   - dembedding: 256 (hidden_dim)
#   - dmodel: 512 (intermediate_dim)
#   - vocab: 50,432
#
# With their GPT-style MLP (2 projections), this gives ~1.05M non-embedding params.
# With our Llama-style SwiGLU MLP (3 projections), this gives ~1.31M non-embedding params.
# We use their exact dimensions for a fair comparison.
regmix_1m_proxy = LlamaConfig(
    max_seq_len=2048,  # RegMix didn't specify, using reasonable default
    hidden_dim=256,  # dembedding from paper
    intermediate_dim=512,  # dmodel from paper
    num_layers=2,
    num_heads=8,
    num_kv_heads=8,
    tie_word_embeddings=True,
    gradient_checkpointing=False,  # Small model, not needed
    scan_layers=True,
)

# RegMix paper Table 9 - 60M proxy config:
#   - nlayers: 10
#   - nheads: 8
#   - dembedding: 768 (hidden_dim)
#   - dmodel: 1536 (intermediate_dim)
#   - vocab: 50,432
#
# With their GPT-style MLP, this gives ~60M non-embedding params.
# With our Llama-style SwiGLU MLP, this gives more non-embedding params.
regmix_60m_proxy = LlamaConfig(
    max_seq_len=2048,
    hidden_dim=768,  # dembedding from paper
    intermediate_dim=1536,  # dmodel from paper
    num_layers=10,
    num_heads=8,
    num_kv_heads=8,
    tie_word_embeddings=True,
    gradient_checkpointing=True,  # Larger model, enable checkpointing
    scan_layers=True,
)


def compute_llama_non_embedding_params(config: LlamaConfig) -> int:
    """
    Compute non-embedding parameter count for Llama config.
    This matches RegMix's convention of counting parameters.
    """
    head_size = config.head_dim if config.head_dim else config.hidden_dim // config.num_heads

    # Attention parameters
    q_proj = config.hidden_dim * head_size * config.num_heads
    k_proj = config.hidden_dim * head_size * config.num_kv_heads
    v_proj = config.hidden_dim * head_size * config.num_kv_heads
    o_proj = head_size * config.num_heads * config.hidden_dim
    attn_params = q_proj + k_proj + v_proj + o_proj

    # MLP parameters (SwiGLU: gate, up, down)
    mlp_params = 3 * config.hidden_dim * config.intermediate_dim

    # Layer norm parameters (2 per layer: input_layernorm, post_attention_layernorm)
    layer_norm_params = 2 * config.hidden_dim

    # Per layer total
    per_layer = attn_params + mlp_params + layer_norm_params

    # Transformer total (includes final layer norm)
    transformer_params = config.num_layers * per_layer + config.hidden_dim

    # LM head is tied to embeddings, so 0 additional params
    # (RegMix counts non-embedding params, so we don't include LM head either way)

    return transformer_params


# =============================================================================
# Training Configuration
# =============================================================================

def get_num_train_steps(total_tokens: int, batch_size: int, seq_len: int) -> int:
    """Compute the number of training steps for a given token budget."""
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


def build_olmo3_proxy_config() -> tuple[str, SpeedrunConfig]:
    """
    Build OLMo3 30M proxy configuration.

    Training setup:
    - 3B tokens (5× Chinchilla for 30M model; Chinchilla optimal is 20× params)
    - MuonH optimizer (hyperparameters scaled from 130M Qwen3 config)
    - Batch size chosen for v5p-8
    """
    model_config = olmo3_30m_proxy

    # Token budget: 3B tokens (as specified in OLMo3 paper)
    total_tokens = 3_000_000_000

    # Batch size and sequence length
    batch_size = 128
    seq_len = model_config.max_seq_len

    num_train_steps = get_num_train_steps(total_tokens, batch_size, seq_len)

    # MuonH optimizer config (adapted from 130M Qwen3 config for small models)
    muon_config = MuonHConfig(
        learning_rate=0.02,
        adam_lr=0.008,
        min_lr_ratio=0,
        momentum=0.95,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1,
        warmup=1000,
    )

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=muon_config.learning_rate,
        optimizer_config=muon_config,
        steps_per_eval=1000,
    )

    # Print parameter counts
    params = compute_olmo3_params(model_config, llama3_tokenizer_vocab_size)
    logger.info(f"OLMo3 30M proxy config:")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Non-embedding parameters: {params['non_embedding']:,}")
    logger.info(f"  Embedding parameters: {params['embeddings']:,}")
    logger.info(f"  Tokens to train: {total_tokens:,}")
    logger.info(f"  Training steps: {num_train_steps:,}")

    config = SpeedrunConfig(
        author=AUTHOR,
        description="OLMo3 ~30M proxy model for swarm data mixing (3B tokens, 5× Chinchilla, MuonH)",
        model_config=model_config,
        train_config=train_config,
    )

    return "olmo3_30m_proxy_3B_muonh_1", config


def build_regmix_proxy_config() -> tuple[str, SpeedrunConfig]:
    """
    Build RegMix 1M proxy configuration.

    Training setup:
    - 1B tokens (as specified in RegMix paper)
    - 1M non-embedding parameters
    - MuonH optimizer (hyperparameters scaled from 130M Qwen3 config)
    - Batch size chosen for v5p-8
    """
    model_config = regmix_1m_proxy

    # Token budget: 1B tokens (as specified in RegMix paper)
    total_tokens = 1_000_000_000

    # Batch size and sequence length
    batch_size = 128
    seq_len = model_config.max_seq_len

    num_train_steps = get_num_train_steps(total_tokens, batch_size, seq_len)

    muon_config = MuonHConfig(
        learning_rate=0.02,
        adam_lr=0.008,
        min_lr_ratio=0,
        momentum=0.95,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1,
        warmup=500,  # Shorter warmup for fewer steps
    )

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=muon_config.learning_rate,
        optimizer_config=muon_config,
        steps_per_eval=1000,
    )

    # Print parameter counts
    non_emb_params = compute_llama_non_embedding_params(model_config)
    total_params = model_config.total_trainable_params(llama3_tokenizer_vocab_size)
    logger.info(f"RegMix 1M proxy config:")
    logger.info(f"  Non-embedding parameters: {non_emb_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Tokens to train: {total_tokens:,}")
    logger.info(f"  Training steps: {num_train_steps:,}")

    config = SpeedrunConfig(
        author=AUTHOR,
        description="RegMix ~1M non-embedding param proxy model (1B tokens, MuonH)",
        model_config=model_config,
        train_config=train_config,
    )

    return "regmix_1m_proxy_1B_muonh_1", config


def build_regmix_60m_proxy_config() -> tuple[str, SpeedrunConfig]:
    """
    Build RegMix 60M proxy configuration.

    Training setup:
    - 1B tokens (same as 1M proxy in RegMix paper)
    - 60M non-embedding parameters (paper Table 9)
    - MuonH optimizer (hyperparameters from 130M Qwen3 config)
    - Batch size chosen for v5p-8
    """
    model_config = regmix_60m_proxy

    # Token budget: 1B tokens (same as their 1M and 60M proxy runs)
    total_tokens = 1_000_000_000

    # Batch size and sequence length
    batch_size = 128
    seq_len = model_config.max_seq_len

    num_train_steps = get_num_train_steps(total_tokens, batch_size, seq_len)

    muon_config = MuonHConfig(
        learning_rate=0.02,
        adam_lr=0.008,
        min_lr_ratio=0,
        momentum=0.95,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1,
        warmup=1000,
    )

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=muon_config.learning_rate,
        optimizer_config=muon_config,
        steps_per_eval=1000,
    )

    # Print parameter counts
    non_emb_params = compute_llama_non_embedding_params(model_config)
    total_params = model_config.total_trainable_params(llama3_tokenizer_vocab_size)
    logger.info(f"RegMix 60M proxy config:")
    logger.info(f"  Non-embedding parameters: {non_emb_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Tokens to train: {total_tokens:,}")
    logger.info(f"  Training steps: {num_train_steps:,}")

    config = SpeedrunConfig(
        author=AUTHOR,
        description="RegMix ~60M non-embedding param proxy model (1B tokens, MuonH)",
        model_config=model_config,
        train_config=train_config,
    )

    return "regmix_60m_proxy_1B_muonh_1", config


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    # Build configurations
    olmo3_name, olmo3_config = build_olmo3_proxy_config()
    regmix_1m_name, regmix_1m_config = build_regmix_proxy_config()
    regmix_60m_name, regmix_60m_config = build_regmix_60m_proxy_config()

    # Print run info
    print("\n" + "=" * 80)
    print("OLMo3 30M Proxy Configuration (3B tokens)")
    print("=" * 80)
    olmo3_config.print_run_info()

    print("\n" + "=" * 80)
    print("RegMix 1M Proxy Configuration (1B tokens)")
    print("=" * 80)
    regmix_1m_config.print_run_info()

    print("\n" + "=" * 80)
    print("RegMix 60M Proxy Configuration (1B tokens)")
    print("=" * 80)
    regmix_60m_config.print_run_info()

    # Create training steps
    steps = []
    steps.extend(default_speedrun(olmo3_name, olmo3_config))
    steps.extend(default_speedrun(regmix_1m_name, regmix_1m_config))
    steps.extend(default_speedrun(regmix_60m_name, regmix_60m_config))

    executor_main(
        steps=steps,
        description="Proxy model training runs for data mixture optimization (OLMo3 + RegMix)"
    )


if __name__ == "__main__":
    main()
