"""Fuzzer test comparing Grug implementation against HuggingFace GPT-OSS.

Tests FAIL when Grug doesn't match GPT-OSS, guiding the porting effort.
Run with: uv run pytest tests/test_grug_vs_gpt_oss.py -v -s

This test imports ONLY from:
  1. levanter.grug.* (the code being tested)
  2. transformers.models.gpt_oss (the reference implementation)
  3. grugfuzz utilities
"""

from dataclasses import dataclass
import os

# Allow simulated multi-device meshes on CPU before JAX initializes.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from grugfuzz import (
    Choice,
    FuzzSpace,
    compare,
    run_hf,
    sample_fuzz_cases,
    torch_to_jax,
)

# Check for torch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Try to import HuggingFace GPT-OSS (the reference)
try:
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssRMSNorm,
        GptOssRotaryEmbedding,
        GptOssAttention,
        GptOssMLP,
        GptOssExperts,
        GptOssTopKRouter,
        GptOssForCausalLM,
    )
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    HAS_GPT_OSS = True
except ImportError:
    HAS_GPT_OSS = False

# Try to import HF mask helpers (used for attention mask parity tests)
try:
    from transformers.masking_utils import create_causal_mask

    HAS_MASK_UTILS = True
except ImportError:
    HAS_MASK_UTILS = False

# Try to import actual Grug implementation (what we're testing)
try:
    from levanter.grug.model import (
        GrugModelConfig,
        GrugModelParameters,
        GrugBlockParams,
        GrugAttentionParams,
        GrugMoEParams,
        activations,
        clipped_gated_activation,
        moe_forward,
        rms_norm,
        mlp,
        forward,
        init_parameters,
    )
    from levanter.grug.attention import (
        RotaryConfig,
        apply_rotary_embedding,
        attention,
        reference_attention,
    )

    HAS_GRUG = True
except ImportError:
    HAS_GRUG = False


@dataclass
class FuzzConfig:
    """Shared test configuration."""

    vocab_size: int = 1024
    hidden_dim: int = 256
    intermediate_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 32
    num_layers: int = 2
    max_seq_len: int = 64
    batch_size: int = 2
    seq_len: int = 32
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    num_experts: int = 4
    num_experts_per_tok: int = 2
    seed: int = 42


CFG = FuzzConfig()
ATOL_SIMPLE = 1e-5
RTOL_SIMPLE = 1e-5
ATOL_COMPLEX = 1e-4
RTOL_COMPLEX = 1e-4

_FUZZ_SPACE = FuzzSpace(
    specs={
        "hidden_dim": Choice((128, 192, 256, 384, 512)),
        "intermediate_dim": Choice((256, 384, 512, 768, 1024)),
        "num_heads": Choice((4, 6, 8)),
        "num_kv_heads": Choice((1, 2, 4)),
        "head_dim": Choice((16, 32, 64)),
        "num_layers": Choice((1, 2, 3)),
        "max_seq_len": Choice((32, 64, 128)),
        "seq_len": Choice((16, 32, 64)),
        "vocab_size": Choice((256, 512, 1024, 2048)),
        "num_experts": Choice((2, 4, 8)),
        "num_experts_per_tok": Choice((1, 2, 4)),
        "batch_size": Choice((1, 2, 4)),
    },
    constraints=(
        lambda c: c["hidden_dim"] == c["num_heads"] * c["head_dim"],
        lambda c: c["num_heads"] % c["num_kv_heads"] == 0,
        lambda c: c["seq_len"] <= c["max_seq_len"],
        lambda c: c["num_experts_per_tok"] <= c["num_experts"],
    ),
)

_FUZZ_COUNT = 16
FUZZ_CASES = tuple(
    (seed, FuzzConfig(**sample))
    for seed, sample in sample_fuzz_cases(_FUZZ_SPACE, seed=9917, count=_FUZZ_COUNT)
)

EDGE_CASES = (
    FuzzConfig(
        vocab_size=256,
        hidden_dim=128,
        intermediate_dim=256,
        num_heads=4,
        num_kv_heads=1,
        head_dim=32,
        num_layers=1,
        max_seq_len=64,
        seq_len=1,
        batch_size=1,
        num_experts=2,
        num_experts_per_tok=1,
    ),
    FuzzConfig(
        vocab_size=2048,
        hidden_dim=512,
        intermediate_dim=1024,
        num_heads=8,
        num_kv_heads=4,
        head_dim=64,
        num_layers=3,
        max_seq_len=128,
        seq_len=128,
        batch_size=2,
        num_experts=8,
        num_experts_per_tok=4,
    ),
)

FULL_FORWARD_CASES = FUZZ_CASES + tuple((9000 + idx, cfg) for idx, cfg in enumerate(EDGE_CASES))

_MASK_FUZZ_SPACE = FuzzSpace(
    specs={
        "hidden_dim": Choice((128, 256)),
        "intermediate_dim": Choice((256, 512)),
        "num_heads": Choice((4, 8)),
        "num_kv_heads": Choice((1, 2, 4)),
        "head_dim": Choice((32, 64)),
        "num_layers": Choice((1, 2)),
        "max_seq_len": Choice((16, 32, 64)),
        "seq_len": Choice((8, 16, 32)),
        "vocab_size": Choice((256, 512, 1024)),
        "num_experts": Choice((2, 4)),
        "num_experts_per_tok": Choice((1, 2)),
        "batch_size": Choice((1, 2, 4)),
    },
    constraints=(
        lambda c: c["hidden_dim"] == c["num_heads"] * c["head_dim"],
        lambda c: c["num_heads"] % c["num_kv_heads"] == 0,
        lambda c: c["seq_len"] <= c["max_seq_len"],
        lambda c: c["num_experts_per_tok"] <= c["num_experts"],
    ),
)

MASK_FUZZ_CASES = tuple(
    (seed, FuzzConfig(**sample))
    for seed, sample in sample_fuzz_cases(_MASK_FUZZ_SPACE, seed=4242, count=10)
)


def _available_mesh_shapes(max_meshes: int = 3) -> tuple[tuple[int, int], ...]:
    device_count = len(jax.devices())
    if device_count <= 0:
        return ((1, 1),)

    shapes = [(data, device_count // data) for data in range(1, device_count + 1) if device_count % data == 0]
    if len(shapes) <= max_meshes:
        return tuple(shapes)

    balanced = min(shapes, key=lambda s: abs(s[0] - s[1]))
    preferred = [shapes[0], shapes[-1], balanced]
    unique = []
    for shape in preferred:
        if shape not in unique:
            unique.append(shape)
    return tuple(unique[:max_meshes])


def _mesh_id(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


MESH_SHAPES = _available_mesh_shapes()


def _random_attention_mask(cfg: FuzzConfig, seed: int) -> "torch.Tensor":
    """Create a padding mask with varied lengths; ensure at least one pad when possible."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, cfg.seq_len + 1, size=cfg.batch_size)
    if cfg.seq_len > 1 and np.all(lengths == cfg.seq_len):
        lengths[0] = cfg.seq_len - 1
    mask = np.zeros((cfg.batch_size, cfg.seq_len), dtype=np.int64)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return torch.from_numpy(mask)


def _build_hf_config(cfg: FuzzConfig) -> "GptOssConfig":
    """Create a GPT-OSS config matching Grug's assumptions."""
    return GptOssConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_dim,
        intermediate_size=cfg.intermediate_dim,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        rope_scaling=None,
        attention_dropout=0.0,
        num_local_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        layer_types=["full_attention"] * cfg.num_layers,
    )


def _build_grug_config(cfg: FuzzConfig) -> GrugModelConfig:
    """Create a Grug config aligned with GPT-OSS features."""
    return GrugModelConfig(
        vocab_size=cfg.vocab_size,
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.max_seq_len,
        layer_norm_eps=cfg.rms_norm_eps,
        rope=RotaryConfig(theta=cfg.rope_theta),
        use_clipped_gated_activation=True,
        clipped_gated_alpha=1.702,
        use_attention_sinks=True,
        use_moe=True,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
    )


def _transpose(weight: "torch.Tensor") -> jax.Array:
    return jnp.transpose(torch_to_jax(weight))


def _build_grug_params_from_hf(hf_model: "GptOssForCausalLM", cfg: FuzzConfig) -> GrugModelParameters:
    """Transfer HF GPT-OSS weights into Grug parameter structures."""
    token_embed = torch_to_jax(hf_model.model.embed_tokens.weight)
    output_proj = _transpose(hf_model.lm_head.weight)
    final_norm = torch_to_jax(hf_model.model.norm.weight)

    blocks: list[GrugBlockParams] = []
    for layer in hf_model.model.layers:
        attn = GrugAttentionParams(
            w_q=_transpose(layer.self_attn.q_proj.weight),
            w_k=_transpose(layer.self_attn.k_proj.weight),
            w_v=_transpose(layer.self_attn.v_proj.weight),
            w_o=_transpose(layer.self_attn.o_proj.weight),
            sinks=torch_to_jax(layer.self_attn.sinks),
        )

        router = _transpose(layer.mlp.router.weight)
        router_bias = torch_to_jax(layer.mlp.router.bias)
        gate_weights: list[jax.Array] = []
        up_weights: list[jax.Array] = []
        down_weights: list[jax.Array] = []
        gate_biases: list[jax.Array] = []
        up_biases: list[jax.Array] = []
        down_biases: list[jax.Array] = []
        gate_up = layer.mlp.experts.gate_up_proj
        down_proj = layer.mlp.experts.down_proj
        gate_up_bias = layer.mlp.experts.gate_up_proj_bias
        down_bias = layer.mlp.experts.down_proj_bias
        for expert_idx in range(gate_up.shape[0]):
            gate_up_weights = gate_up[expert_idx].detach()
            gate_weight = gate_up_weights[:, ::2]
            up_weight = gate_up_weights[:, 1::2]
            down_weight = down_proj[expert_idx].detach()
            gate_up_bias_weights = gate_up_bias[expert_idx].detach()
            gate_bias = gate_up_bias_weights[::2]
            up_bias = gate_up_bias_weights[1::2]
            down_bias_weights = down_bias[expert_idx].detach()
            gate_weights.append(torch_to_jax(gate_weight))
            up_weights.append(torch_to_jax(up_weight))
            down_weights.append(torch_to_jax(down_weight))
            gate_biases.append(torch_to_jax(gate_bias))
            up_biases.append(torch_to_jax(up_bias))
            down_biases.append(torch_to_jax(down_bias_weights))

        blocks.append(
            GrugBlockParams(
                attn=attn,
                rms_attn=torch_to_jax(layer.input_layernorm.weight),
                rms_mlp=torch_to_jax(layer.post_attention_layernorm.weight),
                moe=GrugMoEParams(
                    router=router,
                    router_bias=router_bias,
                    gate=jnp.stack(gate_weights, axis=0),
                    up=jnp.stack(up_weights, axis=0),
                    down=jnp.stack(down_weights, axis=0),
                    gate_bias=jnp.stack(gate_biases, axis=0),
                    up_bias=jnp.stack(up_biases, axis=0),
                    down_bias=jnp.stack(down_biases, axis=0),
                ),
            )
        )

    return GrugModelParameters(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=final_norm,
    )


# ---------------------------------------------------------------------------
# Tests comparing actual implementations
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.skipif(not HAS_GPT_OSS, reason="GPT-OSS not importable")
@pytest.mark.skipif(not HAS_GRUG, reason="Grug not importable")
class TestRMSNorm:
    """RMSNorm should match between GPT-OSS and Grug."""

    def test_rms_norm_matches(self):
        torch.manual_seed(CFG.seed)

        hf_norm = GptOssRMSNorm(CFG.hidden_dim, eps=CFG.rms_norm_eps)
        x_torch = torch.randn(CFG.batch_size, CFG.seq_len, CFG.hidden_dim)
        x_jax = torch_to_jax(x_torch)
        weight_jax = torch_to_jax(hf_norm.weight)

        hf_out = run_hf(hf_norm, x_torch)
        grug_out = rms_norm(x_jax, weight_jax, CFG.rms_norm_eps)

        result = compare(hf_out, grug_out, name="rms_norm", atol=ATOL_SIMPLE, rtol=RTOL_SIMPLE)
        assert result.passed, f"RMSNorm mismatch: {result.failure_summary}"


# ---------------------------------------------------------------------------
# Numerical Comparison Tests - verify implementations match GPT-OSS
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.skipif(not HAS_GPT_OSS, reason="GPT-OSS not importable")
@pytest.mark.skipif(not HAS_GRUG, reason="Grug not importable")
class TestClippedGatedActivationNumerical:
    """Numerically verify clipped gated activation matches GPT-OSS."""

    def test_single_expert_activation_matches(self):
        """Test that a single Grug expert matches GptOssExperts with one expert selected."""
        torch.manual_seed(CFG.seed)

        # Create a minimal GptOssConfig for GptOssExperts
        hf_config = GptOssConfig(
            hidden_size=CFG.hidden_dim,
            intermediate_size=CFG.intermediate_dim,
            num_local_experts=1,  # Single expert for simplicity
            num_experts_per_tok=1,
        )

        # Create and initialize GptOssExperts
        hf_experts = GptOssExperts(hf_config)
        hf_experts.eval()

        # Initialize weights with known values
        torch.nn.init.normal_(hf_experts.gate_up_proj, std=0.02)
        torch.nn.init.zeros_(hf_experts.gate_up_proj_bias)
        torch.nn.init.normal_(hf_experts.down_proj, std=0.02)
        torch.nn.init.zeros_(hf_experts.down_proj_bias)

        # Create input
        batch, seq = 2, 8
        x_torch = torch.randn(batch, seq, CFG.hidden_dim) * 0.1

        # Set up routing: all tokens go to expert 0 with weight 1.0
        num_tokens = batch * seq
        router_indices = torch.zeros(num_tokens, 1, dtype=torch.long)  # (tokens, top_k=1)
        routing_weights = torch.zeros(num_tokens, 1)  # (tokens, num_experts=1)
        routing_weights[:, 0] = 1.0

        # Run HF experts
        with torch.no_grad():
            hf_out = hf_experts(x_torch, router_indices=router_indices, routing_weights=routing_weights)

        # Extract weights for Grug
        # GPT-OSS uses interleaved gate/up: gate_up[..., ::2] and gate_up[..., 1::2]
        gate_up_np = hf_experts.gate_up_proj[0].detach().numpy()  # (hidden, 2*intermediate)
        gate_weight = gate_up_np[:, ::2]  # (hidden, intermediate)
        up_weight = gate_up_np[:, 1::2]   # (hidden, intermediate)
        down_weight = hf_experts.down_proj[0].detach().numpy()  # (intermediate, hidden)

        # Convert to JAX
        x_jax = torch_to_jax(x_torch)
        gate_jax = jnp.array(gate_weight)
        up_jax = jnp.array(up_weight)
        down_jax = jnp.array(down_weight)

        # Run Grug activation (without biases since we zeroed them)
        gate_act = jnp.einsum("bsd,di->bsi", x_jax, gate_jax)
        up_act = jnp.einsum("bsd,di->bsi", x_jax, up_jax)
        activated = clipped_gated_activation(gate_act, up_act, alpha=1.702)
        grug_out = jnp.einsum("bsi,id->bsd", activated, down_jax)

        result = compare(
            torch_to_jax(hf_out),
            grug_out,
            name="single_expert",
            atol=ATOL_SIMPLE,
            rtol=RTOL_SIMPLE,
        )
        assert result.passed, f"Single expert activation mismatch: {result.failure_summary}"

    def test_clipped_gated_activation_values(self):
        """Test clipping behavior with extreme values that should be clipped."""
        # Values that will be clipped - matches GptOssExperts alpha=1.702, limit=7.0
        gate_np = np.array([[[10.0, -10.0, 5.0, 0.0]]]).astype(np.float32)
        up_np = np.array([[[10.0, -10.0, 5.0, 0.0]]]).astype(np.float32)

        # What GPT-OSS does (from GptOssExperts.forward):
        # gate = gate.clamp(max=7.0)  -> [7.0, -10.0, 5.0, 0.0]
        # up = up.clamp(min=-7.0, max=7.0)  -> [7.0, -7.0, 5.0, 0.0]
        # glu = gate * sigmoid(gate * 1.702)
        # output = (up + 1) * glu
        gate_torch = torch.from_numpy(gate_np)
        up_torch = torch.from_numpy(up_np)
        alpha = 1.702
        limit = 7.0

        gate_clipped = gate_torch.clamp(max=limit)
        up_clipped = up_torch.clamp(min=-limit, max=limit)
        glu = gate_clipped * torch.sigmoid(gate_clipped * alpha)
        hf_out = (up_clipped + 1) * glu

        # Grug implementation
        gate_jax = jnp.array(gate_np)
        up_jax = jnp.array(up_np)
        grug_out = clipped_gated_activation(gate_jax, up_jax, alpha=alpha)

        result = compare(
            torch_to_jax(hf_out),
            grug_out,
            name="clipped_activation_extreme",
            atol=ATOL_SIMPLE,
            rtol=RTOL_SIMPLE,
        )
        assert result.passed, f"Clipped gated with extreme values mismatch: {result.failure_summary}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.skipif(not HAS_GPT_OSS, reason="GPT-OSS not importable")
@pytest.mark.skipif(not HAS_GRUG, reason="Grug not importable")
class TestAttentionSinksNumerical:
    """Test attention with sinks."""

    @pytest.mark.skipif(not HAS_MASK_UTILS, reason="transformers.masking_utils not importable")
    def test_attention_with_sinks_matches_grug(self):
        """Compare full attention output (with sinks) between GPT-OSS and Grug."""
        torch.manual_seed(CFG.seed)

        batch, seq = 2, 8

        # Create GptOssConfig with explicit attention implementation
        hf_config = GptOssConfig(
            hidden_size=CFG.hidden_dim,
            intermediate_size=CFG.intermediate_dim,
            num_attention_heads=CFG.num_heads,
            num_key_value_heads=CFG.num_kv_heads,
            head_dim=CFG.head_dim,
            num_hidden_layers=1,
            num_local_experts=1,
            num_experts_per_tok=1,
            attention_bias=False,
            layer_types=["full_attention"],
        )
        hf_config._attn_implementation = "eager"

        # Create GptOssAttention
        hf_attn = GptOssAttention(hf_config, layer_idx=0)

        # Initialize weights (normally done by PreTrainedModel)
        std = 0.02
        for module in [hf_attn.q_proj, hf_attn.k_proj, hf_attn.v_proj, hf_attn.o_proj]:
            torch.nn.init.normal_(module.weight, std=std)
        torch.nn.init.normal_(hf_attn.sinks, std=std)

        hf_attn.eval()

        # Create rotary embeddings
        rotary = GptOssRotaryEmbedding(hf_config)

        # Create input
        x_torch = torch.randn(batch, seq, CFG.hidden_dim) * 0.1
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rotary(x_torch, position_ids)
        position_embeddings = (cos, sin)

        # Build a causal mask so both implementations agree on masking semantics.
        attention_mask = torch.ones((batch, seq), dtype=torch.long)
        causal_mask = create_causal_mask(
            config=hf_config,
            input_embeds=x_torch,
            attention_mask=attention_mask,
            cache_position=torch.arange(seq, device=x_torch.device),
            past_key_values=None,
        )
        assert causal_mask is not None

        # Run HF attention
        with torch.no_grad():
            hf_out, _ = hf_attn(
                hidden_states=x_torch,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
            )

        # Build Grug attention output with the same weights and inputs.
        w_q = _transpose(hf_attn.q_proj.weight)
        w_k = _transpose(hf_attn.k_proj.weight)
        w_v = _transpose(hf_attn.v_proj.weight)
        w_o = _transpose(hf_attn.o_proj.weight)
        sinks = torch_to_jax(hf_attn.sinks)

        x_jax = torch_to_jax(x_torch)
        q = jnp.einsum("bsh,hd->bsd", x_jax, w_q)
        k = jnp.einsum("bsh,hd->bsd", x_jax, w_k)
        v = jnp.einsum("bsh,hd->bsd", x_jax, w_v)
        q = q.reshape(batch, seq, CFG.num_heads, CFG.head_dim)
        k = k.reshape(batch, seq, CFG.num_kv_heads, CFG.head_dim)
        v = v.reshape(batch, seq, CFG.num_kv_heads, CFG.head_dim)

        q, k = apply_rotary_embedding(
            q,
            k,
            seq_len=seq,
            head_dim=CFG.head_dim,
            rope=RotaryConfig(theta=CFG.rope_theta),
        )

        grug_mask = torch_to_jax(causal_mask[:, 0, :, :])
        attn_out = attention(q, k, v, grug_mask, sinks=sinks)
        attn_out = attn_out.reshape(batch, seq, CFG.num_heads * CFG.head_dim)
        grug_out = jnp.einsum("bsh,hd->bsd", attn_out, w_o)

        result = compare(
            torch_to_jax(hf_out),
            grug_out,
            name="attention_sinks_full",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        assert result.passed, f"Attention sinks mismatch: {result.failure_summary}"

    def test_sinks_mechanism_comparison(self):
        """Compare the mathematical effect of GPT-OSS and Grug sinks."""
        torch.manual_seed(CFG.seed)

        batch, seq, heads, head_dim = 2, 8, 4, 32
        scale = 1.0 / np.sqrt(head_dim)

        # Random Q, K, V, sinks
        q_np = np.random.randn(batch, heads, seq, head_dim).astype(np.float32) * 0.1
        k_np = np.random.randn(batch, heads, seq, head_dim).astype(np.float32) * 0.1
        v_np = np.random.randn(batch, heads, seq, head_dim).astype(np.float32) * 0.1
        sinks_np = np.random.randn(heads).astype(np.float32) * 0.1

        # GPT-OSS style: concatenate sink as extra position, then drop
        q_torch = torch.from_numpy(q_np)
        k_torch = torch.from_numpy(k_np)
        v_torch = torch.from_numpy(v_np)
        sinks_torch = torch.from_numpy(sinks_np)

        attn_weights = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
        sinks_expanded = sinks_torch.reshape(1, -1, 1, 1).expand(batch, -1, seq, 1)
        combined_logits = torch.cat([attn_weights, sinks_expanded], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(combined_logits, dim=-1)
        scores = probs[..., :-1]
        hf_out = torch.matmul(scores, v_torch).transpose(1, 2)

        # Grug style: add sinks as bias to attention logits
        q_jax = jnp.array(q_np).transpose(0, 2, 1, 3)  # BHSD -> BSHD
        k_jax = jnp.array(k_np).transpose(0, 2, 1, 3)
        v_jax = jnp.array(v_np).transpose(0, 2, 1, 3)
        sinks_jax = jnp.array(sinks_np)

        grug_out = reference_attention(q_jax, k_jax, v_jax, None, logits_dtype=jnp.float32, sinks=sinks_jax)

        # GPT-OSS: softmax(concat([QK, sinks]))[:-1] @ V
        # Grug: softmax with sink term in denominator (equivalent)
        result = compare(
            torch_to_jax(hf_out),
            grug_out,
            name="sinks_mechanism",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        assert result.passed, f"Sink mechanism mismatch: {result.failure_summary}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.skipif(not HAS_GPT_OSS, reason="GPT-OSS not importable")
@pytest.mark.skipif(not HAS_GRUG, reason="Grug not importable")
@pytest.mark.skipif(not HAS_MASK_UTILS, reason="transformers.masking_utils not importable")
class TestAttentionMaskingNumerical:
    """Numerically verify attention masking parity vs GPT-OSS."""

    def test_full_forward_matches_with_padding_mask(self):
        """Compare HF and Grug with explicit padding + causal mask."""
        torch.manual_seed(CFG.seed)

        cfg = FuzzConfig(
            vocab_size=256,
            hidden_dim=128,
            intermediate_dim=256,
            num_heads=4,
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            max_seq_len=16,
            batch_size=2,
            seq_len=8,
            num_experts=2,
            num_experts_per_tok=1,
        )

        hf_config = _build_hf_config(cfg)
        hf_config._attn_implementation = "eager"
        hf_model = GptOssForCausalLM(hf_config)
        hf_model.eval()

        grug_cfg = _build_grug_config(cfg)
        grug_params = _build_grug_params_from_hf(hf_model, cfg)

        tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)

        with torch.no_grad():
            hf_hidden = hf_model.model(input_ids=tokens, attention_mask=attention_mask).last_hidden_state

        inputs_embeds = hf_model.model.embed_tokens(tokens)
        cache_position = torch.arange(cfg.seq_len, device=inputs_embeds.device)
        causal_mask = create_causal_mask(
            config=hf_config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
        )
        assert causal_mask is not None

        # HF uses [B, 1, Q, K] additive mask; Grug expects [B, Q, K].
        grug_mask = torch_to_jax(causal_mask[:, 0, :, :])

        tokens_jax = torch_to_jax(tokens)
        grug_hidden = activations(grug_params, tokens_jax, grug_cfg, mask=grug_mask)

        result = compare(
            torch_to_jax(hf_hidden),
            grug_hidden,
            name="full_hidden_state_with_mask",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        assert result.passed, f"Masked hidden state mismatch: {result.failure_summary}"

    @pytest.mark.parametrize("seed,cfg", MASK_FUZZ_CASES)
    def test_fuzz_forward_matches_with_padding_mask(self, seed: int, cfg: FuzzConfig):
        """Fuzz parity across random padding masks and configs."""
        torch.manual_seed(seed)

        hf_config = _build_hf_config(cfg)
        hf_config._attn_implementation = "eager"
        hf_model = GptOssForCausalLM(hf_config)
        hf_model.eval()

        grug_cfg = _build_grug_config(cfg)
        grug_params = _build_grug_params_from_hf(hf_model, cfg)

        tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        attention_mask = _random_attention_mask(cfg, seed + 123)

        with torch.no_grad():
            hf_hidden = hf_model.model(input_ids=tokens, attention_mask=attention_mask).last_hidden_state
            hf_logits = hf_model.lm_head(hf_hidden)

        inputs_embeds = hf_model.model.embed_tokens(tokens)
        cache_position = torch.arange(cfg.seq_len, device=inputs_embeds.device)
        causal_mask = create_causal_mask(
            config=hf_config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
        )
        assert causal_mask is not None

        grug_mask = torch_to_jax(causal_mask[:, 0, :, :])
        tokens_jax = torch_to_jax(tokens)
        grug_hidden = activations(grug_params, tokens_jax, grug_cfg, mask=grug_mask)
        grug_logits = forward(grug_params, tokens_jax, grug_cfg, mask=grug_mask)

        hidden_result = compare(
            torch_to_jax(hf_hidden),
            grug_hidden,
            name="fuzz_hidden_state_with_mask",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        assert hidden_result.passed, f"Masked hidden state mismatch: {hidden_result.failure_summary}"

        logits_result = compare(
            torch_to_jax(hf_logits),
            grug_logits,
            name="fuzz_logits_with_mask",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        assert logits_result.passed, f"Masked logits mismatch: {logits_result.failure_summary}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.skipif(not HAS_GPT_OSS, reason="GPT-OSS not importable")
@pytest.mark.skipif(not HAS_GRUG, reason="Grug not importable")
class TestMoENumerical:
    """Numerically verify MoE components match GPT-OSS."""

    def test_gptoss_router_runs(self):
        """Test GptOssTopKRouter produces expected output format."""
        torch.manual_seed(CFG.seed)

        hf_config = GptOssConfig(
            hidden_size=CFG.hidden_dim,
            intermediate_size=CFG.intermediate_dim,
            num_local_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
        )

        router = GptOssTopKRouter(hf_config)
        router.eval()

        batch, seq = 2, 8
        x = torch.randn(batch, seq, CFG.hidden_dim) * 0.1

        with torch.no_grad():
            router_scores, router_indices = router(x)

        # router_scores: (batch*seq, num_experts) - sparse, only top_k non-zero per row
        # router_indices: (batch*seq, top_k)
        assert router_scores.shape == (batch * seq, CFG.num_experts)
        assert router_indices.shape == (batch * seq, CFG.num_experts_per_tok)

        print(f"\nGptOssTopKRouter output:")
        print(f"  router_scores shape: {router_scores.shape}")
        print(f"  router_indices shape: {router_indices.shape}")
        print(f"  Non-zero per row: {(router_scores[0] != 0).sum().item()}")

    def test_router_topk_matches_hf(self):
        """Validate Grug top-k routing matches GPT-OSS router outputs."""
        torch.manual_seed(CFG.seed)

        hf_config = GptOssConfig(
            hidden_size=CFG.hidden_dim,
            intermediate_size=CFG.intermediate_dim,
            num_local_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
        )

        router = GptOssTopKRouter(hf_config)
        std = 0.02
        torch.nn.init.normal_(router.weight, std=std)
        torch.nn.init.normal_(router.bias, std=std)
        router.eval()

        batch, seq = 2, 8
        x_torch = torch.randn(batch, seq, CFG.hidden_dim) * 0.1

        with torch.no_grad():
            hf_scores, hf_indices = router(x_torch)

        x_jax = torch_to_jax(x_torch).reshape(-1, CFG.hidden_dim)
        router_jax = torch_to_jax(router.weight.T)
        router_bias = torch_to_jax(router.bias)
        router_logits = jnp.einsum("td,de->te", x_jax, router_jax) + router_bias
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        topk_weights, topk_indices = jax.lax.top_k(router_probs, CFG.num_experts_per_tok)
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)

        dense_scores = jnp.zeros_like(router_probs)
        dense_scores = dense_scores.at[jnp.arange(dense_scores.shape[0])[:, None], topk_indices].set(topk_weights)

        scores_result = compare(
            torch_to_jax(hf_scores),
            dense_scores,
            name="router_scores_dense",
            atol=ATOL_SIMPLE,
            rtol=RTOL_SIMPLE,
        )
        assert scores_result.passed, f"Router score mismatch: {scores_result.failure_summary}"
        assert jnp.array_equal(torch_to_jax(hf_indices), topk_indices), "Router top-k indices mismatch"

    def test_gptoss_mlp_runs(self):
        """Test GptOssMLP (router + experts) produces expected output."""
        torch.manual_seed(CFG.seed)

        hf_config = GptOssConfig(
            hidden_size=CFG.hidden_dim,
            intermediate_size=CFG.intermediate_dim,
            num_local_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
        )

        mlp = GptOssMLP(hf_config)

        # Initialize weights (normally done by PreTrainedModel._init_weights)
        std = 0.02
        torch.nn.init.normal_(mlp.router.weight, std=std)
        torch.nn.init.normal_(mlp.router.bias, std=std)
        torch.nn.init.normal_(mlp.experts.gate_up_proj, std=std)
        torch.nn.init.zeros_(mlp.experts.gate_up_proj_bias)
        torch.nn.init.normal_(mlp.experts.down_proj, std=std)
        torch.nn.init.zeros_(mlp.experts.down_proj_bias)

        mlp.eval()

        batch, seq = 2, 8
        x = torch.randn(batch, seq, CFG.hidden_dim) * 0.1

        with torch.no_grad():
            output, router_scores = mlp(x)

        assert output.shape == (batch, seq, CFG.hidden_dim)
        assert torch.isfinite(output).all()

        print(f"\nGptOssMLP output:")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    def test_moe_forward_matches_shape(self):
        """Test Grug MoE produces same output shape as GptOssMLP."""
        torch.manual_seed(CFG.seed)

        batch, seq = 2, 8

        # Create GptOssMLP
        hf_config = GptOssConfig(
            hidden_size=CFG.hidden_dim,
            intermediate_size=CFG.intermediate_dim,
            num_local_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
        )
        hf_mlp = GptOssMLP(hf_config)

        # Initialize weights
        std = 0.02
        torch.nn.init.normal_(hf_mlp.router.weight, std=std)
        torch.nn.init.normal_(hf_mlp.router.bias, std=std)
        torch.nn.init.normal_(hf_mlp.experts.gate_up_proj, std=std)
        torch.nn.init.zeros_(hf_mlp.experts.gate_up_proj_bias)
        torch.nn.init.normal_(hf_mlp.experts.down_proj, std=std)
        torch.nn.init.zeros_(hf_mlp.experts.down_proj_bias)

        hf_mlp.eval()

        x_torch = torch.randn(batch, seq, CFG.hidden_dim) * 0.1

        with torch.no_grad():
            hf_out, _ = hf_mlp(x_torch)

        # Create Grug MoE with same architecture
        grug_cfg = GrugModelConfig(
            vocab_size=CFG.vocab_size,
            hidden_dim=CFG.hidden_dim,
            intermediate_dim=CFG.intermediate_dim,
            num_heads=CFG.num_heads,
            num_kv_heads=CFG.num_kv_heads,
            num_layers=1,
            use_moe=True,
            num_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
            use_clipped_gated_activation=True,
        )

        key = jax.random.PRNGKey(CFG.seed)
        keys = jax.random.split(key, 1 + CFG.num_experts * 3)

        router = jax.random.normal(keys[0], (CFG.hidden_dim, CFG.num_experts)) * 0.02
        router_bias = jnp.zeros((CFG.num_experts,), dtype=jnp.float32)
        gate_weights = []
        up_weights = []
        down_weights = []
        gate_biases = []
        up_biases = []
        down_biases = []
        for e in range(CFG.num_experts):
            gate = jax.random.normal(keys[1 + e * 3], (CFG.hidden_dim, CFG.intermediate_dim)) * 0.02
            up = jax.random.normal(keys[2 + e * 3], (CFG.hidden_dim, CFG.intermediate_dim)) * 0.02
            down = jax.random.normal(keys[3 + e * 3], (CFG.intermediate_dim, CFG.hidden_dim)) * 0.02
            gate_weights.append(gate)
            up_weights.append(up)
            down_weights.append(down)
            gate_biases.append(jnp.zeros((CFG.intermediate_dim,), dtype=jnp.float32))
            up_biases.append(jnp.zeros((CFG.intermediate_dim,), dtype=jnp.float32))
            down_biases.append(jnp.zeros((CFG.hidden_dim,), dtype=jnp.float32))

        moe = GrugMoEParams(
            router=router,
            router_bias=router_bias,
            gate=jnp.stack(gate_weights, axis=0),
            up=jnp.stack(up_weights, axis=0),
            down=jnp.stack(down_weights, axis=0),
            gate_bias=jnp.stack(gate_biases, axis=0),
            up_bias=jnp.stack(up_biases, axis=0),
            down_bias=jnp.stack(down_biases, axis=0),
        )

        x_jax = torch_to_jax(x_torch)
        grug_out, extras = moe_forward(moe, x_jax, grug_cfg)

        # Shapes should match
        assert grug_out.shape == hf_out.shape, f"Shape mismatch: Grug {grug_out.shape} vs HF {hf_out.shape}"
        assert jnp.isfinite(grug_out).all()

        # Check aux losses exist
        assert "load_balancing_loss" in extras
        assert "router_z_loss" in extras

        print(f"\nMoE shape comparison:")
        print(f"  HF shape: {hf_out.shape}")
        print(f"  Grug shape: {grug_out.shape}")
        print(f"  Grug aux losses: {list(extras.keys())}")

    def test_moe_weight_transfer(self):
        """Test that transferring weights from GptOssMLP to Grug produces similar output.

        GPT-OSS uses interleaved gate/up weights and expert biases; Grug mirrors them.
        """
        torch.manual_seed(CFG.seed)

        batch, seq = 2, 8

        hf_config = GptOssConfig(
            hidden_size=CFG.hidden_dim,
            intermediate_size=CFG.intermediate_dim,
            num_local_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
        )
        hf_mlp = GptOssMLP(hf_config)

        # Initialize weights properly, then zero biases for fair comparison
        std = 0.02
        torch.nn.init.normal_(hf_mlp.router.weight, std=std)
        torch.nn.init.normal_(hf_mlp.router.bias, std=std)
        torch.nn.init.normal_(hf_mlp.experts.gate_up_proj, std=std)
        torch.nn.init.normal_(hf_mlp.experts.gate_up_proj_bias, std=std)
        torch.nn.init.normal_(hf_mlp.experts.down_proj, std=std)
        torch.nn.init.normal_(hf_mlp.experts.down_proj_bias, std=std)

        hf_mlp.eval()

        x_torch = torch.randn(batch, seq, CFG.hidden_dim) * 0.1

        with torch.no_grad():
            hf_out, _ = hf_mlp(x_torch)

        # Transfer weights to Grug
        # Router: HF (num_experts, hidden) -> Grug (hidden, num_experts), plus bias
        router_jax = torch_to_jax(hf_mlp.router.weight.T)
        router_bias = torch_to_jax(hf_mlp.router.bias)

        # Experts: extract gate/up from interleaved gate_up_proj
        gate_weights = []
        up_weights = []
        down_weights = []
        gate_biases = []
        up_biases = []
        down_biases = []
        for e in range(CFG.num_experts):
            gate_up = hf_mlp.experts.gate_up_proj[e].detach().numpy()  # (hidden, 2*intermediate)
            gate_weight = gate_up[:, ::2]   # (hidden, intermediate)
            up_weight = gate_up[:, 1::2]    # (hidden, intermediate)
            down_weight = hf_mlp.experts.down_proj[e].detach().numpy()  # (intermediate, hidden)
            gate_up_bias = hf_mlp.experts.gate_up_proj_bias[e].detach().numpy()  # (2*intermediate,)
            gate_bias = gate_up_bias[::2]
            up_bias = gate_up_bias[1::2]
            down_bias = hf_mlp.experts.down_proj_bias[e].detach().numpy()  # (hidden,)

            gate_weights.append(jnp.array(gate_weight))
            up_weights.append(jnp.array(up_weight))
            down_weights.append(jnp.array(down_weight))
            gate_biases.append(jnp.array(gate_bias))
            up_biases.append(jnp.array(up_bias))
            down_biases.append(jnp.array(down_bias))

        moe = GrugMoEParams(
            router=router_jax,
            router_bias=router_bias,
            gate=jnp.stack(gate_weights, axis=0),
            up=jnp.stack(up_weights, axis=0),
            down=jnp.stack(down_weights, axis=0),
            gate_bias=jnp.stack(gate_biases, axis=0),
            up_bias=jnp.stack(up_biases, axis=0),
            down_bias=jnp.stack(down_biases, axis=0),
        )

        grug_cfg = GrugModelConfig(
            vocab_size=CFG.vocab_size,
            hidden_dim=CFG.hidden_dim,
            intermediate_dim=CFG.intermediate_dim,
            num_heads=CFG.num_heads,
            num_kv_heads=CFG.num_kv_heads,
            num_layers=1,
            use_moe=True,
            num_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
            use_clipped_gated_activation=True,
        )

        x_jax = torch_to_jax(x_torch)
        grug_out, _ = moe_forward(moe, x_jax, grug_cfg)

        result = compare(
            torch_to_jax(hf_out),
            grug_out,
            name="moe_weight_transfer",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        print(f"\nMoE weight transfer comparison:")
        print(f"  Max diff: {result.max_abs_diff:.6f}")
        print(f"  Passed: {result.passed}")

        # With full bias transfer, should match closely
        assert result.passed, f"MoE weight transfer mismatch: {result.failure_summary}"

    def test_moe_weight_transfer_with_biases_matches(self):
        """Show that GPT-OSS biases match when transferred to Grug."""
        torch.manual_seed(CFG.seed)

        batch, seq = 2, 8

        hf_config = GptOssConfig(
            hidden_size=CFG.hidden_dim,
            intermediate_size=CFG.intermediate_dim,
            num_local_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
        )
        hf_mlp = GptOssMLP(hf_config)

        # Initialize weights and set non-zero biases
        std = 0.02
        torch.nn.init.normal_(hf_mlp.router.weight, std=std)
        torch.nn.init.constant_(hf_mlp.router.bias, 0.5)
        torch.nn.init.normal_(hf_mlp.experts.gate_up_proj, std=std)
        torch.nn.init.constant_(hf_mlp.experts.gate_up_proj_bias, 0.25)
        torch.nn.init.normal_(hf_mlp.experts.down_proj, std=std)
        torch.nn.init.constant_(hf_mlp.experts.down_proj_bias, 0.25)

        hf_mlp.eval()

        x_torch = torch.randn(batch, seq, CFG.hidden_dim) * 0.1

        with torch.no_grad():
            hf_out, _ = hf_mlp(x_torch)

        # Transfer weights to Grug (including biases).
        router_jax = torch_to_jax(hf_mlp.router.weight.T)
        router_bias = torch_to_jax(hf_mlp.router.bias)

        gate_weights = []
        up_weights = []
        down_weights = []
        gate_biases = []
        up_biases = []
        down_biases = []
        for e in range(CFG.num_experts):
            gate_up = hf_mlp.experts.gate_up_proj[e].detach().numpy()
            gate_weight = gate_up[:, ::2]
            up_weight = gate_up[:, 1::2]
            down_weight = hf_mlp.experts.down_proj[e].detach().numpy()
            gate_up_bias = hf_mlp.experts.gate_up_proj_bias[e].detach().numpy()
            gate_bias = gate_up_bias[::2]
            up_bias = gate_up_bias[1::2]
            down_bias = hf_mlp.experts.down_proj_bias[e].detach().numpy()

            gate_weights.append(jnp.array(gate_weight))
            up_weights.append(jnp.array(up_weight))
            down_weights.append(jnp.array(down_weight))
            gate_biases.append(jnp.array(gate_bias))
            up_biases.append(jnp.array(up_bias))
            down_biases.append(jnp.array(down_bias))

        moe = GrugMoEParams(
            router=router_jax,
            router_bias=router_bias,
            gate=jnp.stack(gate_weights, axis=0),
            up=jnp.stack(up_weights, axis=0),
            down=jnp.stack(down_weights, axis=0),
            gate_bias=jnp.stack(gate_biases, axis=0),
            up_bias=jnp.stack(up_biases, axis=0),
            down_bias=jnp.stack(down_biases, axis=0),
        )

        grug_cfg = GrugModelConfig(
            vocab_size=CFG.vocab_size,
            hidden_dim=CFG.hidden_dim,
            intermediate_dim=CFG.intermediate_dim,
            num_heads=CFG.num_heads,
            num_kv_heads=CFG.num_kv_heads,
            num_layers=1,
            use_moe=True,
            num_experts=CFG.num_experts,
            num_experts_per_tok=CFG.num_experts_per_tok,
            use_clipped_gated_activation=True,
        )

        x_jax = torch_to_jax(x_torch)
        grug_out, _ = moe_forward(moe, x_jax, grug_cfg)

        result = compare(
            torch_to_jax(hf_out),
            grug_out,
            name="moe_weight_transfer_with_biases",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        assert result.passed, f"MoE weight transfer with biases mismatch: {result.failure_summary}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.skipif(not HAS_GPT_OSS, reason="GPT-OSS not importable")
@pytest.mark.skipif(not HAS_GRUG, reason="Grug not importable")
class TestFullForwardNumerical:
    """Exact parity tests against GPT-OSS with shared weights."""

    @pytest.mark.parametrize("seed,cfg", FULL_FORWARD_CASES)
    @pytest.mark.parametrize("mesh_shape", MESH_SHAPES, ids=_mesh_id)
    def test_full_forward_matches_hf_init(self, seed: int, cfg: FuzzConfig, mesh_shape: tuple[int, int]):
        """Exact match between Grug and GPT-OSS at init with shared weights."""
        torch.manual_seed(seed)

        hf_config = _build_hf_config(cfg)
        hf_model = GptOssForCausalLM(hf_config)
        hf_model.eval()

        grug_cfg = _build_grug_config(cfg)
        grug_params = _build_grug_params_from_hf(hf_model, cfg)

        tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))

        with torch.no_grad():
            hf_embed = hf_model.model.embed_tokens(tokens)
            hf_hidden = hf_model.model(input_ids=tokens).last_hidden_state
            hf_logits = hf_model.lm_head(hf_hidden)

        tokens_jax = torch_to_jax(tokens)
        data, model = mesh_shape
        devices = np.array(jax.devices()[: data * model]).reshape((data, model))
        mesh = Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        with jax.set_mesh(mesh):
            if cfg.batch_size % data != 0:
                pytest.skip(f"batch_size={cfg.batch_size} not divisible by data mesh size {data}")
            replicated = NamedSharding(mesh, P())
            tokens_jax = jax.device_put(tokens_jax, replicated)
            grug_params = jax.tree_util.tree_map(
                lambda x: x if x is None else jax.device_put(x, replicated),
                grug_params,
            )
            grug_embed = grug_params.token_embed[tokens_jax]
            grug_hidden = activations(grug_params, tokens_jax, grug_cfg)
            grug_logits = forward(grug_params, tokens_jax, grug_cfg)

        embed_result = compare(
            torch_to_jax(hf_embed),
            grug_embed,
            name="token_embed",
            atol=ATOL_SIMPLE,
            rtol=RTOL_SIMPLE,
        )
        assert embed_result.passed, f"Token embedding mismatch: {embed_result.failure_summary}"

        hidden_result = compare(
            torch_to_jax(hf_hidden),
            grug_hidden,
            name="full_hidden_state",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        assert hidden_result.passed, f"Hidden state mismatch: {hidden_result.failure_summary}"

        logits_result = compare(
            torch_to_jax(hf_logits),
            grug_logits,
            name="full_logits",
            atol=ATOL_COMPLEX,
            rtol=RTOL_COMPLEX,
        )
        assert logits_result.passed, f"Logits mismatch: {logits_result.failure_summary}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
