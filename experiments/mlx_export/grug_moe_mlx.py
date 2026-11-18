# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MLX implementation of Marin GrugMoE for the Marin-Snowball export.

This file ships inside the model repo and is loaded by ``mlx-lm`` through the
``model_file`` mechanism (``config.json`` -> ``"model_file":
"grug_moe_mlx.py"``, requires ``trust_remote_code=True``). It implements the
forward pass of the checkpoint ``open-athena/Marin-Snowball-67B-A2B-v0.1``
exactly as the reference vLLM implementation
(``vllm/model_executor/models/grugmoe.py`` in ``marin-community/vllm``)
computes it, for the feature set this export uses.

Not implemented (rejected in ``ModelArgs.__post_init__``): PKO, short conv,
latent experts, split local/global KV heads, fused RoPE, schema version 2
(hero) artifacts, multiple shared experts, tied embeddings.

As a side effect of being imported (which ``mlx_lm.load`` does before building
the tokenizer) this module teaches ``mlx-lm`` the model's thinking markers
``<|start_think|>``/``<|end_think|>``; see ``_install_grug_think_markers``.
Under ``mlx_lm.server`` it also treats text before a bare ``<|end_think|>`` as
reasoning; see ``_install_grug_bare_think_end``.
"""

import sys
import threading
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.tokenizer_utils as tokenizer_utils
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

# The trained GrugMoE checkpoint uses a fixed rank-128 gated-norm bottleneck
# and this exact router combine normalization.
_GATED_NORM_RANK = 128
_ROUTER_COMBINE_WEIGHT_SUM = 2.5
_ROUTER_COMBINE_WEIGHT_EPS = 1e-9
_XSA_EPS = 1e-6

_GRUG_THINK = ("<|start_think|>", "<|end_think|>")


def _install_grug_think_markers() -> None:
    """mlx-lm 0.32.2 detects thinking markers from a fixed list; add Grug's pair.

    Remove once mlx-lm recognizes <|start_think|>/<|end_think|> upstream. This
    relies on mlx_lm.load importing the model file before building the
    tokenizer.
    """
    stock = tokenizer_utils._infer_thinking
    if getattr(stock, "_grug_patch", False):
        return

    def infer_thinking(tokenizer):
        vocab = tokenizer.get_vocab()
        start, end = _GRUG_THINK
        if start in vocab and end in vocab:
            return start, end, (vocab[start],), (vocab[end],)
        return stock(tokenizer)

    infer_thinking._grug_patch = True
    tokenizer_utils._infer_thinking = infer_thinking


_install_grug_think_markers()


def _grug_thinking_enabled(template_kwargs: dict) -> bool:
    """Mirror the chat template: thinking is on unless enable_thinking is False or "/nothink"."""
    return template_kwargs.get("enable_thinking", True) not in (False, "/nothink")


class _UnmarkedTurn:
    """Response-side state for a thinking-mode turn whose reasoning start was assumed.

    Mirrors vLLM's ``grug`` reasoning parser: text is reasoning only once a think
    marker or a tool call confirms it. A turn that ends with neither is an
    answer given without thinking, and is returned as content.
    """

    def __init__(self):
        self.marked = False
        self.pending = ""

    def response_parts(self, text, reasoning_text, tool_calls, finish_reason):
        """Return the (text, reasoning_text) to send for one response or stream chunk."""
        reasoning_text = reasoning_text or ""
        if self.marked or tool_calls:
            reasoning_text, self.pending, self.marked = self.pending + reasoning_text, "", True
            return text, reasoning_text or None
        if finish_reason is None:
            self.pending += reasoning_text
            return text, None
        text, self.pending = self.pending + reasoning_text + text, ""
        return text, None


_grug_turn = threading.local()


def _install_grug_bare_think_end() -> None:
    """Parse thinking-mode responses as reasoning until <|end_think|>.

    With thinking on, the model sometimes reasons without first emitting
    <|start_think|> and then closes with <|end_think|>. mlx-lm 0.32.2 starts the
    response parser in "normal" unless the prompt ends inside a think block, so
    that reasoning and the bare marker land in the content. Start in
    "reasoning" for thinking-mode chat requests instead, and strip a
    <|start_think|> the model does emit. When the model answers without any
    think marker or tool call, return that text as content (``_UnmarkedTurn``);
    streamed responses hold such text back until a marker or the end of the
    turn decides. Remove once mlx-lm supports a per-request initial reasoning
    state. Applies only when mlx_lm.server is already imported, which it is when
    the server loads the model.
    """
    server = sys.modules.get("mlx_lm.server")
    if server is None or getattr(server.ResponseGenerator._tokenize, "_grug_patch", False):
        return
    stock_tokenize = server.ResponseGenerator._tokenize
    stock_generate = server.ResponseGenerator.generate
    stock_generate_response = server.APIHandler.generate_response
    stock_transitions = tokenizer_utils.TokenizerWrapper.response_transitions.fget

    def _tokenize(self, tokenizer, request, args):
        prompt, segments, segment_types, initial_state, vision = stock_tokenize(self, tokenizer, request, args)
        template_kwargs = {**self.model_provider.cli_args.chat_template_args, **(args.chat_template_kwargs or {})}
        if (
            request.request_type == "chat"
            and tokenizer.think_start == _GRUG_THINK[0]
            and initial_state == "normal"
            and _grug_thinking_enabled(template_kwargs)
        ):
            initial_state = "reasoning"
            request._grug_assumed_reasoning = True
        return prompt, segments, segment_types, initial_state, vision

    def generate(self, request, generation_args, progress_callback=None):
        # _tokenize has run on the generation thread by the time generate returns ctx.
        ctx, response = stock_generate(self, request, generation_args, progress_callback)
        if not getattr(request, "_grug_assumed_reasoning", False):
            _grug_turn.state = None
            return ctx, response
        turn = _UnmarkedTurn()
        _grug_turn.state = turn

        def observed():
            for gen in response:
                if any(marker in gen.text for marker in _GRUG_THINK):
                    turn.marked = True
                yield gen

        return ctx, observed()

    def generate_response(self, text, finish_reason, *args, tool_calls=None, reasoning_text=None, **kwargs):
        turn = getattr(_grug_turn, "state", None)
        if turn is not None:
            text, reasoning_text = turn.response_parts(text, reasoning_text, tool_calls, finish_reason)
        return stock_generate_response(
            self, text, finish_reason, *args, tool_calls=tool_calls, reasoning_text=reasoning_text, **kwargs
        )

    def response_transitions(self):
        transitions = stock_transitions(self)
        if self.think_start != _GRUG_THINK[0]:
            return transitions
        return {**transitions, "reasoning": [*transitions.get("reasoning", []), (_GRUG_THINK[0], "reasoning")]}

    _tokenize._grug_patch = True
    server.ResponseGenerator._tokenize = _tokenize
    server.ResponseGenerator.generate = generate
    server.APIHandler.generate_response = generate_response
    tokenizer_utils.TokenizerWrapper.response_transitions = property(response_transitions)


_install_grug_bare_think_end()


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "grug_moe"
    grugmoe_artifact_schema_version: int = 1
    vocab_size: int = 128256
    hidden_size: int = 2560
    num_hidden_layers: int = 26
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    head_dim: int = 128
    moe_intermediate_size: int = 1280
    shared_expert_intermediate_size: int = 2560
    num_experts: int = 256
    num_experts_per_tok: int = 4
    num_shared_experts: int = 1
    sliding_window: int = 2048
    global_every: int = 4
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    qk_mult: float = 1.75
    qk_mult_long_scale: float = 1.0
    disable_long_rope: bool = True
    disable_pko: bool = True
    sconv: bool = False
    rope_fused: bool = False
    latent_dim: int = None
    local_kv_heads: int = None
    global_kv_heads: int = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        unsupported = {
            "grugmoe_artifact_schema_version": self.grugmoe_artifact_schema_version != 1,
            "disable_pko": not self.disable_pko,
            "sconv": self.sconv,
            "rope_fused": self.rope_fused,
            "latent_dim": self.latent_dim is not None,
            "local_kv_heads": self.local_kv_heads is not None,
            "global_kv_heads": self.global_kv_heads is not None,
            "num_shared_experts": self.num_shared_experts != 1,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
        bad = [name for name, enabled in unsupported.items() if enabled]
        if bad:
            raise ValueError(
                f"{', '.join(bad)}: this mlx port does not implement these " "GrugMoE features; the config requests them"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def layer_types(self) -> list:
        return [
            (
                "full_attention"
                if (i + 1) % self.global_every == 0 or i == self.num_hidden_layers - 1
                else "sliding_attention"
            )
            for i in range(self.num_hidden_layers)
        ]


def _unit_rms(x, eps=1e-6):
    """Non-parametric RMS norm over the last axis, computed in float32."""
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class GatedNorm(nn.Module):
    """x * sigmoid(up_proj(silu(down_proj(x)))), rank-128 bottleneck."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, _GATED_NORM_RANK, bias=False)
        self.up_proj = nn.Linear(_GATED_NORM_RANK, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.down_proj(x))
        return x * mx.sigmoid(self.up_proj(gate))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_long: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.n_rep = self.n_heads // self.n_kv

        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_gate = nn.Linear(args.hidden_size, self.n_heads, bias=False)

        # vLLM multiplies q by qk_mult * qk_mult_scale and uses softmax scale
        # head_dim ** -0.5; fold both into the SDPA scale.
        qk_mult_scale = args.qk_mult_long_scale if is_long else 1.0
        self.scale = args.qk_mult * qk_mult_scale * self.head_dim**-0.5
        # Full layers drop RoPE on this export (disable_long_rope).
        self.rope = (
            None
            if (is_long and args.disable_long_rope)
            else nn.RoPE(self.head_dim // 2, traditional=False, base=args.rope_theta)
        )

    def __call__(self, x: mx.array, mask: mx.array, cache=None) -> mx.array:
        B, L, _ = x.shape
        q = _unit_rms(self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = _unit_rms(self.k_proj(x).reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        keys, values = cache.update_and_fetch(k, v) if cache is not None else (k, v)
        y = scaled_dot_product_attention(q, keys, values, cache=cache, scale=self.scale, mask=mask)

        # XSA (exclusive self-attention): remove each head's projection onto
        # its aligned *current-token* value vector, in float32.
        yf = y.astype(mx.float32)
        vf = mx.repeat(v, self.n_rep, axis=1).astype(mx.float32)
        yf = yf - (mx.sum(yf * vf, axis=-1, keepdims=True) / (mx.sum(vf * vf, axis=-1, keepdims=True) + _XSA_EPS)) * vf

        # Per-head gate 2 * sigmoid(attn_gate(x)), from the attention input.
        gate = 2 * mx.sigmoid(self.attn_gate(x).astype(mx.float32))
        yf = yf * gate.transpose(0, 2, 1)[..., None]

        y = yf.astype(x.dtype).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(y)


class Router(nn.Module):
    """QB router: weight [E, D], bias [E]. Not an nn.Linear, so it is never
    quantized and its bias never enters the logits."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, hidden_size))
        self.bias = mx.zeros((num_experts,))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.router = Router(args.hidden_size, args.num_experts)
        self.experts = SwitchGLU(
            input_dims=args.hidden_size,
            hidden_dims=args.moe_intermediate_size,
            num_experts=args.num_experts,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        logits = x.astype(mx.float32) @ self.router.weight.astype(mx.float32).T  # unbiased
        biased = logits + self.router.bias.astype(mx.float32)
        idx = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        idx = mx.stop_gradient(idx)
        w = mx.sigmoid(mx.take_along_axis(logits, idx, axis=-1))
        w = w * (_ROUTER_COMBINE_WEIGHT_SUM / (mx.sum(w, axis=-1, keepdims=True) + _ROUTER_COMBINE_WEIGHT_EPS))
        y = self.experts(x, idx)  # [..., k, hidden]
        return (y.astype(mx.float32) * w[..., None]).sum(axis=-2).astype(x.dtype)


class MLP(nn.Module):
    """Dense SwiGLU shared expert, always on."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.shared_expert_intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, dim, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, dim, bias=False)
        self.down_proj = nn.Linear(dim, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, is_long: bool):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.attn_gated_norm = GatedNorm(args.hidden_size)
        self.self_attn = Attention(args, is_long)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp_gated_norm = GatedNorm(args.hidden_size)
        self.mlp = MoE(args)
        self.shared_expert = MLP(args)

    def __call__(self, x: mx.array, mask: mx.array, cache=None) -> mx.array:
        a_in = self.attn_gated_norm(self.input_layernorm(x))
        x = x + self.self_attn(a_in, mask, cache)
        m_in = self.mlp_gated_norm(self.post_attention_layernorm(x))
        return x + self.mlp(m_in) + self.shared_expert(m_in)


class GrugMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.embed_gated_norm = GatedNorm(args.hidden_size)
        self.layer_types = args.layer_types
        self.layers = [TransformerBlock(args, layer_type == "full_attention") for layer_type in self.layer_types]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.final_gated_norm = GatedNorm(args.hidden_size)
        self.window_size = args.sliding_window
        # Indices only pick a representative cache for mask creation; a model
        # with only one layer type may not have both.
        self.swa_idx = self.layer_types.index("sliding_attention") if "sliding_attention" in self.layer_types else 0
        self.ga_idx = self.layer_types.index("full_attention") if "full_attention" in self.layer_types else 0

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        h = self.embed_gated_norm(self.embed_norm(h))

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self.ga_idx])
        swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.window_size)

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types, strict=True):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            h = layer(h, mask, c)

        h = self.norm(h)
        return self.final_gated_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GrugMoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "gated_norm" in path or path.endswith("attn_gate"):
                # Small multiplicative gates; quantization error compounds.
                return False
            if path.endswith("lm_head") or path.endswith("embed_tokens"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        return [
            KVCache() if lt == "full_attention" else RotatingKVCache(max_size=self.args.sliding_window)
            for lt in self.model.layer_types
        ]
