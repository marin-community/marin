# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FLOP-equivalent scoring for the tokenizer bake-off.

Two things are kept strictly separate so results can be **replayed under different
deployment assumptions** without re-running any training:

* **Measurement (empirical, stored raw).** Training a proxy model on an arm's
  tokenized data yields (training FLOPs, BPB) points and the arm's fertility
  (tokens/byte). These are logged verbatim — nothing about the deployment model is
  baked in.
* **Pricing (a model, applied at analysis time).** A :class:`ServingCostModel`
  turns those raw measurements into a serving cost and a FLOP-equivalent BPB. It
  captures the *deployment* regime — target model shape, context window (default
  16k tokens), local/global attention sparsity, and a hardware/kernel speed factor.
  Change it and re-score the same stored measurements to answer "what if we serve
  at 64k context / a bigger model / denser attention?".

Vocabulary size enters model compute through the output head
``lm_head = 2 * hidden_dim * vocab_size`` (the input embedding is a gather, ~0
FLOPs). Attention enters through the context window: per-token attention FLOPs grow
with the number of positions attended, so at a long serving context (64k) attention
is a large share of cost, and because attention-per-*byte* scales with
fertility * context, a long context amplifies the serving-cost advantage of a
low-fertility tokenizer. Local/global sparsity (most layers attend a bounded window)
dampens the absolute attention cost. See
``.agents/projects/20260703_tokenizer_flop_equivalent_bakeoff.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from experiments.grug.moe.model import GrugModelConfig

# Training FLOPs are forward + backward; the standard 3x over the forward pass.
TRAIN_FLOP_MULTIPLIER = 3.0

# The deployment-scale grug-moe the bake-off prices for: ~250B total / ~20B active. Only
# the shape matters here; context/sparsity/speed live on ServingCostModel. vocab_size is
# a placeholder replaced per arm.
TARGET_MODEL_SHAPE = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=6144,
    num_layers=64,
    num_heads=48,
    num_kv_heads=8,
    head_dim=128,
    intermediate_dim=3072,
    shared_expert_intermediate_dim=6144,
    num_experts=256,
    num_experts_per_token=8,
    max_seq_len=16_384,
    sliding_window=4_096,
)


def _head_dim(model: GrugModelConfig) -> int:
    return model.head_dim or (model.hidden_dim // model.num_heads)


def _attention_flops_per_token(model: GrugModelConfig, context_len: int) -> float:
    """Megatron-style full-attention FLOPs per token at ``context_len`` positions.

    Derived from the per-sequence attention FLOPs (QK^T + softmax + AV = seq^2 *
    (4*H*hd + 3*H)) divided by seq, i.e. ``context_len * (4*H*hd + 3*H)`` — linear in the
    context a token attends over.
    """
    heads = model.num_heads
    hd = _head_dim(model)
    return context_len * (4 * heads * hd + 3 * heads)


@dataclass(frozen=True)
class ServingCostModel:
    """Deployment cost model that prices tokenizers (does not measure BPB).

    Change any field and re-score stored measurements to replay under new assumptions.
    """

    model: GrugModelConfig = TARGET_MODEL_SHAPE
    context_len: int = 16_384  # serving context window, in tokens (replay 4k/64k via ServingCostModel)
    attention_window: int = 4_096  # local (sliding-window) attention span, in tokens
    global_layer_period: int = 6  # one global (full-context) layer per N layers; 6 => 5:1 local:global
    speed_factor: float = 1.0  # effective serving-cost multiplier for hw/kernel speedups (<1 = faster)

    def _global_fraction(self) -> float:
        return 1.0 / self.global_layer_period

    def attention_flops_per_token(self) -> float:
        """Layer-averaged attention FLOPs per token under the local/global mix."""
        frac_global = self._global_fraction()
        window = min(self.attention_window, self.context_len)
        return frac_global * _attention_flops_per_token(self.model, self.context_len) + (
            1 - frac_global
        ) * _attention_flops_per_token(self.model, window)

    def flops_per_token(self, vocab_size: int, *, include_lm_head: bool = True) -> float:
        """Forward FLOPs per token for the target model serving one token at this vocab."""
        m = self.model
        d = m.hidden_dim
        hd = _head_dim(m)
        routed_mlp = 2 * 3 * d * m.intermediate_dim * m.num_experts_per_token  # GLU => factor 3
        has_shared = m.shared_expert_intermediate_dim > 0
        shared_mlp = 2 * 3 * d * m.shared_expert_intermediate_dim * (1 if has_shared else 0)
        mlp = routed_mlp + shared_mlp
        if m.num_experts > 1:
            mlp += 2 * d * m.num_experts  # router
        qkv_proj = 2 * d * (m.num_heads * hd + 2 * m.num_kv_heads * hd)
        dense_proj = 2 * d * d
        attn = self.attention_flops_per_token()
        lm_head = 2 * d * vocab_size if include_lm_head else 0
        return self.speed_factor * (m.num_layers * (mlp + qkv_proj + dense_proj + attn) + lm_head)

    def attention_flop_fraction(self, vocab_size: int) -> float:
        """Share of forward FLOPs spent in attention (diagnostic; rises with context)."""
        m = self.model
        total = self.flops_per_token(vocab_size)
        if total == 0:
            return 0.0
        return self.speed_factor * m.num_layers * self.attention_flops_per_token() / total

    def lm_head_flop_fraction(self, vocab_size: int) -> float:
        """Share of forward FLOPs spent in the output head (rises with vocab)."""
        total = self.flops_per_token(vocab_size)
        if total == 0:
            return 0.0
        return self.speed_factor * 2 * self.model.hidden_dim * vocab_size / total


DEFAULT_SERVING = ServingCostModel()


@dataclass(frozen=True)
class ArmCost:
    """The deployment cost signature of one tokenizer arm.

    ``fertility`` (tokens/byte) is measured; the rest is derived from a
    :class:`ServingCostModel` so a reader can reconstruct the score from logs.
    """

    name: str
    vocab_size: int
    fertility: float  # tokens per byte
    flops_per_token: float  # forward, at serving context
    infer_flops_per_byte: float  # forward FLOPs to serve one byte of text (the serving cost)
    attention_flop_fraction: float
    lm_head_flop_fraction: float


def arm_cost(name: str, vocab_size: int, fertility: float, serving: ServingCostModel = DEFAULT_SERVING) -> ArmCost:
    """Price one arm under ``serving`` (default: the deployment target).

    Only ``vocab_size`` (the head term) and ``fertility`` distinguish arms; the rest of
    the per-token cost is shared across arms at a given ``serving`` model.
    """
    fpt = serving.flops_per_token(vocab_size)
    return ArmCost(
        name=name,
        vocab_size=vocab_size,
        fertility=fertility,
        flops_per_token=fpt,
        infer_flops_per_byte=fpt * fertility,
        attention_flop_fraction=serving.attention_flop_fraction(vocab_size),
        lm_head_flop_fraction=serving.lm_head_flop_fraction(vocab_size),
    )


# --- training-budget planning (proxy runs) ----------------------------------------
# These size the small proxy runs we actually train; they use the PROXY model's own
# forward FLOPs, not the deployment serving model.


def proxy_training_flops_per_token(proxy: GrugModelConfig) -> float:
    """Forward FLOPs/token for the proxy model at its own training context (full attention)."""
    trainer = ServingCostModel(
        model=proxy,
        context_len=proxy.max_seq_len,
        attention_window=proxy.max_seq_len,
        global_layer_period=1,  # all layers full-context at train time
        speed_factor=1.0,
    )
    return trainer.flops_per_token(proxy.vocab_size)


@dataclass(frozen=True)
class ComputePoint:
    """A single proxy training budget: how many tokens/bytes it buys for one arm."""

    total_train_flops: float
    tokens: float
    bytes: float


def compute_point_at_budget(proxy_flops_per_token: float, total_train_flops: float, fertility: float) -> ComputePoint:
    """Tokens and bytes a proxy run trains on at a fixed total training FLOP budget.

    ``tokens = C / (3 * proxy_flops_per_token)``; ``bytes = tokens / fertility``.
    """
    tokens = total_train_flops / (TRAIN_FLOP_MULTIPLIER * proxy_flops_per_token)
    return ComputePoint(total_train_flops=total_train_flops, tokens=tokens, bytes=tokens / fertility)


def budget_for_tokens(proxy_flops_per_token: float, tokens: float) -> float:
    """Total training FLOPs to train a proxy run on ``tokens`` tokens."""
    return TRAIN_FLOP_MULTIPLIER * proxy_flops_per_token * tokens


# --- BPB scaling curve + FLOP-equivalent BPB --------------------------------------


@dataclass(frozen=True)
class ScalingFit:
    """Fit of ``BPB(C) = a * C**(-b) + c`` for one arm across compute points."""

    a: float
    b: float
    c: float

    def bpb_at(self, total_train_flops: float) -> float:
        return self.a * total_train_flops ** (-self.b) + self.c


def fit_bpb_curve(points: Sequence[tuple[float, float]]) -> ScalingFit:
    """Fit ``BPB(C) = a*C^{-b} + c`` from (total_train_flops, bpb) points (>= 3).

    Fits ``b`` by a bounded 1-D search and solves ``a, c`` by least squares at each ``b``.
    """
    if len(points) < 3:
        raise ValueError(f"need >= 3 compute points to fit a scaling curve, got {len(points)}")
    xs = [math.log(c) for c, _ in points]
    ys = [bpb for _, bpb in points]

    def linfit(feat: Sequence[float], target: Sequence[float]) -> tuple[float, float, float]:
        n = len(feat)
        mf = sum(feat) / n
        mt = sum(target) / n
        sff = sum((f - mf) ** 2 for f in feat)
        sft = sum((f - mf) * (t - mt) for f, t in zip(feat, target, strict=True))
        a = sft / sff if sff > 0 else 0.0
        c = mt - a * mf
        sse = sum((t - (a * f + c)) ** 2 for f, t in zip(feat, target, strict=True))
        return a, c, sse

    best: tuple[float, ScalingFit] | None = None
    steps = 500
    for i in range(1, steps + 1):
        b = 0.5 * i / steps  # exponents in (0, 0.5] cover observed LM scaling with margin
        feat = [math.exp(-b * x) for x in xs]  # C^{-b}
        a, c, sse = linfit(feat, ys)
        if best is None or sse < best[0]:
            best = (sse, ScalingFit(a=a, b=b, c=c))
    assert best is not None
    return best[1]


# Default lifetime serving-to-training ratio: how many reference-trainings-worth of FLOPs
# a deployed model spends serving over its life, at the reference tokenizer's serving
# cost. 1.0 weights lifetime serving and training equally (a neutral midpoint); larger
# makes serving efficiency dominate, smaller recovers the equal-training comparison.
DEFAULT_SERVING_RATIO = 1.0


def febpb(
    fit: ScalingFit,
    reference_train_flops: float,
    relative_serving_cost: float,
    serving_ratio: float = DEFAULT_SERVING_RATIO,
) -> float:
    """FLOP-equivalent BPB: BPB after reinvesting serving savings into training.

    Fix a lifetime FLOP budget ``B = reference_train_flops * (1 + serving_ratio)`` and a
    served byte-volume shared across arms. An arm's serving cost scales by
    ``relative_serving_cost = arm.infer_flops_per_byte / reference.infer_flops_per_byte``
    and the remainder funds training:

        train_flops(arm) = reference_train_flops * (1 + serving_ratio*(1 - relative_serving_cost))

    A cheaper-to-serve arm gets more training and a lower BPB; an arm whose serving alone
    exceeds the lifetime budget is **infeasible** (``inf``) — the honest verdict for e.g.
    byte-level. Serving cost genuinely moves the score (unlike a fixed-training compare).
    """
    train_flops = reference_train_flops * (1.0 + serving_ratio * (1.0 - relative_serving_cost))
    if train_flops <= 0:
        return math.inf
    return fit.bpb_at(train_flops)


# --- fertility measurement --------------------------------------------------------


@dataclass(frozen=True)
class FertilityMeasurement:
    """Tokens/byte for one tokenizer over a corpus, with the raw counts behind it."""

    fertility: float  # tokens per byte
    total_tokens: int
    total_bytes: int


def fertility_of(encode: Callable[[str], Sequence[int]], corpus: Iterable[str]) -> FertilityMeasurement:
    """Measure tokens/byte for an encode function over a corpus of strings.

    ``encode`` maps a str to a list of token ids (e.g.
    ``lambda s: tokenizer.encode(s, add_special_tokens=False)``).
    """
    total_tokens = 0
    total_bytes = 0
    for text in corpus:
        total_tokens += len(encode(text))
        total_bytes += len(text.encode("utf-8"))
    if total_bytes == 0:
        raise ValueError("empty corpus: cannot measure fertility")
    return FertilityMeasurement(fertility=total_tokens / total_bytes, total_tokens=total_tokens, total_bytes=total_bytes)


def _self_check() -> None:
    """Demonstrate pricing and show how context length reshapes the tokenizer trade-off."""
    # (name, vocab, fertility) — placeholder fertilities until measured on corpus.
    arms = [
        ("llama3-128k", 128256, 0.260),
        ("qwen3-152k", 151669, 0.255),
        ("gemma-262k", 256000, 0.250),
        ("superbpe-200k", 200000, 0.205),
        ("byte-256", 256, 1.000),
    ]

    # Attention share and relative serving cost at three context windows: the tokenizer
    # trade-off shifts as attention grows with context.
    for ctx in (4_096, 16_384, 65_536):
        serving = ServingCostModel(context_len=ctx)
        costs = [arm_cost(n, v, f, serving) for n, v, f in arms]
        ref = next(c for c in costs if c.name == "llama3-128k").infer_flops_per_byte
        attn_share = serving.attention_flop_fraction(128256) * 100
        print(f"\ncontext={ctx} tokens  (attention = {attn_share:.1f}% of forward FLOPs, 5:1 local:global)")
        print(f"  {'arm':14s} {'vocab':>7s} {'fert':>6s} {'head%':>6s} {'infFLOP/byte':>12s} {'rel_serve':>9s}")
        for c in costs:
            print(
                f"  {c.name:14s} {c.vocab_size:7d} {c.fertility:6.3f} {c.lm_head_flop_fraction * 100:5.1f}% "
                f"{c.infer_flops_per_byte:12.3e} {c.infer_flops_per_byte / ref:9.3f}"
            )

    # feBPB with identical BPB curves => any spread is the serving-cost effect alone.
    serving = DEFAULT_SERVING
    costs = [arm_cost(n, v, f, serving) for n, v, f in arms]
    ref = next(c for c in costs if c.name == "llama3-128k").infer_flops_per_byte
    reference_train_flops = 3e18
    print(f"\nfeBPB at {serving.context_len} context, identical BPB curves (spread = serving-cost only):")
    print(f"  {'arm':14s} {'rel_serve':>9s} {'feBPB':>8s}")
    for c in costs:
        pts = [(cbudget, 0.90 + 3.0 * cbudget ** (-0.08)) for cbudget in (1e18, 3e18, 9e18)]
        fit = fit_bpb_curve(pts)
        fe = febpb(fit, reference_train_flops, c.infer_flops_per_byte / ref)
        fe_str = "inf" if fe == math.inf else f"{fe:8.4f}"
        print(f"  {c.name:14s} {c.infer_flops_per_byte / ref:9.3f} {fe_str:>8s}")


if __name__ == "__main__":
    _self_check()
