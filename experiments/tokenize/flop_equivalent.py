# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FLOP-equivalent scoring for the tokenizer bake-off.

Cross-tokenizer quality is scored with bits-per-byte (BPB), which is byte-anchored
and therefore comparable across vocabularies. The remaining question is *cost*: a
large-vocab or byte-level tokenizer is more expensive to train and serve, so its
BPB must be read against the compute it spends. This module supplies the two cost
quantities every arm is scored on and the compute-matched budget arithmetic.

The FLOP model is Levanter's ``lm_flops_per_token``. Vocabulary size enters exactly
once, through the output head ``lm_head = 2 * hidden_dim * vocab_size``; the input
embedding is a gather (~0 FLOPs). A tokenizer therefore changes model compute in two
places: the LM-head FLOPs per token (rising with vocab) and the *fertility*
``f = tokens/byte`` (how many tokens must be processed to cover a fixed amount of
text). See ``.agents/projects/20260703_tokenizer_flop_equivalent_bakeoff.md``.

Cost is priced at the **target deployment scale**, not at the small proxy model used
to measure BPB. The grug-moe family we care about deploying is a ~250B-total /
~20B-active MoE (:data:`TARGET_MODEL`); at that width the LM-head is a few percent of
FLOPs, so a large vocab is nearly free and fertility dominates the serving cost — the
opposite of the tiny d1024 proxies, where the head is 30-50% of FLOPs and would wildly
over-penalize large vocab. Quality (BPB scaling curves) is measured on the small
proxies; the cost multipliers that turn BPB into feBPB use :data:`TARGET_MODEL`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.moe.model import GrugModelConfig

# Training FLOPs are forward + backward; the standard 3x multiplier over the forward
# pass. Inference (serving) is the forward pass only.
TRAIN_FLOP_MULTIPLIER = 3.0


def flops_per_token(model: GrugModelConfig, *, include_lm_head: bool = True) -> float:
    """Analytic FLOPs per token for a grug MoE config.

    ``include_lm_head=False`` returns the non-embedding FLOPs (the grug heuristic's
    budget convention); the bake-off scores on the full figure so the vocab-dependent
    head cost is not free.
    """
    fpt = lm_flops_per_token(
        hidden_dim=model.hidden_dim,
        intermediate_dim=model.intermediate_dim,
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        num_heads=model.num_heads,
        seq_len=model.max_seq_len,
        vocab_size=model.vocab_size,
        glu=True,
        num_experts=model.num_experts,
        num_shared_experts=1 if model.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model.num_experts_per_token,
        shared_intermediate_dim=model.shared_expert_intermediate_dim,
    )
    if not include_lm_head:
        fpt -= 2 * model.hidden_dim * model.vocab_size
    return fpt


@dataclass(frozen=True)
class ArmCost:
    """The cost signature of one tokenizer arm at a fixed model shape.

    ``fertility`` is tokens/byte measured on the held-out eval corpus. Everything
    else is derived so a reader can reconstruct the score from logs.
    """

    name: str
    vocab_size: int
    fertility: float  # tokens per byte
    flops_per_token_full: float  # includes lm_head
    flops_per_token_nonemb: float  # excludes lm_head

    @property
    def train_flops_per_byte(self) -> float:
        return TRAIN_FLOP_MULTIPLIER * self.flops_per_token_full * self.fertility

    @property
    def infer_flops_per_byte(self) -> float:
        """Forward-pass FLOPs to serve one byte of text — the honest serving cost."""
        return self.flops_per_token_full * self.fertility

    @property
    def lm_head_flop_fraction(self) -> float:
        return 1.0 - self.flops_per_token_nonemb / self.flops_per_token_full


# The deployment-scale grug-moe the bake-off is pricing for: ~250B total / ~20B active.
# Cost multipliers (infer_flops_per_byte, feBPB) are computed at THIS width, not at the
# small proxy models used to measure BPB — at 20B active the LM-head is only a few
# percent of FLOPs, so vocab size is nearly free and fertility dominates the serving
# cost. Pricing at the proxy width (head = 30-50% of FLOPs) would wildly over-penalize
# large vocab. num_experts/top_k/depth are set to land near 20B active / 250B total;
# only hidden_dim and the non-embedding flops_per_token feed the cost math, so the exact
# expert count is not load-bearing for arm *ratios*.
TARGET_MODEL = GrugModelConfig(
    vocab_size=128_256,  # replaced per arm by arm_cost; only the non-head part is shared
    hidden_dim=6144,
    num_layers=64,
    num_heads=48,
    num_kv_heads=8,
    head_dim=128,
    intermediate_dim=3072,
    shared_expert_intermediate_dim=6144,
    num_experts=256,
    num_experts_per_token=8,
    max_seq_len=4096,
    sliding_window=4096,
)


def arm_cost(name: str, vocab_size: int, fertility: float, cost_model: GrugModelConfig = TARGET_MODEL) -> ArmCost:
    """Build the cost signature for one arm, priced at ``cost_model`` (default: deployment scale).

    ``cost_model`` is the model whose FLOPs price the tokenizer — the ~250B/20B-active
    :data:`TARGET_MODEL`, not the small proxy used to measure BPB. Only ``vocab_size``
    (the LM-head term) and ``fertility`` distinguish arms; the non-embedding
    flops_per_token is shared across arms at a given ``cost_model``.
    """
    priced = replace(cost_model, vocab_size=vocab_size)
    return ArmCost(
        name=name,
        vocab_size=vocab_size,
        fertility=fertility,
        flops_per_token_full=flops_per_token(priced, include_lm_head=True),
        flops_per_token_nonemb=flops_per_token(priced, include_lm_head=False),
    )


@dataclass(frozen=True)
class ComputePoint:
    """A single training budget for an arm: how many tokens/bytes it buys."""

    total_train_flops: float
    tokens: float
    bytes: float


def compute_point_at_budget(cost: ArmCost, total_train_flops: float) -> ComputePoint:
    """Tokens and bytes an arm trains on at a fixed *total training FLOP* budget.

    ``tokens = C / (3 * F_full)``; ``bytes = tokens / fertility``. A low-fertility,
    modest-vocab tokenizer covers more bytes for the same budget; a large-vocab or
    byte tokenizer covers fewer. This is the equal-``C_total`` headline comparison.
    """
    tokens = total_train_flops / (TRAIN_FLOP_MULTIPLIER * cost.flops_per_token_full)
    return ComputePoint(total_train_flops=total_train_flops, tokens=tokens, bytes=tokens / cost.fertility)


def budget_for_tokens(cost: ArmCost, tokens: float) -> float:
    """Inverse of :func:`compute_point_at_budget`: total training FLOPs for N tokens."""
    return TRAIN_FLOP_MULTIPLIER * cost.flops_per_token_full * tokens


def budget_for_bytes(cost: ArmCost, num_bytes: float) -> float:
    """Total training FLOPs for an arm to cover ``num_bytes`` bytes of text."""
    return budget_for_tokens(cost, num_bytes * cost.fertility)


@dataclass(frozen=True)
class ScalingFit:
    """Fit of ``BPB(C) = a * C**(-b) + c`` for one arm across compute points."""

    a: float
    b: float
    c: float

    def bpb_at(self, total_train_flops: float) -> float:
        return self.a * total_train_flops ** (-self.b) + self.c


def fit_bpb_curve(points: Sequence[tuple[float, float]]) -> ScalingFit:
    """Fit ``BPB(C) = a*C^{-b} + c`` from (total_train_flops, bpb) points.

    Needs >= 3 points. Fits ``b`` by a 1-D search (bounded, robust for a handful of
    points) and solves ``a, c`` by least squares at each ``b``. Returns the best fit.
    """
    if len(points) < 3:
        raise ValueError(f"need >= 3 compute points to fit a scaling curve, got {len(points)}")
    xs = [math.log(c) for c, _ in points]
    ys = [bpb for _, bpb in points]

    def linfit(feat: Sequence[float], target: Sequence[float]) -> tuple[float, float, float]:
        # least-squares target ~ a*feat + c; returns (a, c, sse)
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
    # b in (0, 0.5] covers observed LM scaling exponents with margin.
    steps = 500
    for i in range(1, steps + 1):
        b = 0.5 * i / steps
        feat = [math.exp(-b * x) for x in xs]  # C^{-b}
        a, c, sse = linfit(feat, ys)
        if best is None or sse < best[0]:
            best = (sse, ScalingFit(a=a, b=b, c=c))
    assert best is not None
    return best[1]


def febpb(cost: ArmCost, fit: ScalingFit, reference_infer_flops_per_byte: float) -> float:
    """FLOP-equivalent BPB: the arm's BPB when trained to a common serving budget.

    An arm cheaper to serve than the reference can afford proportionally more training
    compute (and vice versa). We set each arm's training budget so that
    ``bytes_trained * infer_flops_per_byte`` matches a reference service budget, then
    read BPB off the fitted curve. This rewards cheap-to-serve tokenizers exactly as
    much as their serving discount merits, collapsing the Pareto trade into one number.

    Concretely we anchor the reference budget at ``REFERENCE_SERVICE_BYTES`` bytes at the
    reference serving cost, and give each arm the training FLOPs to reach the same
    service-normalized data volume.
    """
    # Service budget: reference bytes at the reference per-byte serving cost.
    service_budget = REFERENCE_SERVICE_BYTES * reference_infer_flops_per_byte
    arm_bytes = service_budget / cost.infer_flops_per_byte
    total_flops = budget_for_bytes(cost, arm_bytes)
    return fit.bpb_at(total_flops)


# Anchor for feBPB: a fixed held-out corpus size (in bytes) the reference tokenizer is
# "served over". Its absolute value only shifts every arm's training budget by the same
# multiplicative factor, so rankings are invariant; 100 GB is a representative scale.
REFERENCE_SERVICE_BYTES = 100e9


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


def _proxy_model(vocab: int) -> GrugModelConfig:
    """The small model the bake-off actually trains to measure BPB (d1024, ~few-B total)."""
    return GrugModelConfig(
        vocab_size=vocab,
        hidden_dim=1024,
        num_layers=16,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        intermediate_dim=512,
        shared_expert_intermediate_dim=1024,
        num_experts=32,
        num_experts_per_token=4,
        max_seq_len=2048,
        sliding_window=2048,
    )


def _self_check() -> None:
    """Demonstrate the scoring, contrasting proxy-scale vs deployment-scale pricing."""

    # (name, vocab, fertility) — fertilities are placeholders until measured on corpus.
    arms = [
        ("llama3-128k", 128256, 0.260),
        ("qwen3-152k", 151669, 0.255),
        ("gemma-262k", 256000, 0.250),
        ("superbpe-200k", 200000, 0.205),
        ("byte-256", 256, 1.000),
    ]

    # The lm-head FLOP fraction shrinks ~10x from proxy to deployment scale, which is
    # exactly why cost must be priced at TARGET_MODEL, not the proxy.
    print("lm-head FLOP fraction by scale (why we price at deployment scale):")
    print(f"  {'arm':14s} {'proxy d1024':>12s} {'target d6144':>13s}")
    for n, v, _ in arms:
        proxy = arm_cost(n, v, 1.0, cost_model=_proxy_model(v))
        tgt = arm_cost(n, v, 1.0, cost_model=TARGET_MODEL)
        print(f"  {n:14s} {proxy.lm_head_flop_fraction*100:11.1f}% {tgt.lm_head_flop_fraction*100:12.1f}%")

    costs = [arm_cost(n, v, f) for n, v, f in arms]  # priced at TARGET_MODEL
    ref = next(c for c in costs if c.name == "llama3-128k").infer_flops_per_byte

    print("\ndeployment-scale cost signature:")
    print(f"{'arm':14s} {'vocab':>7s} {'fert':>6s} {'lmhead%':>7s} {'infFLOP/byte':>12s} {'rel_serve':>9s}")
    for c in costs:
        print(
            f"{c.name:14s} {c.vocab_size:7d} {c.fertility:6.3f} "
            f"{c.lm_head_flop_fraction*100:6.1f}% {c.infer_flops_per_byte:12.3e} "
            f"{c.infer_flops_per_byte/ref:9.3f}"
        )

    # Synthetic isoFLOP curves (better tokenizers => lower asymptote) to exercise the fit.
    print("\nfeBPB (synthetic BPB curves; lower is better):")
    for c in costs:
        floor = {
            "llama3-128k": 0.90,
            "qwen3-152k": 0.895,
            "gemma-262k": 0.892,
            "superbpe-200k": 0.885,
            "byte-256": 0.95,
        }[c.name]
        pts = []
        for cbudget in (1e18, 3e18, 9e18):
            pts.append((cbudget, floor + 3.0 * cbudget ** (-0.08)))
        fit = fit_bpb_curve(pts)
        print(f"  {c.name:14s} feBPB={febpb(c, fit, ref):.4f}  (fit floor c={fit.c:.4f}, b={fit.b:.3f})")


if __name__ == "__main__":
    _self_check()
