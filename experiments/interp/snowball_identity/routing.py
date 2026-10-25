# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Router rounding isolation for task #310: do BF16 router computations change the routes?

The checkpoint stores ``model.layers.{l}.mlp.router.bias`` in FP32 and the router weight in BF16. A trainer that casts
the whole model to BF16 (Levanter's ``WrappedLossFunction``) routes with a BF16-rounded bias, and Levanter's router
einsum (``snowball.py``, no ``preferred_element_type``) returns BF16 logits before its FP32 cast. At every position of
the pinned #298 sequences, each layer's router input is captured once, in the native forward, and scored three ways:

- ``native``: ``x.float() @ W.float()`` plus the FP32 bias, as in ``GrugMoeRouter.forward``;
- ``bf16_bias``: the same logits plus ``bias.bfloat16()``: only the bias rounded;
- ``bf16_logits_bias``: ``(x.bfloat16() @ W.bfloat16())`` rounded to BF16, plus the BF16 bias: the Levanter-like view.

Selection is the top ``k`` of the biased logits (JAX tie rule, ``grug_moe.jax_top_k``); combine weights are the
sigmoid of each variant's unbiased logits of the selected experts, renormalized to sum to ``GRUG_ROUTING_RENORM_SUM``.
Later layers always see the native residual stream, so each layer's count is the effect of that layer's rounding
alone, not an accumulated drift.

Inputs are the stored token ids of the pinned ``positions.pt`` (``logit_check.stored_sequences``), never a fresh
tokenization. Per variant, layer and sequence group (``check``, ``contrastive``, ``long``, ``long_past_window``) it
reports how many positions select a different expert set than ``native``, how many selected experts differ, the max abs
change of the dense combine weights, and the smallest native margin between the k-th and (k+1)-th biased logits.
Every position whose ordered ids differ is listed (up to ``MAX_CASES`` per variant). Output: ``routing.json``.
"""

import logging
import time
from collections import defaultdict

import torch

from experiments.interp.snowball_identity.harness import Output, SnowballHarness, import_grug
from experiments.interp.snowball_identity.logit_check import WINDOW, stored_sequences
from experiments.interp.snowball_identity.sequences import Sequence

logger = logging.getLogger(__name__)

VARIANTS = ("bf16_bias", "bf16_logits_bias")
MAX_CASES = 500
"""Differing positions listed per variant (all are counted)."""


def routes(logits: torch.Tensor, bias: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selected experts [T, k], dense combine weights [T, E], and the k-th minus (k+1)-th biased logit [T]."""
    grug = import_grug()
    top_values, top_ids = grug.jax_top_k(logits + bias, k + 1)
    selected = top_ids[:, :k]
    weights = torch.sigmoid(torch.gather(logits, -1, selected))
    weights = weights * (grug.GRUG_ROUTING_RENORM_SUM / (weights.sum(-1, keepdim=True) + grug.GRUG_ROUTER_RENORM_EPS))
    dense = torch.zeros_like(logits).scatter_(-1, selected, weights)
    return selected, dense, top_values[:, k - 1] - top_values[:, k]


def compare_routes(
    logits: torch.Tensor, bias: torch.Tensor, k: int, rounded_logits: torch.Tensor | None = None
) -> dict[str, dict[str, torch.Tensor]]:
    """Per variant and position: experts that differ from ``native``, the combine-weight change, the native margin.

    ``rounded_logits`` (default: ``logits`` rounded to BF16) are the ``bf16_logits_bias`` variant's logits."""
    if rounded_logits is None:
        rounded_logits = logits.bfloat16().float()
    native, native_weights, margin = routes(logits, bias, k)
    native_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, native, True)
    bias16 = bias.bfloat16().float()
    results = {}
    for name, (variant_logits, variant_bias) in {
        "bf16_bias": (logits, bias16),
        "bf16_logits_bias": (rounded_logits, bias16),
    }.items():
        selected, weights, _ = routes(variant_logits, variant_bias, k)
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, selected, True)
        results[name] = {
            "native": native,
            "selected": selected,
            "experts_changed": (native_mask & ~mask).sum(-1),
            "order_changed": (native != selected).any(-1),
            "weight_change": (native_weights - weights).abs().max(-1).values,
            "margin": margin,
        }
    return results


def routing_stats(harness: SnowballHarness, sequences: list[Sequence]) -> dict:
    """Per variant, layer and group, over every position of every sequence."""
    captured: dict[int, torch.Tensor] = {}

    def capture(layer: int):
        def hook(module, args, output):
            captured[layer] = args[0].detach()

        return hook

    handles = [block.mlp.router.register_forward_hook(capture(layer)) for layer, block in enumerate(harness.layers)]
    counts: dict = defaultdict(
        lambda: {
            "positions": 0,
            "positions_changed": 0,
            "positions_order_changed": 0,
            "experts_changed": 0,
            "max_weight_change": 0.0,
        }
    )
    cases: dict[str, list] = {variant: [] for variant in VARIANTS}
    margins: dict = defaultdict(lambda: float("inf"))
    started = time.time()
    try:
        for number, sequence in enumerate(sequences):
            captured.clear()
            harness.forward(sequence.tokens, assistant_start=0, logit_positions=torch.tensor([len(sequence.tokens) - 1]))
            groups = {sequence.group: slice(None)}
            if sequence.group == "long" and len(sequence.tokens) > WINDOW:
                groups["long_past_window"] = slice(WINDOW, None)
            for layer, block in enumerate(harness.layers):
                router = block.mlp.router
                x = captured[layer]
                logits = torch.nn.functional.linear(x.float(), router.weight.float())
                rounded = torch.nn.functional.linear(x.bfloat16(), router.weight.bfloat16()).float()
                bias = router.bias.detach().to(device=logits.device, dtype=torch.float32)
                results = compare_routes(logits, bias, router.top_k, rounded)
                for variant, result in results.items():
                    result = {name: value.cpu() for name, value in result.items()}
                    for group, rows in groups.items():
                        changed = result["experts_changed"][rows]
                        count = counts[(variant, layer, group)]
                        count["positions"] += len(changed)
                        count["positions_changed"] += int((changed > 0).sum())
                        count["positions_order_changed"] += int(result["order_changed"][rows].sum())
                        count["experts_changed"] += int(changed.sum())
                        count["max_weight_change"] = max(
                            count["max_weight_change"], float(result["weight_change"][rows].max())
                        )
                        margins[(layer, group)] = min(margins[(layer, group)], float(result["margin"][rows].min()))
                    for position in torch.nonzero(result["order_changed"]).flatten().tolist():
                        if len(cases[variant]) < MAX_CASES:
                            cases[variant].append(
                                {
                                    "sequence": sequence.id,
                                    "position": position,
                                    "layer": layer,
                                    "native": result["native"][position].tolist(),
                                    "variant": result["selected"][position].tolist(),
                                    "set_changed": bool(result["experts_changed"][position] > 0),
                                    "native_margin": float(result["margin"][position]),
                                }
                            )
            if number % 50 == 0:
                logger.info("routing %d/%d %s (%.0fs)", number, len(sequences), sequence.id, time.time() - started)
    finally:
        for handle in handles:
            handle.remove()
    groups = sorted({group for _, _, group in counts})
    layers = []
    for layer, block in enumerate(harness.layers):
        bias = block.mlp.router.bias.detach().float()
        layers.append(
            {
                "layer": layer,
                "bias_dtype": str(block.mlp.router.bias.dtype),
                "weight_dtype": str(block.mlp.router.weight.dtype),
                "bias_rounding_max_abs": float((bias - bias.bfloat16().float()).abs().max()),
                "min_native_margin": {group: margins[(layer, group)] for group in groups},
                **{
                    variant: {
                        group: counts[(variant, layer, group)] for group in groups if (variant, layer, group) in counts
                    }
                    for variant in VARIANTS
                },
            }
        )
    # Totals over check, contrastive and long (long_past_window is a subset of long).
    totals = {}
    for variant in VARIANTS:
        rows = [
            counts[(variant, layer, group)]
            for layer in range(len(layers))
            for group in groups
            if group != "long_past_window"
        ]
        totals[variant] = {
            name: sum(row[name] for row in rows)
            for name in ("positions", "positions_changed", "positions_order_changed", "experts_changed")
        }
    return {
        "sequences": len(sequences),
        "top_k": harness.layers[0].mlp.router.top_k,
        "variants": list(VARIANTS),
        "layers": layers,
        "totals": totals,
        "cases": cases,
    }


def router_rounding(harness: SnowballHarness, tokenizer, spec: dict, output: Output, stop_ids: set[int]) -> None:
    """Spec: ``positions`` (the pinned #298 ``positions.pt``, whose stored token ids are the inputs)."""
    started = time.time()
    report = routing_stats(harness, stored_sequences(spec["positions"]))
    report["positions"] = spec["positions"]
    report["seconds"] = round(time.time() - started, 1)
    output.write_json("routing.json", report)
