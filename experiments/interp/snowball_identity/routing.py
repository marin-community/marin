# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Router rounding isolation for task #310: does rounding the FP32 router bias to BF16 change the routes?

The checkpoint stores ``model.layers.{l}.mlp.router.bias`` in FP32. A trainer that casts the whole model to BF16
(Levanter's ``WrappedLossFunction``) routes with the BF16-rounded bias instead. At every position of the fixed
#298 sequences, each layer's router input is captured once, in the native forward (FP32 bias), and scored twice:

- ``native``: ``logits + bias``, the forward the harness runs;
- ``bf16_bias``: ``logits + bias.bfloat16().float()``, the same input with only the bias rounded.

``logits`` is ``x.float() @ W.float()`` as in ``GrugMoeRouter.forward``. Selection is the top ``k`` of the biased
logits (JAX tie rule, ``grug_moe.jax_top_k``); combine weights are the sigmoid of the unbiased logits of the selected
experts, renormalized to sum to ``GRUG_ROUTING_RENORM_SUM``. Later layers always see the native residual stream, so
each layer's count is the effect of that layer's rounding alone, not an accumulated drift.

Per layer and per sequence group (``check``, ``contrastive``, ``long``, ``long_past_window``) it reports how many
positions select a different expert set, how many selected experts differ, the max abs change of the dense combine
weights, and the smallest native margin between the k-th and (k+1)-th biased logits. Output: ``routing.json``.
"""

import logging
import time
from collections import defaultdict

import torch

from experiments.interp.snowball_identity.harness import Output, SnowballHarness, import_grug
from experiments.interp.snowball_identity.logit_check import WINDOW
from experiments.interp.snowball_identity.sequences import Sequence, build_sequences

logger = logging.getLogger(__name__)


def routes(logits: torch.Tensor, bias: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selected experts [T, k], dense combine weights [T, E], and the k-th minus (k+1)-th biased logit [T]."""
    grug = import_grug()
    top_values, top_ids = grug.jax_top_k(logits + bias, k + 1)
    selected = top_ids[:, :k]
    weights = torch.sigmoid(torch.gather(logits, -1, selected))
    weights = weights * (grug.GRUG_ROUTING_RENORM_SUM / (weights.sum(-1, keepdim=True) + grug.GRUG_ROUTER_RENORM_EPS))
    dense = torch.zeros_like(logits).scatter_(-1, selected, weights)
    return selected, dense, top_values[:, k - 1] - top_values[:, k]


def compare_routes(logits: torch.Tensor, bias: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Per position: experts that differ between the native and BF16-rounded bias, and the combine-weight change."""
    native, native_weights, margin = routes(logits, bias, k)
    rounded, rounded_weights, _ = routes(logits, bias.bfloat16().float(), k)
    native_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, native, True)
    rounded_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, rounded, True)
    return {
        "experts_changed": (native_mask & ~rounded_mask).sum(-1),
        "weight_change": (native_weights - rounded_weights).abs().max(-1).values,
        "margin": margin,
    }


def routing_stats(harness: SnowballHarness, sequences: list[Sequence]) -> dict:
    """Per layer and group, over every position of every sequence."""
    captured: dict[int, torch.Tensor] = {}

    def capture(layer: int):
        def hook(module, args, output):
            captured[layer] = args[0].detach()

        return hook

    handles = [block.mlp.router.register_forward_hook(capture(layer)) for layer, block in enumerate(harness.layers)]
    totals: dict = defaultdict(lambda: {"positions": 0, "positions_changed": 0, "experts_changed": 0})
    extremes: dict = defaultdict(lambda: {"max_weight_change": 0.0, "min_margin": float("inf")})
    started = time.time()
    try:
        for number, sequence in enumerate(sequences):
            captured.clear()
            harness.forward(sequence.tokens, assistant_start=0, logit_positions=torch.tensor([len(sequence.tokens) - 1]))
            for layer, block in enumerate(harness.layers):
                router = block.mlp.router
                x = captured[layer]
                logits = torch.nn.functional.linear(x.float(), router.weight.float())
                bias = router.bias.detach().to(device=logits.device, dtype=torch.float32)
                result = {name: value.cpu() for name, value in compare_routes(logits, bias, router.top_k).items()}
                groups = {sequence.group: slice(None)}
                if sequence.group == "long" and len(sequence.tokens) > WINDOW:
                    groups["long_past_window"] = slice(WINDOW, None)
                for group, rows in groups.items():
                    changed = result["experts_changed"][rows]
                    total = totals[(layer, group)]
                    total["positions"] += len(changed)
                    total["positions_changed"] += int((changed > 0).sum())
                    total["experts_changed"] += int(changed.sum())
                    extreme = extremes[(layer, group)]
                    extreme["max_weight_change"] = max(
                        extreme["max_weight_change"], float(result["weight_change"][rows].max())
                    )
                    extreme["min_margin"] = min(extreme["min_margin"], float(result["margin"][rows].min()))
            if number % 50 == 0:
                logger.info("routing %d/%d %s (%.0fs)", number, len(sequences), sequence.id, time.time() - started)
    finally:
        for handle in handles:
            handle.remove()
    layers = []
    for layer, block in enumerate(harness.layers):
        bias = block.mlp.router.bias.detach().float()
        row = {
            "layer": layer,
            "bias_dtype": str(block.mlp.router.bias.dtype),
            "bias_rounding_max_abs": float((bias - bias.bfloat16().float()).abs().max()),
            "groups": {},
        }
        for (total_layer, group), total in sorted(totals.items()):
            if total_layer == layer:
                row["groups"][group] = {**total, **extremes[(layer, group)]}
        layers.append(row)
    all_groups = [group for group in {g for _, g in totals} if group != "long_past_window"]
    return {
        "sequences": len(sequences),
        "top_k": harness.layers[0].mlp.router.top_k,
        "layers": layers,
        "totals": {
            "positions": sum(totals[(layer, g)]["positions"] for layer in range(len(layers)) for g in all_groups),
            "positions_changed": sum(
                totals[(layer, g)]["positions_changed"] for layer in range(len(layers)) for g in all_groups
            ),
            "experts_changed": sum(
                totals[(layer, g)]["experts_changed"] for layer in range(len(layers)) for g in all_groups
            ),
        },
    }


def router_rounding(harness: SnowballHarness, tokenizer, spec: dict, output: Output, stop_ids: set[int]) -> None:
    """Spec: ``identity`` (the #298 identity.jsonl that builds the fixed sequences)."""
    started = time.time()
    report = routing_stats(harness, build_sequences(tokenizer, spec["identity"]))
    report["identity"] = spec["identity"]
    report["seconds"] = round(time.time() - started, 1)
    output.write_json("routing.json", report)
