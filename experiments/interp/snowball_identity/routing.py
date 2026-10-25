# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Router rounding isolation for task #310: do BF16 router computations change the routes?

The checkpoint stores ``model.layers.{l}.mlp.router.bias`` in FP32 and the router weight in BF16. A trainer that casts
the whole model to BF16 (Levanter's ``WrappedLossFunction``) routes with a BF16-rounded bias, and Levanter's router
einsum (``snowball.py``, no ``preferred_element_type``) returns BF16 logits before its FP32 cast. At every position of
the pinned #298 sequences, each layer's router input is captured once, in the native forward, and scored three ways:

- ``native``: ``x.float() @ W.float()`` plus the FP32 bias, as in ``GrugMoeRouter.forward``;
- ``bf16_bias``: the same logits plus ``bias.bfloat16()``: only the bias rounded;
- ``bf16_logits``: ``(x.bfloat16() @ W.bfloat16())`` rounded to BF16, plus the FP32 bias: only the dot rounded;
- ``bf16_logits_bias``: ``(x.bfloat16() @ W.bfloat16())`` rounded to BF16, plus the BF16 bias: the Levanter-like view.

Selection is the top ``k`` of the biased logits (JAX tie rule, ``grug_moe.jax_top_k``); combine weights are the
sigmoid of each variant's unbiased logits of the selected experts, renormalized to sum to ``GRUG_ROUTING_RENORM_SUM``.
Later layers always see the native residual stream, so each layer's count is the effect of that layer's rounding
alone, not an accumulated drift.

Inputs are the stored token ids of the pinned ``positions.pt`` (``logit_check.stored_sequences``), never a fresh
tokenization. Per variant, layer and sequence group (``check``, ``contrastive``, ``long``, ``long_past_window``) it
reports how many positions select a different expert set than ``native``, how many selected experts differ, the max abs
change of the dense combine weights, the smallest native margin between the k-th and (k+1)-th biased logits, and how
many positions have an exact tie among the native top k+1 biased logits (there the tie rule decides; torch's
``jax_top_k`` and Levanter's ``jax.lax.top_k`` take the lower index, vLLM's ``torch.topk`` has no stated rule).
Every position whose ordered ids differ is listed (up to ``MAX_CASES`` per variant). Output: ``routing.json``.

The two pre-registered #310 routing rows (precision-preflight-preregistration-v1.md):

- row 1, ``bias_exception_required``: any ``bf16_bias`` expert-set change requires keeping the 26 router biases in
  FP32 in the compute view (``compute_dtype_overrides={"router_bias": "float32"}``); ``dot_exception_required``: any
  ``bf16_logits`` (dot-only) expert-set change requires the amendment-v2 FP32 router dot (and the FP32 bias);
- row 2, ``route_gate``: fail-closed check of the cross-stack report (``compare_routes_cross_stack``): incomplete
  coverage, an exception that does not match row 1 and a combined-only change fail. Route differences are bounded by
  amendment G1 instead (``route_proxy_check``: per-layer rate bound and near-tie margins against the torch floor);
  set and ordered agreement are reported here, not gated. Exact ties are counted and reported, never exempted.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import time
from collections import defaultdict

import torch

from experiments.interp.snowball_identity.harness import Output, SnowballHarness, import_grug
from experiments.interp.snowball_identity.logit_check import WINDOW, read, stored_sequences
from experiments.interp.snowball_identity.sequences import Sequence

logger = logging.getLogger(__name__)

VARIANTS = ("bf16_bias", "bf16_logits", "bf16_logits_bias")
MATMUL_ENV = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION")
"""Environment variables that can change matmul precision, recorded with each report."""
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

    ``rounded_logits`` (default: ``logits`` rounded to BF16) are the ``bf16_logits`` and ``bf16_logits_bias`` logits."""
    if rounded_logits is None:
        rounded_logits = logits.bfloat16().float()
    native, native_weights, margin = routes(logits, bias, k)
    native_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, native, True)
    bias16 = bias.bfloat16().float()
    results = {}
    for name, (variant_logits, variant_bias) in {
        "bf16_bias": (logits, bias16),
        "bf16_logits": (rounded_logits, bias),
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
    native_routes: dict[str, torch.Tensor] = {}
    margins: dict = defaultdict(lambda: float("inf"))
    ties: dict = defaultdict(int)
    started = time.time()
    try:
        for number, sequence in enumerate(sequences):
            captured.clear()
            harness.forward(sequence.tokens, assistant_start=0, logit_positions=torch.tensor([len(sequence.tokens) - 1]))
            groups = {sequence.group: slice(None)}
            if sequence.group == "long" and len(sequence.tokens) > WINDOW:
                groups["long_past_window"] = slice(WINDOW, None)
            layer_routes = []
            for layer, block in enumerate(harness.layers):
                router = block.mlp.router
                x = captured[layer]
                logits = torch.nn.functional.linear(x.float(), router.weight.float())
                rounded = torch.nn.functional.linear(x.bfloat16(), router.weight.bfloat16()).float()
                bias = router.bias.detach().to(device=logits.device, dtype=torch.float32)
                results = compare_routes(logits, bias, router.top_k, rounded)
                # Exact ties among the native top k+1 biased logits, where the tie rule alone picks the order.
                top_values = torch.topk(logits + bias, router.top_k + 1).values
                exact_tie = (top_values[:, 1:] == top_values[:, :-1]).any(-1).cpu()
                for group, rows in groups.items():
                    ties[(layer, group)] += int(exact_tie[rows].sum())
                layer_routes.append(results["bf16_bias"]["native"].cpu())
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
                                    "native_exact_tie": bool(exact_tie[position]),
                                }
                            )
            native_routes[sequence.id] = torch.stack(layer_routes, dim=1).to(torch.int16)  # [T, L, K]
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
                "native_exact_tie_positions": {group: ties[(layer, group)] for group in groups},
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
        "native_routes": native_routes,
    }


def router_rounding(harness: SnowballHarness, tokenizer, spec: dict, output: Output, stop_ids: set[int]) -> None:
    """Spec: ``positions`` (the pinned #298 ``positions.pt``, whose stored token ids are the inputs) and its
    ``positions_sha256``, checked before the forward.

    Writes ``routing.json`` and ``routes.pt`` (``{sequence id: native FP32-bias routes [T, L, K] int16}``, the torch
    side of the cross-stack row)."""
    started = time.time()
    actual = hashlib.sha256(read(spec["positions"])).hexdigest()
    if actual != spec["positions_sha256"]:
        raise ValueError(f"{spec['positions']}: sha256 {actual}, expected {spec['positions_sha256']}")
    report = routing_stats(harness, stored_sequences(spec["positions"]))
    output.save("routes.pt", report.pop("native_routes"))
    report["positions"] = spec["positions"]
    report["positions_sha256"] = actual
    report["matmul"] = {
        "torch": torch.__version__,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env": {name: os.environ.get(name) for name in MATMUL_ENV},
    }
    report["seconds"] = round(time.time() - started, 1)
    output.write_json("routing.json", report)


def compare_routes_cross_stack(levanter_path: str, torch_path: str) -> dict:
    """Levanter compute-view routes (``levanter_positions`` with ``routes=True``) vs torch FP32-bias routes
    (``routes.pt``), per layer and group: positions compared, selected-set agreement, ordered-id agreement, and every
    differing position (up to ``MAX_CASES``). Fails if the sequence ids, token counts or layer counts differ."""
    ours = torch.load(io.BytesIO(read(levanter_path)), map_location="cpu")
    theirs = torch.load(io.BytesIO(read(torch_path)), map_location="cpu")
    if set(ours) != set(theirs):
        raise ValueError("Levanter and torch routes cover different sequence ids")
    counts: dict = defaultdict(lambda: {"positions": 0, "set_agree": 0, "ordered_agree": 0})
    cases = []
    for sequence_id in sorted(ours):
        levanter, native = ours[sequence_id]["routes"].long(), theirs[sequence_id].long()
        if levanter.shape != native.shape:
            raise ValueError(f"{sequence_id}: route shapes differ, {tuple(levanter.shape)} vs {tuple(native.shape)}")
        group = sequence_id.split("|")[0]
        ordered = (levanter == native).all(-1)  # [T, L]
        same_set = (levanter.sort(-1).values == native.sort(-1).values).all(-1)
        for layer in range(native.shape[1]):
            spans = {group: slice(None)}
            if group == "long" and native.shape[0] > WINDOW:
                spans["long_past_window"] = slice(WINDOW, None)
            for name, rows in spans.items():
                count = counts[(layer, name)]
                count["positions"] += len(ordered[rows, layer])
                count["set_agree"] += int(same_set[rows, layer].sum())
                count["ordered_agree"] += int(ordered[rows, layer].sum())
        for position, layer in torch.nonzero(~ordered).tolist():
            if len(cases) < MAX_CASES:
                cases.append(
                    {
                        "sequence": sequence_id,
                        "position": position,
                        "layer": layer,
                        "levanter": levanter[position, layer].tolist(),
                        "torch": native[position, layer].tolist(),
                        "set_changed": not bool(same_set[position, layer]),
                    }
                )
    groups = sorted({name for _, name in counts})
    layers = sorted({layer for layer, _ in counts})
    main_groups = [g for g in groups if g != "long_past_window"]
    totals = {
        name: sum(counts[(layer, g)][name] for layer in layers for g in main_groups)
        for name in ("positions", "set_agree", "ordered_agree")
    }
    return {
        "levanter": levanter_path,
        "torch": torch_path,
        "label": "Levanter compute-view routes vs torch FP32-bias routes (not served-vLLM routes)",
        "layers": [{"layer": layer, **{g: counts[(layer, g)] for g in groups}} for layer in layers],
        "totals": totals,
        "cases": cases,
    }


def bias_exception_required(rounding: dict) -> bool:
    """Row 1: whether the ``router_rounding`` report has any position whose BF16-rounded bias selects a different
    expert set than the FP32 bias. Order-only changes are reported in the report's totals but do not trigger it."""
    return rounding["totals"]["bf16_bias"]["positions_changed"] > 0


def dot_exception_required(rounding: dict) -> bool:
    """Row 1, amendment v2: whether rounding only the router dot (FP32 bias kept) changes any expert set."""
    return rounding["totals"]["bf16_logits"]["positions_changed"] > 0


def route_gate(cross: dict, rounding: dict, compute_dtype_overrides: dict | None, router_dot_fp32: bool) -> list[str]:
    """Row 2: every failure of a cross-stack report scored with ``compute_dtype_overrides``; empty is a pass. Route
    differences are G1's (``route_proxy_check``), not this row's.

    ``rounding`` is the torch ``router_rounding`` report from the same job that wrote the compared ``routes.pt``;
    ``router_dot_fp32`` is the measured witness (``levanter_positions.router_dot_dtypes`` of the scored view). The
    FP32 router dot exception also keeps the bias in FP32, so the bias override is expected when either row-1
    exception triggers. An exception applied without its trigger fails, and so does a combined-only change (both
    single-rounding variants clean, ``bf16_logits_bias`` changed): it has no attributed cause."""
    failures = []
    totals = rounding["totals"]
    bias_required, dot_required = bias_exception_required(rounding), dot_exception_required(rounding)
    bias_fp32 = (compute_dtype_overrides or {}).get("router_bias") == "float32"
    changes = {variant: totals[variant]["positions_changed"] for variant in VARIANTS}
    if bias_fp32 != (bias_required or dot_required):
        failures.append(f"router_bias FP32 override {'applied' if bias_fp32 else 'missing'}; row 1 changes {changes}")
    if router_dot_fp32 != dot_required:
        failures.append(f"FP32 router dot {'applied' if router_dot_fp32 else 'missing'}; row 1 changes {changes}")
    if changes["bf16_logits_bias"] and not (bias_required or dot_required):
        failures.append(f"combined-only routing change with no single cause; stop for diagnosis: {changes}")
    covered, expected = cross["totals"], totals["bf16_bias"]["positions"]
    if covered["positions"] != expected or expected == 0:
        failures.append(f"cross-stack covers {covered['positions']} position-layers, torch {expected}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-stack routing row: Levanter vs torch FP32-bias routes.")
    parser.add_argument("--levanter", required=True, help="levanter_positions positions.pt written with routes=True")
    parser.add_argument("--torch", required=True, help="router_rounding routes.pt")
    parser.add_argument("--rounding", help="the same job's routing.json: add the #310 gate result; exit 1 if it fails")
    parser.add_argument("--bias-fp32", action="store_true", help="the Levanter view kept router_bias in FP32")
    parser.add_argument("--router-dot-fp32", action="store_true", help="the measured Levanter router dot is FP32")
    args = parser.parse_args()
    report = compare_routes_cross_stack(args.levanter, args.torch)
    if args.rounding:
        overrides = {"router_bias": "float32"} if args.bias_fp32 else None
        report["gate_failures"] = route_gate(report, json.loads(read(args.rounding)), overrides, args.router_dot_fp32)
    print(json.dumps(report, indent=1))
    if args.rounding and report["gate_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
