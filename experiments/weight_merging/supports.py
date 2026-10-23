# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic whole-expert support for the Base-preserving campaign."""

import hashlib
import math
import re
from collections import defaultdict

from marin.merging.constrained import TensorSupport

EXPERT_NAME = re.compile(r"model.layers.(\d+).mlp.experts.(gate_proj|up_proj|down_proj).weight")


def routed_support(headers: dict, cap: float, seed: int) -> tuple[TensorSupport, ...]:
    tensors = {name: spec for header in headers.values() for name, spec in header.items() if name != "__metadata__"}
    total = sum(math.prod(spec["shape"]) for spec in tensors.values())
    layers = defaultdict(list)
    for name in tensors:
        match = EXPERT_NAME.fullmatch(name)
        if match:
            layers[int(match[1])].append(name)
    orders = {}
    costs = {}
    for layer, names in sorted(layers.items()):
        if len(names) != 3:
            raise ValueError(f"Expected three expert projection banks in layer {layer}")
        counts = {tensors[name]["shape"][0] for name in names}
        if len(counts) != 1:
            raise ValueError("Expert bank dimensions differ")
        count = counts.pop()
        orders[layer] = sorted(range(count), key=lambda row: hashlib.sha256(f"{seed}:{layer}:{row}".encode()).digest())
        costs[layer] = sum(math.prod(tensors[name]["shape"][1:]) for name in names)
    selected = defaultdict(list)
    used = 0
    for rank in range(max(len(order) for order in orders.values())):
        for layer, order in orders.items():
            if rank >= len(order):
                continue
            if used + costs[layer] > math.floor(cap * total):
                return tuple(TensorSupport(name, tuple(rows)) for name, rows in sorted(selected.items()))
            for name in layers[layer]:
                selected[name].append(order[rank])
            used += costs[layer]
    return tuple(TensorSupport(name, tuple(rows)) for name, rows in sorted(selected.items()))


def structured_support(headers: dict, placement: str, cap: float, seed: int) -> tuple[TensorSupport, ...]:
    """Select complete MLP layers or prioritize shared/attention weights, within a scalar budget."""
    tensors = {name: spec for header in headers.values() for name, spec in header.items() if name != "__metadata__"}
    total = sum(math.prod(spec["shape"]) for spec in tensors.values())
    limit = math.floor(cap * total)
    if placement in ("early", "late"):
        layers = defaultdict(list)
        for name in tensors:
            match = re.fullmatch(r"model.layers.(\d+).(?:mlp.experts|shared_expert).+", name)
            if match:
                layers[int(match[1])].append(name)
        selected = []
        used = 0
        for layer in sorted(layers, reverse=placement == "late"):
            names = layers[layer]
            cost = sum(math.prod(tensors[name]["shape"]) for name in names)
            if used + cost > limit:
                break
            selected.extend(TensorSupport(name, None) for name in names)
            used += cost
        return tuple(sorted(selected, key=lambda item: item.name))
    if placement not in ("shared", "attention"):
        raise ValueError(f"Unknown placement: {placement}")
    marker = ".shared_expert." if placement == "shared" else ".self_attn."
    priority = [name for name in tensors if marker in name and len(tensors[name]["shape"]) == 2]
    used = sum(math.prod(tensors[name]["shape"]) for name in priority)
    if used > limit:
        raise ValueError("Prioritized component exceeds support budget")
    routed = routed_support(headers, (limit - used) / total, seed)
    return tuple(sorted([*(TensorSupport(name, None) for name in priority), *routed], key=lambda item: item.name))


def activation_allocation(headers: dict, counts: dict, cap: float) -> list[dict]:
    """Allocate whole experts between domains using normalized routing enrichment."""
    tensors = {name: spec for header in headers.values() for name, spec in header.items() if name != "__metadata__"}
    total = sum(math.prod(spec["shape"]) for spec in tensors.values())
    pools = [[], []]
    for layer, (numeric, tools) in enumerate(zip(counts["numeric"], counts["tools"], strict=True)):
        normalizers = (sum(numeric), sum(tools))
        if min(normalizers) <= 0 or len(numeric) != len(tools):
            raise ValueError("Each layer/domain needs routing observations")
        names = [
            f"model.layers.{layer}.mlp.experts.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        ]
        cost = sum(math.prod(tensors[name]["shape"][1:]) for name in names)
        for expert, (a, b) in enumerate(zip(numeric, tools, strict=True)):
            score = a / normalizers[0] - b / normalizers[1]
            owner = 0 if score >= 0 else 1
            pools[owner].append(
                {"layer": layer, "expert": expert, "donor": owner, "score": abs(score), "elements": cost}
            )
    for pool in pools:
        pool.sort(key=lambda item: (-item["score"], item["layer"], item["expert"]))
    used = [0, 0]
    positions = [0, 0]
    stream = []
    active = [True, True]
    limit = math.floor(total * cap / 2)
    while any(active):
        for donor in (0, 1):
            if not active[donor]:
                continue
            if positions[donor] == len(pools[donor]):
                raise ValueError("Insufficient domain-specific experts to fill the allocation")
            item = pools[donor][positions[donor]]
            if used[donor] + item["elements"] > limit:
                active[donor] = False
                continue
            stream.append(item)
            positions[donor] += 1
            used[donor] += item["elements"]
    return stream


def allocation_prefix(stream: list[dict], total: int, cap: float) -> list[dict]:
    used = [0, 0]
    result = []
    for item in stream:
        donor = item["donor"]
        if used[donor] + item["elements"] > math.floor(total * cap / 2):
            break
        result.append(item)
        used[donor] += item["elements"]
    return result


def matched_random_allocation(stream: list[dict], expert_count: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for item in stream:
        groups[item["layer"]].append(item)
    result = []
    for layer, items in sorted(groups.items()):
        order = sorted(range(expert_count), key=lambda row: hashlib.sha256(f"{seed}:{layer}:{row}".encode()).digest())
        if len(items) > len(order):
            raise ValueError("Cannot sample disjoint experts without replacement")
        result.extend(
            {**item, "expert": row, "score": None} for item, row in zip(items, order[: len(items)], strict=True)
        )
    return result
