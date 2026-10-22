# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug component groups and the keys every weight metric is reported under.

Names are the HF export names written by ``grugmoe_inference_state_dict``
(``experiments/grug/moe/model.py``) and the Levanter bridge
(``lib/levanter/src/levanter/models/snowball.py``). Each name maps to exactly
one :class:`Component` by full-name match; a name no pattern matches raises,
so a new tensor cannot silently fall out of every metric and every hybrid.

Two placement decisions:

- ``mlp.router.bias`` is the top-k selection bias, a separate component from
  the router weights ``mlp.router.weight``. The design treats them as different
  routing factors (DESIGN.md, "Router treatment" and H5).
- ``self_attn.attn_gate.weight`` is the per-head sigmoid output gate of
  attention. It belongs to :attr:`Component.NORMS_AND_GATES` with the gated
  norms, per the design's "norms and gates" component. #9920 counted it under
  attention ("attention q/k/v/o and attention gates"), so attention numbers here
  are not directly comparable with that table; the gate is a few thousand
  weights per layer against tens of millions in q/k/v/o.

Grug stacks routed experts on dim 0 of ``mlp.experts.{gate,up,down}_proj.weight``
(``[E, out, in]``), so the expert index comes from the tensor, never the name.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class Component(StrEnum):
    ATTENTION = "attention"
    ROUTED_EXPERTS = "routed_experts"
    SHARED_EXPERT = "shared_expert"
    ROUTER = "router"
    SELECTION_BIAS = "selection_bias"
    EMBEDDINGS = "embeddings"
    LM_HEAD = "lm_head"
    NORMS_AND_GATES = "norms_and_gates"


_LAYER = r"model\.layers\.(\d+)\."
_COMPONENT_PATTERNS: tuple[tuple[Component, re.Pattern[str]], ...] = tuple(
    (component, re.compile(pattern))
    for component, pattern in (
        (Component.ATTENTION, _LAYER + r"self_attn\.[qkvo]_proj\.weight"),
        (Component.ROUTED_EXPERTS, _LAYER + r"mlp\.experts\.(gate|up|down)_proj\.weight"),
        (Component.SHARED_EXPERT, _LAYER + r"shared_expert\.(gate|up|down)_proj\.weight"),
        (Component.ROUTER, _LAYER + r"mlp\.router\.weight"),
        (Component.SELECTION_BIAS, _LAYER + r"mlp\.router\.bias"),
        (Component.EMBEDDINGS, r"model\.embed_tokens\.weight"),
        (Component.LM_HEAD, r"lm_head\.weight"),
        (Component.NORMS_AND_GATES, _LAYER + r"(input_layernorm|post_attention_layernorm)\.weight"),
        (Component.NORMS_AND_GATES, _LAYER + r"(attn|mlp)_gated_norm\.(up|down)_proj\.weight"),
        (Component.NORMS_AND_GATES, _LAYER + r"self_attn\.attn_gate\.weight"),
        (Component.NORMS_AND_GATES, r"model\.(embed_norm|norm)\.weight"),
        (Component.NORMS_AND_GATES, r"model\.(embed|final)_gated_norm\.(up|down)_proj\.weight"),
    )
)

# Graft groups (WORK_BRIEF WP9): the analysis components with embeddings and the
# LM head grafted together. Seven groups give 7 leave-one-in plus 7
# leave-one-out hybrids per checkpoint, inside the design's 14-16.
GRAFT_GROUPS: dict[str, frozenset[Component]] = {
    "attention": frozenset({Component.ATTENTION}),
    "routed_experts": frozenset({Component.ROUTED_EXPERTS}),
    "shared_expert": frozenset({Component.SHARED_EXPERT}),
    "router": frozenset({Component.ROUTER}),
    "selection_bias": frozenset({Component.SELECTION_BIAS}),
    "embeddings_lm_head": frozenset({Component.EMBEDDINGS, Component.LM_HEAD}),
    "norms_and_gates": frozenset({Component.NORMS_AND_GATES}),
}


def component_of(name: str) -> Component:
    """The component of a Grug tensor name; raises on a name no group claims."""
    for component, pattern in _COMPONENT_PATTERNS:
        if pattern.fullmatch(name):
            return component
    raise ValueError(f"{name!r} matches no Grug component group; classify it in components.py")


def layer_of(name: str) -> int:
    match = re.match(_LAYER, name)
    if match is None:
        raise ValueError(f"{name!r} carries no layer index")
    return int(match.group(1))


def hybrid_plan() -> dict[str, frozenset[Component]]:
    """Leave-one-in (``only_<g>``) and leave-one-out (``without_<g>``) hybrids."""
    every = frozenset().union(*GRAFT_GROUPS.values())
    plan: dict[str, frozenset[Component]] = {}
    for group, members in GRAFT_GROUPS.items():
        plan[f"only_{group}"] = members
        plan[f"without_{group}"] = every - members
    return plan


ExpertKey = tuple[int, int]
"""(layer, expert) for one routed expert."""


@dataclass(frozen=True)
class Block:
    """One unit a metric consumes: a whole tensor, or one expert of a stacked tensor."""

    name: str
    component: Component
    expert: ExpertKey | None
    array: np.ndarray


@dataclass
class Keyed[T]:
    """A metric over all weights, per component, and per (layer, expert)."""

    overall: T
    components: dict[Component, T] = field(default_factory=dict)
    experts: dict[ExpertKey, T] = field(default_factory=dict)

    def map[U](self, fn: Callable[[T], U]) -> "Keyed[U]":
        return Keyed(
            overall=fn(self.overall),
            components={key: fn(value) for key, value in sorted(self.components.items())},
            experts={key: fn(value) for key, value in sorted(self.experts.items())},
        )

    def targets(self, block: Block, factory: Callable[[], T]) -> list[T]:
        """The accumulators one block updates: overall, its component, its expert."""
        if block.component not in self.components:
            self.components[block.component] = factory()
        found = [self.overall, self.components[block.component]]
        if block.expert is not None:
            if block.expert not in self.experts:
                self.experts[block.expert] = factory()
            found.append(self.experts[block.expert])
        return found
