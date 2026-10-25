"""Architecture-neutral observations of MoE router forwards.

Observers run synchronously inside the router forward. A callback that retains
an observation beyond the callback must detach or clone its tensors to avoid
retaining the autograd graph.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RouterObservation:
    """One router decision, expressed independently of the model architecture."""

    router: nn.Module
    router_inputs: torch.Tensor
    selection_logits: torch.Tensor
    selection_log_probs: torch.Tensor
    natural_selected_experts: torch.Tensor
    selected_experts: torch.Tensor
    combine_weights: torch.Tensor


RouterObserver = Callable[[RouterObservation], None]
_active_observer: ContextVar[RouterObserver | None] = ContextVar("router_observer", default=None)


class NativeRouterObserverEmitter:
    """Marker base for routers that emit observations without an adapter hook."""


def emit_router_forward(
    *,
    router: nn.Module,
    router_inputs: torch.Tensor,
    selection_logits: torch.Tensor,
    natural_selected_experts: torch.Tensor,
    selected_experts: torch.Tensor,
    combine_weights: torch.Tensor,
) -> None:
    """Emit an observation when the current execution context has an observer."""

    observer = _active_observer.get()
    if observer is None:
        return
    observer(
        RouterObservation(
            router=router,
            router_inputs=router_inputs,
            selection_logits=selection_logits,
            selection_log_probs=F.log_softmax(selection_logits.float(), dim=-1),
            natural_selected_experts=natural_selected_experts,
            selected_experts=selected_experts,
            combine_weights=combine_weights,
        )
    )
