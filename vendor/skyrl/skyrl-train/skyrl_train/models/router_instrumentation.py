"""Architecture-neutral observations of MoE router forwards.

Observers run synchronously inside the router forward. A callback that retains
an observation beyond the callback must detach or clone its tensors to avoid
retaining the autograd graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import TracebackType

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle


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


@dataclass
class RouterInstrumentation:
    """Own the adapter hooks installed for one model."""

    router_count: int
    _handles: list[RemovableHandle] = field(default_factory=list)

    def close(self) -> None:
        """Remove all adapter hooks owned by this instance."""

        while self._handles:
            self._handles.pop().remove()

    def __enter__(self) -> RouterInstrumentation:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@contextmanager
def observe_router_forwards(observer: RouterObserver) -> Iterator[None]:
    """Send router observations to ``observer`` for the current execution context."""

    token = _active_observer.set(observer)
    try:
        yield
    finally:
        _active_observer.reset(token)


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


def _observe_qwen_router(
    router: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    if _active_observer.get() is None:
        return
    if len(inputs) != 1 or not isinstance(output, tuple) or len(output) != 3:
        raise RuntimeError("instrumented Qwen router returned an unsupported interface")
    router_logits, combine_weights, selected_experts = output
    router_inputs = inputs[0].reshape(-1, inputs[0].shape[-1])
    emit_router_forward(
        router=router,
        router_inputs=router_inputs,
        selection_logits=router_logits,
        natural_selected_experts=selected_experts,
        selected_experts=selected_experts,
        combine_weights=combine_weights,
    )


def _qwen_router_types() -> tuple[type[nn.Module], ...]:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextTopKRouter

    return Qwen3MoeTopKRouter, Qwen3NextTopKRouter, Qwen3_5MoeTopKRouter


def instrument_moe_routers(model: nn.Module) -> RouterInstrumentation:
    """Enable observations for every supported MoE router under ``model``.

    Native MarinSkyRL routers emit observations directly. Hugging Face Qwen
    routers need forward hooks. The returned object owns those hooks and must
    remain alive until instrumentation is no longer needed. Use it as a context
    manager or call :meth:`RouterInstrumentation.close` to remove the hooks.
    """

    router_count = 0
    handles = []
    qwen_router_types = _qwen_router_types()
    for module in model.modules():
        if isinstance(module, NativeRouterObserverEmitter):
            router_count += 1
            continue
        if not isinstance(module, qwen_router_types):
            continue
        router_count += 1
        handles.append(module.register_forward_hook(_observe_qwen_router))
    return RouterInstrumentation(router_count=router_count, _handles=handles)
