"""Keep zero-dropout Grug scoring on the learner's attention forward path."""

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


@contextmanager
def grug_scoring_attention_context(core_attention: nn.Module, attention_dropout: float) -> Iterator[None]:
    """Use training-capable attention for zero-dropout scoring and restore its modes."""
    if attention_dropout != 0.0 or core_attention.training:
        yield
        return
    modes = [(module, module.training) for module in core_attention.modules()]
    try:
        core_attention.train(True)
        yield
    finally:
        for module, was_training in modes:
            module.training = was_training
