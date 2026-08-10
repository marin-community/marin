# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conservative JAX donation matching for mixed device/host training state.

JAX currently discards every memory-kind tag when any input or output has an
unspecified kind. A metric with inferred placement can therefore let a donated
pinned-host optimizer buffer alias a device-resident model output. Keep known
memory kinds matchable and give every unknown leaf a distinct marker so it is
not donated in a mixed-memory computation.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

_PATCH_MARKER = "_marin_mixed_memory_donation_safe"


def _mixed_memory_safe_alias_setup(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(
        input_output_aliases,
        avals_in,
        avals_out,
        donated_args,
        arg_memory_kinds: Sequence[str | None] | None,
        result_memory_kinds: Sequence[str | None] | None,
        in_layouts,
        out_layouts,
        result_shardings,
    ):
        if arg_memory_kinds is not None and result_memory_kinds is not None:
            arg_memory_kinds = list(arg_memory_kinds)
            result_memory_kinds = list(result_memory_kinds)
            has_known_kind = any(kind is not None for kind in arg_memory_kinds) or any(
                kind is not None for kind in result_memory_kinds
            )
            if has_known_kind:
                # The private JAX helper only uses these values as dictionary keys.
                # Per-leaf objects cannot match each other, so unknown placement is
                # conservatively non-donatable while known device/pinned_host kinds
                # continue to match normally.
                arg_memory_kinds = [kind if kind is not None else object() for kind in arg_memory_kinds]
                result_memory_kinds = [kind if kind is not None else object() for kind in result_memory_kinds]

        return original(
            input_output_aliases,
            avals_in,
            avals_out,
            donated_args,
            arg_memory_kinds,
            result_memory_kinds,
            in_layouts,
            out_layouts,
            result_shardings,
        )

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def install_mixed_memory_donation_safety() -> None:
    """Install the process-local JAX compatibility shim once."""

    from jax._src.interpreters import mlir  # noqa: PLC0415

    current = mlir._set_up_aliases
    if getattr(current, _PATCH_MARKER, False):
        return
    mlir._set_up_aliases = _mixed_memory_safe_alias_setup(current)


__all__ = ["install_mixed_memory_donation_safety"]
