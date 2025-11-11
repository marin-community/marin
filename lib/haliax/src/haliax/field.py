# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Any, Callable

import equinox as eqx


def field(
    *,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
    axis_names: tuple[str, ...] | None = None,
    **kwargs,
):
    """Wrapper around :func:`equinox.field` with optional ``axis_names`` metadata.

    Args:
        converter: Optional function applied to the value during dataclass initialisation.
        static: Whether the field is static in the PyTree.
        axis_names: Optional axis names associated with array fields. Cannot be
            specified together with ``static=True``.
        **kwargs: Additional keyword arguments forwarded to :func:`dataclasses.field`.

    Returns:
        A dataclasses field configured like :func:`equinox.field` with additional
        ``axis_names`` metadata.
    """
    if static and axis_names is not None:
        raise ValueError("axis_names cannot be specified together with static=True")

    metadata = dict(kwargs.pop("metadata", {}))
    metadata["axis_names"] = axis_names

    field_kwargs = {}
    if converter is not None:
        field_kwargs["converter"] = converter

    return eqx.field(static=static, metadata=metadata, **field_kwargs, **kwargs)
