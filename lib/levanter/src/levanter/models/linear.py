# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import equinox as eqx


class LinearLikeModule(eqx.Module):
    """
    Marker base class for linear-like modules.

    Subclasses should expose `weight` and optional `bias` leaves that optimizers
    and adapters can target without depending on concrete layer implementations.
    """


def has_linear_like_marker(node: Any) -> bool:
    """Return True if an object inherits Levanter's linear-like marker base class."""
    return isinstance(node, LinearLikeModule)
