# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import warnings as _warnings
from typing import TYPE_CHECKING


# Re-export of the rotary embedding utilities that were moved to `levanter.layers.rotary`.
# Accessing them through this module is still supported for backward-compatibility but
# will raise a `DeprecationWarning`. Please update your imports to
#   ``from levanter.layers.rotary import ...``
# in the future.

if TYPE_CHECKING:  # pragma: no cover -- typing helpers
    # Re-import with their public names so static analysis tools see them.
    from levanter.layers.rotary import (
        RotaryEmbeddings,
        RotaryEmbeddingsConfig,
        DefaultRotaryEmbeddingsConfig,
        Llama3RotaryEmbeddingsConfig,
        YarnRotaryEmbeddingsConfig,
        rotary_pos_emb,
    )

from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig as _DefaultRotaryEmbeddingsConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig as _Llama3RotaryEmbeddingsConfig
from levanter.layers.rotary import RotaryEmbeddings as _RotaryEmbeddings
from levanter.layers.rotary import RotaryEmbeddingsConfig as _RotaryEmbeddingsConfig
from levanter.layers.rotary import YarnRotaryEmbeddingsConfig as _YarnRotaryEmbeddingsConfig
from levanter.layers.rotary import rotary_pos_emb as _rotary_pos_emb


__all__ = [
    "RotaryEmbeddings",
    "RotaryEmbeddingsConfig",
    "DefaultRotaryEmbeddingsConfig",
    "Llama3RotaryEmbeddingsConfig",
    "YarnRotaryEmbeddingsConfig",
    "rotary_pos_emb",
]

# Internal mapping from public names to their real objects in the new location.
_mapping = {
    "RotaryEmbeddings": _RotaryEmbeddings,
    "RotaryEmbeddingsConfig": _RotaryEmbeddingsConfig,
    "DefaultRotaryEmbeddingsConfig": _DefaultRotaryEmbeddingsConfig,
    "Llama3RotaryEmbeddingsConfig": _Llama3RotaryEmbeddingsConfig,
    "YarnRotaryEmbeddingsConfig": _YarnRotaryEmbeddingsConfig,
    "rotary_pos_emb": _rotary_pos_emb,
}

_deprecation_message = (
    "Importing {name} from 'levanter.models.rotary' is deprecated and will be removed "
    "in a future release. Please import it from 'levanter.layers.rotary' instead."
)


def __getattr__(name):  # noqa: D401   # simple function; flake8 D401 false-positive
    """Dynamically fetch attributes while emitting a deprecation warning.

    This hook (PEP 562) lets us lazily forward attribute access so we can emit the
    warning only when the symbol is actually used, rather than at import time.
    """
    if name in _mapping:
        _warnings.warn(_deprecation_message.format(name=name), DeprecationWarning, stacklevel=2)
        return _mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# We purposely do not eagerly inject the forwarded attributes into the module namespace.
# This way, the `DeprecationWarning` is emitted the first time each symbol is actually
# accessed, keeping noise to a minimum while still guiding users towards the new import
# location.
