# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Proxy package: re-exports the native Rust extension when available,
# otherwise provides stubs that raise ImportError on use. This allows
# `import dupekit` to succeed without building the Rust extension, deferring
# the error to when native functionality is actually called.

import sys
import types

__path__: list[str]

try:
    from dupekit._native import *  # noqa: F403
    from dupekit._native import (
        Bloom,
        Document,
        HashAlgorithm,
        Transformation,
    )
except ImportError:
    _INSTALL_MSG = (
        "dupekit native extension is not installed. "
        "Build it with: cd lib/dupekit && uv sync && maturin develop --release"
    )

    class _StubModule(types.ModuleType):
        """Module replacement that raises ImportError on any attribute access."""

        def __getattr__(self, name: str):
            if name.startswith("_"):
                raise AttributeError(name)
            raise ImportError(_INSTALL_MSG)

    _stub = _StubModule(__name__)
    _stub.__file__ = __file__
    _stub.__path__ = __path__
    _stub.__package__ = __package__
    sys.modules[__name__] = _stub
