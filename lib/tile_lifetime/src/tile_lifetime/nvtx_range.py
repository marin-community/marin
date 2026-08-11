# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed NVTX push/pop ranges with bounded diagnostics."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NVTX_NO_PUSH_POP_TRACKING = -2
_MAX_DIAGNOSTIC_CHARS = 2048
_INJECTION_ENVIRONMENT = ("NVTX_INJECTION64_PATH", "NVTX_INJECTION32_PATH")


@dataclass(frozen=True)
class NvtxLibraryIdentity:
    """Runtime identity of the requested and loaded NVTX library."""

    requested_path: str
    resolved_path: str
    dladdr_path: str


class _DlInfo(ctypes.Structure):
    _fields_ = (
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    )


def nvtx_return_classification(result: int) -> str:
    """Classify one signed NVTX push/pop result."""

    if result == NVTX_NO_PUSH_POP_TRACKING:
        return "no_push_pop_tracking"
    if result < 0:
        return "negative_error"
    if result == 0:
        return "outermost_level"
    return "nested_level"


def _symbol_library_path(symbol: Any) -> str:
    try:
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(_DlInfo))
        dladdr.restype = ctypes.c_int
        address = ctypes.cast(symbol, ctypes.c_void_p)
        info = _DlInfo()
        if not address.value or dladdr(address, ctypes.byref(info)) == 0 or info.dli_fname is None:
            return "unavailable"
        return info.dli_fname.decode(errors="replace")
    except (AttributeError, OSError, TypeError, ValueError):
        return "unavailable"


class NvtxRange:
    """Annotate one balanced thread-local NVTX push/pop range."""

    def __init__(
        self,
        name: str,
        library_path: Path,
        *,
        library_loader: Callable[[str], Any] | None = None,
        environment: Mapping[str, str] = os.environ,
        symbol_library_path: Callable[[Any], str] | None = None,
    ):
        self._name = name
        self._environment = environment
        requested = str(library_path)
        self._library = (library_loader or ctypes.CDLL)(requested)
        self._library.nvtxRangePushA.argtypes = (ctypes.c_char_p,)
        self._library.nvtxRangePushA.restype = ctypes.c_int
        self._library.nvtxRangePop.argtypes = ()
        self._library.nvtxRangePop.restype = ctypes.c_int
        self.library_identity = NvtxLibraryIdentity(
            requested_path=requested,
            resolved_path=str(library_path.resolve(strict=False)),
            dladdr_path=(symbol_library_path or _symbol_library_path)(self._library.nvtxRangePushA),
        )
        self.push_level: int | None = None
        self.pop_level: int | None = None

    def __enter__(self) -> NvtxRange:
        self.push_level = int(self._library.nvtxRangePushA(self._name.encode()))
        if self.push_level < 0:
            raise RuntimeError(self._failure_message("push", self.push_level))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.pop_level = int(self._library.nvtxRangePop())
        if self.pop_level < 0:
            raise RuntimeError(self._failure_message("pop", self.pop_level))
        if self.pop_level != self.push_level:
            raise RuntimeError(
                self._bounded_message(
                    f"NVTX range level mismatch for {self._name!r}: "
                    f"push_level={self.push_level} pop_level={self.pop_level}"
                )
            )

    def _failure_message(self, operation: str, result: int) -> str:
        injection = " ".join(
            f"{name}={'present' if name in self._environment else 'absent'}" for name in _INJECTION_ENVIRONMENT
        )
        identity = self.library_identity
        return self._bounded_message(
            f"NVTX {operation} rejected range {self._name!r}: "
            f"return_code={result} classification={nvtx_return_classification(result)} "
            f"requested_library={identity.requested_path!r} resolved_library={identity.resolved_path!r} "
            f"dladdr_library={identity.dladdr_path!r} {injection}"
        )

    @staticmethod
    def _bounded_message(message: str) -> str:
        if len(message) <= _MAX_DIAGNOSTIC_CHARS:
            return message
        return "NVTX diagnostic exceeded the closed 2048-character bound"
