# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import tile_lifetime.cuda_toolchain as cuda_toolchain
import tile_lifetime.nvtx_range as nvtx_range
from tile_lifetime.nvtx_range import NvtxRange, NvtxRangeResultKind

_RUNNER_PATH = Path(__file__).parents[1] / "benchmarks" / "h100_contract_map_backend_runner.py"


def _current_runner() -> ModuleType:
    module_name = "test_nvtx_range_current_h100_contract_map_backend_runner"
    spec = importlib.util.spec_from_file_location(module_name, _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeFunction:
    def __init__(self, *results: int):
        self._results = list(results)
        self.calls: list[tuple[object, ...]] = []
        self.argtypes: tuple[object, ...] | None = None
        self.restype: object | None = None

    def __call__(self, *arguments: object) -> int:
        self.calls.append(arguments)
        return self._results.pop(0)


class _FakeLibrary:
    def __init__(self, *, pushes: tuple[int, ...], pops: tuple[int, ...]):
        self.nvtxRangePushA = _FakeFunction(*pushes)
        self.nvtxRangePop = _FakeFunction(*pops)


def _range(
    library: _FakeLibrary,
    *,
    name: str = "contract_map.steady.0.ordinary_xla",
    environment: dict[str, str] | None = None,
) -> NvtxRange:
    return NvtxRange(
        name,
        Path("/toolkit/libnvToolsExt.so"),
        library_loader=lambda path: library,
        environment=environment or {},
        symbol_library_path=lambda symbol: "/toolkit/libnvtx3interop.so.1",
    )


@pytest.mark.parametrize("level", (0, 3))
def test_nvtx_range_accepts_balanced_outermost_and_nested_levels(level: int) -> None:
    library = _FakeLibrary(pushes=(level,), pops=(level,))

    with _range(library) as active:
        assert active.push_result is not None
        assert active.push_result.return_code == level
        assert active.push_result.kind is NvtxRangeResultKind.TRACKED_LEVEL
        assert active.pop_result is None

    assert active.pop_result is not None
    assert active.pop_result.return_code == level
    assert active.pop_result.kind is NvtxRangeResultKind.TRACKED_LEVEL
    assert library.nvtxRangePushA.calls == [(b"contract_map.steady.0.ordinary_xla",)]
    assert library.nvtxRangePop.calls == [()]
    assert library.nvtxRangePushA.argtypes == (ctypes.c_char_p,)
    assert library.nvtxRangePushA.restype is ctypes.c_int
    assert library.nvtxRangePop.argtypes == ()
    assert library.nvtxRangePop.restype is ctypes.c_int


@pytest.mark.parametrize(
    ("result", "classification"),
    ((-2, "no_push_pop_tracking"), (-3, "negative_error")),
)
def test_nvtx_range_reports_bounded_signed_push_failure_without_environment_values(
    result: int, classification: str
) -> None:
    library = _FakeLibrary(pushes=(result,), pops=())
    secret = "must-not-appear"

    with pytest.raises(RuntimeError) as raised:
        with _range(library, environment={"NVTX_INJECTION64_PATH": secret}):
            pass

    message = str(raised.value)
    assert f"return_code={result}" in message
    assert f"classification={classification}" in message
    assert "requested_library='/toolkit/libnvToolsExt.so'" in message
    assert "dladdr_library='/toolkit/libnvtx3interop.so.1'" in message
    assert "NVTX_INJECTION64_PATH=present" in message
    assert "NVTX_INJECTION32_PATH=absent" in message
    assert secret not in message
    assert len(message) <= 2048
    assert library.nvtxRangePop.calls == []


@pytest.mark.parametrize("result", (-3, -2))
def test_nvtx_range_rejects_negative_pop_after_successful_push(result: int) -> None:
    library = _FakeLibrary(pushes=(0,), pops=(result,))

    with pytest.raises(RuntimeError, match=rf"NVTX pop rejected.*return_code={result}"):
        with _range(library):
            pass


def test_nvtx_range_rejects_unbalanced_nonnegative_pop() -> None:
    library = _FakeLibrary(pushes=(2,), pops=(1,))

    with pytest.raises(RuntimeError, match=r"push_return_code=2.*pop_return_code=1"):
        with _range(library):
            pass


def test_nvtx_range_accepts_exact_untracked_success_and_still_pops() -> None:
    library = _FakeLibrary(pushes=(-1,), pops=(-1,))

    with _range(library) as active:
        assert active.push_result is not None
        assert active.push_result.return_code == -1
        assert active.push_result.kind is NvtxRangeResultKind.UNTRACKED_SUCCESS

    assert active.pop_result is not None
    assert active.pop_result.return_code == -1
    assert active.pop_result.kind is NvtxRangeResultKind.UNTRACKED_SUCCESS
    assert library.nvtxRangePop.calls == [()]


@pytest.mark.parametrize(
    ("push", "pop"),
    ((-1, 0), (-1, -2), (-1, -3), (0, -1), (2, -1)),
)
def test_nvtx_range_rejects_untracked_tracked_or_error_mismatch(push: int, pop: int) -> None:
    library = _FakeLibrary(pushes=(push,), pops=(pop,))

    with pytest.raises(RuntimeError, match=r"NVTX (range result mismatch|pop rejected)"):
        with _range(library):
            pass


def test_nvtx_range_pops_when_the_annotated_body_raises() -> None:
    library = _FakeLibrary(pushes=(0,), pops=(0,))

    with pytest.raises(LookupError, match="body failed"):
        with _range(library):
            raise LookupError("body failed")

    assert library.nvtxRangePop.calls == [()]


def test_nvtx_range_untracked_success_pops_and_preserves_body_exception() -> None:
    library = _FakeLibrary(pushes=(-1,), pops=(-1,))

    with pytest.raises(LookupError, match="body failed"):
        with _range(library):
            raise LookupError("body failed")

    assert library.nvtxRangePop.calls == [()]


def test_nvtx_range_replaces_oversized_diagnostic() -> None:
    library = _FakeLibrary(pushes=(-2,), pops=())

    with pytest.raises(RuntimeError, match=r"^NVTX diagnostic exceeded the closed 2048-character bound$"):
        with _range(library, name="x" * 4096):
            pass


def test_runner_nvtx_wrapper_uses_the_toolkit_library_and_exact_range_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _current_runner()
    library = _FakeLibrary(pushes=(0,), pops=(0,))
    expected_library = Path("/pinned/cuda/lib64/libnvToolsExt.so")
    monkeypatch.setattr(cuda_toolchain, "cuda_toolkit_shared_library", lambda nvcc, name: expected_library)
    monkeypatch.setattr(nvtx_range.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(nvtx_range, "_symbol_library_path", lambda symbol: "/pinned/libnvtx3interop.so.1")

    with runner._NvtxRange("production-boundary", Path("/pinned/cuda/bin/nvcc")) as active:
        assert active.library_identity.requested_path == str(expected_library)

    assert active.push_result is not None
    assert active.pop_result is not None
    assert active.push_result.return_code == 0
    assert active.pop_result.return_code == 0


def test_runner_nvtx_wrapper_accepts_only_exact_untracked_success_at_production_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _current_runner()
    library = _FakeLibrary(pushes=(-1,), pops=(-1,))
    expected_library = Path("/pinned/cuda/lib64/libnvToolsExt.so")
    monkeypatch.setattr(cuda_toolchain, "cuda_toolkit_shared_library", lambda nvcc, name: expected_library)
    monkeypatch.setattr(nvtx_range.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(nvtx_range, "_symbol_library_path", lambda symbol: "/pinned/libnvtx3interop.so.1")

    with runner._NvtxRange("production-boundary", Path("/pinned/cuda/bin/nvcc")) as active:
        assert active.push_result is not None
        assert active.push_result.kind is NvtxRangeResultKind.UNTRACKED_SUCCESS

    assert active.pop_result is not None
    assert active.pop_result.kind is NvtxRangeResultKind.UNTRACKED_SUCCESS
    assert library.nvtxRangePop.calls == [()]
