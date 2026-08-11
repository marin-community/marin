# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
VALIDATOR = REPO_ROOT / "lib" / "iris" / "images" / "h100_evidence_nvtx_smoke.py"
NVTX_RANGE = REPO_ROOT / "lib" / "tile_lifetime" / "src" / "tile_lifetime" / "nvtx_range.py"


class _FakeFunction:
    def __init__(self, result: int):
        self._result = result
        self.calls: list[tuple[object, ...]] = []
        self.argtypes: tuple[object, ...] | None = None
        self.restype: object | None = None

    def __call__(self, *arguments: object) -> int:
        self.calls.append(arguments)
        return self._result


class _FakeNvtxLibrary:
    def __init__(self):
        self.nvtxRangePushA = _FakeFunction(-1)
        self.nvtxRangePop = _FakeFunction(-1)


def _validator_module():
    spec = importlib.util.spec_from_file_location("h100_evidence_nvtx_smoke", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_result(
    path: Path,
    *,
    push_return_code: int = 0,
    pop_return_code: int = 0,
    push_result_kind: str = "tracked_level",
    pop_result_kind: str = "tracked_level",
) -> None:
    path.write_text(
        json.dumps(
            {
                "dladdr_library": "/usr/local/cuda-13.2/lib64/libnvtx3interop.so.1",
                "name": "h100-evidence-nvtx-smoke",
                "pop_result_kind": pop_result_kind,
                "pop_return_code": pop_return_code,
                "push_result_kind": push_result_kind,
                "push_return_code": push_return_code,
                "requested_library": "/usr/local/cuda-13.2/lib64/libnvToolsExt.so",
                "resolved_library": "/usr/local/cuda-13.2/lib64/libnvtx3interop.so.1.1.0",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _write_sqlite(path: Path, rows: tuple[tuple[int, int, str], ...]) -> None:
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT)")
        database.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)", rows)


def test_nvtx_smoke_validator_accepts_one_balanced_exported_range(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    database = tmp_path / "trace.sqlite"
    _write_result(result)
    _write_sqlite(database, ((10, 20, "h100-evidence-nvtx-smoke"),))

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "validate",
            "--result",
            str(result),
            "--sqlite",
            str(database),
            "--name",
            "h100-evidence-nvtx-smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["push_result_kind"] == "tracked_level"


def test_nvtx_smoke_validator_accepts_exact_untracked_success_only_with_exported_range(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    database = tmp_path / "trace.sqlite"
    _write_result(
        result,
        push_return_code=-1,
        pop_return_code=-1,
        push_result_kind="untracked_success",
        pop_result_kind="untracked_success",
    )
    _write_sqlite(database, ((10, 20, "h100-evidence-nvtx-smoke"),))

    validated = _validator_module().validate_report(result, database, "h100-evidence-nvtx-smoke")

    assert validated["push_return_code"] == -1
    assert validated["pop_return_code"] == -1


def test_nvtx_smoke_emitter_pops_and_persists_exact_untracked_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _validator_module()
    library = _FakeNvtxLibrary()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    result_path = tmp_path / "result.json"

    validator.emit_range(
        NVTX_RANGE,
        Path("/usr/local/cuda-13.2/lib64/libnvToolsExt.so"),
        result_path,
        "h100-evidence-nvtx-smoke",
    )

    result = json.loads(result_path.read_bytes())
    assert result["push_return_code"] == -1
    assert result["push_result_kind"] == "untracked_success"
    assert result["pop_return_code"] == -1
    assert result["pop_result_kind"] == "untracked_success"
    assert library.nvtxRangePushA.calls == [(b"h100-evidence-nvtx-smoke",)]
    assert library.nvtxRangePop.calls == [()]
    assert library.nvtxRangePushA.restype is ctypes.c_int
    assert library.nvtxRangePop.restype is ctypes.c_int


@pytest.mark.parametrize(
    ("push", "pop", "push_kind", "pop_kind"),
    (
        (-2, -2, "error", "error"),
        (-3, -3, "error", "error"),
        (0, 1, "tracked_level", "tracked_level"),
        (-1, 0, "untracked_success", "tracked_level"),
        (0, -1, "tracked_level", "untracked_success"),
        (-1, -1, "tracked_level", "tracked_level"),
        (-1, -1, "unknown", "unknown"),
    ),
)
def test_nvtx_smoke_validator_rejects_untracked_error_or_unbalanced_result(
    tmp_path: Path, push: int, pop: int, push_kind: str, pop_kind: str
) -> None:
    result = tmp_path / "result.json"
    database = tmp_path / "trace.sqlite"
    _write_result(
        result,
        push_return_code=push,
        pop_return_code=pop,
        push_result_kind=push_kind,
        pop_result_kind=pop_kind,
    )
    _write_sqlite(database, ((10, 20, "h100-evidence-nvtx-smoke"),))

    with pytest.raises(ValueError, match="balanced accepted"):
        _validator_module().validate_report(result, database, "h100-evidence-nvtx-smoke")


@pytest.mark.parametrize(
    "rows",
    (
        (),
        ((10, 20, "lookalike-h100-evidence-nvtx-smoke"),),
        ((10, 20, "h100-evidence-nvtx-smoke"), (30, 40, "h100-evidence-nvtx-smoke")),
        ((20, 10, "h100-evidence-nvtx-smoke"),),
    ),
)
def test_nvtx_smoke_validator_rejects_missing_duplicate_or_malformed_trace_range(
    tmp_path: Path, rows: tuple[tuple[int, int, str], ...]
) -> None:
    result = tmp_path / "result.json"
    database = tmp_path / "trace.sqlite"
    _write_result(result)
    _write_sqlite(database, rows)

    with pytest.raises(ValueError, match=r"exactly one|malformed"):
        _validator_module().validate_report(result, database, "h100-evidence-nvtx-smoke")


def test_nvtx_smoke_validator_rejects_oversized_artifact_before_reading(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_bytes(b"{}" + b" " * (1 << 20))
    database = tmp_path / "trace.sqlite"
    _write_sqlite(database, ())

    with pytest.raises(ValueError, match="exceeds"):
        _validator_module().validate_report(result, database, "h100-evidence-nvtx-smoke")


@pytest.mark.parametrize(
    "rows",
    (
        (),
        ((10, 20, "lookalike-h100-evidence-nvtx-smoke"),),
        ((10, 20, "h100-evidence-nvtx-smoke"), (30, 40, "h100-evidence-nvtx-smoke")),
        ((20, 10, "h100-evidence-nvtx-smoke"),),
    ),
)
def test_nvtx_smoke_validator_requires_exact_exported_event_for_untracked_success(
    tmp_path: Path, rows: tuple[tuple[int, int, str], ...]
) -> None:
    result = tmp_path / "result.json"
    database = tmp_path / "trace.sqlite"
    _write_result(
        result,
        push_return_code=-1,
        pop_return_code=-1,
        push_result_kind="untracked_success",
        pop_result_kind="untracked_success",
    )
    _write_sqlite(database, rows)

    with pytest.raises(ValueError, match=r"exactly one|malformed"):
        _validator_module().validate_report(result, database, "h100-evidence-nvtx-smoke")


def test_nvtx_smoke_validator_rejects_missing_event_table(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    database = tmp_path / "trace.sqlite"
    _write_result(result)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated (value INTEGER)")

    with pytest.raises(ValueError, match="omits NVTX_EVENTS"):
        _validator_module().validate_report(result, database, "h100-evidence-nvtx-smoke")


def test_nvtx_smoke_validator_rejects_missing_export(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    _write_result(result)

    with pytest.raises(ValueError, match="must be a regular file"):
        _validator_module().validate_report(result, tmp_path / "missing.sqlite", "h100-evidence-nvtx-smoke")
