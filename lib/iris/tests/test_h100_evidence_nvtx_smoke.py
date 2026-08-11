# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
VALIDATOR = REPO_ROOT / "lib" / "iris" / "images" / "h100_evidence_nvtx_smoke.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location("h100_evidence_nvtx_smoke", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_result(path: Path, *, push_level: int = 0, pop_level: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "dladdr_library": "/usr/local/cuda-13.2/lib64/libnvtx3interop.so.1",
                "name": "h100-evidence-nvtx-smoke",
                "pop_level": pop_level,
                "push_level": push_level,
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
    assert json.loads(completed.stdout)["push_level"] == 0


@pytest.mark.parametrize(
    ("push_level", "pop_level"),
    ((-2, -2), (-1, -1), (0, 1)),
)
def test_nvtx_smoke_validator_rejects_untracked_error_or_unbalanced_result(
    tmp_path: Path, push_level: int, pop_level: int
) -> None:
    result = tmp_path / "result.json"
    database = tmp_path / "trace.sqlite"
    _write_result(result, push_level=push_level, pop_level=pop_level)
    _write_sqlite(database, ((10, 20, "h100-evidence-nvtx-smoke"),))

    with pytest.raises(ValueError, match="balanced nonnegative"):
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
