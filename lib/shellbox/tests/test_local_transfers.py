# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The LOCAL transfer path preserves the declared file metadata graders rely on."""

import os
import stat
import subprocess
from pathlib import Path

from shellbox.backends.local.machine import _archive

MTIME = 1700000000.123456


def _round_trip(tmp_path: Path, source: Path, name: str) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    subprocess.run(["tar", "-xof", "-", "-C", str(out)], input=_archive(source, name), check=True)
    return out / name


def test_archive_round_trip_preserves_mode(tmp_path: Path):
    source = tmp_path / "grade.py"
    source.write_text("print('graded')\n")
    os.chmod(source, 0o640)
    restored = _round_trip(tmp_path, source, "grade.py")
    assert stat.S_IMODE(restored.stat().st_mode) == 0o640


def test_archive_round_trip_preserves_subsecond_mtime(tmp_path: Path):
    source = tmp_path / "answer.key"
    source.write_text("42\n")
    os.utime(source, (MTIME, MTIME))
    restored = _round_trip(tmp_path, source, "answer.key")
    # pax archives carry sub-second precision; the restored value must not be
    # truncated to whole seconds the way a plain ustar archive would be.
    assert abs(restored.stat().st_mtime - MTIME) < 1e-3
    assert restored.stat().st_mtime_ns % 1_000_000_000 != 0


def test_archive_round_trip_preserves_directory_contents(tmp_path: Path):
    tree = tmp_path / "grader-files"
    tree.mkdir()
    (tree / "rubric.txt").write_text("criteria\n")
    os.chmod(tree / "rubric.txt", 0o600)
    os.utime(tree / "rubric.txt", (MTIME, MTIME))
    restored_root = _round_trip(tmp_path, tree, ".")
    restored = restored_root / "rubric.txt"
    assert restored.read_text() == "criteria\n"
    assert stat.S_IMODE(restored.stat().st_mode) == 0o600
    assert abs(restored.stat().st_mtime - MTIME) < 1e-3


def test_archive_round_trip_preserves_nanosecond_mtime(tmp_path: Path):
    source = tmp_path / "marker"
    source.write_bytes(b"mimo-local-native-qualification\n")
    source.chmod(0o640)
    mtime_ns = 1700000000123456789
    os.utime(source, ns=(mtime_ns, mtime_ns))
    restored = _round_trip(tmp_path, source, "marker")
    assert stat.S_IMODE(restored.stat().st_mode) == 0o640
    assert restored.stat().st_mtime_ns == mtime_ns
