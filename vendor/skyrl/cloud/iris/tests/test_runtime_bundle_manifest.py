"""Regression tests for the runtime bundle file manifest."""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from cloud.iris.runtime_bundle import BUNDLE_FILE_MANIFEST, read_manifest_paths  # noqa: E402


def test_every_runtime_module_is_in_the_runtime_bundle() -> None:
    """Ship both Python packages imported by the task runtime."""
    manifest = set(read_manifest_paths(_REPOSITORY_ROOT))
    on_disk = {
        path.relative_to(_REPOSITORY_ROOT).as_posix()
        for package in ("cloud/iris", "marinskyrl")
        for path in (_REPOSITORY_ROOT / package).glob("*.py")
    }
    missing = on_disk - manifest
    assert not missing, (
        f"runtime modules not shipped to Iris tasks: {sorted(missing)}. "
        f"Add them to {BUNDLE_FILE_MANIFEST} or add an explicit exclusion in this test."
    )
