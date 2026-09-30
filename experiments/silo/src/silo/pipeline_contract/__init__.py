# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The capability pipeline's own contract code, vendored byte for byte.

The provider is judged by the consumer's real code -- its error classifier, its
readiness and deletion waits, its recipe identity check, its cgroup telemetry
parser -- not by a restatement of them. A drift on either side then shows up as
a failing test or a failing acceptance run.

The sources live in *.py.orig files so no formatter or linter can rewrite
them (black already did once, to a .py copy), and each is checked against the
sha256 of the upstream file at load time: if the bytes differ, loading fails.

Source: capability_env_gen/capability_pipeline/ (vendored 2026-09-22).
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PINNED_SHA256 = {
    "daytona_snapshot": "a74497facf887768dc77d1593452fe2c7fb659837121ff93d6a0c8bc5cc7e644",
    "daytona_resources": "2f74a782c717b02c723d709b5dfd89ac765c44228ab1948db57113ee17d29f71",
    "daytona_telemetry": "fbb00617f89345681cf79f07e35f39a79910dee499269c6f3457a9a81590f14e",
}


def _load(name: str) -> types.ModuleType:
    source = (_HERE / f"{name}.py.orig").read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != PINNED_SHA256[name]:
        raise ImportError(f"vendored {name} changed: sha256 {actual} != pinned {PINNED_SHA256[name]}")
    qualified = f"{__name__}.{name}"
    module = types.ModuleType(qualified)
    module.__file__ = str(_HERE / f"{name}.py.orig")
    # Lets the upstream relative import (daytona_telemetry -> .daytona_resources) resolve.
    module.__package__ = __name__
    sys.modules[qualified] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


# Dependency order: telemetry imports resources.
daytona_resources = _load("daytona_resources")
daytona_snapshot = _load("daytona_snapshot")
daytona_telemetry = _load("daytona_telemetry")
