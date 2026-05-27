# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hatchling build hook: stamp ``iris/_build_info.py`` on every wheel build.

The Iris controller rejects clients whose ``client_revision_date`` predates a
server-side floor (see :mod:`iris.version`). That date comes from
``BUILD_DATE``, which this hook sets from ``git log`` on the iris source tree
(so a wheel reports the age of the code it was built from), falling back to the
build day when no git history is reachable. The stamped file is force-included
into the wheel rather than written to the tracked source, keeping working trees
clean.
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        # Only the wheel ships importable code. Editable installs keep the
        # git-log fallback (their tree is the live repo), so leave them alone.
        if self.target_name != "wheel" or build_data.get("editable"):
            return

        stamped = (
            "# Copyright The Marin Authors\n"
            "# SPDX-License-Identifier: Apache-2.0\n\n"
            "# Auto-generated at build time by lib/iris/hatch_build.py.\n\n"
            f'BUILD_DATE = "{self._resolve_date()}"\n'
        )
        tmp = Path(tempfile.mkdtemp(prefix="iris-build-info-")) / "_build_info.py"
        tmp.write_text(stamped)
        build_data.setdefault("force_include", {})[str(tmp)] = "iris/_build_info.py"

    def _resolve_date(self) -> str:
        """Committer date (YYYY-MM-DD) of the iris source tree, or build day."""
        root = Path(self.root)
        try:
            out = subprocess.check_output(
                ["git", "log", "-1", "--format=%cs", "--", (root / "src" / "iris").as_posix()],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            ).strip()
            if out:
                return out
        except (subprocess.SubprocessError, OSError):
            pass
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
