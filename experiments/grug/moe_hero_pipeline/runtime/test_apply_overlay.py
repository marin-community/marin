# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

from experiments.grug.moe_hero_pipeline.runtime.apply_overlay import apply_overlay


def test_apply_overlay_inside_checkout_changes_installed_package(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    installed = site_packages / "jaxpp" / "core.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("original = True\n")
    patch = b"""diff --git a/src/jaxpp/core.py b/src/jaxpp/core.py
--- a/src/jaxpp/core.py
+++ b/src/jaxpp/core.py
@@ -1 +1 @@
-original = True
+patched = True
"""

    apply_overlay(site_packages, patch)

    assert installed.read_text() == "patched = True\n"
