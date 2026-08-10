# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

from iris.cli.build import build_dashboard_assets


def test_build_dashboard_assets_produces_controller_bundle(tmp_path, monkeypatch):
    dashboard_dir = tmp_path / "dashboard"
    dashboard_dir.mkdir()
    (dashboard_dir / "package.json").write_text("{}")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npm = bin_dir / "npm"
    npm.write_text(
        """#!/bin/sh
set -eu
if [ "$1" = "ci" ]; then
  mkdir -p node_modules
  exit 0
fi
test -d node_modules
mkdir -p dist
printf 'current dashboard' > dist/controller.html
"""
    )
    npm.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    build_dashboard_assets(dashboard_dir)

    assert (dashboard_dir / "dist" / "controller.html").read_text() == "current dashboard"
