# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tomllib
from pathlib import Path


def _load_pyproject() -> dict:
    return tomllib.loads(Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text())


def test_chex_is_runtime_dependency():
    pyproject = _load_pyproject()
    dependencies = pyproject["project"]["dependencies"]
    assert any(dep.startswith("chex") for dep in dependencies)
