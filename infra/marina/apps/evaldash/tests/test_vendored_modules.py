# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The vendored evaluation modules must match lib/marin, and nothing else may import marin."""

import ast
from pathlib import Path

import pytest
from evaldash import vendor_marin_evaluation as vendoring

APP = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", vendoring.MODULES)
def test_vendored_module_matches_the_library(module: str):
    vendored = (vendoring.VENDORED / f"{module}.py").read_text(encoding="utf-8")
    assert vendored == vendoring.vendored_source(
        module
    ), f"marin_evaluation/{module}.py differs from lib/marin; run vendor_marin_evaluation.py"


def test_the_app_imports_no_installed_marin_module():
    offenders = []
    for source in APP.rglob("*.py"):
        if "marin_evaluation" in source.parts:
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            offenders += [f"{source.relative_to(APP)}: {n}" for n in names if n == "marin" or n.startswith("marin.")]
    assert not offenders, "\n".join(offenders)
