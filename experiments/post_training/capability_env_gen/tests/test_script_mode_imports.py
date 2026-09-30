"""Modules that run as scripts must not use relative imports.

runtime.py is executed directly (`python .../runtime.py`) inside the composite
runtime venv, where it has no parent package; it supports that by putting the
repo on sys.path and importing absolutely.  A relative import there raises
"attempted relative import with no known parent package" -- but only in
production, because every unit test imports the module as a package member.
2026-09-22: exactly this failed the first silo gate validation in 12 s.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "capability_pipeline"


def _script_mode_modules():
    for path in sorted(PACKAGE.glob("*.py")):
        if "__package__ in" in path.read_text():
            yield path


def test_script_mode_modules_exist():
    assert any(p.name == "runtime.py" for p in _script_mode_modules())


def test_script_mode_modules_use_only_absolute_imports():
    offenders = []
    for path in _script_mode_modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"relative imports in script-mode modules: {offenders}"
