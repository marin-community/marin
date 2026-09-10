# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the SWE-bench-shaped converters (``swe_patched``, ``swe_trusted_paths``)."""

import json
from collections.abc import Iterable
from pathlib import PurePosixPath

PLUGIN = "pytest-json-report"
CONFIG_JSON = "tests/config.json"
TRUSTED_TEST_PATHS = "tests/trusted_test_paths.txt"
TESTBED = "/testbed"
"""Where the SWE images and the environment-setup step in ``instruction.md`` put the repository."""


def test_file(node_id: str) -> str:
    return node_id.split("::", 1)[0]


def uncollectable(node_id: str) -> bool:
    """True when the pytest mode cannot collect ``node_id``.

    The mode clears the repository's ``addopts``, which drops ``--doctest-glob`` and
    ``--doctest-modules``, so ids in non-Python files (``tests/tests.md::tests.md``) and doctest items,
    whose name is the dotted object path (``parso/__init__.py::parso``,
    ``parso/tree.py::parso.tree.NodeOrLeaf.dump``), never run and count as failures.
    """
    file, _, rest = node_id.partition("::")
    if not file.endswith(".py"):
        return True
    name = rest.split("[", 1)[0]
    if "::" in name:
        return False
    path = PurePosixPath(file)
    module = path.parent.name if path.name == "__init__.py" else path.stem
    # A module doctest is named after its module; a test module's own name also matches pytest's
    # ``test*`` function pattern, so only a non-test name identifies a doctest.
    return "." in name or (name == module and not name.startswith("test"))


def uncovered_files(graded_files: set[str], manifests: Iterable[str | None]) -> list[str]:
    """Graded test files that no trusted manifest restores, so an agent could rewrite them."""
    manifest = {line.strip() for text in manifests for line in (text or "").splitlines() if line.strip()}
    return sorted(graded_files - manifest)


def test_ids(value: object) -> list[str]:
    """``FAIL_TO_PASS``/``PASS_TO_PASS`` as a list of node ids; the field is sometimes a JSON-encoded string."""
    if isinstance(value, str):
        decoded: object = json.loads(value) if value.strip() else []
    else:
        decoded = value if value is not None else []
    if not isinstance(decoded, list):
        raise ValueError(f"expected a list of test ids, got {type(decoded).__name__}")
    return [str(v) for v in decoded]


def ensure_pytest_json_report(dockerfile: str, conda_lines: tuple[str, ...] = ()) -> str:
    """Add the ``pytest-json-report`` plugin the ``pytest`` mode needs into the repo's own Python.

    Installed at image build time because grading runs with no network. An existing
    ``RUN ... pip install ... pytest`` line gets the plugin appended, so it lands in the same
    interpreter; otherwise a conda repo installs it inside its activated env and a plain repo
    through ``pip`` or ``pip3``.
    """
    if PLUGIN in dockerfile:
        return dockerfile
    lines = dockerfile.splitlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        if tokens[:1] == ["RUN"] and "pip" in tokens and "install" in tokens and "pytest" in tokens:
            lines[index] = line + f" {PLUGIN}"
            return "\n".join(lines) + "\n"
    if conda_lines:
        activate = " && ".join(conda_lines)
        install = f'RUN bash -lc "{activate} && pip install --no-cache-dir {PLUGIN}"\n'
    else:
        install = f"RUN (pip install --no-cache-dir {PLUGIN} || pip3 install --no-cache-dir {PLUGIN})\n"
    return dockerfile.rstrip("\n") + "\n" + install
