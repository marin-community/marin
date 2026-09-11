# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the SWE-bench-shaped converters (``swe_patched``, ``swe_trusted_paths``)."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus, Rejected

PLUGIN = "pytest-json-report"
CONFIG_JSON = "tests/config.json"
TRUSTED_TEST_PATHS = "tests/trusted_test_paths.txt"
TESTBED = "/testbed"
"""Where the SWE images and the environment-setup step in ``instruction.md`` put the repository."""

_RESTORE_SETUP = """set -euo pipefail
ws="$TASKTROVE_WORKSPACE"
cd "$ws"
git -c safe.directory="$ws" cat-file -e TRUSTED_SHA^{commit}
restore_path() {
    path="$1"
    git -c safe.directory="$ws" clean -ffdx -- "$path" >/dev/null 2>&1 || true
    rm -rf -- "$path"
    if git -c safe.directory="$ws" cat-file -e TRUSTED_SHA:"$path" 2>/dev/null; then
        git -c safe.directory="$ws" archive --format=tar TRUSTED_SHA -- "$path" | tar -xf - -C "$ws"
    elif [ -n "FALLBACK_SHA" ] && git -c safe.directory="$ws" cat-file -e FALLBACK_SHA:"$path" 2>/dev/null; then
        git -c safe.directory="$ws" archive --format=tar FALLBACK_SHA -- "$path" | tar -xf - -C "$ws"
    fi
}
restore_manifest() {
    while IFS= read -r path || [ -n "$path" ]; do
        [ -z "$path" ] && continue
        case "$path" in
            ""|/*|.|..|../*|*/..|*/../*) exit 1 ;;
        esac
        restore_path "$path"
    done < "$1"
}
RESTORE_MANIFESTS
APPLY_PATCH
"""


@dataclass(frozen=True)
class PytestSelection:
    must_pass: tuple[str, ...]
    must_not_break: tuple[str, ...]
    files: tuple[str, ...]


def restore_setup(trusted: str, manifests: tuple[str, ...], fallback: str = "", patch: str = "") -> str:
    manifest_commands = "\n".join(
        f'restore_manifest "$TASKTROVE_TESTS_DIR/{PurePosixPath(path).name}"' for path in manifests
    )
    patch_command = (
        f'git -c safe.directory="$ws" apply --whitespace=nowarn "$TASKTROVE_TESTS_DIR/{PurePosixPath(patch).name}"'
        if patch
        else ""
    )
    return (
        _RESTORE_SETUP.replace("TRUSTED_SHA", trusted)
        .replace("FALLBACK_SHA", fallback)
        .replace("RESTORE_MANIFESTS", manifest_commands)
        .replace("APPLY_PATCH", patch_command)
    )


def pytest_selection(
    fail_to_pass: list[str], pass_to_pass: list[str], manifests: Iterable[str | None]
) -> PytestSelection | Rejected:
    foreign = [node_id for node_id in fail_to_pass if uncollectable(node_id)]
    if foreign:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT, f"FAIL_TO_PASS ids the pytest mode cannot collect: {foreign[:3]}"
        )
    retained_pass_to_pass = [node_id for node_id in pass_to_pass if not uncollectable(node_id)]
    files = {test_file(node_id) for node_id in [*fail_to_pass, *retained_pass_to_pass]}
    uncovered = uncovered_files(files, manifests)
    if uncovered:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT, f"graded test files missing from trusted manifest: {uncovered[:5]}"
        )
    return PytestSelection(tuple(fail_to_pass), tuple(retained_pass_to_pass), tuple(sorted(files)))


def test_file(node_id: str) -> str:
    return node_id.split("::", 1)[0]


def uncollectable(node_id: str) -> bool:
    """True when the pytest mode cannot collect ``node_id``.

    The mode clears the repository's ``addopts``, which drops ``--doctest-glob`` and
    ``--doctest-modules``, so ids in non-Python files (``tests/tests.md::tests.md``) and doctest items,
    whose name is the dotted object path (``parso/__init__.py::parso``,
    ``parso/tree.py::parso.tree.NodeOrLeaf.dump``), never run and count as failures. Some sources
    split their id lists on whitespace, so a parametrized id whose parameter contains a space arrives
    truncated (``test_ddl[add-kwargs0-ALTER``) and never matches a collected item either.
    """
    file, _, rest = node_id.partition("::")
    if not file.endswith(".py"):
        return True
    if "[" in rest and not rest.endswith("]"):
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
