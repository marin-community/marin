# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the SWE-bench-shaped converters (``swe_patched``, ``swe_trusted_paths``)."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

from taskcompendium.convert.tasktrove_converted_task import ConvertStatus, Rejected

from experiments.post_training.task_curation.datasets.tasktrove.repository_pytest import uncollectable

CONFIG_JSON = "tests/config.json"
TRUSTED_TEST_PATHS = "tests/trusted_test_paths.txt"
TESTBED = "/testbed"
"""Where the SWE images and the environment-setup step in ``instruction.md`` put the repository."""

_RESTORE_SETUP = """set -euo pipefail
ws="$VERIFYIT_WORKSPACE"
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
        f'restore_manifest "$VERIFYIT_TESTS_DIR/{PurePosixPath(path).name}"' for path in manifests
    )
    patch_command = (
        f'git -c safe.directory="$ws" apply --whitespace=nowarn "$VERIFYIT_TESTS_DIR/{PurePosixPath(patch).name}"'
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
