# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the SWE-bench-shaped converters (``swe_patched``, ``swe_trusted_paths``)."""

import json
from pathlib import PurePosixPath

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


def test_ids(value: object) -> list[str]:
    """``FAIL_TO_PASS``/``PASS_TO_PASS`` as a list of node ids; the field is sometimes a JSON-encoded string."""
    if isinstance(value, str):
        decoded: object = json.loads(value) if value.strip() else []
    else:
        decoded = value if value is not None else []
    if not isinstance(decoded, list):
        raise ValueError(f"expected a list of test ids, got {type(decoded).__name__}")
    return [str(v) for v in decoded]
