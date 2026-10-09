# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted repository tests carried from TaskTrove into deferred build tasks."""

import json
import re
from pathlib import PurePosixPath

from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.tasktrove import TaskFiles
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection
from verifyit.spec import PytestSpec

TRUSTED_PATHS = "trusted_test_paths.txt"
TRUSTED_INVOCATION = re.compile(
    r"install_trusted_test_paths\.sh\s*\\?\s*\n?\s*"
    r"\S+\s+(?P<trusted>[0-9a-f]{7,40})\s+\S+"
    r"(?:\s+(?:\"\"|\S+))?"
    r"(?:\s*\\?\s*\n?\s*(?P<fallback>[0-9a-f]{7,40}))?",
    re.MULTILINE,
)
SWESMITH_REPOSITORY = re.compile(r"https://github\.com/swesmith/(?P<repo>[^\s/]+)")
RESTORE_TESTS = """set -euo pipefail
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1
ws="$VERIFYIT_WORKSPACE"
cd "$ws"
trusted_git() {
    command git -c safe.directory="$ws" -c core.hooksPath=/dev/null -c core.fsmonitor=false "$@"
}
trusted_git cat-file -e TRUSTED_SHA^{commit}
restore_path() {
    path="$1"
    case "$path" in
        ""|/*|.|..|../*|*/..|*/../*) exit 1 ;;
    esac
    trusted_git clean -ffdx -- "$path"
    rm -rf -- "$path"
    if trusted_git cat-file -e TRUSTED_SHA:"$path" 2>/dev/null; then
        trusted_git archive --format=tar TRUSTED_SHA -- "$path" | tar -xf - -C "$ws"
    elif [ -n "FALLBACK_SHA" ] && trusted_git cat-file -e FALLBACK_SHA:"$path" 2>/dev/null; then
        trusted_git archive --format=tar FALLBACK_SHA -- "$path" | tar -xf - -C "$ws"
    fi
}
is_test_control_path() {
    path="$1"
    basename=${path##*/}
    case "/$path/" in
        */test/*|*/tests/*|*/testing/*|*/test-data/*|*/testdata/*|*/spec/*|*/specs/*|*/__tests__/*) return 0 ;;
    esac
    case "$basename" in
        conftest.py|pytest.ini|tox.ini|.coveragerc|jest.config.*|vitest.config.*|test_*|*_test.*|*.test.*|*.spec.*)
            return 0 ;;
    esac
    return 1
}
# Match the source installer: candidate hooks outside the manifest can alter collection and grading.
while IFS= read -r -d '' path; do
    if is_test_control_path "$path"; then
        restore_path "$path"
    fi
done < <(
    {
        trusted_git diff --name-only -z TRUSTED_SHA
        trusted_git ls-files --others --exclude-standard -z
        trusted_git ls-files --others --ignored --exclude-standard -z
    } | sort -zu
)
# Index flags can hide modified hooks from ordinary diff discovery.
while IFS= read -r -d '' record; do
    flag=${record%% *}
    path=${record#? }
    case "$flag" in
        [a-z]|S)
            if is_test_control_path "$path"; then
                restore_path "$path"
            fi ;;
    esac
done < <(trusted_git ls-files -v -z)
restore_manifest() {
    while IFS= read -r path || [ -n "$path" ]; do
        [ -z "$path" ] && continue
        restore_path "$path"
    done < "$1"
}
restore_manifest "$VERIFYIT_TESTS_DIR/trusted_test_paths.txt"

"""


def repository_test_ids(value: object) -> tuple[str, ...]:
    """Read the source's list or JSON-encoded list of pytest node ids."""
    if isinstance(value, str):
        decoded = json.loads(value) if value.strip() else []
    else:
        decoded = value if value is not None else []
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError("Repository test ids must be a list of strings")
    return tuple(decoded)


def uncollectable(node_id: str) -> bool:
    """Source doctest and truncated parametrized ids cannot run with cleared pytest addopts."""
    file, _, rest = node_id.partition("::")
    if not file.endswith(".py") or ("[" in rest and not rest.endswith("]")):
        return True
    name = rest.split("[", 1)[0]
    if "::" in name:
        return False
    path = PurePosixPath(file)
    module = path.parent.name if path.name == "__init__.py" else path.stem
    return "." in name or (name == module and not name.startswith("test"))


def trusted_pytest(task: TaskFiles) -> PytestSpec | ImportRejection:
    """Recover the legacy SWE-smith pytest contract, including trusted-test restoration."""
    config = json.loads(task.text("tests/config.json"))
    fail_to_pass = repository_test_ids(config.get("FAIL_TO_PASS"))
    pass_to_pass = repository_test_ids(config.get("PASS_TO_PASS"))
    if not fail_to_pass and not pass_to_pass:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="null_grader",
            detail="config.json has no FAIL_TO_PASS or PASS_TO_PASS tests",
        )
    match = TRUSTED_INVOCATION.search(task.text("tests/test.sh"))
    if match is None:
        return unsupported("unsupported_variant", "tests/test.sh does not call install_trusted_test_paths.sh")
    foreign = [node_id for node_id in fail_to_pass if uncollectable(node_id)]
    if foreign:
        return unsupported("unsupported_variant", f"FAIL_TO_PASS ids the pytest mode cannot collect: {foreign[:3]}")
    retained = tuple(node_id for node_id in pass_to_pass if not uncollectable(node_id))
    files = {node_id.split("::", 1)[0] for node_id in (*fail_to_pass, *retained)}
    manifest = {line.strip() for line in task.text(f"tests/{TRUSTED_PATHS}").splitlines() if line.strip()}
    uncovered = sorted(files - manifest)
    if uncovered:
        return unsupported("unsupported_variant", f"graded test files missing from trusted manifest: {uncovered[:5]}")
    return PytestSpec(
        paths=tuple(sorted(files)),
        must_pass=fail_to_pass,
        must_not_break=retained,
        setup=RESTORE_TESTS.replace("TRUSTED_SHA", match["trusted"]).replace("FALLBACK_SHA", match["fallback"] or ""),
        protected_paths_files=(TRUSTED_PATHS,),
        workspace="/testbed",
    )


def repository_dockerfile(dockerfile: str, instruction: str) -> str:
    """Carry the legacy pytest plugin and repository dependency fixes into the build recipe."""
    if "pytest-json-report" not in dockerfile:
        lines = dockerfile.splitlines()
        for index, line in enumerate(lines):
            tokens = line.split()
            if tokens[:1] == ["RUN"] and "pip" in tokens and "install" in tokens and "pytest" in tokens:
                lines[index] = line + " pytest-json-report"
                dockerfile = "\n".join(lines) + "\n"
                break
        else:
            dockerfile = dockerfile.rstrip("\n") + (
                "\nRUN (pip install --no-cache-dir pytest-json-report"
                " || pip3 install --no-cache-dir pytest-json-report)\n"
            )
    match = SWESMITH_REPOSITORY.search(instruction)
    if match is None:
        return dockerfile
    packages = '"pytest<9"'
    additions = {
        "marshmallow-code__marshmallow.9716fc62": " simplejson",
        "conan-io__conan.86f29e13": " mock webtest PyJWT bottle parameterized",
        "oauthlib__oauthlib.1fd52536": " PyJWT cryptography blinker",
    }
    packages += additions.get(match["repo"], "")
    return (
        dockerfile.rstrip("\n")
        + "\nRUN printf 'pytest<9\\n' > /opt/verifyit-pytest-constraints.txt\n"
        + "ENV PIP_CONSTRAINT=/opt/verifyit-pytest-constraints.txt\n"
        + f"RUN python -m pip install --no-cache-dir {packages}\n"
    )
