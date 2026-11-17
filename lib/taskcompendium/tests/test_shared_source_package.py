# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable regressions for the protected SHARED source-scorer packaging."""

import ast
import json
import tempfile
import zlib
from pathlib import Path

import pytest

from taskcompendium.harbor.shared_source import (
    SCALE_SWE_PROFILE,
    SWE_LEGO_PROFILE,
    SWE_REBENCH_PROFILE,
    harden_shared_source,
    patch_paths,
    published_contract,
)
from taskcompendium.integrity import SharedDependencyState, SharedSourceFamily
from taskcompendium.runtime import shared_source
from taskcompendium.runtime.shared_source import parse_source_reward

SCALE_TEST_SH = b"#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\npython /tests/grade.py\n"


def scale_files(workdir: str = "/repo") -> dict[str, bytes]:
    config = {
        "workdir": workdir,
        "base_commit": "a" * 40,
        "f2p_patch": "",
        "f2p_script": "",
        "test_ids": ["tests/test_x.py::test_x"],
    }
    return {
        "instruction.md": b"Solve the issue.\n",
        "task.toml": (
            b'[environment]\ncpus = 4\nmemory_mb = 4096\nstorage_mb = 10240\n[verifier]\nenvironment_mode = "shared"\n'
        ),
        "environment/Dockerfile": (
            ("FROM example.test/scale@sha256:" + "a" * 64 + "\nWORKDIR " + workdir + "\n").encode()
        ),
        "tests/test.sh": SCALE_TEST_SH,
        "tests/grade.py": b"grade",
        "tests/native_scripts.py": b"ENV = {}\nRESTORE = 'true'\n",
        "tests/scale_score.py": b"score",
        "tests/config.json": json.dumps(config).encode(),
    }


def lego_files() -> dict[str, bytes]:
    config = {
        "base_commit": "b" * 40,
        "test_patch": "--- /dev/null\n+++ b/tests/test_new.py\n@@ -0,0 +1 @@\n+def test_new(): pass\n",
        "test_cmd": "python -m pytest -rA tests/test_new.py",
        "test_ids": ["tests/test_new.py::test_new"],
    }
    return {
        "instruction.md": b"Solve the issue.\n",
        "task.toml": b'[environment]\n[verifier]\nenvironment_mode = "shared"\ntimeout_sec = 900.0\n',
        "environment/Dockerfile": ("FROM example.test/lego@sha256:" + "b" * 64 + "\n").encode(),
        "tests/test.sh": SCALE_TEST_SH,
        "tests/grade.py": b"grade",
        "tests/native_scripts.py": b"ENV = {}\nREVERT_TEST_FILES = 'true'\n",
        "tests/config.json": json.dumps(config).encode(),
    }


def test_scale_package_keeps_body_and_hides_all_tests_material():
    files = scale_files()
    contract = harden_shared_source(
        files,
        profile=SCALE_SWE_PROFILE,
        dependency_state=SharedDependencyState.AGENT_MUTABLE,
    )
    assert (contract.id, contract.version, contract.native.family) == ("shared-source", 2, SharedSourceFamily.SCALE_SWE)
    assert contract.workspace == "/repo"
    assert published_contract(files) == contract
    instruction = files["instruction.md"].decode()
    assert instruction.startswith("Grading rules (shared-source v2, scale-swe, agent_mutable):\n")
    assert instruction.endswith("Solve the issue.\n")
    assert sorted(name for name in files if name.startswith("tests/")) == ["tests/test.sh"]
    assert files["tests/test.sh"].decode().startswith('#!/bin/bash\nset -euo pipefail\nexec "')
    payload = json.loads(zlib.decompress(files["setup_files/taskcompendium-source-input.zlib"]))
    assert sorted(payload["tests"]) == ["config.json", "grade.py", "native_scripts.py", "scale_score.py", "test.sh"]
    assert payload["tests"]["config.json"] == files_original_config_hex(files)
    assert payload["spec"] == {"timeout": None}
    assert "timeout_sec" not in files["task.toml"].decode().split("[verifier]")[-1].split("[agent]")[0]
    setup = "\n".join(line for line in files["task.toml"].decode().splitlines() if "bootstrap" in line)
    assert "taskcompendium-source-input.json" in setup
    assert files["environment/taskcompendium-shared/control.py"]
    assert "taskcompendium-grade" in files["environment/Dockerfile"].decode()
    # UID handover happens at build time: sandbox fs may deny runtime uid changes.
    assert "RUN mkdir -p /repo && chown -R 10001:10001 /repo" in files["environment/Dockerfile"].decode()
    assert files["task.toml"].decode().count("agent_mutable_v2") == 1


def files_original_config_hex(files: dict[str, bytes]) -> str:
    payload = json.loads(zlib.decompress(files["setup_files/taskcompendium-source-input.zlib"]))
    return payload["tests"]["config.json"]


def test_lego_contract_captures_patch_paths_and_declares_its_policy():
    files = lego_files()
    contract = harden_shared_source(
        files,
        profile=SWE_LEGO_PROFILE,
        dependency_state=SharedDependencyState.AGENT_MUTABLE,
    )
    assert contract.workspace == "/testbed"
    assert "tests/test_new.py" in contract.native.capture_patterns
    assert contract.native.patch_config_keys == ("test_patch",)
    assert contract.known_open_gaps == (
        "Agent-installed pytest plugins and imports in the shared interpreter environments",
        "Candidate-produced test-output injection",
    )
    assert "RUN mkdir -p /testbed && chown -R 10001:10001 /testbed" in files["environment/Dockerfile"].decode()
    payload = json.loads(zlib.decompress(files["setup_files/taskcompendium-source-input.zlib"]))
    assert sorted(payload["tests"]) == ["config.json", "grade.py", "native_scripts.py", "test.sh"]
    task_toml = files["task.toml"].decode()
    # The declared source deadline passes through verbatim.
    assert payload["spec"] == {"timeout": 900.0}
    assert "timeout_sec = 900.0" in task_toml
    assert 'user = "root"' in task_toml
    assert "restart_environment = false" in task_toml
    assert 'user = "10001"' in task_toml
    assert 'environment_mode = "shared"' in task_toml


def test_unknown_config_keys_and_missing_files_fail_closed():
    files = scale_files()
    files["tests/config.json"] = json.dumps({**json.loads(files["tests/config.json"]), "extra": 1}).encode()
    with pytest.raises(ValueError, match="config keys differ"):
        harden_shared_source(files, profile=SCALE_SWE_PROFILE, dependency_state=SharedDependencyState.AGENT_MUTABLE)
    files = scale_files()
    del files["tests/scale_score.py"]
    with pytest.raises(ValueError, match="lacks protected grading files"):
        harden_shared_source(files, profile=SCALE_SWE_PROFILE, dependency_state=SharedDependencyState.AGENT_MUTABLE)
    files = scale_files()
    files["tests/surprise.py"] = b"x"
    with pytest.raises(ValueError, match="undeclared tests files"):
        harden_shared_source(files, profile=SCALE_SWE_PROFILE, dependency_state=SharedDependencyState.AGENT_MUTABLE)


def test_patch_paths_cover_new_and_modified_files():
    diff = (
        "--- a/tests/test_old.py\n"
        "+++ b/tests/test_old.py\n"
        "@@ -1 +1 @@\n"
        "-x\n+y\n"
        "--- /dev/null\n"
        "+++ b/tests/nested/test_new.py\n"
        "@@ -0,0 +1 @@\n"
        "+pass\n"
    )
    assert patch_paths(diff) == ["tests/test_old.py", "tests/nested/test_new.py"]


def test_source_reward_validation_fails_closed(tmp_path):
    regular = tmp_path / "reward.txt"
    regular.write_text("0.5\n")
    assert parse_source_reward(regular) == 0.5
    for content in ("nan\n", "inf\n", "-inf\n", "2.0\n", "-0.1\n", "abc\n", ""):
        regular.write_text(content)
        assert parse_source_reward(regular) is None
    dangling = tmp_path / "dangling.txt"
    dangling.symlink_to(tmp_path / "missing")
    assert parse_source_reward(dangling) is None
    assert parse_source_reward(tmp_path / "absent.txt") is None


def test_scale_requires_absolute_workdir_and_pinned_commit():
    files = scale_files(workdir="relative/repo")
    with pytest.raises(ValueError, match="absolute workspace path"):
        harden_shared_source(files, profile=SCALE_SWE_PROFILE, dependency_state=SharedDependencyState.AGENT_MUTABLE)
    files = scale_files()
    config = json.loads(files["tests/config.json"])
    config["base_commit"] = "not-a-commit"
    files["tests/config.json"] = json.dumps(config).encode()
    with pytest.raises(ValueError, match="pinned base commit"):
        harden_shared_source(files, profile=SCALE_SWE_PROFILE, dependency_state=SharedDependencyState.AGENT_MUTABLE)


def test_guest_controller_stays_runtime_compatible_with_python_39():
    """The controller must load on task images whose best interpreter is 3.9.

    Annotations stay lazy via ``from __future__ import annotations`` (PEP 604
    unions never evaluate), so the only 3.9 hazards are runtime constructs:
    ``datetime.UTC`` exists only on 3.11+ and broke the matplotlib-20009
    image's bootstrap in the freeze-v3 QA (ImportError at control.py import).
    """
    source = Path(shared_source.__file__).read_text()
    assert "from __future__ import annotations" in source
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("from datetime import"):
            names = {name.strip() for name in stripped[len("from datetime import") :].split(",")}
            assert "UTC" not in names, f"datetime.UTC is 3.11+ only; controller line: {stripped}"
    # Syntax-level guard: 3.10+ grammar (e.g. match statements) must not enter
    # the controller either.
    ast.parse(source, feature_version=(3, 9))


def swe_rebench_files(workdir: str = "/spectree") -> dict[str, bytes]:
    """A SWE-rebench-shaped converted package: explicit test_files, rendered test.sh."""
    config = {
        "workdir": workdir,
        "base_commit": "b" * 40,
        "test_patch": "--- /dev/null\n+++ b/tests/test_new.py\n@@ -0,0 +1,1 @@\n+value = 1\n",
        "test_cmd": "python -m pytest tests/test_new.py",
        "test_files": ["tests/test_app.py", "tests/conftest.py"],
        "log_parser": "parse_log_pytest",
        "language": "python",
        "test_ids": ["tests/test_new.py::test_resolved"],
        "FAIL_TO_PASS": ["tests/test_new.py::test_resolved"],
        "PASS_TO_PASS": [],
    }
    return {
        "instruction.md": b"Fix the module so the tests pass.\n",
        "task.toml": (
            b"[environment]\ncpus = 6\nmemory_mb = 24576\nstorage_mb = 32768\n"
            b"[agent]\ntimeout_sec = 3000.0\n"
            b"[verifier]\ntimeout_sec = 2400.0\n"
        ),
        "environment/Dockerfile": b"FROM fixture.test/swe-rebench-base@sha256:" + b"b" * 64 + b"\n",
        "tests/test.sh": b"#!/bin/sh\nset -uo pipefail\nexit 0\n",
        "tests/test.patch": config["test_patch"].encode(),
        "tests/config.json": json.dumps(config).encode(),
        "tests/grade.py": b"print('grade')\n",
        "solution/solve.sh": b"#!/bin/sh\nexit 0\n",
        "solution/gold.patch": b"--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
    }


def test_swe_rebench_profile_binds_the_agreed_contract():
    contract = harden_shared_source(
        swe_rebench_files(), profile=SWE_REBENCH_PROFILE, dependency_state=SharedDependencyState.AGENT_MUTABLE
    )
    assert contract.workspace == "/spectree"
    assert contract.id == "shared-source"
    native = contract.native
    assert native.family == SharedSourceFamily.SWE_REBENCH
    assert native.patch_config_keys == ("test_patch",)
    assert native.path_config_keys == ("test_files",)
    assert native.capture_dirs == ()
    assert native.capture_patterns == ("tests/test_new.py",)
    # Protected test patterns mirror the grading-material scope: the test patch
    # paths (Lego precedent) plus the row's explicit test_files list.
    assert "tests/test_new.py" in contract.protected_test_patterns
    assert "tests/test_app.py" in contract.protected_test_patterns
    # The upstream-declared verifier deadline passes through verbatim; the
    # quiesce binding is the POSIX shell variant for this family.
    assert SWE_REBENCH_PROFILE.quiesce_variant == "shell"
    assert SWE_REBENCH_PROFILE.path_config_keys == ("test_files",)


def test_swe_rebench_rejects_undeclared_config_keys_and_missing_private_files():
    files = swe_rebench_files()
    config = json.loads(files["tests/config.json"])
    config["unexpected"] = 1
    files["tests/config.json"] = json.dumps(config).encode()
    with pytest.raises(ValueError, match="config"):
        harden_shared_source(files, profile=SWE_REBENCH_PROFILE, dependency_state=SharedDependencyState.AGENT_MUTABLE)
    files = swe_rebench_files()
    del files["tests/test.patch"]
    with pytest.raises(ValueError, match=r"test\.patch"):
        harden_shared_source(files, profile=SWE_REBENCH_PROFILE, dependency_state=SharedDependencyState.AGENT_MUTABLE)


def test_capture_scope_joins_explicit_config_path_lists():
    workspace = tmp_scope()
    runner = {
        "capture_dirs": (),
        "capture_patterns": (),
        "patch_config_keys": ("test_patch",),
        "path_config_keys": ("test_files",),
    }
    config = {
        "test_patch": "--- /dev/null\n+++ b/tests/test_new.py\n@@ -0,0 +1,1 @@\n+value = 1\n",
        "test_files": ["tests/test_app.py", "tests/absent.py", "/etc/passwd"],
    }
    (workspace / "tests").mkdir(parents=True)
    (workspace / "tests" / "test_app.py").write_text("value = 0\n")
    scope = shared_source.capture_scope(workspace, runner, config)
    # Existing test_files entries join; absent entries and absolute paths do
    # not. The test_patch path does NOT join here (it does not exist before
    # the grading-time patch application; the patch-path ancestor handling
    # covers it then).
    assert scope == ["tests/test_app.py"]


def tmp_scope() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="capture-scope-"))
    return directory
