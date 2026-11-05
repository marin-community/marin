# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shlex
from pathlib import Path

import yaml
from iris.cluster.client.bundle import collect_workspace_files


def test_nightly_bundle_contains_pytest_entrypoint_and_ancestor_fixtures():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/marin-canary-serving-parity.yaml").read_text())
    submit = next(step for step in workflow["jobs"]["logprob-check"]["steps"] if step.get("id") == "submit")
    argv = shlex.split(submit["run"])
    includes = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--include"]
    files = {path.relative_to(root).as_posix() for path in collect_workspace_files(root, extra_includes=includes)}
    required = {
        "tests/conftest.py",
        "tests/cluster/conftest.py",
        "tests/cluster/vllm/conftest.py",
        "tests/cluster/vllm/test_snowball_backend_parity.py",
    }
    required.update(
        str(path.relative_to(root))
        for path in (
            root / "tests/__init__.py",
            root / "tests/cluster/__init__.py",
            root / "tests/cluster/vllm/__init__.py",
        )
        if path.exists()
    )
    assert required <= files
