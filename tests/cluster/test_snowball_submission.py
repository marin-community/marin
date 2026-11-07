# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
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


@pytest.mark.parametrize(
    "arm,expected",
    [
        ("gpu", {"vllm-gpu-pp1", "vllm-gpu-pp2"}),
        ("tpu", {"vllm-tpu-v4-tp16"}),
    ],
)
def test_each_nightly_leg_collects_only_its_local_vllm_arms(arm, expected):
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/marin-canary-serving-parity.yaml").read_text())
    job = workflow["jobs"]["logprob-check"]
    leg = next(case for case in job["strategy"]["matrix"]["include"] if case["arm"] == arm)
    assert job["env"]["SELECTOR"] == "${{ matrix.selector }}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/cluster/vllm/test_snowball_backend_parity.py",
            "-m",
            "cluster",
            "-o",
            "addopts=",
            "--import-mode=importlib",
            "-k",
            leg["selector"],
            "--collect-only",
            "-q",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selected = {
        line.rsplit("[", 1)[1].removesuffix("]")
        for line in result.stdout.splitlines()
        if "::test_snowball_export_matches_representative_goldens[" in line
    }
    assert selected == expected
