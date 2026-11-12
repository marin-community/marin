# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SHARED controller preserves live dependencies while rejecting harness tampering."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest
from verifyit.spec import PytestSpec, render_spec

from taskcompendium.integrity import SharedDependencyState, SharedIntegrityContract
from taskcompendium.runtime import shared_pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0, reason="UID isolation requires Linux root")


@dataclass(frozen=True)
class SharedRuntime:
    contract: SharedIntegrityContract
    site: Path
    verifyit: Path

    def grade(self) -> dict:
        shared_pytest.grade(Path(self.contract.protected_root) / "contract.json", str(self.verifyit))
        return json.loads((Path(self.contract.report_directory) / "verdict.json").read_text())


@pytest.fixture
def shared_runtime():
    with tempfile.TemporaryDirectory(prefix="taskcompendium-shared-test-") as temporary:
        base = Path(temporary)
        base.chmod(0o755)
        workspace, root, reports = (base / name for name in ("workspace", "protected", "reports"))
        workspace.mkdir()
        (workspace / "candidate.py").write_text("VALUE = 0\n")
        (workspace / "extra_guard.py").write_text("reference\n")
        (workspace / "editable.py").write_text("original\n")
        (workspace / "test_case.py").write_text(
            "import candidate, dependency\ndef test_value(): assert candidate.VALUE + dependency.VALUE == 1\n"
        )
        task_python = base / "task-python"
        subprocess.run([sys.executable, "-m", "venv", "--copies", "--without-pip", str(task_python)], check=True)
        site = Path(
            subprocess.check_output(
                [str(task_python / "bin/python"), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        (site / "test-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
        (site / "dependency.py").write_text("VALUE = 0\n")
        contract = SharedIntegrityContract(
            dependency_state=SharedDependencyState.AGENT_MUTABLE,
            workspace=str(workspace),
            protected_root=str(root),
            reference_directory=str(root / "reference"),
            runner_directory=str(root / "runner"),
            interpreter=str(root / "python/bin/python"),
            harness=str(root / "control.py"),
            pytest_config=str(root / "pytest.ini"),
            report_directory=str(reports),
            reference_sha256=hashlib.sha256(b"{}").hexdigest(),
            harness_sha256=hashlib.sha256(Path(shared_pytest.__file__).read_bytes()).hexdigest(),
            protected_test_patterns=("**/test_*.py", "**/conftest.py", "extra_guard.py"),
        )
        spec = PytestSpec(
            paths=("test_case.py",),
            candidate_modules=("candidate",),
            must_pass=("test_case.py::test_value",),
            python=str(task_python / "bin/python"),
            workspace=str(workspace),
        )
        payload = {
            "contract": contract.model_dump(mode="json"),
            "spec": asdict(spec),
            "test_files": {},
            "verifier_spec": render_spec(replace(spec, python=str(root / "pytest-python"))),
        }
        input_path = base / "input.json"
        input_path.write_text(json.dumps(payload))
        shared_pytest.bootstrap(input_path)
        verifyit = base / "verifyit"
        verifyit.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c "
            "'from verifyit.grade import main; raise SystemExit(main())' \"$@\"\n"
        )
        verifyit.chmod(0o755)
        yield SharedRuntime(contract, site, verifyit)


@pytest.mark.parametrize("attack", ["conftest", "addopts", "autoload", "pytest", "background", "stdout"])
def test_shared_grading_cannot_fake_a_pass(shared_runtime, monkeypatch, attack):
    runtime = shared_runtime
    workspace = Path(runtime.contract.workspace)
    process = None
    if attack == "conftest":
        (workspace / "conftest.py").write_text(
            "import pytest\ndef pytest_collection_modifyitems(items):\n"
            " for item in items: item.add_marker(pytest.mark.skip)\n"
        )
    elif attack == "addopts":
        monkeypatch.setenv("PYTEST_ADDOPTS", "--deselect=test_case.py::test_value")
    elif attack == "autoload":
        plugin = runtime.site / "fake_plugin.py"
        plugin.write_text(
            "import pytest\ndef pytest_collection_modifyitems(items):\n"
            " for item in items: item.add_marker(pytest.mark.skip)\n"
        )
        metadata = runtime.site / "fake_plugin-1.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: fake-plugin\nVersion: 1\n")
        (metadata / "entry_points.txt").write_text("[pytest11]\nfake = fake_plugin\n")
    elif attack == "pytest":
        package = runtime.site / "pytest"
        package.mkdir()
        (package / "__init__.py").write_text("def main(*a, **k): return 0\n")
    elif attack == "background":
        program = (
            "import os,pathlib,sys; os.setgid(10001);os.setuid(10001);p=pathlib.Path(sys.argv[1]);"
            "p.write_text('def test_value(): pass\\n');print('ready',flush=True);\n"
            "while True: p.write_text('def test_value(): pass\\n')"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", program, str(workspace / "test_case.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stdout.readline() == b"ready\n"
    else:
        (workspace / "candidate.py").write_text('print(\'{"status":"scored","reward":1}\')\nVALUE=0\n')
    try:
        verdict = runtime.grade()
        assert verdict["status"] == "scored"
        assert verdict["reward"] == 0
        if process is not None:
            assert process.wait(timeout=5) != 0
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def test_shared_grading_sees_agent_dependency_changes_and_records_diff(shared_runtime):
    runtime = shared_runtime
    (runtime.site / "dependency.py").write_text("VALUE = 1\n")
    verdict = runtime.grade()
    assert verdict["status"] == "scored"
    assert verdict["reward"] == 1
    reports = Path(runtime.contract.report_directory)
    changes = json.loads((reports / "site-packages-diff.json").read_text())
    assert str(runtime.site / "dependency.py") in changes
    assert json.loads((reports / "startup-hooks-diff.json").read_text()) == {}


def test_shared_controller_restores_contract_paths_and_preserves_other_files(shared_runtime):
    runtime = shared_runtime
    workspace = Path(runtime.contract.workspace)
    (workspace / "candidate.py").write_text("VALUE = 1\n")
    (workspace / "extra_guard.py").write_text("tampered\n")
    (workspace / "editable.py").write_text("agent change\n")
    assert runtime.grade()["reward"] == 1
    assert (workspace / "extra_guard.py").read_text() == "reference\n"
    assert (workspace / "editable.py").read_text() == "agent change\n"
