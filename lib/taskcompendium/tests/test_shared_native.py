# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Linux root regressions against real language runners and UID-separated agents."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from taskcompendium.integrity import NativeSharedRunner, SharedDependencyState, SharedIntegrityContract
from taskcompendium.native_integrity import FAMILIES, TEST_PATTERNS
from taskcompendium.runtime import shared_native, shared_pytest

from .native_projects import PROJECTS

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0, reason="UID isolation requires Linux root"),
]


@pytest.fixture(scope="module", params=PROJECTS, ids=PROJECTS)
def native_runtime(request):
    project = PROJECTS[request.param]
    profile = FAMILIES[project.family]
    with tempfile.TemporaryDirectory(prefix="taskcompendium-native-test-") as temporary:
        base = Path(temporary)
        base.chmod(0o755)
        workspace, root, reports, dependencies = (
            base / name for name in ("workspace", "protected", "reports", "dependencies")
        )
        workspace.mkdir()
        dependencies.mkdir()
        for directory, files in ((workspace, project.files), (dependencies, project.dependency_files)):
            for name, content in files.items():
                if name == "dependency":
                    (directory / name).symlink_to(dependencies, target_is_directory=True)
                    continue
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content.replace("{dependency}", str(dependencies)))
        (workspace / "tests").mkdir(exist_ok=True)
        (workspace / "tests/run.sh").write_text("set -euo pipefail\n" + project.command + "\necho TASK_PASSED\n")
        tests = {
            "config.json": json.dumps({"marker": "TASK_PASSED"}),
            "test_state.py": (
                "import json\nfrom pathlib import Path\n"
                "def evaluate_test_results(output, config):\n"
                " return {'resolved': json.loads(Path(config).read_text())['marker'] in Path(output).read_text()}\n"
            ),
            "trusted_test_paths.txt": "tests/run.sh\n",
            "trusted_patch_paths.txt": "tests/run.sh\n",
            "install_trusted_test_paths.sh": "true\n",
            "install_trusted_test_patch.sh": "true\n",
            "test_patch.diff": "",
        }
        encoded = {name: value.encode().hex() for name, value in tests.items()}
        contract = SharedIntegrityContract(
            id="shared-native",
            dependency_state=SharedDependencyState.AGENT_MUTABLE,
            workspace=str(workspace),
            protected_root=str(root),
            reference_directory=str(root / "reference"),
            runner_directory=str(root / "runner"),
            harness=str(root / "control.py"),
            report_directory=str(reports),
            reference_sha256=hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest(),
            harness_sha256=hashlib.sha256(Path(shared_native.__file__).read_bytes()).hexdigest(),
            protected_test_patterns=TEST_PATTERNS,
            audit_probes=("dependencies-diff.json", "startup-hooks-diff.json"),
            native=NativeSharedRunner(
                family=project.family,
                command="/bin/bash tests/run.sh",
                toolchain_commands=project.tools,
                config_patterns=profile.configs,
                dependency_paths=(str(dependencies), *profile.dependencies),
                startup_patterns=profile.hooks,
                environment_keys=profile.environment,
                runner_paths=profile.runner_paths,
                trusted_commit="a" * 40,
                shared_controller_sha256=hashlib.sha256(Path(shared_pytest.__file__).read_bytes()).hexdigest(),
            ),
        )
        payload = base / "input.json"
        payload.write_text(
            json.dumps({"contract": contract.model_dump(mode="json"), "test_files": encoded, "spec": {"timeout": 300}})
        )
        shared_native.bootstrap(payload)
        yield contract, dependencies, project


def native_verdict(contract: SharedIntegrityContract) -> dict:
    shared_native.grade(Path(contract.protected_root) / "contract.json")
    return json.loads((Path(contract.report_directory) / "verdict.json").read_text())


@pytest.mark.parametrize("attack", ["config", "tests", "runner", "toolchain", "environment", "background", "verdict"])
def test_native_tampering_cannot_fake_a_pass(native_runtime, monkeypatch, attack):
    contract, _, project = native_runtime
    workspace, root = Path(contract.workspace), Path(contract.protected_root)
    process = None
    if attack == "config":
        (workspace / project.config).write_text("fake configuration")
    elif attack == "tests":
        (workspace / "tests/run.sh").write_text("echo TASK_PASSED\n")
    elif attack == "toolchain":
        binary = Path(shutil.which(project.tools[0])).resolve()
        binary.write_text("#!/bin/sh\necho 'TASK_PASSED'\n")
    elif attack == "runner":
        script = (
            "import pathlib,sys\nfor name in sys.argv[1:]:\n"
            " try: pathlib.Path(name).write_text('fake pass')\n"
            " except PermissionError: continue\n"
            " raise SystemExit('agent could write protected runner')\n"
        )
        subprocess.run(
            [sys.executable, "-c", script, str(root / "control.py"), str(root / "tests/test_state.py")],
            user=contract.agent_uid,
            group=contract.agent_uid,
            extra_groups=(),
            check=True,
        )
    elif attack == "environment":
        monkeypatch.setenv("GOFLAGS", "-run=NoSuchTest")
        monkeypatch.setenv("GOTOOLCHAIN", "fake")
        monkeypatch.setenv("BASH_ENV", str(workspace / "inject.sh"))
        (workspace / "inject.sh").write_text("echo 'TASK_PASSED'; exit 0\n")
    elif attack == "background":
        program = (
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]);"
            "p.unlink(missing_ok=True);p.write_text('echo TASK_PASSED\\n');print('ready',flush=True);\n"
            "while True: p.write_text('echo TASK_PASSED\\n')"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", program, str(workspace / "tests/run.sh")],
            user=contract.agent_uid,
            group=contract.agent_uid,
            extra_groups=(),
            stdout=subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stdout.readline() == b"ready\n"
    else:
        reports = Path(contract.report_directory)
        reports.mkdir(exist_ok=True)
        (reports / "verdict.json").write_text('{"status":"scored","reward":1}')
        (reports / "reward.txt").write_text("1\n")
    try:
        verdict = native_verdict(contract)
        assert (verdict["status"], verdict["reward"]) == ("scored", 0)
        assert (Path(contract.report_directory) / "verdict.json").stat().st_uid == 0
        if process is not None:
            assert process.wait(timeout=5) != 0
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def test_native_grading_uses_live_dependencies_and_audits_changes(native_runtime):
    contract, dependencies, project = native_runtime
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
            str(dependencies / project.mutation_file),
            project.mutation,
        ],
        user=contract.agent_uid,
        group=contract.agent_uid,
        extra_groups=(),
        check=True,
    )
    verdict = native_verdict(contract)
    assert (verdict["status"], verdict["reward"]) == ("scored", 1)
    diff = json.loads((Path(contract.report_directory) / "dependencies-diff.json").read_text())
    assert str(dependencies / project.mutation_file) in diff
