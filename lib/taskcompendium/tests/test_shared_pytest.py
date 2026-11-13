# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SHARED controller preserves live dependencies while rejecting harness tampering."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
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
    candidate: Path

    def grade(self) -> dict:
        shared_pytest.grade(Path(self.contract.protected_root) / "contract.json", str(self.verifyit))
        return json.loads((Path(self.contract.report_directory) / "verdict.json").read_text())


@pytest.fixture(params=["base"])
def shared_runtime(request):
    with tempfile.TemporaryDirectory(prefix="taskcompendium-shared-test-") as temporary:
        base = Path(temporary)
        base.chmod(0o755)
        workspace, root, reports = (base / name for name in ("workspace", "protected", "reports"))
        workspace.mkdir()
        candidate = workspace / "candidate.py"
        candidate.write_text("VALUE = 0\n")
        (workspace / "extra_guard.py").write_text("reference\n")
        (workspace / "editable.py").write_text("original\n")
        (workspace / "conftest.py").write_text("pytest_plugins: tuple[str, ...]\n")
        (workspace / "test_case.py").write_text(
            "import candidate, dependency\ndef test_value(): assert candidate.VALUE + dependency.VALUE == 1\n"
        )
        test_path = "test_case.py"
        if request.param in {"entrypoints", "entrypoint-block", "entrypoint-required"}:
            blocked = request.param == "entrypoint-block"
            (workspace / "pytest.ini").write_text(
                "[pytest]\nfilterwarnings = error\n"
                + ("required_plugins = baseline-plugin\n" if request.param == "entrypoint-required" else "")
                + "addopts = --baseline-enabled"
                + (" -p no:baseline-z" if blocked else "")
                + "\n"
            )
            (workspace / "test_case.py").write_text(
                "import candidate\ndef test_value(source_value, pytestconfig):\n"
                "    assert candidate.VALUE == source_value\n"
                "    assert pytestconfig.pluginmanager.hasplugin('baseline')\n"
                f"    assert pytestconfig.pluginmanager.hasplugin('baseline-z') == {not blocked}\n"
            )
        elif request.param == "local-plugin":
            plugin = workspace / "tests/local_fixture/plugin.py"
            plugin.parent.mkdir(parents=True)
            plugin.write_text("import pytest\n@pytest.fixture\ndef source_value(): return 1\n")
            (workspace / "tests/conftest.py").write_text('pytest_plugins = ["local_fixture.plugin"]\n')
            test_path = "tests/test_case.py"
            (workspace / test_path).write_text(
                "import candidate\ndef test_value(source_value): assert candidate.VALUE == source_value\n"
            )
        elif request.param in {"pyproject.toml", "setup.cfg", "pytest.ini"}:
            (workspace / "extras").mkdir()
            (workspace / "extras/source_dependency.py").write_text("VALUE = 1\n")
            (workspace / "test_case.py").write_text(
                "import candidate, source_dependency\n"
                "def test_value(): assert candidate.VALUE == source_dependency.VALUE\n"
            )
            if request.param == "pyproject.toml":
                configuration = '[tool.pytest.ini_options]\npythonpath=["extras"]\n'
            elif request.param == "setup.cfg":
                configuration = "[tool:pytest]\npythonpath = extras\n"
            else:
                configuration = "[pytest]\npythonpath = extras\n"
            (workspace / request.param).write_text(configuration)
        task_python = base / "task-python"
        source_python = request.config.getoption("--shared-source-python") or sys.executable
        subprocess.run(
            [Path(source_python).resolve(), "-m", "venv", "--copies", "--without-pip", str(task_python)], check=True
        )
        site = Path(
            subprocess.check_output(
                [str(task_python / "bin/python"), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        source_site = subprocess.check_output(
            [source_python, "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
        (site / "test-dependencies.pth").write_text(source_site + "\n")
        (site / "dependency.py").write_text("VALUE = 0\n")
        (site / "baseline_plugin.py").write_text("# Source-required pytest plugin\n")
        metadata = site / "baseline_plugin-1.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: baseline-plugin\nVersion: 1\n")
        (metadata / "entry_points.txt").write_text("[pytest11]\nbaseline = baseline_plugin\n")
        (metadata / "RECORD").write_text(
            "baseline_plugin.py,,\nbaseline_plugin-1.dist-info/METADATA,,\n"
            "baseline_plugin-1.dist-info/entry_points.txt,,\nbaseline_plugin-1.dist-info/RECORD,,\n"
        )
        if request.param.startswith("workspace-plugin-"):
            (workspace / "candidate.py").unlink()
            package = workspace / "candidate"
            package.mkdir()
            namespace = request.param == "workspace-plugin-namespace"
            candidate = package / ("value.py" if namespace else "__init__.py")
            candidate.write_text("VALUE = 0\n")
            (package / "plugin.py").write_text("# Repository-owned pytest plugin\n")
            (workspace / "test_case.py").write_text(
                (
                    "import candidate.value as candidate\nimport baseline_plugin\n"
                    if namespace
                    else "import candidate, baseline_plugin\n"
                )
                + "def test_value():\n"
                "    assert candidate.VALUE == 1, candidate.__file__\n"
                "    assert baseline_plugin.SENTINEL == 1\n"
            )
            (site / "baseline_plugin.py").write_text("SENTINEL = 1\n")
            owner = workspace if request.param == "workspace-plugin-editable" else site
            product_metadata = owner / "candidate_product-1.dist-info"
            product_metadata.mkdir()
            (product_metadata / "METADATA").write_text("Name: candidate-product\nVersion: 1\n")
            (product_metadata / "top_level.txt").write_text("candidate\n")
            (product_metadata / "RECORD").write_text(
                f"candidate/{candidate.name},,\ncandidate/plugin.py,,\n"
                "candidate_product-1.dist-info/METADATA,,\ncandidate_product-1.dist-info/RECORD,,\n"
            )
            if owner == workspace:
                (site / "source-product.pth").write_text(str(workspace) + "\n")
            else:
                shutil.copytree(package, site / "candidate")
            if request.param == "workspace-plugin-dependency":
                (metadata / "METADATA").write_text(
                    "Name: baseline-plugin\nVersion: 1\nRequires-Dist: candidate-product\n"
                )
            else:
                (product_metadata / "entry_points.txt").write_text("[pytest11]\nproduct = candidate.plugin\n")
        if request.param in {"entrypoints", "entrypoint-block", "entrypoint-required"}:
            (site / "baseline_plugin.py").unlink()
            package = site / "baseline_plugin"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "a.py").write_text(
                "import baseline_plugin.z, pytest\n"
                "def pytest_addoption(parser): parser.addoption('--baseline-enabled', action='store_true')\n"
                "@pytest.fixture\ndef source_value(): return 1\n"
            )
            (package / "z.py").write_text("VALUE = 1\n")
            (metadata / "entry_points.txt").write_text(
                "[pytest11]\nbaseline = baseline_plugin.a\nbaseline-z = baseline_plugin.z\n"
            )
            (metadata / "RECORD").write_text(
                "baseline_plugin/__init__.py,,\nbaseline_plugin/a.py,,\nbaseline_plugin/z.py,,\n"
                "baseline_plugin-1.dist-info/METADATA,,\nbaseline_plugin-1.dist-info/entry_points.txt,,\n"
                "baseline_plugin-1.dist-info/RECORD,,\n"
            )
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
            paths=(test_path,),
            candidate_modules=("candidate",),
            must_pass=(f"{test_path}::test_value",),
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
        yield SharedRuntime(contract, site, verifyit, candidate)


@pytest.mark.parametrize(
    "shared_runtime",
    ["workspace-plugin-editable", "workspace-plugin-wheel", "workspace-plugin-dependency", "workspace-plugin-namespace"],
    indirect=True,
)
def test_shared_grading_keeps_repository_product_live_and_third_party_plugin_pinned(shared_runtime):
    runtime = shared_runtime
    candidate = runtime.candidate
    candidate.write_text("VALUE = 1\n")
    (runtime.site / "baseline_plugin.py").write_text("SENTINEL = 0\n")
    verdict = runtime.grade()
    assert verdict["reward"] == 1.0, verdict
    reports = Path(runtime.contract.report_directory)
    ownership = json.loads((reports / "workspace-owned-distributions.json").read_text())
    product = next(row for row in ownership["distributions"] if row["distribution"] == "candidate-product")
    assert product["grader_dependency"]
    changes = json.loads((reports / "workspace-owned-diff.json").read_text())
    assert changes[str(candidate)] == {
        "before": hashlib.sha256(b"VALUE = 0\n").hexdigest(),
        "after": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
    }


@pytest.mark.parametrize(
    "attack", ["conftest", "addopts", "autoload", "pinned-plugin", "pytest", "background", "stdout"]
)
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
    elif attack == "pinned-plugin":
        (runtime.site / "baseline_plugin.py").write_text(
            "import pytest\ndef pytest_collection_modifyitems(items):\n"
            " for item in items: item.add_marker(pytest.mark.skip)\n"
        )
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


@pytest.mark.parametrize(
    "shared_runtime",
    [
        "local-plugin",
        "pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "setup.cfg",
        "entrypoints",
        "entrypoint-block",
        "entrypoint-required",
    ],
    indirect=True,
)
def test_shared_grading_restores_source_plugin_and_config_semantics(shared_runtime):
    runtime = shared_runtime
    workspace = Path(runtime.contract.workspace)
    # An agent cannot replace source-owned configuration or a local fixture plugin.
    for name in ("pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"):
        path = workspace / name
        if path.exists():
            path.write_text("[pytest]\naddopts = --deselect=test_case.py::test_value\n")
    plugin = workspace / "tests/local_fixture/plugin.py"
    if plugin.exists():
        plugin.write_text("import pytest\n@pytest.fixture\ndef source_value(): return 0\n")
        (plugin.parent / "__init__.py").write_text("raise RuntimeError('agent package initializer')\n")
        (plugin.parent.with_suffix(".py")).write_text("raise RuntimeError('agent plugin shadow')\n")
    assert runtime.grade()["reward"] == 0
    (workspace / "candidate.py").write_text("VALUE = 1\n")
    verdict = runtime.grade()
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == 1
