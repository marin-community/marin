# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SHARED controller preserves live dependencies while rejecting harness tampering."""

import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest
from verifyit.spec import PytestSpec, render_spec

from taskcompendium.harbor.shared_package import SharedPytestFormat, render_shared_pytest_spec
from taskcompendium.harbor.shared_reference import (
    REFERENCE_BUNDLE,
    REFERENCE_INPUT,
    ActorEnvironment,
    FrozenSharedReference,
    actor_environment_for,
    harden_shared_swe_reference,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.integrity import SharedDependencyState, SharedIntegrityContract
from taskcompendium.runtime import shared_pytest, shared_reference

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0, reason="UID isolation requires Linux root"),
]


@dataclass(frozen=True)
class SharedRuntime:
    contract: SharedIntegrityContract
    site: Path
    verifyit: Path
    candidate: Path
    plugin: Path
    actor: ActorEnvironment | None = None

    def grade(self) -> dict:
        try:
            shared_pytest.grade(Path(self.contract.protected_root) / "contract.json", str(self.verifyit))
        except RuntimeError as error:
            stderr = Path(self.contract.report_directory) / "verifyit.stderr"
            if stderr.is_file():
                raise RuntimeError(f"{error}\n{stderr.read_text()}") from error
            raise
        return json.loads((Path(self.contract.report_directory) / "verdict.json").read_text())


@pytest.fixture(params=["base"])
def shared_runtime(request, monkeypatch):
    with tempfile.TemporaryDirectory(prefix="taskcompendium-shared-test-") as temporary:
        base = Path(temporary)
        base.chmod(0o755)
        workspace, root, reports = (base / name for name in ("workspace", "protected", "reports"))
        workspace.mkdir()
        candidate = workspace / "candidate.py"
        candidate.write_text("VALUE = 0\n")
        if request.param == "site-builtins":
            candidate.write_text("class CandidateHelp(type(help)):\n    pass\nVALUE = 0\n")
        (workspace / "extra_guard.py").write_text("reference\n")
        (workspace / "editable.py").write_text("original\n")
        (workspace / "conftest.py").write_text("pytest_plugins: tuple[str, ...]\n")
        (workspace / "test_case.py").write_text(
            "import candidate, dependency\ndef test_value(): assert candidate.VALUE + dependency.VALUE == 1\n"
        )
        if request.param == "child-interpreter":
            (workspace / "test_case.py").write_text(
                "import candidate, subprocess, sys\n"
                "def test_value():\n"
                "    value = subprocess.check_output([sys.executable, '-c', "
                "'import dependency; print(dependency.VALUE)'], text=True)\n"
                "    assert candidate.VALUE + int(value) == 1\n"
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
        task_python = workspace / "runtime" if request.param == "workspace-plugin-interpreter" else base / "task-python"
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
        plugin_path = site / "baseline_plugin.py"
        plugin_path.write_text("# Source-required pytest plugin\n")
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
            if request.param == "workspace-plugin-interpreter":
                plugin_path.unlink()
                plugin_path = site / "baseline_plugin/__init__.py"
                plugin_path.parent.mkdir()
                plugin_path.write_text("SENTINEL = 1\n")
                (metadata / "RECORD").write_text(
                    (metadata / "RECORD").read_text().replace("baseline_plugin.py", "baseline_plugin/__init__.py")
                )
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
            plugin_path = package / "a.py"
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
            reference_sha256=hashlib.sha256(
                json.dumps(
                    {"trusted.txt": b"test_case.py\n".hex()} if request.param == "source-pinned-cli" else {},
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            harness_sha256=hashlib.sha256(Path(shared_pytest.__file__).read_bytes()).hexdigest(),
            protected_test_patterns=("**/test_*.py", "**/conftest.py", "extra_guard.py"),
        )
        spec = PytestSpec(
            paths=(test_path,),
            candidate_modules=() if request.param == "source-pinned-cli" else ("candidate",),
            protected_paths_files=("trusted.txt",) if request.param == "source-pinned-cli" else (),
            must_pass=(f"{test_path}::test_value",),
            python=str(task_python / "bin/python"),
            workspace=str(workspace),
        )
        pinned_source = (
            request.config.getoption("--shared-verifyit-source") if request.param == "source-pinned-cli" else None
        )
        protected_spec = replace(spec, python=str(root / "pytest-python"))
        payload = {
            "contract": contract.model_dump(mode="json"),
            "spec": asdict(spec),
            "test_files": {"trusted.txt": b"test_case.py\n".hex()} if request.param == "source-pinned-cli" else {},
            "verifier_spec": (
                render_shared_pytest_spec(protected_spec, SharedPytestFormat.SINGLE_PROTECTED_MANIFEST)
                if pinned_source is not None
                else render_spec(protected_spec)
            ),
        }
        input_path = base / "input.json"
        input_path.write_text(json.dumps(payload))
        shared_pytest.bootstrap(input_path)
        verifyit = base / "verifyit"
        verifyit_main = "from verifyit.grade import main; raise SystemExit(main())"
        if pinned_source is not None:
            verifyit_main = f"import sys; sys.path.insert(0, {str(pinned_source.resolve())!r}); " + verifyit_main
        verifyit.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c " + shlex.quote(verifyit_main) + ' "$@"\n'
        )
        verifyit.chmod(0o755)
        actor = None
        if request.param.startswith("frozen-reference"):
            if request.param == "frozen-reference-dual-site":
                extra_site = base / "extra-site"
                extra_site.mkdir()
                (extra_site / "second_baseline.py").write_text("VALUE = 72\n")
                runtime_path = root / "runtime.json"
                captured_runtime = json.loads(runtime_path.read_text())
                captured_runtime["sites"].append(str(extra_site))
                runtime_path.write_text(json.dumps(captured_runtime))
            actor = actor_environment_for(contract, prefix=str(base / "agent-prefix"))
            files = {
                "instruction.md": b"Keep the original clone and installation in the actor phase.\n",
                "environment/Dockerfile": b"FROM unchanged-source-image\n",
                "task.toml": b"[metadata]\n[environment]\n[agent]\ntimeout_sec=900\n[verifier]\ntimeout_sec=600\n",
                "solution/solve.sh": b"#!/bin/bash\n# original oracle bytes\n",
            }
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                for path in sorted(root.iterdir()):
                    archive.add(path, arcname=path.name)
            bundle = stream.getvalue()
            reference = FrozenSharedReference(
                original_files_sha256=canonical_sha256({k: v.hex() for k, v in files.items()}),
                original_task_sha256="1" * 64,
                original_runtime_sha256="2" * 64,
                source_spec_sha256=canonical_sha256(asdict(spec)),
                image_context_sha256=canonical_sha256(
                    {k: v.hex() for k, v in files.items() if k.startswith("environment/")}
                ),
                source_cli_revision="a" * 40,
                capture_trial_uri="fixture://trusted-original-control",
                capture_trial_sha256="3" * 64,
                bundle_sha256=hashlib.sha256(bundle).hexdigest(),
                contract=contract,
                actor_environment=actor,
            )
            harden_shared_swe_reference(
                files,
                spec=spec,
                reference=reference,
                bundle=bundle,
                pytest_format=SharedPytestFormat.CURRENT,
                host_init_timeout=259200,
            )
            workspace.rename(base / "captured-workspace")
            root.rename(base / "captured-protected")
            workspace.mkdir()
            input_path.write_bytes(files[REFERENCE_INPUT])
            bundle_path = base / "reference.tar.gz"
            bundle_path.write_bytes(files[REFERENCE_BUNDLE])
            monkeypatch.setenv("PATH", str(base) + ":" + os.environ["PATH"])
            shared_reference.initialize(input_path, bundle_path)
            assert list(workspace.iterdir()) == []
            candidate.write_text("VALUE = 0\n")
        yield SharedRuntime(contract, site, verifyit, candidate, plugin_path, actor)


@pytest.mark.parametrize("shared_runtime", ["source-pinned-cli"], indirect=True)
def test_shared_grading_source_cli_rejects_broken_product_and_accepts_repair(shared_runtime):
    runtime = shared_runtime
    (runtime.candidate.parent / "test_case.py").write_text("def test_value(): pass\n")
    verdict = runtime.grade()
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == 0, verdict
    write_observable(runtime.candidate, "VALUE = 1\n")
    verdict = runtime.grade()
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == 1, verdict


@pytest.mark.parametrize("shared_runtime", ["frozen-reference"], indirect=True)
def test_frozen_reference_restores_grading_without_preparing_actor_workspace(shared_runtime):
    runtime = shared_runtime
    assert runtime.grade()["reward"] == 0
    write_observable(runtime.candidate, "VALUE = 1\n")
    assert runtime.grade()["reward"] == 1
    assert os.stat(runtime.contract.protected_root).st_uid == 0
    assert os.stat(runtime.contract.protected_root).st_mode & 0o007 == 0


def agent_command(runtime, code: str) -> subprocess.CompletedProcess:
    agent = runtime.contract.agent_uid
    return subprocess.run([sys.executable, "-c", code], user=agent, group=agent, capture_output=True, text=True)


def write_observable(path: Path, text: str) -> None:
    """Write product text and advance mtime so a same-second pyc cannot shadow it at grading."""
    path.write_text(text)
    stamped = time.time() + 5
    os.utime(path, (stamped, stamped))


@pytest.mark.parametrize("shared_runtime", ["frozen-reference"], indirect=True)
def test_frozen_reference_materializes_declared_actor_environment(shared_runtime):
    runtime = shared_runtime
    actor = runtime.actor
    venv = Path(actor.python)
    info = os.stat(actor.prefix)
    assert info.st_uid == actor.uid == runtime.contract.agent_uid
    assert info.st_gid == actor.gid
    assert info.st_mode & 0o777 == actor.directory_mode
    assert (venv / "bin" / "pip").read_text() == actor.pip_shim
    actor_python = venv / "bin" / "python"
    source_python = (
        Path(json.loads((Path(runtime.contract.protected_root) / "runtime.json").read_text())["sites"][0]).parents[2]
        / "bin/python"
    )
    assert (
        hashlib.sha256(actor_python.read_bytes()).hexdigest() == hashlib.sha256(source_python.read_bytes()).hexdigest()
    )
    probe = json.loads(
        subprocess.check_output(
            [str(actor_python), "-I", "-c", "import json,sys;print(json.dumps({'prefix':sys.prefix,'path':sys.path}))"],
            text=True,
        )
    )
    assert probe["prefix"] == str(venv)
    runtime_data = json.loads((Path(runtime.contract.protected_root) / "runtime.json").read_text())
    for site in runtime_data["sites"]:
        assert site in probe["path"]
    actor_sites = json.loads(
        subprocess.check_output(
            [
                str(actor_python),
                "-I",
                "-c",
                "import json,sysconfig;"
                "print(json.dumps(sorted({sysconfig.get_path('purelib'),sysconfig.get_path('platlib')})))",
            ],
            text=True,
        )
    )
    for site in actor_sites:
        assert site in runtime_data["sites"]


@pytest.mark.parametrize("shared_runtime", ["frozen-reference"], indirect=True)
def test_agent_installs_land_in_declared_environment_not_root_sites(shared_runtime):
    runtime = shared_runtime
    actor = runtime.actor
    # Repaired: the agent-owned bin accepts writes, so task-time console entrypoints can land there.
    repaired = agent_command(runtime, f"open({str(Path(actor.python) / 'bin' / 'probe')!r}, 'w').write('')")
    assert repaired.returncode == 0, repaired.stderr
    # Broken: the root-owned source task python stays protected — the sealed Errno 13 world.
    source_python = Path(runtime.site).parents[2] / "bin/python"
    broken = agent_command(runtime, f"open({str(source_python.parents[1] / 'bin' / 'probe')!r}, 'w').write('')")
    assert broken.returncode != 0
    assert not os.stat(source_python.parents[1] / "bin").st_mode & 0o022


@pytest.mark.parametrize("shared_runtime", ["frozen-reference"], indirect=True)
def test_grading_observes_agent_environment_installs(shared_runtime):
    runtime = shared_runtime
    actor_python = Path(runtime.actor.python) / "bin" / "python"
    site = subprocess.check_output(
        [str(actor_python), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    install = Path(site) / "dependency.py"
    write = agent_command(runtime, f"open({str(install)!r}, 'w').write('VALUE = 1\\n')")
    assert write.returncode == 0, write.stderr
    observed = subprocess.check_output(
        [str(actor_python), "-I", "-c", "import dependency;print(dependency.VALUE)"], text=True
    )
    assert observed.strip() == "1"
    verdict = runtime.grade()
    diff = json.loads((Path(runtime.contract.report_directory) / "site-packages-diff.json").read_text())
    assert diff[str(install)]["after"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == 1


@pytest.mark.parametrize("shared_runtime", ["frozen-reference"], indirect=True)
def test_frozen_reference_keeps_root_executables_protected_during_editable_console_install(shared_runtime):
    runtime = shared_runtime
    workspace = runtime.candidate.parent
    source_python = runtime.site.parents[2] / "bin/python"
    (workspace / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='source-entrypoint-fixture',version='0.0',py_modules=['candidate'],"
        "entry_points={'console_scripts':['source-fixture-cli=candidate:main']})\n"
    )
    environment = shared_pytest.clean_environment()
    ready = subprocess.run(
        [str(source_python), "-c", "import setuptools, pip, wheel"],
        cwd=workspace,
        env=environment,
        capture_output=True,
        user=runtime.contract.agent_uid,
        group=runtime.contract.agent_uid,
        extra_groups=[],
    )
    assert ready.returncode == 0, ready.stderr.decode()
    result = subprocess.run(
        [str(source_python), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", "."],
        cwd=workspace,
        env=environment,
        capture_output=True,
        user=runtime.contract.agent_uid,
        group=runtime.contract.agent_uid,
        extra_groups=[],
    )
    assert result.returncode != 0
    assert not (source_python.parent / "source-fixture-cli").exists()
    assert not os.stat(source_python.parent).st_mode & 0o022


@pytest.mark.parametrize(
    "shared_runtime",
    [
        "workspace-plugin-editable",
        "workspace-plugin-wheel",
        "workspace-plugin-dependency",
        "workspace-plugin-namespace",
        "workspace-plugin-interpreter",
    ],
    indirect=True,
)
def test_shared_grading_keeps_repository_product_live_and_third_party_plugin_pinned(shared_runtime):
    runtime = shared_runtime
    candidate = runtime.candidate
    candidate.write_text("VALUE = 1\n")
    runtime.plugin.write_text("SENTINEL = 0\n")
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


@pytest.mark.parametrize("shared_runtime", ["site-builtins"], indirect=True)
def test_shared_grading_keeps_standard_site_builtins_available_to_product(shared_runtime):
    runtime = shared_runtime
    verdict = runtime.grade()
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == 0
    write_observable(runtime.candidate, runtime.candidate.read_text().replace("VALUE = 0", "VALUE = 1"))
    verdict = runtime.grade()
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == 1


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


@pytest.mark.parametrize("shared_runtime", ["child-interpreter"], indirect=True)
def test_shared_grading_product_subprocess_sees_live_source_dependencies(shared_runtime):
    runtime = shared_runtime
    assert runtime.grade()["reward"] == 0
    (runtime.site / "dependency.py").write_text("VALUE = 1\n")
    verdict = runtime.grade()
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == 1, verdict
    changes = json.loads((Path(runtime.contract.report_directory) / "site-packages-diff.json").read_text())
    assert str(runtime.site / "dependency.py") in changes


def test_shared_controller_restores_contract_paths_and_preserves_other_files(shared_runtime):
    runtime = shared_runtime
    workspace = Path(runtime.contract.workspace)
    (workspace / "candidate.py").write_text("VALUE = 1\n")
    (workspace / "extra_guard.py").write_text("tampered\n")
    (workspace / "editable.py").write_text("agent change\n")
    reports = Path(runtime.contract.report_directory)
    reports.mkdir(exist_ok=True)
    reports.chmod(0o777)
    baseline = Path(runtime.contract.protected_root) / "workspace-owned-baseline.json"
    before = baseline.read_bytes()
    planted = agent_command(
        runtime,
        "from pathlib import Path;"
        f"Path({str(reports / 'site-packages-diff.json')!r}).symlink_to({str(baseline)!r});"
        f"Path({str(reports / 'verifyit.stdout')!r}).symlink_to({str(reports / 'reward.json')!r})",
    )
    assert planted.returncode == 0, planted.stderr
    assert runtime.grade()["reward"] == 1
    assert (workspace / "extra_guard.py").read_text() == "reference\n"
    assert (workspace / "editable.py").read_text() == "agent change\n"
    assert baseline.read_bytes() == before


@pytest.mark.parametrize("shared_runtime", ["frozen-reference-dual-site"], indirect=True)
def test_frozen_reference_exposes_all_captured_source_sites(shared_runtime):
    python = Path(shared_runtime.actor.python) / "bin/python"
    observed = subprocess.check_output(
        [str(python), "-I", "-c", "import second_baseline;print(second_baseline.VALUE)"], text=True
    )
    assert observed.strip() == "72"
    assert shared_runtime.grade()["reward"] == 0


@pytest.mark.parametrize("destination", ["external", "protected"])
def test_shared_cleanup_preserves_files_behind_actor_symlink(shared_runtime, destination):
    runtime = shared_runtime
    workspace = Path(runtime.contract.workspace)
    target = (
        Path(runtime.contract.protected_root) / "hidden-extra"
        if destination == "protected"
        else workspace.parent / "external"
    )
    target.mkdir(exist_ok=True)
    victim = target / "test_untouched.py"
    victim.write_text("private material\n")
    attack = agent_command(
        runtime, f"from pathlib import Path;Path({str(workspace / 'tests')!r}).symlink_to({str(target)!r})"
    )
    assert attack.returncode == 0, attack.stderr
    contract = runtime.contract.model_dump(mode="json")
    contract["protected_test_patterns"] = ["**/tests/**/*", *contract["protected_test_patterns"]]
    shared_pytest.prepare_grading(contract)
    assert victim.read_text() == "private material\n"
    assert not (workspace / "tests").is_symlink()
    assert runtime.grade()["reward"] == 0


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
    write_observable(workspace / "candidate.py", "VALUE = 1\n")
    verdict = runtime.grade()
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == 1
