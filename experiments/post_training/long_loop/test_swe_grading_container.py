# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the trusted grader against branch, plugin and package-shadow attacks."""

import base64
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from experiments.post_training.long_loop.swe_grading import TaskPins
from experiments.post_training.long_loop.swe_verified_data import RUNTIME
from experiments.post_training.task_curation.swesmith_staged import DockerPilot

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.docker
@pytest.mark.timeout(600)
def test_trusted_grading_ignores_mutable_agent_packages_and_pytest_hooks(tmp_path):
    context = tmp_path / "image"
    context.mkdir()
    shutil.copyfile(RUNTIME, context / "grading.py")
    shutil.copytree(REPOSITORY_ROOT / "lib/verifyit", context / "verifyit", ignore=shutil.ignore_patterns("__pycache__"))
    repository = context / "fixture"
    repository.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (repository / "solution.py").write_text("VALUE = 1\n")
    (repository / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='fixture',version='1',"
        "py_modules=['solution','fixture_cli','generated_version'],"
        "entry_points={'console_scripts':['fixture-value=fixture_cli:main']})\n"
    )
    (repository / "fixture_cli.py").write_text(
        "import json,sys\nfrom solution import VALUE\n"
        "def main(): print(json.dumps({'value':VALUE,'prefix':sys.prefix}))\n"
    )
    (repository / "noxfile.py").write_text("import unavailable_build_tool\n")
    (repository / ".gitignore").write_text("generated_version.py\n")
    (repository / "tests").mkdir()
    (repository / "tests/test_value.py").write_text(
        "import os,sys,json,subprocess,pathlib,time\nfrom solution import VALUE\nfrom generated_version import VERSION\n"
        "def test_value():\n    if VALUE == 0: time.sleep(10)\n    assert VALUE == 1\n    assert os.getuid() == 10002\n"
        "    assert sys.prefix == '/opt/long-loop-trusted/python'\n"
        "    cli=json.loads(subprocess.check_output(['fixture-value'],text=True))\n"
        "    assert cli == {'value':1,'prefix':'/opt/long-loop-trusted/python'}\n"
        "    assert VERSION == 7\n"
        "    pathlib.Path('scratch.log').write_text('test scratch')\n"
        "    pathlib.Path('tests/generated-fixture.key').write_text('fixture')\n"
        "    for path in ['tests/test_value.py','setup.py']:\n"
        "        try: pathlib.Path(path).write_text('corrupted')\n"
        "        except PermissionError: pass\n"
        "        else: raise AssertionError('grader overwrote a root-owned file')\n"
    )
    git("add", ".")
    git("commit", "-qm", "reference")
    reference = git("rev-parse", "HEAD")
    (repository / "solution.py").write_text("VALUE = 0\n")
    git("commit", "-qam", "bug")
    bug = git("rev-parse", "HEAD")
    (context / "Dockerfile").write_text(
        "FROM python:3.12-slim-bookworm\n"
        "RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*\n"
        "COPY verifyit /opt/verifyit-source\n"
        "RUN python -m venv /opt/verifyit-control && /opt/verifyit-control/bin/pip install /opt/verifyit-source "
        "&& ln -s /opt/verifyit-control/bin/verifyit /usr/local/bin/verifyit "
        "&& pip install setuptools pytest pytest-json-report\n"
        "RUN useradd -m -U -u 10001 longloop-agent && useradd -m -U -u 10002 longloop-grade "
        "&& chgrp longloop-agent /usr/local/bin && chmod g+w,+t /usr/local/bin\n"
        "COPY grading.py /opt/long-loop-swe/grading.py\nCOPY fixture /fixture\nRUN mkdir /testbed\nWORKDIR /\n"
    )
    pilot = DockerPilot(tmp_path)
    image = (
        pilot.command(["docker", "build", "--quiet", "--platform", "linux/amd64", str(context)], timeout=450)
        .stdout.decode()
        .strip()
    )
    container = (
        pilot.command(["docker", "run", "-d", "--platform", "linux/amd64", "--entrypoint", "sleep", image, "infinity"])
        .stdout.decode()
        .strip()
    )
    pins = TaskPins(
        "/fixture",
        bug,
        reference,
        "task-bug",
        "printf 'VERSION = 7\\n' > generated_version.py\npython -m pip install --no-build-isolation --no-deps -e .",
        "",
        60,
        3,
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "trusted_test_paths.txt").write_text("tests\n")
    (tests / "verifier.toml").write_text(
        'mode="pytest"\npaths=["tests/test_value.py"]\nmust_pass=["tests/test_value.py::test_value"]\n'
        'python="/opt/long-loop-trusted/grade-python"\nworkspace="/testbed"\ntimeout=3\nsetup=""\n'
        'protected_paths_files=["trusted_test_paths.txt"]\n'
    )

    def grade(name):
        pilot.command(
            [
                "docker",
                "exec",
                container,
                "/opt/long-loop-control-bin/python3",
                "-I",
                "-S",
                "/opt/long-loop-swe/grading.py",
                "grade",
                "--tests",
                "/tests",
                "--logs",
                f"/logs/{name}",
            ]
        )
        return json.loads(pilot.command(["docker", "exec", container, "cat", f"/logs/{name}/verdict.json"]).stdout)

    try:
        payload = base64.b64encode(json.dumps(asdict(pins)).encode()).decode()
        pilot.command(
            ["docker", "exec", container, "python3", "-I", "-S", "/opt/long-loop-swe/grading.py", "bootstrap", payload]
        )
        pilot.command(["docker", "cp", str(tests), f"{container}:/tests"])
        inject_commands = (
            "import pathlib; "
            "payload='#!/bin/sh\\nid -u > /root/agent-command-executed\\nexit 0\\n'; "
            "[(p.write_text(payload),p.chmod(0o755)) "
            "for p in [pathlib.Path('/usr/local/bin')/n for n in ('git','tar','setpriv')]]"
        )
        pilot.command(["docker", "exec", "-u", "10001", container, "python3", "-c", inject_commands])
        baseline = grade("bug")
        assert baseline["reward"] == 0 and baseline["detail"]["reason"] == "timeout"
        forge = (
            "import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_path('purelib')); "
            "(p/'solution.py').write_text('VALUE=1\\n'); "
            "(p/'forged_plugin.py').write_text('raise RuntimeError(\"agent plugin loaded\")\\n'); "
            "d=p/'forged-1.dist-info'; d.mkdir(); "
            "(d/'METADATA').write_text('Name: forged\\nVersion: 1\\n'); "
            "(d/'entry_points.txt').write_text('[pytest11]\\nforged=forged_plugin\\n'); "
            "pathlib.Path('/testbed/conftest.py').write_text('raise RuntimeError(\"agent hook loaded\")\\n')"
        )
        pilot.command(["docker", "exec", "-u", "10001", container, "python3", "-c", forge])
        hacked = grade("forged")
        assert hacked["status"] == "scored" and hacked["reward"] == 0
        pilot.command(
            ["docker", "exec", "-u", "10001", container, "sh", "-c", "printf 'OTHER=1\\n' > /testbed/solution.py"]
        )
        missing_symbol = grade("candidate-collection-failure")
        assert missing_symbol["status"] == "scored" and missing_symbol["reward"] == 0
        pilot.command(
            ["docker", "exec", "-u", "10001", container, "sh", "-c", "printf 'VALUE=1\\n' > /testbed/solution.py"]
        )
        assert grade("fixed")["reward"] == 1
        pilot.command(
            ["docker", "exec", "-u", "10001", container, "/usr/bin/git", "-C", "/testbed", "checkout", "-f", reference]
        )
        bypass = grade("wrong-branch")
        assert bypass["reward"] == 0 and bypass["detail"]["reason"] == "wrong_ancestry"
        marker = pilot.command(["docker", "exec", container, "test", "-e", "/root/agent-command-executed"], check=False)
        assert marker.returncode == 1
        baseline = json.loads(
            pilot.command(["docker", "exec", container, "cat", "/opt/long-loop-trusted/dependencies.json"]).stdout
        )
        lock = "".join(f"{name}=={version}\n" for name, version in baseline["packages"])
        pilot.command(["docker", "rm", "-f", container])
        container = (
            pilot.command(
                ["docker", "run", "-d", "--platform", "linux/amd64", "--entrypoint", "sleep", image, "infinity"]
            )
            .stdout.decode()
            .strip()
        )
        payload = base64.b64encode(json.dumps(asdict(replace(pins, dependency_constraints=lock))).encode()).decode()
        pilot.command(
            ["docker", "exec", container, "python3", "-I", "-S", "/opt/long-loop-swe/grading.py", "bootstrap", payload]
        )
        pilot.command(["docker", "cp", str(tests), f"{container}:/tests"])
        assert grade("locked-bug")["reward"] == 0
        pilot.command(
            ["docker", "exec", "-u", "10001", container, "sh", "-c", "printf 'VALUE=1\\n' > /testbed/solution.py"]
        )
        assert grade("locked-fix")["reward"] == 1
    finally:
        pilot.command(["docker", "rm", "-f", container], check=False)
