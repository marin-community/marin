# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guest-side SWE bootstrap and trusted-worktree grading.

This file is copied into each original task image. It intentionally uses only
the standard library and runs with ``python -I -S``: the agent can install task
dependencies, but cannot replace the harness's imports or grading interpreter.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TRUST = Path("/opt/long-loop-trusted")
CONTROL_BIN = Path("/opt/long-loop-control-bin")
WORKSPACE = Path("/testbed")
AGENT_UID = 10001
GRADER_UID = 10002
PYTEST_CONFIGS = frozenset({"pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "conftest.py"})
SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class TaskPins:
    repository: str
    bug: str
    reference: str
    branch: str
    dependency_setup: str
    dependency_constraints: str
    setup_timeout: int
    verifier_timeout: int

    def __post_init__(self) -> None:
        if not SHA.fullmatch(self.bug) or not SHA.fullmatch(self.reference):
            raise ValueError("SWE bug and reference must be full commit SHAs")


def command(argv: list[str], *, cwd: Path, timeout: float, uid: int | None = None) -> subprocess.CompletedProcess:
    environment = {
        **os.environ,
        "PATH": f"{CONTROL_BIN}:/usr/bin:/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
    ):
        environment.pop(name, None)
    if uid is not None:
        argv = ["setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--no-new-privs", *argv]
        environment["HOME"] = f"/home/{'longloop-agent' if uid == AGENT_UID else 'longloop-grade'}"
    return subprocess.run(argv, cwd=cwd, env=environment, capture_output=True, timeout=timeout, check=False)


def git(directory: Path, *args: str, timeout: float = 120, uid: int | None = None) -> subprocess.CompletedProcess:
    return command(
        [
            "git",
            "-c",
            f"safe.directory={directory}",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *args,
        ],
        cwd=directory,
        timeout=timeout,
        uid=uid,
    )


def checked(result: subprocess.CompletedProcess) -> bytes:
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}: {result.stderr.decode(errors='replace')[-6000:]}")
    return result.stdout


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def protected_path(path: str, trusted_paths: list[str]) -> bool:
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or any(part in ("..", ".git") for part in parts):
        raise ValueError(f"Unsafe candidate path: {path}")
    if parts[-1] in PYTEST_CONFIGS:
        return True
    return any(path == item or path.startswith(item.rstrip("/") + "/") for item in trusted_paths)


def candidate_patch(workspace: Path, bug: str, trusted_paths: list[str], *, uid: int | None) -> tuple[dict, bytes]:
    """Check ancestry before extracting any patch; never execute agent Git as root."""
    head_result = git(workspace, "rev-parse", "HEAD", uid=uid)
    if head_result.returncode:
        return {"accepted": False, "reason": "missing_repository"}, b""
    head = head_result.stdout.decode().strip()
    ancestry = git(workspace, "merge-base", "--is-ancestor", bug, head, uid=uid)
    if ancestry.returncode:
        return {"accepted": False, "reason": "wrong_ancestry", "head": head, "bug": bug}, b""
    changed = checked(git(workspace, "diff", "--name-only", "-z", bug, "--", uid=uid)).decode().split("\0")
    untracked = checked(git(workspace, "ls-files", "--others", "--exclude-standard", "-z", uid=uid)).decode().split("\0")
    paths = sorted({p for p in (*changed, *untracked) if p and not protected_path(p, trusted_paths)})
    patch = b""
    if paths:
        patch = checked(git(workspace, "diff", "--binary", "--no-ext-diff", "--no-textconv", bug, "--", *paths, uid=uid))
    for path in untracked:
        if path and path in paths:
            result = git(workspace, "diff", "--no-index", "--binary", "--no-ext-diff", "/dev/null", path, uid=uid)
            if result.returncode != 1:
                raise RuntimeError(f"Cannot capture untracked candidate file {path}")
            patch += result.stdout
    if re.search(rb"^(?:new|old) mode 120000$", patch, re.MULTILINE) or b" 120000\n" in patch:
        return {"accepted": False, "reason": "candidate_symlink", "head": head}, b""
    return {
        "accepted": True,
        "head": head,
        "bug": bug,
        "paths": paths,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }, patch


def restore_paths(repo: Path, workspace: Path, pins: TaskPins, paths: list[str]) -> None:
    """Restore trusted tests and pytest configuration from pinned objects only."""
    tree = checked(git(repo, "ls-tree", "-r", "--name-only", pins.bug)).decode().splitlines()
    config_paths = [p for p in tree if PurePosixPath(p).name in PYTEST_CONFIGS]
    for path in sorted(set(paths + config_paths)):
        protected_path(path, [])  # Validate even a path originating in the trusted manifest.
        target = workspace / path
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()
        for revision in (pins.bug, pins.reference):
            if git(repo, "cat-file", "-e", f"{revision}:{path}").returncode:
                continue
            archive = checked(git(repo, "archive", revision, "--", path))
            # git archive's paths come from a trusted, pinned tree, not the agent.
            subprocess.run(["tar", "-xf", "-", "-C", str(workspace)], input=archive, check=True)
            break


def bootstrap(pins: TaskPins) -> None:
    """Materialize the task and freeze its declared dependencies before agent access."""
    started = time.time()
    phases = {}
    TRUST.mkdir(mode=0o700)
    CONTROL_BIN.mkdir(mode=0o755)
    CONTROL_BIN.chmod(0o755)
    executables = {}
    for name in ("git", "tar", "setpriv", "python3", "verifyit", "chown", "chmod", "chgrp"):
        executable = shutil.which(name)
        if executable is None:
            raise FileNotFoundError(f"Required task control executable is missing: {name}")
        original = Path(executable).resolve()
        if original.stat().st_mode & 0o022:
            raise ValueError(f"Task control executable is writable by a non-root user: {original}")
        (CONTROL_BIN / name).symlink_to(original)
        executables[name] = str(Path(executable).absolute())
    write_json(TRUST / "control-executables.json", executables)
    repo = TRUST / "repository.git"
    checked(command(["git", "init", "--bare", str(repo)], cwd=TRUST, timeout=120))
    checked(git(repo, "fetch", "--depth=2", pins.repository, pins.bug, pins.reference, timeout=pins.setup_timeout))
    for revision in (pins.bug, pins.reference):
        actual = checked(git(repo, "rev-parse", f"{revision}^{{commit}}")).decode().strip()
        if actual != revision:
            raise ValueError("Fetched SWE revision differs from the pinned commit")
    phases["repository_fetch"] = time.time() - started
    checkout_started = time.time()
    checked(git(repo, "update-ref", "refs/heads/task-bug", pins.bug))
    # The upstream images create an empty lower-layer /testbed. Recreate it
    # before the agent starts so overlayfs can later rename the workspace into
    # the private grading area without EXDEV or a full repository copy.
    if WORKSPACE.exists():
        WORKSPACE.rmdir()
    checked(
        command(
            ["git", "clone", "--no-local", "--single-branch", "--branch", "task-bug", str(repo), str(WORKSPACE)],
            cwd=TRUST,
            timeout=pins.setup_timeout,
        )
    )
    checked(git(WORKSPACE, "checkout", "-B", pins.branch, pins.bug))
    checked(git(WORKSPACE, "remote", "set-url", "origin", pins.repository))
    phases["clone_checkout"] = time.time() - checkout_started
    dependencies_started = time.time()
    if pins.dependency_constraints:
        constraint = TRUST / "dependency-constraints.txt"
        constraint.write_text(pins.dependency_constraints)
        os.environ["PIP_CONSTRAINT"] = str(constraint)
    # Upstream setup runs before agent access in the task's original environment.
    setup = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", pins.dependency_setup],
        cwd=WORKSPACE,
        capture_output=True,
        timeout=pins.setup_timeout,
        check=False,
    )
    (TRUST / "setup.stdout").write_bytes(setup.stdout)
    (TRUST / "setup.stderr").write_bytes(setup.stderr)
    checked(setup)
    phases["declared_dependency_install"] = time.time() - dependencies_started
    freeze_started = time.time()
    python = executables["python3"]
    info = json.loads(
        checked(
            command(
                [python, "-I", "-c", "import json,sysconfig; print(json.dumps(sysconfig.get_paths()))"],
                cwd=TRUST,
                timeout=30,
            )
        )
    )
    frozen = TRUST / "python"
    checked(command([python, "-I", "-m", "venv", "--without-pip", str(frozen)], cwd=TRUST, timeout=120))
    target = json.loads(
        checked(
            command(
                [
                    str(frozen / "bin/python"),
                    "-I",
                    "-c",
                    "import json,sysconfig; print(json.dumps(sysconfig.get_paths()))",
                ],
                cwd=TRUST,
                timeout=30,
            )
        )
    )
    for source, destination in {(info[k], target[k]) for k in ("purelib", "platlib")}:
        shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=False)
    # Console scripts must use the frozen interpreter too; retaining their
    # original shebang would re-enter agent-writable installed packages.
    for source in Path(info["scripts"]).iterdir():
        if not source.is_file() or not os.access(source, os.X_OK):
            continue
        with source.open("rb") as stream:
            first = stream.readline(4096)
            if not first.startswith(b"#!"):
                continue
            interpreter = first[2:].strip().split()[0].decode(errors="replace")
            if (
                not interpreter.startswith("/")
                or Path(interpreter).parent.resolve() != Path(python).parent.resolve()
                or Path(interpreter).resolve() != Path(python).resolve()
            ):
                continue
            destination = Path(target["scripts"]) / source.name
            destination.write_bytes(f"#!{frozen}/bin/python\n".encode() + stream.read())
            destination.chmod(0o755)
    inventory = (
        "import importlib.metadata as m,json,sys,pathlib; ds=list(m.distributions()); "
        "modules={n for d in ds "
        "if json.loads(d.read_text('direct_url.json') or '{}').get('url','').rstrip('/')=='file:///testbed' "
        "for n in (d.read_text('top_level.txt') or '').splitlines()}; "
        "roots=[pathlib.Path(p) for p in sys.path if p=='/testbed' or p.startswith('/testbed/')]; "
        "modules.update(p.parent.name for root in roots for p in root.glob('*/__init__.py')); "
        "modules.update(p.stem for root in roots for p in root.glob('*.py') if p.stem not in ('setup','conftest')); "
        "print(json.dumps({'packages':sorted((d.metadata.get('Name') or '',d.version) for d in ds),"
        "'plugins':sorted(e.value for e in m.entry_points(group='pytest11')),'candidate_modules':sorted(modules)}))"
    )
    packages = checked(command([str(frozen / "bin/python"), "-I", "-c", inventory], cwd=TRUST, timeout=60))
    (TRUST / "dependencies.json").write_bytes(packages)
    if pins.dependency_constraints:
        actual = "".join(f"{name}=={version}\n" for name, version in json.loads(packages)["packages"])
        if actual != pins.dependency_constraints:
            raise ValueError("Trusted dependency environment differs from the admitted task lock")
    module_files = [*WORKSPACE.glob("*.egg-info/top_level.txt"), *WORKSPACE.glob("src/*.egg-info/top_level.txt")]
    discovered = {name for path in module_files for name in path.read_text().splitlines()}
    modules = sorted(
        {
            name
            for name in (*discovered, *json.loads(packages)["candidate_modules"])
            if name and not name.startswith("_") and name not in ("tests", "test")
        }
    )
    if not modules:
        raise ValueError("No declared editable project import roots; task needs an explicit import contract")
    write_json(TRUST / "candidate-modules.json", modules)
    phases["trusted_python_snapshot"] = time.time() - freeze_started
    write_json(TRUST / "pins.json", pins.__dict__)
    # The harness stays private; only the frozen interpreter is visible to the
    # separate grader UID. The agent has no membership in that group.
    checked(command(["chown", "-R", "root:longloop-grade", str(TRUST)], cwd=Path("/"), timeout=120))
    checked(command(["chmod", "-R", "u=rwX,g=rX,o=", str(TRUST)], cwd=Path("/"), timeout=120))
    checked(command(["chown", "-R", "longloop-agent:longloop-agent", str(WORKSPACE)], cwd=Path("/"), timeout=120))
    for source in {info["purelib"], info["platlib"]}:
        checked(command(["chgrp", "-R", "longloop-agent", source], cwd=Path("/"), timeout=120))
        checked(command(["chmod", "-R", "g+rwX", source], cwd=Path("/"), timeout=120))
    write_json(
        TRUST / "bootstrap.json",
        {
            "started_unix": started,
            "ended_unix": time.time(),
            "pins": pins.__dict__,
            "dependency_sha256": hashlib.sha256(packages).hexdigest(),
            "phases": phases,
        },
    )


def grade(tests: Path, logs: Path) -> None:
    """Grade only a descended candidate diff in a clean trusted repository."""
    os.environ["PATH"] = f"{CONTROL_BIN}:/usr/bin:/bin"
    logs.mkdir(parents=True, exist_ok=True)
    pins = TaskPins(**json.loads((TRUST / "pins.json").read_text()))
    trusted_paths = (tests / "trusted_test_paths.txt").read_text().splitlines()
    receipt, patch = candidate_patch(WORKSPACE, pins.bug, trusted_paths, uid=AGENT_UID)
    write_json(logs / "candidate.json", receipt)
    (logs / "candidate.patch").write_bytes(patch)
    for name in (
        "pins.json",
        "dependencies.json",
        "candidate-modules.json",
        "bootstrap.json",
        "control-executables.json",
        "setup.stdout",
        "setup.stderr",
    ):
        shutil.copyfile(TRUST / name, logs / name)
    if not receipt["accepted"]:
        write_json(logs / "reward.json", {"reward": 0.0})
        write_json(logs / "verdict.json", {"status": "scored", "reward": 0.0, "detail": receipt})
        return
    original = TRUST / "agent-workspace"
    WORKSPACE.rename(original)
    try:
        checked(git(TRUST / "repository.git", "worktree", "add", "--detach", str(WORKSPACE), pins.bug))
        if patch:
            result = subprocess.run(
                ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
                cwd=WORKSPACE,
                input=patch,
                capture_output=True,
                check=False,
            )
            checked(result)
        restore_paths(TRUST / "repository.git", WORKSPACE, pins, trusted_paths)
        module_probe = (
            "import importlib,json,pathlib; "
            f"names={json.loads((TRUST / 'candidate-modules.json').read_text())!r}; "
            "origins={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names}; "
            "print(json.dumps(origins)); assert all(p.startswith('/testbed/') for p in origins.values()),origins"
        )
        imports = command(
            [str(TRUST / "python/bin/python"), "-I", "-c", module_probe],
            cwd=Path("/"),
            timeout=pins.verifier_timeout,
            uid=GRADER_UID,
        )
        (logs / "imports.stdout").write_bytes(imports.stdout)
        (logs / "imports.stderr").write_bytes(imports.stderr)
        if imports.returncode < 0 or imports.returncode in (125, 126, 127):
            raise RuntimeError(f"Trusted import probe was interrupted: exit {imports.returncode}")
        if imports.returncode:
            write_json(logs / "reward.json", {"reward": 0.0})
            write_json(
                logs / "verdict.json",
                {
                    "status": "scored",
                    "reward": 0.0,
                    "detail": {"reason": "candidate_import", "stderr": imports.stderr.decode(errors="replace")[-6000:]},
                },
            )
            return
        # verifyit itself uses its image-owned uv tool environment. Only pytest
        # runs under the task's frozen Python, with no writable agent packages.
        wrapper = TRUST / "grade-python"
        wrapper.write_text(f'#!/bin/sh\nexec {CONTROL_BIN}/python3 -I -S /opt/long-loop-swe/grading.py pytest -- "$@"\n')
        wrapper.chmod(0o750)
        private_tests = TRUST / "tests"
        shutil.copytree(tests, private_tests, dirs_exist_ok=True)
        private_spec = private_tests / "verifier.toml"
        # uv's protected control interpreter is >=3.11 even when task Python is
        # 3.10. Use its TOML parser; never import agent-writable site-packages.
        executable = shutil.which("verifyit")
        if executable is None:
            raise FileNotFoundError("The pinned verifyit control runtime is missing")
        control_python = Path(executable).resolve().read_text().splitlines()[0].removeprefix("#!")
        if not Path(control_python).is_absolute():
            raise ValueError("verifyit must have an absolute protected interpreter")
        compose = (
            "import json,pathlib,sys,tomllib,tomli_w; "
            "config=tomllib.loads(pathlib.Path(sys.argv[1]).read_text()); "
            "config['candidate_modules']=json.loads(sys.argv[3]); "
            "pathlib.Path(sys.argv[2]).write_text(tomli_w.dumps(config))"
        )
        checked(
            command(
                [
                    control_python,
                    "-I",
                    "-c",
                    compose,
                    str(tests / "verifier.toml"),
                    str(private_spec),
                    (TRUST / "candidate-modules.json").read_text(),
                ],
                cwd=Path("/"),
                timeout=30,
            )
        )
        plugins = json.loads((TRUST / "dependencies.json").read_text())["plugins"]
        environment = {
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_PLUGINS": ",".join(plugins),
            "PYTHONNOUSERSITE": "1",
        }
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"):
            environment.pop(name, None)
        result = subprocess.run(
            ["verifyit", str(private_spec), "--logs-dir", str(logs)],
            cwd=Path("/"),
            env=environment,
            capture_output=True,
            timeout=pins.verifier_timeout,
            check=False,
        )
        (logs / "verifyit.stdout").write_bytes(result.stdout)
        (logs / "verifyit.stderr").write_bytes(result.stderr)
        checked(result)
        if not (logs / "verdict.json").is_file():
            raise RuntimeError("Trusted verifier returned without a verdict")
    finally:
        if WORKSPACE.exists():
            checked(git(TRUST / "repository.git", "worktree", "remove", "--force", str(WORKSPACE)))
        original.rename(WORKSPACE)


def pytest_process(arguments: list[str]) -> None:
    """Hand verifyit's private report directory to the unprivileged pytest child."""
    # verifyit creates its scratch as root/0700 before calling spec.python.
    # Accept only that known report namespace; never chown a candidate path.
    reports = [arg.split("=", 1)[1] for arg in arguments if arg.startswith("--json-report-file=")]
    reports += [arg for arg in arguments if arg.endswith("-failures.txt")]
    for report in reports:
        parent = Path(report).parent
        if parent.parent != Path("/tmp") or not parent.name.startswith("tasktrove-pytest-"):
            raise ValueError(f"Unexpected verifyit scratch path: {parent}")
        os.chown(parent, GRADER_UID, GRADER_UID)
    os.environ["HOME"] = "/home/longloop-grade"
    os.environ["PATH"] = f"{TRUST}/python/bin:{CONTROL_BIN}:/usr/bin:/bin"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"):
        os.environ.pop(name, None)
    os.execvp(
        "setpriv",
        [
            "setpriv",
            "--reuid=10002",
            "--regid=10002",
            "--clear-groups",
            "--no-new-privs",
            str(TRUST / "python/bin/python"),
            "-I",
            *arguments,
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    setup_parser = subparsers.add_parser("bootstrap")
    setup_parser.add_argument("pins_base64")
    grade_parser = subparsers.add_parser("grade")
    grade_parser.add_argument("--tests", type=Path, required=True)
    grade_parser.add_argument("--logs", type=Path, required=True)
    pytest_parser = subparsers.add_parser("pytest")
    pytest_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == "bootstrap":
        bootstrap(TaskPins(**json.loads(base64.b64decode(args.pins_base64))))
    elif args.operation == "grade":
        grade(args.tests, args.logs)
    else:
        pytest_process(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)


if __name__ == "__main__":
    main()
