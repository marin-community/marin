# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guest-side SHARED controller, launched with the protected Python and -I -S.

Only the standard library is imported here. The published integrity contract
configures paths, identities, plugins, test selection and audit evidence.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def clean_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def checked(argv: list[str], *, cwd: Path, timeout: float, env: dict[str, str] | None = None) -> bytes:
    result = subprocess.run(
        argv, cwd=cwd, env=env or clean_environment(), capture_output=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"Control command failed ({result.returncode}): {result.stderr.decode(errors='replace')[-4000:]}"
        )
    return result.stdout


def inventory(roots: list[str]) -> dict[str, str]:
    result = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if path.is_symlink():
                result[str(path)] = "symlink:" + os.readlink(path)
            elif path.is_file() and "__pycache__" not in path.parts:
                result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def inventory_diff(before: dict[str, str], after: dict[str, str]) -> dict:
    return {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    }


def hook_inventory(files: dict[str, str]) -> dict[str, str]:
    return {
        path: digest
        for path, digest in files.items()
        if Path(path).suffix == ".pth"
        or any(part.split(".")[0] in {"sitecustomize", "usercustomize"} for part in Path(path).parts)
    }


def selected_tests(workspace: Path, contract: dict, tests: Path, spec: dict) -> list[str]:
    selected = {
        path.relative_to(workspace).as_posix()
        for pattern in contract["protected_test_patterns"]
        for path in workspace.glob(pattern)
        if path.is_file()
    }
    for manifest in spec.get("protected_paths_files", []):
        for name in (tests / manifest).read_text().splitlines():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid reference test path: {name}")
            target = workspace / path
            if target.is_dir():
                selected.update(p.relative_to(workspace).as_posix() for p in target.rglob("*") if p.is_file())
            elif target.is_file():
                selected.add(name)
    return sorted(selected)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def restore_file(workspace: Path, relative: str, reference: Path) -> None:
    target = workspace / relative
    for parent in reversed(target.parents):
        if parent == workspace or not parent.is_relative_to(workspace):
            continue
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            remove_path(parent)
        parent.mkdir(exist_ok=True)
    remove_path(target)
    shutil.copy2(reference, target)


def stop_agent_processes(uid: int) -> None:
    """Terminate all processes owned by the agent before touching grading state."""
    if uid == os.getuid():
        raise ValueError("The controller cannot run as the agent UID")
    deadline = time.monotonic() + 5
    while True:
        remaining = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                owner = next(line for line in (entry / "status").read_text().splitlines() if line.startswith("Uid:"))
                if uid not in map(int, owner.split()[1:]):
                    continue
                # A killed zombie cannot modify the filesystem and waits for its parent.
                state = next(line for line in (entry / "status").read_text().splitlines() if line.startswith("State:"))
                if "Z (zombie)" in state:
                    continue
                pid = int(entry.name)
                os.kill(pid, signal.SIGKILL)
                remaining.append(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Agent processes survived SIGKILL: {remaining}")
        time.sleep(0.01)


def bootstrap(input_path: Path) -> None:
    payload = json.loads(input_path.read_text())
    contract, spec = payload["contract"], payload["spec"]
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    if root.exists():
        raise FileExistsError(f"Shared controller already bootstrapped: {root}")
    root.mkdir(mode=0o750, parents=True)
    tests = root / "tests"
    tests.mkdir()
    for name, content in payload["test_files"].items():
        target = tests / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes.fromhex(content))
    write_json(root / "contract.json", contract)
    shutil.copy2(__file__, contract["harness"])
    if hashlib.sha256(Path(contract["harness"]).read_bytes()).hexdigest() != contract["harness_sha256"]:
        raise ValueError("Published harness hash differs from installed controller")
    reference_digest = hashlib.sha256(json.dumps(payload["test_files"], sort_keys=True).encode()).hexdigest()
    if reference_digest != contract["reference_sha256"]:
        raise ValueError("Published test reference differs from bootstrap payload")

    # Capture the agent-visible test tree before applying the source's trusted test patch.
    original = {name: (workspace / name).read_bytes() for name in selected_tests(workspace, contract, tests, spec)}
    if spec.get("setup"):
        environment = {**clean_environment(), "VERIFYIT_WORKSPACE": str(workspace), "VERIFYIT_TESTS_DIR": str(tests)}
        checked(
            ["bash", "-euo", "pipefail", "-c", spec["setup"]], cwd=workspace, timeout=spec["timeout"], env=environment
        )
    reference = Path(contract["reference_directory"])
    reference.mkdir()
    paths = selected_tests(workspace, contract, tests, spec)
    for name in paths:
        target = reference / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / name, target)
    write_json(root / "reference-paths.json", paths)
    for name in paths:
        target = workspace / name
        remove_path(target)
        if name in original:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original[name])

    python = shutil.which(spec["python"])
    if python is None:
        raise FileNotFoundError(f"Task interpreter is missing: {spec['python']}")
    probe = (
        "import json,sys,sysconfig;print(json.dumps({'sites':"
        "list({sysconfig.get_path('purelib'),sysconfig.get_path('platlib')}),'executable':sys.executable}))"
    )
    info = json.loads(checked([python, "-I", "-c", probe], cwd=root, timeout=30))
    checked(
        [python, "-I", "-m", "venv", "--copies", "--without-pip", str(Path(contract["interpreter"]).parent.parent)],
        cwd=root,
        timeout=120,
    )
    # Freeze pytest, explicitly allowed plugins and their implementation dependencies only.
    # Candidate dependencies remain in the live site directories recorded below.
    package_probe = """import importlib.metadata as m,json,pathlib,packaging.requirements
plugins=__import__('sys').argv[1:]
roots={'pytest','pytest-json-report'}
for d in m.distributions():
 if any(e.group=='pytest11' and e.value.split(':')[0] in plugins for e in d.entry_points): roots.add(d.metadata['Name'])
seen=set(); files=[]
while roots:
 name=roots.pop(); key=name.lower().replace('_','-')
 if key in seen: continue
 seen.add(key); d=m.distribution(name)
 for requirement in d.requires or []:
  r=packaging.requirements.Requirement(requirement)
  if r.marker is None or r.marker.evaluate({'extra':''}): roots.add(r.name)
 for f in d.files or []:
  p=pathlib.Path(d.locate_file(f)).resolve()
  if not str(f).startswith('..') and p.is_file(): files.append([str(f),str(p)])
print(json.dumps({'distributions':sorted(seen),'files':files}))"""
    packages = json.loads(
        checked([python, "-I", "-c", package_probe, *contract["pinned_plugins"]], cwd=root, timeout=60)
    )
    runner = Path(contract["runner_directory"])
    runner.mkdir()
    for relative, source in packages["files"]:
        destination = runner / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    Path(contract["pytest_config"]).write_text("[pytest]\naddopts =\n")
    write_json(root / "runtime.json", {"sites": info["sites"], "pinned_distributions": packages["distributions"]})
    write_json(root / "site-packages-baseline.json", inventory(info["sites"]))
    spec.update({"setup": "", "python": str(root / "pytest-python")})
    write_json(root / "spec.json", spec)
    (root / "verifier.toml").write_text(payload["verifier_spec"])
    wrapper = root / "pytest-python"
    wrapper.write_text(
        f"#!/bin/sh\nexec {sys.executable} -I -S {contract['harness']} pytest "
        f"--contract {root / 'contract.json'} -- \"$@\"\n"
    )
    wrapper.chmod(0o750)
    # All private control paths belong to root and the grading group, never the agent.
    for directory in (root, *root.rglob("*")):
        os.chown(directory, 0, contract["grader_uid"])
        directory.chmod(0o750 if directory.is_dir() or os.access(directory, os.X_OK) else 0o640)
    checked(["chown", "-R", f"{contract['agent_uid']}:{contract['agent_uid']}", str(workspace)], cwd=root, timeout=120)
    for site in info["sites"]:
        checked(["chgrp", "-R", str(contract["agent_uid"]), site], cwd=root, timeout=120)
        checked(["chmod", "-R", "g+rwX", site], cwd=root, timeout=120)


def prepare_grading(contract: dict) -> None:
    stop_agent_processes(contract["agent_uid"])
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    runtime = json.loads((root / "runtime.json").read_text())
    before = json.loads((root / "site-packages-baseline.json").read_text())
    after = inventory(runtime["sites"])
    reports = Path(contract["report_directory"])
    reports.mkdir(mode=0o750, parents=True, exist_ok=True)
    if reports.is_symlink() or reports.stat().st_uid != 0:
        raise ValueError("The structured report directory must be owned by root")
    reports.chmod(0o750)
    write_json(reports / contract["audit_probes"][0], inventory_diff(before, after))
    write_json(reports / contract["audit_probes"][1], inventory_diff(hook_inventory(before), hook_inventory(after)))
    # Remove every newly introduced conftest as well as restoring every reference conftest.
    paths = json.loads((root / "reference-paths.json").read_text())
    for path in workspace.rglob("conftest.py"):
        if path.relative_to(workspace).as_posix() not in paths:
            remove_path(path)
    for name in paths:
        restore_file(workspace, name, Path(contract["reference_directory"]) / name)
    checked(["chgrp", "-R", str(contract["grader_uid"]), str(workspace)], cwd=root, timeout=120)
    checked(["chmod", "-R", "g+rwX", str(workspace)], cwd=root, timeout=120)


def pytest_process(contract: dict, arguments: list[str]) -> None:
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    runtime = json.loads((root / "runtime.json").read_text())
    # Verifyit owns the scratch namespace; only its per-launch test-report parent is writable by the grader.
    for argument in arguments:
        report = (
            argument.partition("=")[2]
            if argument.startswith("--json-report-file=")
            else argument if argument.endswith("-failures.txt") else None
        )
        if report:
            parent = Path(report).parent
            if parent.parent != Path("/tmp") or not parent.name.startswith("tasktrove-pytest-"):
                raise ValueError(f"Unexpected structured report namespace: {parent}")
            os.chown(parent, 0, contract["grader_uid"])
            parent.chmod(0o770)
    pinned = [
        "-c",
        contract["pytest_config"],
        "--rootdir",
        str(workspace),
        "--confcutdir",
        str(workspace),
        "-o",
        "addopts=",
    ]
    for plugin in contract["pinned_plugins"]:
        pinned.extend(["-p", plugin])
    arguments.extend(pinned)
    launch = """import sys,site,runpy
runner,workspace,sites,arguments=__import__('json').loads(sys.argv[1])
sys.path[:0]=[runner,workspace]
for directory in sites: site.addsitedir(directory)
try: __import__('sitecustomize')
except ModuleNotFoundError as error:
 if error.name!='sitecustomize': raise
sys.path.remove(runner);sys.path.insert(0,runner)
sys.argv=['-c',*arguments[2:]]
if arguments[:1]==['-c']: exec(compile(arguments[1],'<string>','exec'))
else: raise ValueError('Verifyit must invoke the structured pytest runner with -c')
"""
    os.setgroups([])
    os.setgid(contract["grader_uid"])
    os.setuid(contract["grader_uid"])
    os.execve(
        contract["interpreter"],
        [
            contract["interpreter"],
            "-I",
            "-S",
            "-c",
            launch,
            json.dumps([contract["runner_directory"], str(workspace), runtime["sites"], arguments]),
        ],
        clean_environment(),
    )


def grade(contract_path: Path, verifyit: str) -> None:
    contract = json.loads(contract_path.read_text())
    prepare_grading(contract)
    for name in ("verdict.json", "reward.json", "reward.txt"):
        remove_path(Path(contract["report_directory"]) / name)
    root = Path(contract["protected_root"])
    # The Verifyit controller is protected and runs as root; only its pytest child drops UID.
    result = subprocess.run(
        [verifyit, str(root / "verifier.toml"), "--logs-dir", contract["report_directory"]],
        cwd=Path("/"),
        env=clean_environment(),
        capture_output=True,
        timeout=json.loads((root / "spec.json").read_text())["timeout"],
        check=False,
    )
    reports = Path(contract["report_directory"])
    (reports / "verifyit.stdout").write_bytes(result.stdout)
    (reports / "verifyit.stderr").write_bytes(result.stderr)
    if result.returncode or not (reports / "verdict.json").is_file():
        raise RuntimeError("Protected Verifyit controller did not produce its structured verdict")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    bootstrap_parser = commands.add_parser("bootstrap")
    bootstrap_parser.add_argument("input", type=Path)
    grade_parser = commands.add_parser("grade")
    grade_parser.add_argument("--contract", type=Path, required=True)
    grade_parser.add_argument("--verifyit", required=True)
    pytest_parser = commands.add_parser("pytest")
    pytest_parser.add_argument("--contract", type=Path, required=True)
    pytest_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == "bootstrap":
        bootstrap(args.input)
    elif args.operation == "grade":
        grade(args.contract, args.verifyit)
    else:
        pytest_process(
            json.loads(args.contract.read_text()), args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        )


if __name__ == "__main__":
    main()
