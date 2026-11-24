# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted initialization of an immutable captured reference without source setup.

This standalone guest program uses only the standard library. It never derives
grading references from the final actor's workspace or dependency installation.
It materializes the declared agent-owned environment from the exact source
interpreter and keeps every protected grading path out of actor-writable space.
"""

import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath


def validate_protected_target(target: Path, actor_prefix: Path, agent_gid: int) -> Path:
    """Reject protected references that traverse, resolve into, or sit in actor-writable space.

    Returns the resolved target so callers link to one validated path.
    """
    real = target.resolve()
    for candidate in (target, real):
        if candidate.is_relative_to(actor_prefix):
            raise ValueError(f"Protected grading must not reference the actor environment: {target}")
        if any(part.is_relative_to(actor_prefix) for part in candidate.parents):
            raise ValueError(f"Protected grading traverses the actor environment: {target}")
    for checked in (*reversed(real.parents), real):
        info = checked.stat()
        # A sticky world-writable directory (e.g. /tmp) cannot be emptied by the actor;
        # a plain world-writable or agent-group-writable component would be.
        if info.st_mode & 0o002 and not info.st_mode & 0o1000:
            raise ValueError(f"Protected grading target has a world-writable component: {checked}")
        if info.st_gid == agent_gid and info.st_mode & 0o020:
            raise ValueError(f"Protected grading target is agent-group-writable: {checked}")
    return real


def inventory(roots: list[str]) -> dict[str, str]:
    result = {}
    for root in roots:
        source = Path(root)
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                result[str(path)] = "symlink:" + os.readlink(path)
            elif path.is_file():
                result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def resolve_source_interpreter(runtime: dict, control_python: Path) -> Path:
    """Identify the interpreter that owns the captured baseline sites; fail loudly otherwise."""
    sites = [Path(site) for site in runtime["sites"]]
    probe = "import sysconfig;print(sysconfig.get_path('purelib'))"

    failures: list[str] = []

    def owns(python: Path) -> bool:
        try:
            purelib = subprocess.check_output([str(python), "-I", "-c", probe], text=True, timeout=60).strip()
        except (OSError, subprocess.SubprocessError) as error:
            failures.append(f"{python}: {error}")
            return False
        return Path(purelib) in sites

    if owns(control_python):
        return control_python
    candidate = sites[0].parents[2] / "bin/python"
    if candidate.is_file() and owns(candidate):
        return candidate
    raise ValueError(f"The captured baseline sites do not identify the source interpreter; probes: {failures}")


def materialize_actor_environment(actor: dict, interpreter: Path, source_sites: list[str]) -> list[str]:
    """Build the declared agent-owned venv from the exact source interpreter; return its site directories."""
    prefix = Path(actor["prefix"])
    venv = Path(actor["python"])
    if prefix.exists():
        raise FileExistsError(prefix)
    uid, gid = int(actor["uid"]), int(actor["gid"])
    prefix.mkdir(mode=int(actor["directory_mode"]))
    os.chown(prefix, uid, gid)
    subprocess.run(
        [str(interpreter), "-I", "-m", "venv", *actor["venv_flags"], str(venv)],
        check=True,
        timeout=300,
    )
    for base, directories, files in os.walk(venv, topdown=True, followlinks=False):
        for directory in (base, *(Path(base) / name for name in directories)):
            path = Path(directory)
            os.lchown(path, uid, gid)
            if not path.is_symlink():
                path.chmod(int(actor["directory_mode"]))
        for name in files:
            path = Path(base) / name
            os.lchown(path, uid, gid)
            if path.is_symlink():
                continue
            path.chmod(int(actor["executable_mode"] if path.stat().st_mode & 0o111 else actor["file_mode"]))
    purelib = Path(
        subprocess.check_output(
            [str(venv / "bin/python"), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
            text=True,
            timeout=60,
        ).strip()
    )
    # The declared baseline: the captured source site directories, read-only, in import order.
    baseline = purelib / "taskcompendium-source-baseline.pth"
    baseline.write_text("import site;" + ";".join(f"site.addsitedir({site!r})" for site in source_sites) + "\n")
    os.chown(baseline, uid, gid)
    baseline.chmod(int(actor["file_mode"]))
    probe = (
        "import json,sys,sysconfig;"
        "print(json.dumps({'prefix':sys.prefix,'path':sys.path,"
        "'sites':sorted({sysconfig.get_path('purelib'),sysconfig.get_path('platlib')})}))"
    )
    info = json.loads(subprocess.check_output([str(venv / "bin/python"), "-I", "-c", probe], text=True, timeout=60))
    if info["prefix"] != str(venv):
        raise ValueError("The actor environment prefix does not resolve to its declared venv")
    actor_python = venv / "bin/python"
    if hashlib.sha256(actor_python.read_bytes()).hexdigest() != hashlib.sha256(interpreter.read_bytes()).hexdigest():
        raise ValueError("The actor interpreter is not the exact source interpreter")
    if any(site not in info["path"] for site in (*source_sites, *info["sites"])):
        raise ValueError("The actor environment does not expose its sites and the declared source baseline")
    shim = venv / "bin" / "pip"
    shim.write_text(actor["pip_shim"])
    shim.chmod(int(actor["executable_mode"]))
    os.chown(shim, uid, gid)
    return info["sites"]


def merge_actor_environment(root: Path, actor_sites: list[str]) -> None:
    """Record the actor sites in the grading runtime so verdicts observe agent-owned installs."""
    runtime_path = root / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["sites"] = list(dict.fromkeys((*actor_sites, *runtime["sites"])))
    runtime_path.write_text(json.dumps(runtime, sort_keys=True) + "\n")
    baseline_path = root / "site-packages-baseline.json"
    baseline = json.loads(baseline_path.read_text())
    baseline.update(inventory(actor_sites))
    baseline_path.write_text(json.dumps(baseline, sort_keys=True) + "\n")


def initialize(reference_path: Path, bundle_path: Path) -> None:
    reference = json.loads(reference_path.read_text())
    contract = reference["contract"]
    actor = reference["actor_environment"]
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    if root.exists():
        raise FileExistsError(root)
    if not workspace.is_dir() or any(workspace.iterdir()):
        raise ValueError("Reference-only initialization requires the original empty workspace")
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != reference["bundle_sha256"]:
        raise ValueError("Frozen reference changed before trusted initialization")
    verifyit = shutil.which("verifyit")
    if verifyit is None:
        raise FileNotFoundError("The original source Verifyit CLI is missing")
    root.mkdir(mode=0o700, parents=True)
    with tarfile.open(bundle_path) as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.isdev() or member.isfifo():
                raise ValueError(f"Unsafe frozen reference member: {member.name}")
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError(f"Unsafe frozen reference link: {member.name}")
        # The guest control interpreter may predate tarfile extraction filters;
        # membership is fully validated above, so plain extraction is safe.
        archive.extractall(root)
    if json.loads((root / "contract.json").read_text()) != contract:
        raise ValueError("Installed frozen reference differs from its declared contract")
    if hashlib.sha256(Path(contract["harness"]).read_bytes()).hexdigest() != contract["harness_sha256"]:
        raise ValueError("Installed grading controller differs from its pinned source")
    for uid, name in ((contract["agent_uid"], "taskcompendium-agent"), (contract["grader_uid"], "taskcompendium-grade")):
        try:
            pwd.getpwuid(uid)
        except KeyError:
            subprocess.run(["/usr/sbin/useradd", "--create-home", "--uid", str(uid), "--user-group", name], check=True)
    actor_prefix = Path(actor["prefix"])
    if actor_prefix.is_relative_to(root) or root.is_relative_to(actor_prefix):
        raise ValueError("The actor prefix and the protected root must be disjoint")
    bootstrap = Path(contract["protected_root"] + "-bootstrap")
    bootstrap.mkdir(mode=0o750)
    shutil.copy2(Path(contract["harness"]), bootstrap / "control.py")
    interpreter = validate_protected_target(Path(sys.executable), actor_prefix, int(actor["gid"]))
    validate_protected_target(Path(verifyit), actor_prefix, int(actor["gid"]))
    (bootstrap / "python").symlink_to(interpreter)
    (bootstrap / "verifyit").symlink_to(verifyit)
    runtime = json.loads((root / "runtime.json").read_text())
    source_python = resolve_source_interpreter(runtime, interpreter)
    actor_sites = materialize_actor_environment(actor, source_python, runtime["sites"])
    merge_actor_environment(root, actor_sites)
    os.chown(workspace, contract["agent_uid"], contract["agent_uid"])
    for directory in (root, bootstrap):
        for path in (directory, *directory.rglob("*")):
            os.lchown(path, 0, contract["grader_uid"])
            if path.is_symlink():
                continue
            path.chmod(0o750 if path.is_dir() or os.access(path, os.X_OK) else 0o640)
    reference_path.unlink()
    bundle_path.unlink()


if __name__ == "__main__":
    initialize(Path(sys.argv[1]), Path(sys.argv[2]))
