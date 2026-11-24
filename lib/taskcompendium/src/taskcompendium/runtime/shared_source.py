# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only Linux controller for protected SHARED source scorers.

Annotations stay lazy so the module also loads on task images whose best
interpreter is 3.9. The source's own ``grade.py`` stays byte-identical: bootstrap freezes the
family's test-restore scope as a grading reference, pins a pristine ``.git``
(a declared versioned security change: committed agent history never reaches
grading), preserves the exact pre-bootstrap workspace for the agent, and hides
the private scorer material in the protected root. Grading restores the frozen
state, then executes the original ``tests/test.sh`` bytes as the grading UID.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

NATIVE_SCRIPTS = "native_scripts.py"
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
CONTROL_RUNTIME_PROBE = """import json,os,pathlib,sys,sysconfig
mutable=[pathlib.Path(p).resolve() for p in json.loads(sys.argv[1])]
sites={pathlib.Path(sysconfig.get_path(k)).resolve() for k in ('purelib','platlib')}
paths={pathlib.Path(sys.executable).resolve(),pathlib.Path(sys.base_prefix).resolve()}
paths.update(pathlib.Path(p).resolve() for p in sys.path if p)
stdlib=pathlib.Path(sysconfig.get_path('stdlib')).resolve()
for directory,children,files in os.walk(stdlib):
    parent=pathlib.Path(directory)
    children[:]=[name for name in children if (parent/name).resolve() not in sites]
    paths.add(parent)
    paths.update(parent/name for name in files)
for path in paths:
    if path in sites: continue
    for candidate in (path,*path.parents):
        if any(candidate.is_relative_to(root) for root in mutable) or os.access(candidate,os.W_OK):
            raise SystemExit('Actor can modify root control runtime: '+str(candidate))
"""


def protect_control_runtime(common, contract: dict, mutable: list[str]) -> None:
    """Refuse a root interpreter, stdlib, or ancestor the actor can modify."""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", CONTROL_RUNTIME_PROBE, json.dumps(mutable)],
        user=contract["agent_uid"],
        group=contract["agent_uid"],
        extra_groups=(),
        cwd="/",
        env=common.clean_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError(f"Unsafe root control runtime: {result.stderr.strip()}")


def dependency_sites(common, dependencies: list[str], root: Path) -> list[str]:
    """Locate mutable package directories without granting writes to Python itself."""
    sites: set[str] = set()
    probe = "import json,sysconfig;print(json.dumps([sysconfig.get_path(k) for k in ('purelib','platlib')]))"
    for directory in dependencies:
        python = Path(directory) / "bin/python"
        sites.update(json.loads(common.checked([str(python), "-I", "-c", probe], cwd=root, timeout=30)))
    return sorted(sites)


def shared_controller():
    path = Path(__file__).with_name("shared_pytest.py")
    spec = importlib.util.spec_from_file_location("shared_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves lazy annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_source_module(tests: Path, name: str):
    """Import one pinned source helper module from the protected tests copy."""
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), tests / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patch_paths(diff: str) -> list[str]:
    """Workspace-relative paths a unified diff touches, in first-seen order."""
    selected: list[str] = []
    seen: set[str] = set()
    lines = diff.splitlines()
    old_remaining = new_remaining = 0
    for index, line in enumerate(lines):
        if old_remaining or new_remaining:
            if line.startswith((" ", "-")):
                old_remaining -= 1
            if line.startswith((" ", "+")):
                new_remaining -= 1
            continue
        hunk = HUNK_HEADER.match(line)
        if hunk:
            old_remaining, new_remaining = (int(count) if count is not None else 1 for count in hunk.groups())
            continue
        if not line.startswith("--- ") or index + 1 == len(lines) or not lines[index + 1].startswith("+++ "):
            continue
        for header in (line, lines[index + 1]):
            path = header[4:].partition("\t")[0].strip('"')
            if path == "/dev/null":
                continue
            path = PurePosixPath(path.removeprefix("a/").removeprefix("b/")).as_posix()
            if path == "." or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                raise ValueError(f"Invalid patch path: {path}")
            if path not in seen:
                seen.add(path)
                selected.append(path)
    return selected


def entry(workspace: Path, name: str) -> dict:
    """Record one workspace path's kind, mode and content without following symlinks."""
    path = workspace / name
    if path.is_symlink():
        # Pre-agent image state is preserved verbatim, including symlinks whose
        # targets sit outside the workspace. Agent-created redirects are still
        # refused at restore time by the ancestor repair checks.
        return {"kind": "symlink", "target": os.readlink(path)}
    if path.is_dir():
        return {"kind": "dir", "mode": stat.S_IMODE(path.lstat().st_mode)}
    if path.is_file():
        return {
            "kind": "file",
            "mode": stat.S_IMODE(path.lstat().st_mode),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content": path.read_bytes().hex(),
        }
    raise ValueError(f"Captured path disappeared: {name}")


def capture_scope(workspace: Path, runner: dict, config: dict) -> list[str]:
    """Select every workspace path inside the family's original restore scope."""
    selected: set[str] = set()
    for pattern in runner["capture_patterns"]:
        for path in workspace.glob(pattern):
            selected.add(path.relative_to(workspace).as_posix())
    for directory in runner["capture_dirs"]:
        base = workspace / directory
        if base.is_dir():
            selected.add(directory)
            for path in base.rglob("*"):
                selected.add(path.relative_to(workspace).as_posix())
    for key in runner["patch_config_keys"]:
        for name in patch_paths(config[key]):
            if (workspace / name).exists() and not PurePosixPath(name).is_absolute():
                selected.add(name)
    for key in runner.get("path_config_keys", ()):
        # Explicit per-row path lists (e.g. a source's trusted test files):
        # entries that exist at capture time join the scope; patch-created
        # files that only materialize at grading are covered by the patch-path
        # handling instead.
        for name in config.get(key) or []:
            if (workspace / name).exists() and not PurePosixPath(name).is_absolute():
                selected.add(name)
    return sorted(selected)


def snapshot(workspace: Path, names: list[str]) -> dict[str, dict]:
    return {name: entry(workspace, name) for name in names}


def materialize(common, root: Path, workspace: Path, runner: dict, config: dict, tests: Path, timeout: float) -> None:
    """Run the family's original test-restore logic with its original snippets.

    Any temporary patch material lives inside the protected root and is removed
    before the agent phase; the agent can neither read nor keep it.
    """
    if runner["family"] == "swe-rebench":
        # The declared per-task setup (clone + checkout of the pinned base)
        # already reproduces the base state on the shared base image, and the
        # source restores test_files itself inside its own test.sh at grading
        # time - there is no image-side test drift for this family to undo.
        return
    module = load_source_module(tests, NATIVE_SCRIPTS)
    # The original snippets read the shell variable $base, exactly as grade.py sets it.
    environment = {**os.environ, **module.ENV, "base": config["base_commit"]}
    patch = None
    try:
        if runner["family"] == "scale-swe":
            common.checked(["sh", "-c", module.RESTORE], cwd=workspace, timeout=timeout, env=environment)
            return
        if not config["test_patch"].strip():
            return
        patch = root / "materialize.patch"
        patch.write_text(config["test_patch"], encoding="utf-8")
        patch.chmod(0o600)
        environment["patch"] = str(patch)
        common.checked(["sh", "-c", module.REVERT_TEST_FILES], cwd=workspace, timeout=timeout, env=environment)
    finally:
        if patch is not None:
            common.remove_path(patch)


def write_entries(root: Path, entries: dict[str, dict]) -> None:
    for name, record in entries.items():
        target = root / name
        if target.exists() or target.is_symlink():
            shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else os.unlink(target)
        if record["kind"] == "dir":
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(record["mode"])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if record["kind"] == "symlink":
            os.symlink(record["target"], target)
            continue
        target.write_bytes(bytes.fromhex(record["content"]))
        target.chmod(record["mode"])


def repair_ancestors(common, workspace: Path, name: str, reference: dict[str, dict]) -> None:
    """Make every ancestor of ``name`` a safe directory before removing or writing it.

    An agent-replaced ancestor symlink must not redirect a restore or removal
    outside the workspace. Ancestors the reference itself recorded as symlinks
    are the image's own state and are kept after re-checking they stay inside.
    """
    workspace_resolved = workspace.resolve()
    for parent in reversed((workspace / name).parents):
        if parent == workspace or not parent.is_relative_to(workspace):
            continue
        relative = parent.relative_to(workspace).as_posix()
        recorded = reference.get(relative)
        if parent.is_symlink():
            resolved = parent.resolve()
            if recorded is not None and recorded["kind"] == "symlink" and resolved.is_relative_to(workspace_resolved):
                continue
            common.remove_path(parent)
            parent.mkdir(parents=True, exist_ok=True)
        elif parent.exists() and not parent.is_dir():
            common.remove_path(parent)
            parent.mkdir(parents=True, exist_ok=True)
        elif not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)


def quarantine_remove(common, path: Path, quarantine_root: str | None) -> None:
    """Remove a workspace path, quarantining it when the sandbox fs refuses.

    Daytona sandboxes were observed returning EOVERFLOW for ``unlink`` of
    specific image inodes. ``rename`` moves the whole tree without touching the
    refused inode, so the quarantine preserves the workspace semantics (the
    member is gone) and keeps the refused inode for post-mortem inside the
    protected root. A failed quarantine re-raises: the restore cannot proceed
    faithfully.
    """
    try:
        common.remove_path(path)
    except OSError:
        if quarantine_root is None:
            raise
        destination = Path(quarantine_root) / datetime.now(timezone.utc).strftime(  # noqa: UP017 - guest Python 3.9
            "quarantine-%Y%m%d%H%M%S%f"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(path, destination)


def restore_entries(
    common, workspace: Path, entries: dict[str, dict], reference: dict[str, dict], quarantine_root: str | None = None
) -> None:
    """Recreate recorded workspace paths exactly, including modes and symlinks."""
    directories = [name for name, record in entries.items() if record["kind"] == "dir"]
    files = [name for name, record in entries.items() if record["kind"] != "dir"]
    for name in sorted(directories, key=lambda item: item.count("/")):
        repair_ancestors(common, workspace, name, reference)
        target = workspace / name
        quarantine_remove(common, target, quarantine_root)
        target.mkdir(parents=True, exist_ok=True)
        target.chmod(entries[name]["mode"])
    for name in sorted(files, key=lambda item: item.count("/"), reverse=True):
        repair_ancestors(common, workspace, name, reference)
        record = entries[name]
        target = workspace / name
        quarantine_remove(common, target, quarantine_root)
        if record["kind"] == "symlink":
            os.symlink(record["target"], target)
            continue
        target.write_bytes(bytes.fromhex(record["content"]))
        target.chmod(record["mode"])


def remove_entries(
    common, workspace: Path, entries: dict[str, dict], keep: dict[str, dict], quarantine_root: str | None = None
) -> None:
    """Remove recorded paths absent from ``keep``, cleaning up empty parents."""
    for name in sorted(entries, key=lambda item: item.count("/"), reverse=True):
        if name in keep:
            continue
        repair_ancestors(common, workspace, name, keep)
        quarantine_remove(common, workspace / name, quarantine_root)
    for name in sorted(entries, key=lambda item: item.count("/")):
        if name in keep:
            continue
        parent = workspace / name
        while parent != workspace and parent.is_relative_to(workspace):
            parent = parent.parent
            try:
                parent.rmdir()
            except OSError as error:
                if error.errno in {errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT}:
                    break
                raise


def pin_git(root: Path, workspace: Path) -> None:
    """Freeze the pristine post-setup repository state before the agent runs."""
    source = workspace / ".git"
    if not source.exists():
        raise ValueError("Source grading requires a git repository in the workspace")
    destination = root / "git-pin"
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def restore_git(common, root: Path, workspace: Path) -> None:
    """Replace the repository state with its pristine pinned copy."""
    pin = root / "git-pin"
    target = workspace / ".git"
    common.remove_path(target)
    if pin.is_dir():
        shutil.copytree(pin, target, symlinks=True)
    else:
        shutil.copy2(pin, target)


def restore_tests(common, contract: dict, payload: dict) -> None:
    """Rebuild /tests with the original private scorer material, root-owned."""
    tests = Path("/tests")
    common.remove_path(tests)
    tests.mkdir(mode=0o750)
    os.chown(tests, 0, contract["grader_uid"])
    tests.chmod(0o750)
    for name, content in payload["tests"].items():
        if name == "test.sh":
            continue
        target = tests / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes.fromhex(content))
        os.chown(target, 0, contract["grader_uid"])
        target.chmod(0o644)


def bootstrap(input_path: Path) -> None:
    common = shared_controller()
    payload = json.loads(input_path.read_text())
    contract = payload["contract"]
    runner, spec = contract["native"], payload["spec"]
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    root.mkdir(mode=0o750, parents=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != contract["harness_sha256"]:
        raise ValueError("Published source controller hash differs from installed controller")
    common_source = Path(__file__).with_name("shared_pytest.py")
    if hashlib.sha256(common_source.read_bytes()).hexdigest() != runner["shared_controller_sha256"]:
        raise ValueError("Published shared controller hash differs from installed controller")
    if hashlib.sha256(json.dumps(payload["tests"], sort_keys=True).encode()).hexdigest() != contract["reference_sha256"]:
        raise ValueError("Published test reference differs from bootstrap payload")
    shutil.copy2(__file__, contract["harness"])
    shutil.copy2(common_source, root / "shared_pytest.py")
    tests = root / "tests"
    tests.mkdir()
    for name, content in payload["tests"].items():
        target = tests / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes.fromhex(content))
    common.write_json(root / "contract.json", contract)
    common.write_json(root / "spec.json", spec)

    config = json.loads((tests / "config.json").read_text())
    if runner["family"] in ("scale-swe", "swe-rebench"):
        if config.get("workdir") != contract["workspace"]:
            raise ValueError("Source workdir differs from the published contract workspace")
    pin_git(root, workspace)

    scope = capture_scope(workspace, runner, config)
    before = snapshot(workspace, scope)
    materialize(common, root, workspace, runner, config, tests, spec["timeout"])
    after_names = capture_scope(workspace, runner, config)
    after = snapshot(workspace, after_names)
    reference = {name: record for name, record in after.items() if record["kind"] != "dir"}
    common.write_json(root / "reference-paths.json", reference)
    reference_directory = root / "reference"
    reference_directory.mkdir()
    write_entries(reference_directory, reference)
    # Preserve the exact pre-bootstrap state for the agent, including untracked
    # files, modes and symlinks; only the frozen reference keeps restored bytes.
    keep = {name: record for name, record in before.items()}
    quarantine = str(root / "quarantine")
    remove_entries(common, workspace, after, keep, quarantine)
    restore_entries(common, workspace, keep, keep, quarantine)

    dependencies = [path for path in runner["dependency_paths"] if Path(path).exists()]
    home = os.environ.get("HOME", "/root")
    mutable = [*dependency_sites(common, dependencies, root), str(Path(home) / ".cache")]
    for directory in mutable:
        Path(directory).mkdir(parents=True, exist_ok=True)
        common.checked(["chgrp", "-R", str(contract["agent_uid"]), directory], cwd=root, timeout=120)
        common.checked(["chmod", "-R", "g+rwX", directory], cwd=root, timeout=120)

    protect_control_runtime(common, contract, [*mutable, str(workspace)])

    baseline = common.inventory(dependencies)
    common.write_json(root / common.DEPENDENCY_BASELINE, baseline)
    common.write_json(root / common.STARTUP_BASELINE, common.hook_inventory(baseline))
    for item in (root, *root.rglob("*")):
        if item.is_symlink():
            continue
        os.chown(item, 0, contract["grader_uid"])
        item.chmod(0o750 if item.is_dir() or os.access(item, os.X_OK) else 0o640)
    # Hand the workspace to the agent without touching file modes; grading
    # normalizes group access itself after the agent finishes.
    common.checked(
        ["chown", "-R", f"{contract['agent_uid']}:{contract['agent_uid']}", str(workspace)], cwd=root, timeout=120
    )


def prepare_grading(contract: dict, payload: dict) -> None:
    common = shared_controller()
    common.stop_agent_processes(contract["agent_uid"])
    common.stop_agent_processes(contract["grader_uid"])
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    reports = Path(contract["report_directory"])
    common.reset_report_directory(contract, 0o770)
    dependencies = [path for path in contract["native"]["dependency_paths"] if Path(path).exists()]
    baseline = json.loads((root / common.DEPENDENCY_BASELINE).read_text())
    inventory = common.inventory(dependencies)
    common.write_json(reports / contract["audit_probes"][0], common.inventory_diff(baseline, inventory))
    common.write_json(
        reports / contract["audit_probes"][1],
        common.inventory_diff(common.hook_inventory(baseline), common.hook_inventory(inventory)),
    )
    restore_git(common, root, workspace)
    quarantine = str(root / "quarantine")
    reference = json.loads((root / "reference-paths.json").read_text())
    runner, config = contract["native"], json.loads((root / "tests/config.json").read_text())
    # Agent-created files inside the declared capture scope cannot alter grading.
    for name in capture_scope(workspace, runner, config):
        if name not in reference:
            repair_ancestors(common, workspace, name, reference)
            quarantine_remove(common, workspace / name, quarantine)
    # A patch path may not exist yet; its ancestors still must not redirect the
    # source's own patch application outside the workspace.
    for key in runner["patch_config_keys"]:
        for name in patch_paths(config[key]):
            if PurePosixPath(name).is_absolute():
                continue
            repair_ancestors(common, workspace, name, reference)
            if name not in reference:
                common.remove_path(workspace / name)
    restore_entries(common, workspace, reference, reference, quarantine)
    restore_tests(common, contract, payload)
    common.checked(
        ["chown", "-R", f"{contract['grader_uid']}:{contract['agent_uid']}", str(workspace)], cwd=root, timeout=120
    )
    common.checked(["chmod", "-R", "g+rwX", str(workspace)], cwd=root, timeout=120)


def grade(contract_path: Path) -> None:
    common = shared_controller()
    contract = json.loads(contract_path.read_text())
    payload = {"tests": contract_tests(contract)}
    prepare_grading(contract, payload)
    root, reports = Path(contract["protected_root"]), Path(contract["report_directory"])
    timeout = json.loads((root / "spec.json").read_text())["timeout"]
    original = (root / "tests/test.sh").read_bytes()
    script = root / "original-test.sh"
    script.write_bytes(original)
    script.chmod(0o755)
    environment = {**os.environ, "HOME": "/home/taskcompendium-grade"}
    output = reports / "test_output.log"
    with output.open("wb") as stream:
        # The capture stream is opened by root before the setuid child starts;
        # hand the file to the grader group so a source test script that writes
        # its own command log to the same path (the upstream SWE-rebench
        # rendered test.sh redirects test_cmd there) can reopen and truncate it
        # under the declared trust model (the source scorer owns its verdict).
        os.fchown(stream.fileno(), 0, contract["grader_uid"])
        os.fchmod(stream.fileno(), 0o660)
        process = subprocess.Popen(
            [str(script)],
            cwd=contract["workspace"],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            user=contract["grader_uid"],
            group=contract["grader_uid"],
            extra_groups=(contract["agent_uid"],),
            start_new_session=True,
        )
        try:
            # timeout=None means the source declares no verifier deadline;
            # wait like the source does under the same outer enforcement.
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            common.stop_agent_processes(contract["grader_uid"])
            common.publish_verdict(contract, {"status": "scored", "reward": 0, "detail": {"reason": "timeout"}})
            return
    common.stop_agent_processes(contract["grader_uid"])
    reward_file = Path(contract["report_directory"]) / "reward.txt"
    if code != 0:
        # An aborted scorer cannot vouch for a reward its same-UID children may
        # have written before dying; fail closed instead of trusting the file.
        common.publish_verdict(
            contract,
            {"status": "scored", "reward": 0, "detail": {"exit_code": code, "reason": "scorer_aborted"}},
        )
        return
    reward = parse_source_reward(reward_file)
    if reward is None:
        common.publish_verdict(
            contract,
            {
                "status": "scored",
                "reward": 0,
                "detail": {"exit_code": code, "reason": "invalid_reward", "exists": reward_file.is_file()},
            },
        )
        return
    common.publish_verdict(
        contract,
        {"status": "scored", "reward": reward, "detail": {"exit_code": code}},
    )


def parse_source_reward(reward_file: Path) -> float | None:
    """Return a finite reward in [0, 1], or None for a missing or invalid file."""
    if reward_file.is_symlink() or not reward_file.is_file():
        return None
    try:
        reward = float(reward_file.read_text().strip())
    except ValueError:
        return None
    if not math.isfinite(reward) or reward < 0.0 or reward > 1.0:
        return None
    return reward


def contract_tests(contract: dict) -> dict[str, str]:
    """Reload the private tests material the bootstrap froze into the protected root."""
    tests = Path(contract["protected_root"]) / "tests"
    return {
        path.relative_to(tests).as_posix(): path.read_bytes().hex()
        for path in sorted(tests.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    boot = commands.add_parser("bootstrap")
    boot.add_argument("input", type=Path)
    grading = commands.add_parser("grade")
    grading.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "bootstrap":
        bootstrap(args.input)
    else:
        grade(args.contract)


if __name__ == "__main__":
    main()
