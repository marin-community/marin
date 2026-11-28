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
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPORT_FILES = ("verdict.json", "reward.txt", "reward.json", "test_output.log")
NATIVE_SCRIPTS = "native_scripts.py"


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
    for line in diff.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        path = line[4:].partition("\t")[0].strip('"')
        if path == "/dev/null":
            continue
        path = PurePosixPath(path.removeprefix("a/").removeprefix("b/")).as_posix()
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
        destination = Path(quarantine_root) / datetime.now(timezone.utc).strftime("quarantine-%Y%m%d%H%M%S%f")
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
            except OSError:
                break


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


def ensure_verifier_directory(common, contract: dict) -> None:
    """Give the grading UID a controlled report directory without losing captures."""
    reports = Path(contract["report_directory"])
    if reports.is_symlink() or not reports.is_dir():
        common.remove_path(reports)
        reports.mkdir(mode=0o770, parents=True)
    os.chown(reports, 0, contract["grader_uid"])
    reports.chmod(0o770)
    for name in REPORT_FILES:
        common.remove_path(reports / name)


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
    elif contract["workspace"] != "/testbed":
        raise ValueError("SWE-Lego grades in /testbed")
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
    for directory in [*dependencies, str(Path(home) / ".cache")]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        common.checked(["chgrp", "-R", str(contract["agent_uid"]), directory], cwd=root, timeout=120)
        common.checked(["chmod", "-R", "g+rwX", directory], cwd=root, timeout=120)

    baseline = common.inventory(dependencies)
    common.write_json(root / "dependency-baseline.json", baseline)
    common.write_json(root / "startup-baseline.json", common.hook_inventory(baseline))
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
    ensure_verifier_directory(common, contract)
    dependencies = [path for path in contract["native"]["dependency_paths"] if Path(path).exists()]
    baseline = json.loads((root / "dependency-baseline.json").read_text())
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
    with output.open("ab") as stream:
        # The capture stream is opened by root before the setuid child starts;
        # hand the file to the grader group so a source test script that writes
        # its own command log to the same path (the upstream SWE-rebench
        # rendered test.sh redirects test_cmd there) can reopen and truncate it
        # under the declared trust model (the source scorer owns its verdict).
        # Append mode: the source's own `cat > test_output.log` still opens and
        # truncates the path, but this handle's writes (the harness echo and the
        # standalone grader's stdout) append AFTER the source's content instead
        # of stomping it at an independent offset — the two writers on one file
        # were cutting the source's test-name lines mid-stream (observed on the
        # DELTA5/DELTA6 zero-capture fixtures: the echo glued into `✓ <name> (NN`
        # and whole test lines lost at the stomp boundaries).
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
            common.write_json(
                reports / "verdict.json", {"status": "scored", "reward": 0, "detail": {"reason": "timeout"}}
            )
            _write_canonical_reward(reports, 0, contract)
            return
    common.stop_agent_processes(contract["grader_uid"])
    reward_file = Path(contract["report_directory"]) / "reward.txt"
    if code != 0:
        # An aborted scorer cannot vouch for a reward its same-UID children may
        # have written before dying; fail closed instead of trusting the file.
        common.write_json(
            reports / "verdict.json",
            {"status": "scored", "reward": 0, "detail": {"exit_code": code, "reason": "scorer_aborted"}},
        )
        _write_canonical_reward(reports, 0, contract)
        return
    reward = parse_source_reward(reward_file)
    if reward is None:
        common.write_json(
            reports / "verdict.json",
            {
                "status": "scored",
                "reward": 0,
                "detail": {"exit_code": code, "reason": "invalid_reward", "exists": reward_file.is_file()},
            },
        )
        _write_canonical_reward(reports, 0, contract)
        return
    common.write_json(
        reports / "verdict.json",
        {"status": "scored", "reward": reward, "detail": {"exit_code": code}},
    )
    _write_canonical_reward(reports, reward, contract)


def parse_source_reward(reward_file: Path) -> float | None:
    """Validate the source scorer's reward file: regular, finite, in [0, 1]."""
    if reward_file.is_symlink() or not reward_file.is_file():
        return None
    try:
        reward = float(reward_file.read_text().strip())
    except (OSError, ValueError):
        return None
    # The declared reward domain of these source scorers is a finite [0, 1] float.
    if not math.isfinite(reward) or reward < 0.0 or reward > 1.0:
        return None
    return reward


def _write_canonical_reward(reports: Path, reward: float, contract: dict) -> None:
    reward_file = reports / "reward.txt"
    reward_file.write_text(f"{reward}\n")
    os.chown(reward_file, 0, contract["grader_uid"])
    reward_file.chmod(0o640)


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
