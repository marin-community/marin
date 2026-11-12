# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only Linux controller for source-declared non-Python tests.

The test process never owns its output log or verdict. The trusted source parser
runs in a fresh isolated root Python after all grading-UID processes are dead.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def shared_controller():
    path = Path(__file__).with_name("shared_pytest.py")
    spec = importlib.util.spec_from_file_location("shared_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paths_matching(workspace: Path, patterns: list[str]) -> list[str]:
    return sorted(
        {
            p.relative_to(workspace).as_posix()
            for pattern in patterns
            for p in workspace.glob(pattern)
            if p.is_file() or p.is_symlink()
        }
    )


def freeze_path(source: Path, pins: Path, entries: list[dict]) -> None:
    source = source.resolve(strict=True)
    if any(source.is_relative_to(Path(entry["source"])) for entry in entries):
        return
    destination = pins / str(len(entries))
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)
    entries.append({"source": str(source), "reference": str(destination)})


def toolchain_root(executable: Path) -> Path:
    """Keep relocatable compiler distributions together; system binaries use the OS image."""
    executable = executable.resolve(strict=True)
    root = executable.parent.parent if executable.parent.name in {"bin", "sbin"} else executable.parent
    if root in {Path("/"), Path("/usr"), Path("/usr/local")} or ".cargo" in root.parts or ".opam" in root.parts:
        return executable
    return root


def protect_tree(path: Path, group: int) -> None:
    for item in (path, *path.rglob("*")) if path.is_dir() else (path,):
        if item.is_symlink():
            continue
        os.chown(item, 0, group)
        item.chmod(0o755 if item.is_dir() or os.access(item, os.X_OK) else 0o644)


def bootstrap(input_path: Path) -> None:
    common = shared_controller()
    payload = json.loads(input_path.read_text())
    contract = payload["contract"]
    native = contract["native"]
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    root.mkdir(mode=0o755, parents=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != contract["harness_sha256"]:
        raise ValueError("Published native controller hash differs from installed controller")
    common_source = Path(__file__).with_name("shared_pytest.py")
    if hashlib.sha256(common_source.read_bytes()).hexdigest() != native["shared_controller_sha256"]:
        raise ValueError("Published shared controller hash differs from installed controller")
    if (
        hashlib.sha256(json.dumps(payload["test_files"], sort_keys=True).encode()).hexdigest()
        != contract["reference_sha256"]
    ):
        raise ValueError("Published test reference differs from bootstrap payload")
    shutil.copy2(__file__, contract["harness"])
    shutil.copy2(common_source, root / "shared_pytest.py")
    tests = root / "tests"
    tests.mkdir()
    for name, content in payload["test_files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid private test path: {name}")
        target = tests / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes.fromhex(content))
    common.write_json(root / "contract.json", contract)
    common.write_json(root / "spec.json", payload["spec"])
    environment = common.clean_environment()
    home = os.environ.get("HOME", "/root")
    Path(home).chmod(0o755)
    environment["HOME"] = home
    for key in native["environment_keys"]:
        if key in os.environ:
            environment[key] = os.environ[key]
    patterns = [*contract["protected_test_patterns"], *native["config_patterns"], *native["startup_patterns"]]
    command_files = [
        name.removeprefix("./")
        for name in native["command_files"]
        if not Path(name).is_absolute() and ".." not in Path(name).parts
    ]
    initial = set(paths_matching(workspace, patterns)) | {name for name in command_files if (workspace / name).is_file()}
    original = {name: (workspace / name).read_bytes() for name in initial}
    # Source restoration happens once, before the agent. Subsequent grades restore
    # these bytes without trusting agent-editable Git objects, index or hooks.
    for script, arguments in (
        (
            "install_trusted_test_paths.sh",
            [str(workspace), native["trusted_commit"], str(tests / "trusted_test_paths.txt")],
        ),
        ("install_trusted_test_patch.sh", [str(workspace), str(tests / "test_patch.diff"), native["trusted_commit"]]),
    ):
        source = (tests / script).read_text().replace("/tests/", str(tests) + "/")
        common.checked(
            ["/bin/bash", "-euo", "pipefail", "-c", source, script, *arguments],
            cwd=workspace,
            timeout=payload["spec"]["timeout"],
            env=environment,
        )
    selected = set(paths_matching(workspace, patterns)) | {
        name for name in command_files if (workspace / name).is_file()
    }
    for manifest in ("trusted_test_paths.txt", "trusted_patch_paths.txt"):
        for name in (tests / manifest).read_text().splitlines():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Invalid protected path: {name}")
            target = workspace / name
            if target.is_dir():
                selected.update(p.relative_to(workspace).as_posix() for p in target.rglob("*") if p.is_file())
            elif target.is_file():
                selected.add(name)
    reference = Path(contract["reference_directory"])
    reference.mkdir()
    for name in sorted(selected):
        common.restore_path(reference, name, workspace / name)
        common.remove_path(workspace / name)
        if name in original:
            common.restore_path(workspace, name, reference / name)
            (workspace / name).write_bytes(original[name])
    common.write_json(root / "reference-paths.json", sorted(selected))

    dependencies = [
        str(Path(value.replace("~", home, 1).replace("{workspace}", str(workspace))))
        for value in native["dependency_paths"]
    ]
    for directory in [*dependencies, str(Path(home) / ".cache")]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    for directory in [*dependencies, str(Path(home) / ".cache")]:
        common.checked(["chgrp", "-R", str(contract["agent_uid"]), directory], cwd=root, timeout=120)
        common.checked(["chmod", "-R", "g+rwX", directory], cwd=root, timeout=120)
    common.checked(["chgrp", "-R", str(contract["grader_uid"]), str(workspace)], cwd=root, timeout=120)
    common.checked(["chmod", "-R", "g+rwX", str(workspace)], cwd=root, timeout=120)

    # Install source-declared wrapper distributions before the agent can seed
    # their caches. Version probes do not execute the source test command.
    if native["family"] == "jvm":
        probes = []
        wrapper_environment = {**environment, "PATH": os.environ["PATH"]}
        for wrapper in ("gradlew", "mvnw"):
            if wrapper not in native["command"]:
                continue
            for path in workspace.rglob(wrapper):
                if path.is_file():
                    probes.append((["/bin/bash", str(path), "--version"], path.parent))
        if "sbt" in native["toolchain_commands"]:
            probes.append((["sbt", "about"], workspace))
        for arguments, directory in probes:
            subprocess.run(
                arguments,
                cwd=directory,
                env=wrapper_environment,
                capture_output=True,
                timeout=300,
                check=True,
                user=contract["grader_uid"],
                group=contract["grader_uid"],
                extra_groups=(contract["agent_uid"],),
            )

    pins = root / "pins"
    pins.mkdir()
    entries = []
    tool_directories = []
    for name in native["toolchain_commands"]:
        binary = shutil.which(name)
        if binary is None:
            continue
        executable = Path(binary)
        tool_directories.append(str(executable.parent))
        # rustup proxies would otherwise select a new or agent-installed toolchain.
        if name in {"cargo", "rustc", "rustdoc"} and executable.resolve().name == "rustup":
            actual = (
                common.checked(
                    [str(executable), "which", name],
                    cwd=root,
                    timeout=30,
                    env={**environment, "PATH": os.environ["PATH"]},
                )
                .decode()
                .strip()
            )
            freeze_path(Path(actual).parent.parent, pins, entries)
            environment["RUSTUP_TOOLCHAIN"] = str(Path(actual).parent.parent)
        freeze_path(toolchain_root(executable), pins, entries)
    if not tool_directories:
        raise ValueError("No declared language toolchain is installed")
    environment["PATH"] = ":".join(dict.fromkeys([*tool_directories, "/usr/local/bin", "/usr/bin", "/bin"]))
    for pattern in native["runner_paths"]:
        expanded = pattern.replace("~", home, 1).replace("{workspace}", str(workspace))
        for path in Path("/").glob(expanded.removeprefix("/")):
            freeze_path(path, pins, entries)
    for name in native["command_files"]:
        path = Path(name)
        if path.is_absolute() and path.is_file() and os.access(path, os.X_OK):
            freeze_path(toolchain_root(path), pins, entries)
    if native["family"] == "rust":
        rustup = Path(environment.get("RUSTUP_HOME", str(Path(home) / ".rustup")))
        if rustup.exists():
            freeze_path(rustup, pins, entries)
    # Pin executable startup selectors outside the workspace as well.
    startup_roots = [Path(home), *[Path(value) for key, value in environment.items() if key.endswith("_HOME")]]
    for directory in startup_roots:
        for name in paths_matching(directory, native["startup_patterns"]):
            freeze_path(directory / name, pins, entries)
    common.write_json(
        root / "runtime.json",
        {
            "environment": environment,
            "pins": entries,
            "dependencies": dependencies,
            "startup_roots": [str(p) for p in startup_roots],
        },
    )
    common.write_json(root / "dependency-baseline.json", common.inventory(dependencies))
    common.write_json(root / "startup-baseline.json", startup_inventory(contract, startup_roots))
    for item in (root, *root.rglob("*")):
        if item.is_symlink():
            continue
        os.chown(item, 0, contract["grader_uid"])
        item.chmod(0o750 if item.is_dir() or os.access(item, os.X_OK) else 0o640)
    common.checked(
        ["chown", "-R", f"{contract['agent_uid']}:{contract['agent_uid']}", str(workspace)], cwd=root, timeout=120
    )
    for directory in [*dependencies, str(Path(home) / ".cache")]:
        if Path(directory).exists():
            common.checked(["chgrp", "-R", str(contract["agent_uid"]), directory], cwd=root, timeout=120)
            common.checked(["chmod", "-R", "g+rwX", directory], cwd=root, timeout=120)
    for entry in entries:
        protect_tree(Path(entry["source"]), 0)


def startup_inventory(contract: dict, roots: list[Path]) -> dict[str, str]:
    selected = {}
    for directory in [Path(contract["workspace"]), *roots]:
        for name in paths_matching(directory, contract["native"]["startup_patterns"]):
            path = directory / name
            selected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return selected


def prepare_grading(contract: dict) -> dict:
    common = shared_controller()
    common.stop_agent_processes(contract["agent_uid"])
    common.stop_agent_processes(contract["grader_uid"])
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    runtime = json.loads((root / "runtime.json").read_text())
    reports = Path(contract["report_directory"])
    reports.mkdir(mode=0o700, parents=True, exist_ok=True)
    if reports.is_symlink() or reports.stat().st_uid != 0:
        raise ValueError("The structured report directory must be owned by root")
    reports.chmod(0o700)
    common.write_json(
        reports / contract["audit_probes"][0],
        common.inventory_diff(
            json.loads((root / "dependency-baseline.json").read_text()), common.inventory(runtime["dependencies"])
        ),
    )
    common.write_json(
        reports / contract["audit_probes"][1],
        common.inventory_diff(
            json.loads((root / "startup-baseline.json").read_text()),
            startup_inventory(contract, [Path(p) for p in runtime["startup_roots"]]),
        ),
    )
    native = contract["native"]
    paths = json.loads((root / "reference-paths.json").read_text())
    patterns = [*contract["protected_test_patterns"], *native["config_patterns"], *native["startup_patterns"]]
    for name in paths_matching(workspace, patterns):
        if name not in paths:
            common.remove_path(workspace / name)
    for name in paths:
        common.restore_path(workspace, name, Path(contract["reference_directory"]) / name)
    for entry in runtime["pins"]:
        target, reference = Path(entry["source"]), Path(entry["reference"])
        common.restore_path(Path("/"), target.relative_to("/").as_posix(), reference)
        protect_tree(target, 0)
        if target.is_dir():
            for lock in target.rglob("*"):
                if (
                    not lock.is_symlink()
                    and lock.is_file()
                    and (lock.name.endswith(".lck") or lock.name == "sbt.boot.lock")
                ):
                    os.chown(lock, 0, contract["grader_uid"])
                    lock.chmod(0o660)
    for directory in runtime["startup_roots"]:
        for name in paths_matching(Path(directory), native["startup_patterns"]):
            path = (Path(directory) / name).resolve()
            if not any(path == Path(p["source"]) for p in runtime["pins"]):
                common.remove_path(path)
    # Build caches may require explicit timestamp updates, which group write alone cannot grant.
    common.checked(
        ["chown", "-R", f"{contract['grader_uid']}:{contract['agent_uid']}", str(workspace)], cwd=root, timeout=120
    )
    common.checked(["chmod", "-R", "g+rwX", str(workspace)], cwd=root, timeout=120)
    return runtime


def grade(contract_path: Path) -> None:
    common = shared_controller()
    contract = json.loads(contract_path.read_text())
    runtime = prepare_grading(contract)
    root, reports = Path(contract["protected_root"]), Path(contract["report_directory"])
    for name in ("verdict.json", "reward.json", "reward.txt", "test_output.log"):
        common.remove_path(reports / name)
    timeout = json.loads((root / "spec.json").read_text())["timeout"]
    output = reports / "test_output.log"
    with output.open("wb") as stream:
        process = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-o", "pipefail", "-c", contract["native"]["command"]],
            cwd=contract["workspace"],
            env=runtime["environment"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            user=contract["grader_uid"],
            group=contract["grader_uid"],
            extra_groups=(contract["agent_uid"],),
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            common.stop_agent_processes(contract["grader_uid"])
            common.write_json(
                reports / "verdict.json", {"status": "scored", "reward": 0, "detail": {"reason": "timeout"}}
            )
            (reports / "reward.txt").write_text("0\n")
            return
    common.stop_agent_processes(contract["grader_uid"])
    # A fresh Python process imports only the pinned source parser and stdlib.
    program = (
        "import runpy,json,sys; module=runpy.run_path(sys.argv[1]); "
        "report=module['evaluate_test_results'](sys.argv[2],sys.argv[3]); "
        "print(json.dumps(report,allow_nan=False))"
    )
    report = json.loads(
        common.checked(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                program,
                str(root / "tests/test_state.py"),
                str(output),
                str(root / "tests/config.json"),
            ],
            cwd=root,
            env=common.clean_environment(),
            timeout=60,
        )
    )
    reward = int(report["resolved"] is True)
    common.write_json(
        reports / "verdict.json", {"status": "scored", "reward": reward, "detail": {"exit_code": code, **report}}
    )
    (reports / "reward.txt").write_text(f"{reward}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    boot = commands.add_parser("bootstrap")
    boot.add_argument("input", type=Path)
    grading = commands.add_parser("grade")
    grading.add_argument("--contract", type=Path, required=True)
    grading.add_argument("--verifyit")
    args = parser.parse_args()
    if args.operation == "bootstrap":
        bootstrap(args.input)
    else:
        grade(args.contract)


if __name__ == "__main__":
    main()
