# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A CPU-only Docker CLI boundary backed by temporary files and real shell exec."""

import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path


def _run_compose_exec(command: list[str], container: Path, workspace: Path, mapped: Callable[[str], str]) -> None:
    cwd = mapped(command[command.index("-w") + 1]) if "-w" in command else None
    shell = command[command.index("main") + 1 :]
    user = command[command.index("-u") + 1] if "-u" in command else None
    if user is None:
        uid = int(os.environ.get("TASKCOMPENDIUM_FAKE_IMAGE_UID", "0"))
        gid = int(os.environ.get("TASKCOMPENDIUM_FAKE_IMAGE_GID", "0"))
    elif user == "root":
        uid = gid = 0
    else:
        user_id, _, group_id = user.partition(":")
        uid = int(user_id)
        gid = int(group_id or user_id)
    owners_path = container / "owners.json"
    owners = json.loads(owners_path.read_text()) if owners_path.exists() else {}
    if shell[-1] == "id -u; id -g":
        print(f"{uid}\n{gid}")
        return
    arguments = shlex.split(mapped(shell[-1]))
    if arguments[0] == "chown":
        if uid != 0:
            sys.exit(1)
        recursive = "-R" in arguments
        operands = [value for value in arguments[1:] if not value.startswith("-")]
        owner = [int(value) for value in operands[0].split(":")]
        for name in operands[1:]:
            path = Path(name)
            targets = [path, *path.rglob("*")] if recursive else [path]
            for target in targets:
                owners[str(target)] = owner
        owners_path.write_text(json.dumps(owners))
        return
    if os.environ.get("TASKCOMPENDIUM_FAKE_NO_WORKER_PYTHON") and "python3" in shell[-1]:
        sys.exit(127)
    permissions = {}
    # The host cannot switch UID. Project the container user's permission bits
    # onto host-owner bits for real shell/file operations, restoring modes afterward.
    if uid != 0:
        for path in [workspace, *workspace.rglob("*")]:
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            permissions[path] = mode
            owner_uid, owner_gid = owners.get(str(path), [0, 0])
            shift = 6 if uid == owner_uid else 3 if gid == owner_gid else 0
            path.chmod(((mode >> shift) & 0o7) << 6)
    try:
        result = subprocess.run([*shell[:-1], mapped(shell[-1])], cwd=cwd, capture_output=True)
    finally:
        for path, mode in permissions.items():
            if path.exists() and not path.is_symlink():
                path.chmod(mode)
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)


def main() -> None:
    args = sys.argv[1:]
    root = Path(os.environ["TASKCOMPENDIUM_FAKE_DOCKER_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    if args[:2] == ["context", "inspect"]:
        print(json.dumps({"Host": "unix://" + os.environ["TASKCOMPENDIUM_FAKE_DOCKER_SOCKET"]}))
        return
    if args[0] == "version":
        print("1.51")
        return
    if args[0] == "inspect" or args[:2] == ["image", "inspect"]:
        print("linux")
        return
    if args == ["info"]:
        return
    if args[0] != "compose":
        raise ValueError(args)
    project = args[args.index("--project-name") + 1]
    container = root / project
    container.mkdir(exist_ok=True)
    files = []
    index = 1
    while args[index].startswith("-"):
        option, value = args[index : index + 2]
        if option == "-f":
            files.append(Path(value))
        index += 2
    command = args[index:]
    with (root / "events.jsonl").open("a") as events:
        events.write(
            json.dumps({"project": project, "command": command, "compose": [str(path) for path in files]}) + "\n"
        )
    mounts = []
    for path in files:
        if path.suffix == ".json":
            mounts.extend(json.loads(path.read_text()).get("services", {}).get("main", {}).get("volumes", []))
    definition = Path(args[args.index("--project-directory") + 1]).parent / "task.toml"
    workdir = tomllib.loads(definition.read_text())["environment"]["workdir"]
    mapping = {workdir: str(container / workdir.lstrip("/"))}
    for mount in mounts:
        mapping[mount["target"]] = mount["source"]

    def mapped(value):
        for target, source in mapping.items():
            value = value.replace(target, source)
        return value

    if command[0] == "ps":
        print(hashlib.sha256(project.encode()).hexdigest())
        return
    if command[0] == "up":
        identifier = hashlib.sha256(project.encode()).hexdigest()
        (root / f"{identifier}.json").write_text(json.dumps({"project": project}))
        (container / workdir.lstrip("/")).mkdir(exist_ok=True)
        (container / "mounts.json").write_text(json.dumps(mounts))
        if "--pull" not in command or command[command.index("--pull") + 1] != "never":
            raise ValueError("CPU fixture must never pull an image")
        return
    if command[0] in ("down", "stop"):
        if "--rmi" in command:
            raise ValueError("Shared images must be retained")
        return
    if command[0] == "exec":
        _run_compose_exec(command, container, Path(mapping[workdir]), mapped)
        return
    if command[0] == "cp":
        source, destination = command[1:]
        source = mapped(source.removeprefix("main:"))
        destination = mapped(destination.removeprefix("main:"))
        if source.endswith("/."):
            shutil.copytree(source[:-2], destination, dirs_exist_ok=True)
        else:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return
    raise ValueError(command)


if __name__ == "__main__":
    main()
