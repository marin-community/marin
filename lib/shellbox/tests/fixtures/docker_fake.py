# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A CPU-only Docker CLI boundary backed by temporary files and real shell exec."""

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def _signal_ready() -> None:
    with socket.socket(socket.AF_UNIX) as ready:
        ready.connect(os.environ["TASKCOMPENDIUM_FAKE_DOCKER_SOCKET"])
        ready.sendall(b"POST /fixture-ready HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
        ready.recv(1024)


def _run_exec(command: list[str], container: Path, workspace: Path, mapped: Callable[[str], str]) -> None:
    cwd = mapped(command[command.index("-w") + 1]) if "-w" in command else str(container)
    marker = next(index for index, value in enumerate(command) if value == container.name)
    shell = command[marker + 1 :]
    user = command[command.index("--user") + 1] if "--user" in command else None
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
    if "fixture-block" in shell[-1]:
        workspace.joinpath("answer.txt").write_text("12")
        _signal_ready()
        signal.pause()
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
    if os.environ.get("TASKCOMPENDIUM_FAKE_LOG_FAILURE") and "mkdir" in shell[-1] and "/logs/agent" in shell[-1]:
        sys.stderr.write("fixture diagnostic directory unavailable")
        sys.exit(1)
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
    if args[0] == "info":
        print("linux")
        return
    if args[0] == "run":
        name = args[args.index("--name") + 1]
    elif args[0] == "exec":
        name = next(value for value in args if value.startswith("harbor-machine-"))
    elif args[0] == "cp":
        name = next(value.split(":", 1)[0] for value in args if value.startswith("harbor-machine-"))
    else:
        name = args[-1]
    container = root / name
    with (root / "events.jsonl").open("a") as events:
        events.write(json.dumps({"project": name, "command": args}) + "\n")
    if args[0] == "run":
        if "--pull=never" not in args:
            raise ValueError("CPU fixture must never pull an image")
        container.mkdir()
        if os.environ.get("TASKCOMPENDIUM_FAKE_IMAGE_WORKSPACE_SYMLINK"):
            target = container / "image-target"
            target.mkdir()
            (target / "sentinel").write_text("image-original")
            (container / "workspace").symlink_to(target)
        else:
            (container / "workspace").mkdir()
        (container / "running").touch()
        (root / f"{name}.json").write_text(json.dumps({"project": name, "auto_remove": "--rm" in args}))
        if os.environ.get("TASKCOMPENDIUM_FAKE_START_FAILURE"):
            sys.stderr.write("fixture startup unavailable")
            sys.exit(1)
        if os.environ.get("TASKCOMPENDIUM_FAKE_START_BLOCK"):
            _signal_ready()
            signal.pause()
        print(hashlib.sha256(name.encode()).hexdigest())
        return
    if args[0] == "stop":
        if os.environ.get("TASKCOMPENDIUM_FAKE_STOP_FAILURE"):
            sys.stderr.write("fixture stop unavailable")
            sys.exit(1)
        (container / "running").unlink()
        definition = json.loads((root / f"{name}.json").read_text())
        if definition["auto_remove"]:
            shutil.rmtree(container)
        return
    if args[0] == "rm":
        if container.exists():
            closed = root / "closed"
            closed.mkdir(exist_ok=True)
            container.rename(closed / name)
        return

    def mapped(value):
        if value == "/":
            return str(container)
        return re.sub(
            r"/(workspace|logstuff|logs|real-workspace)(?=/|$|[\s\"';])",
            lambda match: str(container / match.group(1)),
            value,
        )

    if args[0] == "exec":
        if not (container / "running").exists():
            raise ValueError("Stopped machines cannot execute commands")
        workspace = Path(mapped(args[args.index("-w") + 1])) if "-w" in args else container
        _run_exec(args, container, workspace, mapped)
        return
    if args[0] == "cp":
        source, destination = args[1:]
        source = mapped(source.split(":", 1)[1]) if source.startswith(name + ":") else source
        destination = mapped(destination.split(":", 1)[1]) if destination.startswith(name + ":") else destination
        if source.endswith("/."):
            shutil.copytree(source[:-2], destination, dirs_exist_ok=True)
        else:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return
    raise ValueError(args)


if __name__ == "__main__":
    main()
