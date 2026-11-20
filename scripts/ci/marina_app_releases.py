# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select, release, and audit maintained Marina apps in CI."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_MAIN_REFRESHES = 3


@dataclass(frozen=True)
class AppPolicy:
    name: str
    release_on_merge: bool
    tests: tuple[str, ...]
    source_paths: tuple[str, ...]

    @property
    def path(self) -> str:
        return f"infra/marina/apps/{self.name}"


class StaleMainError(Exception):
    pass


def policies(root: Path = ROOT) -> dict[str, AppPolicy]:
    """Load the complete release policy and the apps' declared shared inputs."""
    raw = tomllib.loads((root / ".github/marina-app-releases.toml").read_text())
    entries = raw["apps"]
    app_root = root / "infra/marina/apps"
    discovered = {path.parent.name for path in app_root.glob("*/applet.toml")}
    if set(entries) != discovered:
        raise ValueError(f"Release policy apps differ from maintained apps: {sorted(set(entries) ^ discovered)}")
    result = {}
    for name, entry in entries.items():
        if not isinstance(entry.get("release_on_merge"), bool):
            raise ValueError(f"{name}: release_on_merge must be a boolean")
        tests = entry.get("tests")
        if not isinstance(tests, list) or not tests or not all(isinstance(path, str) for path in tests):
            raise ValueError(f"{name}: tests must be a nonempty list of paths")
        if any(not (root / "infra/marina" / path).exists() for path in tests):
            raise ValueError(f"{name}: a declared test path does not exist")
        manifest = tomllib.loads((app_root / name / "applet.toml").read_text())
        result[name] = AppPolicy(
            name=name,
            release_on_merge=entry["release_on_merge"],
            tests=tuple(tests),
            source_paths=tuple(manifest.get("source_paths", [])),
        )
    return result


def changed_apps(changed: set[str], configured: dict[str, AppPolicy]) -> list[str]:
    """Select opted-in apps whose source or declared shared inputs changed."""
    return sorted(
        name
        for name, policy in configured.items()
        if policy.release_on_merge
        and any(
            path.startswith(f"{policy.path}/")
            or any(path == source or path.startswith(f"{source}/") for source in policy.source_paths)
            for path in changed
        )
    )


def git(*arguments: str, root: Path = ROOT) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def marina(*arguments: str, root: Path | None = None) -> dict:
    result = subprocess.run(
        ["uv", "run", "--no-sync", "marina", *arguments, "--json"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if result.stderr:
        sys.stderr.write(result.stderr)
    result.check_returncode()
    return json.loads(result.stdout)


def write_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, sort_keys=True) + "\n")


def safe_markers(root: Path) -> str:
    options = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]["addopts"]
    arguments = shlex.split(options)
    return arguments[arguments.index("-m") + 1]


def refresh_main(root: Path) -> None:
    """Refresh an ephemeral CI checkout after Marina rejects a stale main head."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Refusing to reset a checkout outside GitHub Actions")
    if git("status", "--porcelain", "--untracked-files=all", root=root):
        raise RuntimeError("Refusing to reset a modified checkout")
    git("fetch", "origin", "main", root=root)
    git("reset", "--hard", "FETCH_HEAD", root=root)
    subprocess.run(
        ["uv", "sync", "--locked", "--package", "marin-marina", "--group", "test", "--no-default-groups"],
        cwd=root,
        check=True,
    )


def release(policy: AppPolicy, record_path: Path, *, root: Path = ROOT) -> dict:
    """Validate current main and publish with one guarded attempt per source head."""
    initial_version = None
    for refresh_count in range(MAX_MAIN_REFRESHES + 1):
        current_policy = policies(root)[policy.name] if refresh_count else policy
        if not current_policy.release_on_merge:
            result = {"app": policy.name, "action": "disabled_after_refresh"}
            write_record(record_path, result)
            return result
        try:
            return release_once(current_policy, record_path, initial_version=initial_version, root=root)
        except StaleMainError:
            if refresh_count == MAX_MAIN_REFRESHES:
                raise
            record = json.loads(record_path.read_text())
            initial_version = record["before"]["version"]
            record.update(action="refreshing_main", refresh_count=refresh_count + 1)
            write_record(record_path, record)
            refresh_main(root)
    raise AssertionError("unreachable")


def release_once(policy: AppPolicy, record_path: Path, *, initial_version: int | None, root: Path) -> dict:
    before = marina("release-status", policy.path, root=root)
    record = {"app": policy.name, "action": "inspected", "before": before}
    write_record(record_path, record)
    state = before["state"]
    if state == "current":
        record["action"] = "unchanged"
        write_record(record_path, record)
        return record
    if initial_version is not None and before["version"] != initial_version:
        raise RuntimeError(f"{policy.name}: live version changed during source refresh; inspect before releasing")
    if state == "unrecorded":
        raise RuntimeError(f"{policy.name}: live source has no Git provenance; release manually before automation")
    if state != "different":
        raise RuntimeError(f"{policy.name}: refusing to release source in state {state!r}")

    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-c",
            "infra/marina/pyproject.toml",
            "-m",
            safe_markers(root),
            *(f"infra/marina/{path}" for path in policy.tests),
        ],
        cwd=root,
        check=True,
    )
    record["action"] = "tested"
    write_record(record_path, record)
    validation = marina("release", policy.path, "--dry-run", root=root)
    record.update(
        action="validated", validation={"checks": validation["checks"], "file_count": validation["file_count"]}
    )
    write_record(record_path, record)

    try:
        published = marina("release", policy.path, "--base-version", str(before["version"]), root=root)
    except subprocess.CalledProcessError as error:
        if error.returncode == os.EX_TEMPFAIL:
            raise StaleMainError(f"{policy.name}: source head moved during publication") from error
        raise
    record.update(action="released", published=published, verification="pending")
    write_record(record_path, record)
    try:
        after = marina("release-status", policy.path, root=root)
    except subprocess.CalledProcessError:
        record["verification"] = "failed"
        write_record(record_path, record)
        raise
    record["after"] = after
    record["verification"] = (
        "passed" if after["state"] == "current" and after["version"] == published["version"] else "failed"
    )
    write_record(record_path, record)
    if record["verification"] != "passed":
        raise RuntimeError(f"{policy.name}: published revision is no longer current")
    return record


def drift(policy: AppPolicy, *, root: Path = ROOT) -> dict:
    """Report source commits absent from the selected release."""
    status = marina("release-status", policy.path, root=root)
    result = {"app": policy.name, "state": status["state"], "version": status["version"], "commits": []}
    if status["state"] == "different":
        released = status["released"]
        paths = sorted({policy.path, released["path"], *policy.source_paths, *released["inputs"]})
        output = git(
            "log", "--format=%h %s", f'{released["commit"]}..{status["checkout"]["commit"]}', "--", *paths, root=root
        )
        result["commits"] = output.splitlines()
    elif status["state"] == "unrecorded":
        result["detail"] = "Selected release has no Git provenance; unreleased commits cannot be named."
    elif status["state"] == "modified":
        result["detail"] = "CI checkout has local changes."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--before", required=True)
    select.add_argument("--after", required=True)
    commands.add_parser("all")
    release_command = commands.add_parser("release")
    release_command.add_argument("app")
    release_command.add_argument("--record", type=Path, required=True)
    drift_command = commands.add_parser("drift")
    drift_command.add_argument("app")
    arguments = parser.parse_args()
    configured = policies()
    if arguments.command == "select":
        changed = set(git("diff", "--name-only", arguments.before, arguments.after).splitlines())
        print(
            json.dumps(
                {"include": [{"name": name} for name in changed_apps(changed, configured)]}, separators=(",", ":")
            )
        )
    elif arguments.command == "all":
        print(json.dumps({"include": [{"name": name} for name in sorted(configured)]}, separators=(",", ":")))
    elif arguments.command == "release":
        print(json.dumps(release(configured[arguments.app], arguments.record)))
    else:
        report = drift(configured[arguments.app])
        print(json.dumps(report, sort_keys=True))
        if report["state"] != "current":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
