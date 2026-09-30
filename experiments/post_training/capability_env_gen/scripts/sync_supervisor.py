#!/usr/bin/env python3
"""Bound and supervise durable result syncs without leaking transport errors."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def _stop_group(child: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit while a descendant remains in its process group.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def _read_capture(capture: Path) -> str:
    try:
        return capture.read_text(errors="replace")
    except OSError:
        return "SyncCaptureMissingError\n"


# ---------------------------------------------------------------------------
# Network-config sentinel (2026-09-29).
#
# shard-033-g2: a builder's local selfcheck unpacked a fixture rootfs whose tar members
# were absolute (/etc/resolv.conf, /etc/hosts, ...) over the job container's own /etc.
# DNS broke and every durable sync failed for 30 minutes until a human cancelled the
# job. Agents are now confined (capability_pipeline/builder_confinement.py). This is
# the cheap second line: remember the files the transport depends on, say so loudly the
# moment they change, and put the job's own bytes back so the syncs keep working.
# ---------------------------------------------------------------------------
WATCHED_ETC = ("/etc/resolv.conf", "/etc/hosts")


def _digest(data: bytes | None) -> str | None:
    import hashlib

    return None if data is None else hashlib.sha256(data).hexdigest()


class EtcSentinel:
    def __init__(self, paths: tuple[str, ...] = WATCHED_ETC, *, restore: bool = True) -> None:
        self.restore = restore
        self.baseline: dict[Path, bytes] = {}
        for name in paths:
            try:
                self.baseline[Path(name)] = Path(name).read_bytes()
            except OSError:
                continue
        self._reported: set[tuple[str, str | None]] = set()

    def check(self) -> list[dict]:
        """Events for watched files that differ from their start-of-job bytes."""
        events = []
        for path, original in self.baseline.items():
            try:
                current: bytes | None = path.read_bytes()
            except OSError:
                current = None
            if current == original:
                continue
            event: dict = {
                "event": "etc_changed",
                "path": str(path),
                "baseline_sha256": _digest(original),
                "current_sha256": _digest(current),
            }
            if self.restore:
                try:
                    # In place: kubelet bind-mounts these files, so a rename cannot land.
                    with path.open("wb") as output:
                        output.write(original)
                    event["restored"] = True
                except OSError as error:
                    event["restored"] = False
                    event["restore_error"] = type(error).__name__
            # A restored change is reported once per occurrence; an unrestorable one once
            # per distinct content, so a persistent fault does not flood the log.
            key = (str(path), event["current_sha256"])
            if event.get("restored") or key not in self._reported:
                events.append(event)
            self._reported.add(key)
        return events


def _write_capture(capture: Path, text: str) -> None:
    try:
        capture.write_text(text)
    except OSError:
        pass


def run_sync(command: list[str], capture: Path, *, deadline_seconds: int, cwd: str | None = None) -> int:
    """Run one sync in its own process group; TERM and deadlines reap descendants."""
    with capture.open("wb") as output:
        child: subprocess.Popen[bytes] | None = None
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        previous_handler = signal.getsignal(signal.SIGTERM)

        def stop(_signum: int, _frame: object) -> None:
            if child is not None:
                _stop_group(child)
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, stop)
        try:
            child = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask),  # noqa: PLW1509 - single-threaded supervisor
            )
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            try:
                return child.wait(timeout=deadline_seconds)
            except subprocess.TimeoutExpired:
                _stop_group(child)
                output.write(b"\nSyncDeadlineExceededError\n")
                return 124
        finally:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            if child is not None:
                _stop_group(child)
            signal.signal(signal.SIGTERM, previous_handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


# ---------------------------------------------------------------------------
# Self-contained sync runtime (2026-09-29).
#
# Agents running local shell commands with /app as their workdir were observed deleting
# /app mid-job: first the stage dir, then /app/.venv and /app/lib (rigging). Every later
# sync then died with ModuleNotFoundError ('fsspec', 'rigging') and the final sync failed,
# losing the job's results. `build-runtime` runs ONCE at worker startup, inside the intact
# /app environment, and copies exactly the installed distributions sync_results.py
# imports (plus their declared dependency closure, the editable rigging source and the
# cluster-config YAMLs rigging reads) into a fresh venv outside /app. Versions are
# therefore exactly those of /app/.venv; no network access is needed.
# ---------------------------------------------------------------------------

# Everything sync_results.py touches. The probe also constructs an S3 filesystem so the
# lazy imports on that path (s3fs -> aiobotocore -> botocore) are recorded too.
_RUNTIME_PROBE = """
import fsspec, s3fs, aiohttp, aiobotocore
from aiobotocore.httpsession import AIOHTTPSession
from rigging.filesystem.cluster_config import StoreType, store_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
store_config(StoreType.COREWEAVE)  # fails unless the cluster-config YAMLs are reachable
configure_coreweave_s3()
fsspec.core.url_to_fs("s3://sync-runtime-probe/probe")  # builds the client; no request
"""


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def build_runtime(target: Path, config_dir: Path, *, outside: tuple[Path, ...] = (), uv: str = "uv") -> Path:
    """Build a venv at *target* holding only the sync dependencies; return its python.

    Must run under the interpreter of the full project environment (the source of the
    copied files). Fails loudly on anything it cannot place outside that environment, and
    when the base interpreter itself lives under any *outside* directory (e.g. /app).
    """
    import importlib.metadata as md
    import shutil
    import sysconfig

    # Run the probe in a fresh interpreter so only what it imports is attributed, not
    # whatever this process (or a test runner hosting it) happens to have loaded.
    report = _RUNTIME_PROBE + (
        "\nimport json, sys\n"
        "def _top(name):\n"
        "    m = sys.modules.get(name.split('.')[0])\n"
        "    paths = list(getattr(m, '__path__', []) or [])\n"
        "    return paths[0] if paths else getattr(m, '__file__', None)\n"
        "print(json.dumps([[n, m.__file__, _top(n)] for n, m in list(sys.modules.items())\n"
        "                  if getattr(m, '__file__', None)]))\n"
    )
    imported = json.loads(subprocess.run(
        [sys.executable, "-c", report], check=True, stdout=subprocess.PIPE, text=True,  # stderr to the log
    ).stdout.strip().splitlines()[-1])
    base = Path(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))
    project_prefix = Path(os.path.realpath(sys.prefix))
    for forbidden in (project_prefix, *(Path(os.path.realpath(p)) for p in outside)):
        if base.is_relative_to(forbidden):
            raise RuntimeError(f"base interpreter {base} lives under {forbidden}")
    stdlib = tuple(os.path.realpath(sysconfig.get_paths()[k]) for k in ("stdlib", "platstdlib"))

    # Map every installed file to its distribution, then attribute imported modules.
    owner: dict[str, md.Distribution] = {}
    dists: dict[str, md.Distribution] = {}
    for dist in md.distributions():
        name = _norm(dist.metadata["Name"] or "")
        if not name or name in dists:
            continue  # first on sys.path wins, matching the import system
        dists[name] = dist
        for file in dist.files or ():
            owner[os.path.realpath(dist.locate_file(file))] = dist
    wanted: set[str] = set()
    source_packages: dict[str, Path] = {}
    for module_name, file, top_path in imported:
        real = os.path.realpath(file)
        if real.startswith(stdlib) and "site-packages" not in real:
            continue
        if real in owner:
            wanted.add(_norm(owner[real].metadata["Name"]))
            continue
        # Not a recorded file: an editable install (rigging lives at /app/lib/rigging/src).
        top = module_name.split(".")[0]
        if top in {"_virtualenv", "sitecustomize", "usercustomize"}:
            continue  # interpreter start-up hooks; the new venv has its own
        if not top_path:
            raise RuntimeError(f"cannot place imported module {module_name} ({file})")
        source_packages[top] = Path(top_path)  # a package dir, or a single-file module

    def editable(dist: md.Distribution) -> bool:
        return '"editable": true' in (dist.read_text("direct_url.json") or "").replace("'", '"')

    # Close over declared runtime requirements (extras skipped) so cross-distribution
    # lazy imports the probe did not exercise are present too. Editable (source) packages
    # are placed by source above; their full requirement lists are not needed.
    pending = [name for name in wanted if not editable(dists[name])]
    while pending:
        for requirement in dists[pending.pop()].requires or ():
            if re.search(r"\bextra\s*==", requirement):
                continue
            match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
            name = _norm(match.group(1)) if match else ""
            if name in dists and name not in wanted:
                wanted.add(name)
                if not editable(dists[name]):
                    pending.append(name)

    if target.exists():
        shutil.rmtree(target)
    subprocess.run([uv, "venv", "--quiet", "--no-project", "--python", str(base), str(target)], check=True)
    python = target / "bin" / "python"
    site = Path(subprocess.run(
        [str(python), "-E", "-s", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    for name in sorted(wanted):
        dist = dists[name]
        for file in dist.files or ():
            parts = Path(str(file)).parts
            # Skip scripts outside site-packages, bytecode (regenerated on first import) and,
            # for editable installs, the .pth/finder that would point back into /app.
            if ".." in parts or "__pycache__" in parts:
                continue
            if editable(dist) and (str(file).endswith(".pth") or parts[-1].startswith("__editable__")):
                continue
            src = Path(dist.locate_file(file))
            if not src.is_file():
                continue
            dest = site / Path(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    for top, path in source_packages.items():
        if path.is_file():
            shutil.copy2(path, site / path.name)
        else:
            shutil.copytree(path, site / top, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    if "rigging" in source_packages:
        # rigging's installed-wheel layout: cluster configs at rigging/clusters (see
        # rigging.filesystem.cluster_config._bundled_cluster_config_dir).
        clusters = site / "rigging" / "clusters"
        clusters.mkdir(parents=True, exist_ok=True)
        configs = sorted([*config_dir.glob("*.yaml"), *config_dir.glob("*.yml")])
        if not configs:
            raise RuntimeError(f"no cluster-config YAMLs in {config_dir}")
        for config in configs:
            shutil.copy2(config, clusters / config.name)
    for pth in site.glob("*.pth"):
        for line in pth.read_text(errors="replace").splitlines():
            if line.startswith("/") and not Path(line).resolve().is_relative_to(target.resolve()):
                raise RuntimeError(f"{pth.name} points outside the sync runtime: {line}")
    verify_runtime(python, cwd=target)
    return python


def verify_runtime(python: Path, *, cwd: Path) -> None:
    """Run the probe in *python* and fail unless every imported module is its own."""
    check = _RUNTIME_PROBE + (
        "\nimport os, sys\n"
        "roots = tuple(os.path.realpath(p) + os.sep for p in (sys.prefix, sys.base_prefix))\n"
        "leaks = sorted({os.path.realpath(m.__file__) for m in list(sys.modules.values())\n"
        "                if getattr(m, '__file__', None) and not os.path.realpath(m.__file__).startswith(roots)})\n"
        "assert not leaks, f'modules outside the sync runtime: {leaks[:5]}'\n"
        "print(f'sync runtime ok: {len(sys.modules)} modules under {sys.prefix}')\n"
    )
    # -E -s: ignore PYTHONPATH (the worker later exports the stage dir) and user site.
    subprocess.run([str(python), "-E", "-s", "-c", check], check=True, cwd=cwd)


def build_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="sync_supervisor.py build-runtime")
    parser.add_argument("--target", required=True, help="new venv directory (replaced)")
    parser.add_argument("--config-dir", required=True, help="cluster-config YAMLs, e.g. <project>/config")
    parser.add_argument("--outside", action="append", default=[], help="directory the base interpreter must not be under")
    args = parser.parse_args(argv)
    started = time.monotonic()
    python = build_runtime(Path(args.target), Path(args.config_dir), outside=tuple(Path(p) for p in args.outside))
    print(f"sync runtime built: python={python} seconds={time.monotonic() - started:.1f}", flush=True)
    return 0


def main() -> int:
    if sys.argv[1:2] == ["build-runtime"]:
        return build_main(sys.argv[2:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--state", required=True)
    interpreter_group = parser.add_mutually_exclusive_group()
    interpreter_group.add_argument("--project", help="frozen project environment for each sync (legacy)")
    interpreter_group.add_argument(
        "--python",
        help="self-contained sync runtime interpreter (see build-runtime); each sync runs "
        "with it directly, from this script's directory, independent of the project tree",
    )
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--deadline-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.final and args.loop:
        parser.error("final sync cannot loop")
    if args.interval_seconds < 1 or args.deadline_seconds < 1:
        parser.error("interval and deadline must be positive")
    source, state = Path(args.source), Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    log = source / "controller" / "uploader.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    capture = state / "sync-last.log"
    mode = "final" if args.final else "incremental"
    # With --python the cwd is this script's private copy directory (worker.sh: under
    # $WORK_ROOT), never the project tree that can vanish under a long-lived supervisor.
    stable_cwd = (
        args.project if args.project and Path(args.project).is_dir() else str(Path(__file__).resolve().parent)
    )
    os.chdir(stable_cwd)
    sentinel = EtcSentinel()
    while True:
        for event in sentinel.check():
            print(
                f"ALERT: {event['path']} changed since job start "
                f"(restored={event.get('restored')}); something in the job overwrote it",
                flush=True,
            )
            try:
                with log.open("a") as output:
                    output.write(json.dumps(event, separators=(",", ":")) + "\n")
            except OSError:
                pass
        # Something in the job can delete the state (and even the log) directory under a
        # long-lived supervisor (2026-09-29: shard-053-h1 died on a missing sync-last.log
        # and synced nothing more that attempt). Recreate both every round; a vanished
        # capture is a failed round, never a supervisor crash.
        state.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        if args.python:
            # -E -s: PYTHONPATH (the worker exports the stage dir before the final sync)
            # and user site must not pull anything back in from the project tree.
            interpreter = [args.python, "-E", "-s"]
        elif args.project:
            interpreter = ["uv", "run", "--project", args.project, "--frozen", "python3"]
        else:
            interpreter = [sys.executable]
        command = [
            *interpreter,
            str(Path(__file__).with_name("sync_results.py")),
            "sync", "--source", str(source), "--destination", args.destination,
            "--state", str(state),
        ]
        if args.final:
            command.append("--final")
        # Run each sync from a directory that cannot disappear. The worker's own cwd can be
        # removed or replaced under a long-lived supervisor (seen 2026-09-28: every sync then
        # failed with "getcwd: cannot access parent directories" and uv exiting rc=2).
        try:
            result = run_sync(command, capture, deadline_seconds=args.deadline_seconds, cwd=stable_cwd)
        except OSError as error:
            result = 1
            _write_capture(capture, f"SyncCaptureUnavailableError {type(error).__name__}\n")
        captured = _read_capture(capture)
        classes = sorted(set(re.findall(r"\b([A-Za-z0-9_]*(?:Error|Exception))\b", captured)))
        receipt = {"event": "durable_sync", "ok": result == 0, "mode": mode}
        if result != 0:
            receipt["error_classes"] = ",".join(classes) or "unknown"
        try:
            with log.open("a") as output:
                output.write(json.dumps(receipt, separators=(",", ":")) + "\n")
        except OSError as error:
            print(f"uploader log unavailable: {type(error).__name__}", flush=True)
        if not args.loop:
            return 0 if result == 0 else 1
        if result != 0:
            # The capture holds only sync_results' own JSON/stderr lines (file paths and
            # error class names, never credentials); its tail makes an "unknown" failure
            # diagnosable from the job log instead of silently repeating.
            tail = " | ".join(
                line.strip()[:200]
                for line in captured.strip().splitlines()[-3:]
            )
            print(f"upload heartbeat failed; will retry (rc={result} error_classes={receipt.get('error_classes')}) tail: {tail}", flush=True)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
