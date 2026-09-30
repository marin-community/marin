"""Confine agent (omp) processes to their own workspace in the construction job.

Why (catalog-full-construct-003, 2026-09-29).  omp's read/write/edit/bash tools run
in THIS container, not in the sandbox.  Before this module that container ran every
agent as root, with SYS_PTRACE, host networking, a world-writable /app, a non-sticky
/tmp, node-shared hostPath caches (/uv/cache, /cargo, /hf/cache, /cache) and the full
job environment (object-store keys included).  Builder code written to run "inside
the sandbox as root" was then run on the host instead:

* shard-033-g2 ~21:17Z: d02.security.isolation-3's verifier selfcheck unpacked its
  fixture tarball with ``shutil.unpack_archive``.  build-fixture.py had written the
  members as absolute names (``/etc/resolv.conf``, ``/etc/passwd``, ...) and Python
  3.12's default tar filter honours them, so the job's own /etc was overwritten.
  DNS broke and every durable sync failed until the job was cancelled.
* shard-055-c6 18:44:32Z: d03.astronomy.gravitational_waves-3 ran its
  ``ContainerPath`` "dry run" on the host; ``reset_workspace()`` rmtree'd the
  TaskSpec workdir, /app, i.e. the stage, /app/.venv and /app/lib.
* A read-only census of the 90 live pods found apt installs, a useradd with a
  ``[trusted=yes]`` apt source, /etc/cron.d entries, pip installs into the system
  interpreter and sandbox layouts (/input /opt/runtime /snapshot /work ...) at /.

Every file operation an agent performs is a syscall from omp or a descendant, so the
class is closed at the process boundary rather than per tool or per archive format:
each session runs as an unprivileged uid derived from its workspace, with no
capabilities, ``no_new_privs`` and an allowlisted environment.  Only its workspace,
its session directory, explicitly granted directories and a private state directory
(home, tmp, private toolchain copy) are owned by that uid.  Everything else is
root-owned and read-only to it.  Shared world-writable directories get the sticky
bit, so one uid cannot delete another's (or root's) entries, and /app loses o+w.

Modes (``CAPABILITY_BUILDER_CONFINEMENT``):
  unset/"uid"  when the controller is root: drop privileges (fail closed if we cannot)
  "off"        explicit operator escape hatch: no privilege drop (env is still filtered)
When the controller is not root there is nothing to drop; the environment is still
filtered.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

MODE_ENV = "CAPABILITY_BUILDER_CONFINEMENT"
STATE_ROOT_ENV = "CAPABILITY_BUILDER_STATE_ROOT"
DEFAULT_STATE_ROOT = Path("/tmp/capability-agents")
# Agent uids live far above any image or cluster account.  One deterministic slot per
# workspace; concurrent collisions inside this process probe to the next free slot.
UID_BASE = 2_000_000
UID_SPAN = 1 << 20
PREFLIGHT_UID = UID_BASE - 1
AGENT_UMASK = 0o022

# World-writable directories that must be sticky, and directories no agent may write.
STICKY_DIRECTORIES = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", "/iris/outputs")
PRIVATE_DIRECTORIES = ("/app",)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Credentials an agent legitimately needs.  Everything else that looks like a
# credential is dropped even if a prefix rule would admit it.
ALLOWED_SECRETS = frozenset(
    {
        # The model token.  omp reads it from the file models.yml names; builders'
        # native judge transport reads it from this variable.  Same credential.
        "GLM_API_TOKEN",
        # Sandbox provider credentials: builders reach silo/Daytona through dt.py.
        "SILO_API_TOKEN",
        "DAYTONA_API_KEY",
        # Research providers the omp overlay selects (submit.sh passes them for this).
        "PARALLEL_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    }
)
_ALLOWED_EXACT = ALLOWED_SECRETS | frozenset(
    {
        "PATH",
        "LANG",
        "LANGUAGE",
        "TZ",
        "TERM",
        "NO_COLOR",
        "FORCE_COLOR",
        "GLM_BASE_URL",
        "GLM_TIER",
        "CAPABILITY_SANDBOX_PROVIDER",
        "SILO_BROKER_RESOLVE_URL",
        "SILO_BROKER_URL",
        "DAYTONA_API_URL",
        "DAYTONA_SERVER_URL",
        "DAYTONA_TARGET",
        "DT_KEY",  # a sandbox label, not a credential (dt.py labels())
        # Read-only tool and contract locations named by prompts and helpers.
        "CAPABILITY_DAYTONA_TOOLS",
        "CAPABILITY_OMP_CONFIG",
        "CAPABILITY_TASK_CONTRACT",
        "CAPABILITY_TASK_SPEC_LOCK",
        "TASKCOMPENDIUM_SOURCE",
        "TASKCOMPENDIUM_SHELLSIM_BRIDGE",
        "TOOLS_DIR",
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONIOENCODING",
        "UV_PYTHON_INSTALL_DIR",
        "UV_CONCURRENT_BUILDS",
        "UV_CONCURRENT_INSTALLS",
        # Parallelism pins: nproc reports the node's cores, not the task's request.
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "CARGO_BUILD_JOBS",
        "MAKEFLAGS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "GOMAXPROCS",
        "PYTEST_XDIST_AUTO_NUM_WORKERS",
        "NPM_CONFIG_JOBS",
        "TOKENIZERS_PARALLELISM",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
    }
)
_ALLOWED_PREFIXES = ("LC_", "DT_")
# Who the agent is.  Replaced by private values when privileges are dropped; kept as
# they are when the agent runs as the controller's own user (off / unprivileged).
_IDENTITY = (
    "HOME",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
    "UV_CACHE_DIR",
    "CARGO_HOME",
    "RUSTUP_HOME",
)
_SECRETISH = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|PRIVATE|AUTH|(^|_)KEY($|_)|API_KEY|ACCESS_KEY", re.IGNORECASE)
# Never admissible, whatever an allowlist edit says: object-store and cloud keys, the
# Iris job env (which embeds every -e value) and filesystem credential blobs.
_FORBIDDEN = re.compile(r"^(CW_|AWS_|FSSPEC_|WANDB_|IRIS_|GOOGLE_|GCLOUD_|AZURE_|HF_TOKEN|HUGGING_FACE)", re.IGNORECASE)


class ConfinementError(RuntimeError):
    pass


def agent_environment(
    base: Mapping[str, str] | None = None, overrides: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The allowlisted environment an agent process may see."""
    base = os.environ if base is None else base
    env: dict[str, str] = {}
    for name, value in base.items():
        if name not in _ALLOWED_EXACT and not name.startswith(_ALLOWED_PREFIXES):
            continue
        if _SECRETISH.search(name) and name not in ALLOWED_SECRETS and name != "DT_KEY":
            continue
        env[name] = value
    env.update(overrides or {})
    leaked = sorted(name for name in env if _FORBIDDEN.search(name))
    if leaked:
        raise ConfinementError(f"agent environment would carry forbidden variables: {leaked}")
    return env


# ---------------------------------------------------------------------------
# Mode, identity and paths
# ---------------------------------------------------------------------------
def mode() -> str:
    """"uid" (drop privileges), "off" (explicitly disabled) or "unprivileged"."""
    value = os.environ.get(MODE_ENV, "").strip().lower()
    if value in {"off", "0", "false", "none", "disabled"}:
        return "off"
    if value not in {"", "uid", "1", "true", "on"}:
        raise ConfinementError(f"unknown {MODE_ENV}={value!r}; use 'uid' or 'off'")
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return "unprivileged"
    return "uid"


def state_root() -> Path:
    return Path(os.environ.get(STATE_ROOT_ENV) or DEFAULT_STATE_ROOT)


def workspace_key(workspace: Path) -> str:
    return hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()


def agent_state_dir(workspace: Path) -> Path:
    """Private per-workspace directory: home/, tmp/ and toolchain/ live here."""
    return state_root() / workspace_key(workspace)[:20]


def preferred_uid(workspace: Path) -> int:
    return UID_BASE + int(workspace_key(workspace)[:12], 16) % UID_SPAN


_LOCK = threading.Lock()
_ACTIVE: dict[int, list] = {}  # uid -> [workspace key, reference count]
_GRANTS: dict[str, list[Path]] = {}  # workspace key -> create-only output directories
_PROCESS_SETUP: dict[str, object] = {}


def _allocate_uid(key: str) -> int:
    start = int(key[:12], 16) % UID_SPAN
    with _LOCK:
        for probe in range(UID_SPAN):
            uid = UID_BASE + (start + probe) % UID_SPAN
            entry = _ACTIVE.get(uid)
            if entry is None:
                _ACTIVE[uid] = [key, 1]
                return uid
            if entry[0] == key:
                entry[1] += 1
                return uid
    raise ConfinementError("no free agent uid")


def _release_uid(uid: int) -> None:
    with _LOCK:
        entry = _ACTIVE.get(uid)
        if entry is not None:
            entry[1] -= 1
            if entry[1] <= 0:
                del _ACTIVE[uid]


def _sole_holder(uid: int) -> bool:
    with _LOCK:
        entry = _ACTIVE.get(uid)
        return entry is not None and entry[1] == 1


@contextlib.contextmanager
def grant_create(workspace: Path, *roots: Path) -> Iterator[None]:
    """Let the agent session for ``workspace`` create new files directly in ``roots``.

    For callers whose agent must write an output outside its cwd (repair writes its
    receipt into a repair root that also holds the controller's immutable before-state).
    During the session each root is root-owned with mode 1777: the agent can create
    entries but cannot delete or rename root's (sticky bit), and the kernel's
    protected_symlinks refuses to let root follow a link the agent plants there.
    Ownership of the existing contents is never handed over.
    """
    key = workspace_key(workspace)
    added = [Path(root) for root in roots]
    with _LOCK:
        _GRANTS.setdefault(key, []).extend(added)
    try:
        yield
    finally:
        with _LOCK:
            current = _GRANTS.get(key, [])
            for root in added:
                if root in current:
                    current.remove(root)
            if not current:
                _GRANTS.pop(key, None)


def _granted(key: str) -> list[Path]:
    with _LOCK:
        return list(_GRANTS.get(key, []))


# ---------------------------------------------------------------------------
# Filesystem operations (root side)
# ---------------------------------------------------------------------------
def _log(event: dict) -> None:
    print(json.dumps({"component": "builder_confinement", **event}, sort_keys=True), file=sys.stderr, flush=True)


def chown_tree(root: Path, uid: int, gid: int, *, chown=None) -> dict[str, int]:
    """Give ``uid`` ownership of ``root`` without ever following a link out of it.

    Symlinks are re-owned themselves (lchown) and never traversed; a regular file with
    more than one link that the agent does not already own is skipped, because it
    may be the same inode as a file elsewhere (a root-era hard link to /etc/shadow).
    """
    chown = chown or os.chown
    counts = {"changed": 0, "skipped_hardlinks": 0}
    root = Path(root)
    if not root.exists() and not root.is_symlink():
        return counts
    stack = [root]
    while stack:
        path = stack.pop()
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1 and info.st_uid != uid:
            counts["skipped_hardlinks"] += 1
            continue
        if info.st_uid != uid or info.st_gid != gid:
            chown(path, uid, gid, follow_symlinks=False)
            counts["changed"] += 1
        if stat.S_ISDIR(info.st_mode):
            try:
                stack.extend(path.iterdir())
            except (FileNotFoundError, NotADirectoryError):
                continue
    return counts


def ensure_traversable(path: Path) -> list[str]:
    """Add o+x to root-owned ancestors of ``path`` that an agent could not pass."""
    changed = []
    current = Path(path).resolve().parent
    while True:
        try:
            info = current.stat()
        except FileNotFoundError:
            info = None
        if info is not None and info.st_uid == 0 and not info.st_mode & stat.S_IXOTH:
            os.chmod(current, stat.S_IMODE(info.st_mode) | stat.S_IXOTH)
            changed.append(str(current))
        if current.parent == current:
            return changed
        current = current.parent


def ensure_readable(path: Path) -> None:
    """A controller-written input the agent must read (prompt, overlay, binary)."""
    path = Path(path)
    ensure_traversable(path)
    info = path.stat()
    wanted = stat.S_IROTH | (stat.S_IXOTH if stat.S_ISDIR(info.st_mode) or info.st_mode & stat.S_IXUSR else 0)
    if info.st_mode & wanted != wanted:
        os.chmod(path, stat.S_IMODE(info.st_mode) | wanted)


def harden_shared_directories(
    sticky: Iterable[str] = STICKY_DIRECTORIES, private: Iterable[str] = PRIVATE_DIRECTORIES
) -> list[dict]:
    """Sticky bit on world-writable shared dirs; no group/other write on private ones."""
    changes = []
    for name in sticky:
        path = Path(name)
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_IWOTH and not info.st_mode & stat.S_ISVTX:
            os.chmod(path, stat.S_IMODE(info.st_mode) | stat.S_ISVTX)
            changes.append({"path": name, "before": oct(stat.S_IMODE(info.st_mode)), "added": "sticky"})
    for name in private:
        path = Path(name)
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(info.st_mode) and info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            os.chmod(path, stat.S_IMODE(info.st_mode) & ~(stat.S_IWGRP | stat.S_IWOTH))
            changes.append({"path": name, "before": oct(stat.S_IMODE(info.st_mode)), "removed": "group/other write"})
    return changes


def remove_escaping_symlinks(root: Path, allowed: Iterable[Path] = ()) -> list[str]:
    """Delete symlinks directly inside ``root`` that point outside the agent's trees.

    The controller writes top-level files into agent-writable roots after a session
    (attempt logs, correction prompts, retained receipts).  Run only after every agent
    process is gone, so nothing can re-plant a link in between.
    """
    root = Path(root)
    removed = []
    bounds = [root.resolve(), *(Path(path).resolve() for path in allowed)]
    try:
        entries = list(root.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return removed
    for entry in entries:
        if not entry.is_symlink():
            continue
        try:
            # Resolve through every link in the chain, as the kernel would on open().
            target = entry.resolve()
        except (OSError, RuntimeError):  # loops count as escaping
            target = None
        if target is None or not any(target == bound or target.is_relative_to(bound) for bound in bounds):
            entry.unlink()
            removed.append(str(entry))
    return removed


def remove_agent_symlinks(root: Path, uid: int) -> list[str]:
    """Delete every symlink the agent created directly inside a create-only root."""
    removed = []
    try:
        entries = list(Path(root).iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return removed
    for entry in entries:
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) and info.st_uid == uid:
            entry.unlink()
            removed.append(str(entry))
    return removed


def _real_parent(root: Path, relative: str) -> Path:
    """``root/relative``'s parent with every component a real directory (no links)."""
    parts = Path(relative).parts
    if Path(relative).is_absolute() or ".." in parts or not parts:
        raise ValueError(f"unsafe path under agent-owned root: {relative!r}")
    current = Path(root)
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            current.unlink()
        current.mkdir(exist_ok=True)
    return current


def _clear(destination: Path) -> None:
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.is_dir():
        shutil.rmtree(destination)
    elif destination.exists():
        destination.unlink()


def write_private(root: Path, relative: str, data: bytes, mode: int, *, owner: int | None = None) -> Path:
    """Root writes a file inside an agent-owned tree without following any link."""
    destination = _real_parent(root, relative) / Path(relative).name
    _clear(destination)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        os.write(descriptor, data)
        os.fchmod(descriptor, mode)
        if owner is not None:
            os.fchown(descriptor, owner, owner)
    finally:
        os.close(descriptor)
    return destination


def stage_file(root: Path, relative: str, source: Path) -> Path:
    """Controller copy into an agent-owned tree that cannot be redirected by a link.

    Any symlinked or non-directory component under ``root`` is replaced by a real
    directory, and an existing destination (link or not) is removed before the copy.
    """
    destination = _real_parent(root, relative) / Path(relative).name
    _clear(destination)
    shutil.copy2(source, destination)
    return destination


def private_toolchain(shared_root: Path | None, workspace: Path) -> Path | None:
    """A per-agent copy of the builder toolchain, when agents cannot share one.

    Agents run ``uv`` and edit files in the "official package root".  As distinct uids
    they cannot share one writable tree (uv stages 0700 directories), so each agent
    gets its own copy inside its private state directory, owned by it at session start.
    """
    if shared_root is None or mode() != "uid":
        return shared_root
    target = agent_state_dir(workspace) / "toolchain"
    marker = target / ".capability-toolchain-source"
    source_id = str(Path(shared_root).resolve())
    if target.is_symlink():  # the state dir is agent-owned: never copy through a link
        target.unlink()
    if target.is_dir() and not marker.is_symlink() and marker.is_file() and marker.read_text() == source_id:
        return target
    _clear(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        shared_root,
        target,
        symlinks=True,
        ignore=shutil.ignore_patterns(".venv*", "__pycache__", ".pytest_cache", "target"),
    )
    marker.write_text(source_id)
    return target


# ---------------------------------------------------------------------------
# omp home
# ---------------------------------------------------------------------------
_CAT_PATH = re.compile(r"!cat\s+([^\s\"']+)")


def render_agent_home(source_home: Path, home: Path, *, owner: int | None = None) -> dict:
    """Copy omp's provider config into the agent home, rewriting secret file paths.

    models.yml names its token as ``!cat <path>``; the root copy of that file is
    under /root (0700), which an agent uid cannot read.  The home is agent-owned, so
    every write here refuses to follow a link the agent may have planted.
    """
    source_agent = Path(source_home) / ".omp" / "agent"
    copied: list[str] = []
    if not source_agent.is_dir():
        return {"copied": copied}
    for source in sorted(source_agent.iterdir()):
        if not source.is_file() or source.suffix not in {".yml", ".yaml", ".json"}:
            continue
        text = source.read_text()
        for index, match in enumerate(_CAT_PATH.finditer(text)):
            secret = Path(match.group(1))
            if not secret.is_file():
                continue
            private = write_private(
                home, f".omp/secret-{index}-{secret.name}", secret.read_bytes(), 0o400, owner=owner
            )
            text = text.replace(match.group(0), f"!cat {private}")
        write_private(home, f".omp/agent/{source.name}", text.encode(), 0o600, owner=owner)
        copied.append(source.name)
    return {"copied": copied}


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------
def processes_of(uid: int, proc: Path = Path("/proc")) -> list[int]:
    pids = []
    try:
        entries = list(proc.iterdir())
    except FileNotFoundError:
        return pids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("Uid:"):
                    if str(uid) in line.split()[1:]:
                        pids.append(int(entry.name))
                    break
        except (OSError, ValueError):
            continue
    return pids


def reap(uid: int) -> int:
    """SIGKILL every process still running as the agent uid (background daemons)."""
    killed = 0
    for _ in range(3):
        pids = processes_of(uid)
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
    return killed


def privilege_drop_argv(setpriv: str, uid: int, gid: int) -> list[str]:
    return [
        setpriv,
        f"--reuid={uid}",
        f"--regid={gid}",
        "--clear-groups",
        "--no-new-privs",
        "--inh-caps=-all",
        "--bounding-set=-all",
        "--",
    ]


def _preflight(setpriv: str, root: Path) -> dict:
    """Demonstrate the drop works: the probe must be denied a root-owned file."""
    probe = root / f".preflight-{os.getpid()}"
    probe.write_text("root-owned\n")
    os.chmod(probe, 0o644)
    script = (
        'id -u; awk \'/^(NoNewPrivs|CapEff):/{print $1 $2}\' /proc/self/status; '
        'if printf x >> "$1" 2>/dev/null; then echo WRITABLE; else echo DENIED; fi'
    )
    try:
        completed = subprocess.run(
            [*privilege_drop_argv(setpriv, PREFLIGHT_UID, PREFLIGHT_UID), "/bin/sh", "-c", script, "sh", str(probe)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
            cwd="/",
        )
        tokens = completed.stdout.split()
        intact = probe.read_text() == "root-owned\n"
    finally:
        probe.unlink(missing_ok=True)
    record = {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-400:],
        "stderr": completed.stderr.strip()[-400:],
    }
    ok = (
        completed.returncode == 0
        and tokens[:1] == [str(PREFLIGHT_UID)]
        and "NoNewPrivs:1" in tokens
        and "CapEff:0000000000000000" in tokens
        and "DENIED" in tokens
        and intact
    )
    if not ok:
        raise ConfinementError(f"agent privilege drop did not take effect: {record}")
    return record


def open_root_toolchains(home: Path | None = None) -> dict[str, str]:
    """Let agents run the rustup toolchain installed under root's home, read-only.

    The worker image installs rustup in /root (0700), and builders do build Rust
    locally.  Root's home becomes traverse-only (no listing) and omp's own state under
    it is closed to others; agents then get RUSTUP_HOME and a private CARGO_HOME.
    """
    home = Path(home or os.environ.get("HOME") or "/root")
    rustup = Path(os.environ.get("RUSTUP_HOME") or home / ".rustup")
    if not rustup.is_dir():
        return {}
    omp_state = home / ".omp"
    if omp_state.is_dir() and not omp_state.is_symlink():
        os.chmod(omp_state, stat.S_IMODE(omp_state.stat().st_mode) & ~0o077)
    if rustup.is_relative_to(home):
        info = home.stat()
        os.chmod(home, stat.S_IMODE(info.st_mode) | stat.S_IXOTH)
    return {"RUSTUP_HOME": str(rustup)}


def _setup_process() -> dict:
    """Once per controller process: harden shared dirs, create the state root, preflight."""
    with _LOCK:
        if "setup" in _PROCESS_SETUP:
            return _PROCESS_SETUP["setup"]  # type: ignore[return-value]
        setpriv = shutil.which("setpriv", path="/usr/bin:/bin:/usr/sbin:/sbin") or shutil.which("setpriv")
        if not setpriv:
            raise ConfinementError(
                "setpriv (util-linux) is unavailable; refusing to run agents as root. "
                f"Set {MODE_ENV}=off only to accept unconfined agents deliberately."
            )
        changes = harden_shared_directories()
        toolchain_env = open_root_toolchains()
        root = state_root()
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            root.unlink()
        root.mkdir(parents=True, exist_ok=True)
        os.chown(root, 0, 0, follow_symlinks=False)
        os.chmod(root, 0o711)  # traversable to reach one's own dir, not listable
        preflight = _preflight(setpriv, root)
        setup = {"setpriv": setpriv, "hardened": changes, "toolchain_env": toolchain_env, "preflight": preflight}
        _log({"event": "confinement_ready", **setup})
        _PROCESS_SETUP["setup"] = setup
        return setup


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@dataclass
class AgentSession:
    mode: str
    env: dict[str, str]
    uid: int | None = None
    gid: int | None = None
    prefix: list[str] = field(default_factory=list)
    umask: int = AGENT_UMASK
    writable: list[Path] = field(default_factory=list)
    create_only: list[tuple[Path, int]] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    closed: bool = False

    def wrap(self, argv: list[str]) -> list[str]:
        return [*self.prefix, *argv]

    def close(self) -> dict:
        """Kill leftover agent processes, then drop links the controller could follow."""
        if self.closed:
            return self.detail
        self.closed = True
        if self.uid is not None:
            try:
                if not _sole_holder(self.uid):
                    # Another live session of the same workspace still uses this uid.
                    return self.detail
                self.detail["reaped"] = reap(self.uid)
                removed = []
                bounds = [*self.writable, *(root for root, _ in self.create_only)]
                for root in self.writable:
                    removed.extend(remove_escaping_symlinks(root, bounds))
                for root, _ in self.create_only:
                    removed.extend(remove_agent_symlinks(root, self.uid))
                if removed:
                    self.detail["removed_escaping_links"] = removed[:20]
                    _log({"event": "escaping_links_removed", "uid": self.uid, "paths": removed[:20]})
            finally:
                for root, previous in self.create_only:
                    with contextlib.suppress(FileNotFoundError):
                        os.chmod(root, previous)
                _release_uid(self.uid)
        return self.detail

    def describe(self) -> dict:
        return {"mode": self.mode, "uid": self.uid, **self.detail}


def open_session(
    workspace: Path,
    session_dir: Path,
    *,
    readable: Iterable[Path] = (),
    base_env: Mapping[str, str] | None = None,
) -> AgentSession:
    """Prepare an agent session whose cwd is ``workspace``."""
    workspace, session_dir = Path(workspace), Path(session_dir)
    selected = mode()
    base_env = os.environ if base_env is None else base_env
    if selected != "uid":
        if selected == "off" and os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0:
            _log({"event": "confinement_disabled", "workspace": str(workspace), "detail": f"{MODE_ENV}=off"})
        identity = {name: base_env[name] for name in _IDENTITY if name in base_env}
        return AgentSession(mode=selected, env=agent_environment(base_env, identity))
    setup = _setup_process()
    key = workspace_key(workspace)
    uid = _allocate_uid(key)
    try:
        state = agent_state_dir(workspace)
        home, tmp = state / "home", state / "tmp"
        state.mkdir(parents=True, exist_ok=True)
        for path in (home, tmp):
            _real_parent(state, f"{path.name}/.keep")  # a real directory, never a link
        render_agent_home(Path(base_env.get("HOME") or "/root"), home, owner=uid)
        writable = [workspace, session_dir]
        counts = {}
        for root in [state, *writable]:
            Path(root).mkdir(parents=True, exist_ok=True)
            counts[str(root)] = chown_tree(Path(root), uid, uid)
        os.chmod(state, 0o700)
        create_only = []
        for root in _granted(key):
            root.mkdir(parents=True, exist_ok=True)
            info = root.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise ConfinementError(f"granted output path is not a directory: {root}")
            os.chown(root, 0, 0, follow_symlinks=False)
            create_only.append((root, stat.S_IMODE(info.st_mode)))
            os.chmod(root, 0o1777)
        for path in [workspace, session_dir, *(root for root, _ in create_only), *readable]:
            if Path(path).exists():
                ensure_traversable(Path(path))
        for path in readable:
            if Path(path).exists():
                ensure_readable(Path(path))
        env = agent_environment(
            base_env,
            {
                "HOME": str(home),
                "TMPDIR": str(tmp),
                "USER": f"agent{uid}",
                "LOGNAME": f"agent{uid}",
                "XDG_CACHE_HOME": str(home / ".cache"),
                "UV_CACHE_DIR": str(home / ".cache" / "uv"),
                "CARGO_HOME": str(home / ".cargo"),
                **dict(setup.get("toolchain_env") or {}),
            },
        )
        skipped = sum(value.get("skipped_hardlinks", 0) for value in counts.values())
        detail = {"state_dir": str(state)}
        if skipped:
            detail["skipped_hardlinks"] = skipped
        return AgentSession(
            mode="uid",
            env=env,
            uid=uid,
            gid=uid,
            prefix=privilege_drop_argv(str(setup["setpriv"]), uid, uid),
            writable=[state, *writable],
            create_only=create_only,
            detail=detail,
        )
    except BaseException:
        for root, previous in locals().get("create_only", []):
            with contextlib.suppress(FileNotFoundError):
                os.chmod(root, previous)
        _release_uid(uid)
        raise
