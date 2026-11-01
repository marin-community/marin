# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local end-to-end Belay demo: the Marina ``belay`` app, its watcher and dispatcher, and agent sessions.

    uv run python -m experiments.belay.demo up        # Postgres + local Marina + release the app + watch/dispatch
    uv run python -m experiments.belay.demo down      # stop Marina and Postgres (state is kept)
    uv run python -m experiments.belay.demo export-public out/   # static public page + public.json snapshot

The page has a public read-only view at ``/belay/`` and an admin view at ``/belay/admin`` (approvers only). ``up``
stores a service token in ``<state-dir>/service_token`` and its hash in the app's settings; the watcher, dispatcher
and sessions send it on every ``agent/`` call. ``--demo-approver`` names the identity a local, unauthenticated browser
acts as.

Stand-ins for the production pieces (see ``.agents/projects/belay-agent-on-loom.md``):

- **Marina:** a loopback ``marina serve`` from a clean origin/main checkout against a local Postgres 17, with the
  app released from this checkout with ``--allow-dirty``. Production is the named app on marina.oa.dev.
- **Watcher job:** the loop in ``up``. Every ``--interval`` seconds it lists Belay's Iris jobs (``/<user>/belay-*``)
  and their W&B progress, reads the intake issue's comments, sums ``tmp/ttl=30d/belay/`` hourly, and posts all of it
  to the app, which turns newly terminal runs into ``run_finished`` events.
- **Dispatcher:** every few seconds it asks the app to claim queued events for a new session (the app refuses when
  dispatch is off or a session is active) and runs the session.
- **Loom session:** headless Claude Code (``claude -p``) in the agent's experiment checkout, with
  ``experiments/belay/RUNBOOK.md`` appended to its system prompt, its queued events in
  ``<state-dir>/sessions/<id>.events.json``, and a tool allowlist that admits read-only tools plus
  ``python -m experiments.belay`` (every write a session makes goes through those commands). Its transcript and summary
  go back to the app.
"""

import argparse
import datetime
import getpass
import hashlib
import importlib.util
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import s3fs
import sqlalchemy
import wandb

from experiments.belay.config import CONFIG
from experiments.belay.page import BOT_MARK, TOKEN_ENV, call
from experiments.belay.page import SESSION_ENV as SESSION_ID_ENV
from experiments.belay.submit import iris_jobs

logger = logging.getLogger("belay.demo")

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "infra/marina/apps/belay"
# The app's build script (not a package): its asset versioning also stamps the static export.
_build_spec = importlib.util.spec_from_file_location("belay_build", APP_DIR / "build.py")
BUILD = importlib.util.module_from_spec(_build_spec)
_build_spec.loader.exec_module(BUILD)
APP_ID = "10534466-6e92-4305-a354-217d0e4f5038"
HOME = Path.home()
DEFAULT_STATE_DIR = HOME / ".belay_demo"
DEFAULT_MARINA_CHECKOUT = HOME / "marin_wt_marina_main"
DEFAULT_AGENT_DIR = HOME / "marin_wt_loomdemo_agent"
DEFAULT_REGISTRY_DIR = CONFIG.registry_checkout
DEFAULT_PG_BIN = Path("/opt/homebrew/opt/postgresql@17/bin")
PG_PORT = 55433
MARINA_PORT = 8091
WATCH_INTERVAL = 120
DISPATCH_INTERVAL = 5
STORAGE_INTERVAL = 3600
INTAKE_ISSUE = 9514
REPOSITORY = CONFIG.repository
WANDB_PATH = CONFIG.wandb_project
STORAGE_BUCKET = "marin-us-east-02a"
CW_ENDPOINT = "https://cwobject.com"
SESSION_MAX_TURNS = 150
APP_SCHEMA = "belay"
PUBLIC_FILES = (
    "boot.js",
    "common.js",
    "front.js",
    "status.js",
    "mfu.js",
    "mfu.json",
    "scaling.json",
    "speedrun.json",
    "tracks.json",
    "fonts.css",
    "belay.css",
)
# The public page's speedrun chart and results show macro loss as placeholder BPB until BPB evals land in the registry.
PLACEHOLDER_BPB_RATIO = "0.4147"


# ---------- local Postgres and Marina ----------


@dataclass(frozen=True)
class Stack:
    state_dir: Path
    marina_checkout: Path
    pg_bin: Path
    pg_port: int = PG_PORT
    marina_port: int = MARINA_PORT

    @property
    def database_url(self) -> str:
        return f"postgresql+pg8000://postgres@127.0.0.1:{self.pg_port}/marina"

    @property
    def page_url(self) -> str:
        return f"http://127.0.0.1:{self.marina_port}/belay/"

    def marina_env(self) -> dict[str, str]:
        data_root = self.state_dir / "marina_data"
        data_root.mkdir(parents=True, exist_ok=True)
        return {
            **os.environ,
            "MARINA_DATABASE_URL": self.database_url,
            "MARINA_DATA_ROOT": str(data_root),
            "PORT": str(self.marina_port),
        }

    def marina(self, *args: str, capture: bool = True) -> str:
        result = subprocess.run(
            ["uv", "run", "marina", *args],
            cwd=self.marina_checkout,
            env=self.marina_env(),
            check=True,
            capture_output=capture,
            text=True,
        )
        return result.stdout if capture else ""


def _pg(stack: Stack, tool: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([str(stack.pg_bin / tool), *args], check=check, capture_output=True, text=True)


def start_postgres(stack: Stack) -> None:
    data = stack.state_dir / "pg"
    if not data.exists():
        logger.info("initializing Postgres in %s", data)
        _pg(stack, "initdb", "-D", str(data), "-U", "postgres", "--auth=trust", "-E", "UTF8")
    if _pg(stack, "pg_ctl", "-D", str(data), "status", check=False).returncode != 0:
        options = f"-p {stack.pg_port} -c unix_socket_directories='' -c listen_addresses=127.0.0.1"
        _pg(stack, "pg_ctl", "-D", str(data), "-o", options, "-l", str(stack.state_dir / "postgres.log"), "-w", "start")
    exists = _pg(
        stack,
        "psql",
        "-h",
        "127.0.0.1",
        "-p",
        str(stack.pg_port),
        "-U",
        "postgres",
        "-tAc",
        "SELECT 1 FROM pg_database WHERE datname = 'marina'",
    ).stdout.strip()
    if exists != "1":
        _pg(stack, "psql", "-h", "127.0.0.1", "-p", str(stack.pg_port), "-U", "postgres", "-c", "CREATE DATABASE marina")


def marina_up(stack: Stack) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{stack.marina_port}/api/marina/me", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def start_marina(stack: Stack) -> None:
    if marina_up(stack):
        return
    stack.marina("migrate")
    # The server outlives this process (``demo down`` stops it), so it keeps its own copy of the log handle.
    with open(stack.state_dir / "marina.log", "a") as log:
        process = subprocess.Popen(
            ["uv", "run", "marina", "serve", "--host", "127.0.0.1", "--local-publisher", f"{getpass.getuser()}@local"],
            cwd=stack.marina_checkout,
            env=stack.marina_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (stack.state_dir / "marina.pid").write_text(str(process.pid))
    for _ in range(60):
        if marina_up(stack):
            return
        time.sleep(1)
    raise RuntimeError(f"Marina did not start; see {stack.state_dir / 'marina.log'}")


def release_app(stack: Stack) -> int:
    """Release the app from committed source only, so a release never ships a half-edited page."""
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", str(APP_DIR)],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise SystemExit(f"refusing to release: uncommitted changes under {APP_DIR}:\n{dirty}")
    url = f"http://127.0.0.1:{stack.marina_port}"
    try:
        current = json.loads(stack.marina("apps", "versions", APP_ID, "--url", url, "--json"))["current_version"]
    except subprocess.CalledProcessError:
        current = 0
    out = stack.marina("release", str(APP_DIR), "--base-version", str(current), "--url", url, "--allow-dirty", "--json")
    return json.loads(out.strip().splitlines()[-1])["version"]


def bootstrap_access(stack: Stack, demo_approver: str) -> str:
    """Set the service token and the local demo approver directly in the app's settings; returns the token.

    This is the operator step that production does through secrets and the admin view: the token's hash is stored,
    the token itself stays in ``<state-dir>/service_token`` (mode 600), and ``demo_approver`` is added to the
    approver list so an unauthenticated loopback browser can use the admin view.
    """
    token_file = stack.state_dir / "service_token"
    if not token_file.exists():
        token_file.write_text(secrets.token_urlsafe(32))
        token_file.chmod(0o600)
    token = token_file.read_text().strip()
    engine = sqlalchemy.create_engine(stack.database_url)
    try:
        with engine.begin() as conn:
            approvers = conn.execute(
                sqlalchemy.text(f"SELECT value FROM {APP_SCHEMA}.settings WHERE key = 'approvers'")
            ).scalar_one()
            if not any(a.get("identity") == demo_approver for a in approvers):
                approvers.append({"github": "local-demo", "identity": demo_approver})
            values = {
                "service_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "demo_approver": demo_approver,
                "approvers": approvers,
            }
            for key, value in values.items():
                conn.execute(
                    sqlalchemy.text(
                        f"INSERT INTO {APP_SCHEMA}.settings (key, value, updated_by, updated_at) "
                        "VALUES (:k, CAST(:v AS JSONB), 'demo up', now()) ON CONFLICT (key) DO UPDATE SET "
                        "value = EXCLUDED.value, updated_by = 'demo up', updated_at = now()"
                    ),
                    {"k": key, "v": json.dumps(value)},
                )
    finally:
        engine.dispose()
    return token


def export_public(directory: Path) -> Path:
    """Write the front and status pages with JSON snapshots of the public state and registry results, for static
    hosting (GitHub Pages)."""
    directory.mkdir(parents=True, exist_ok=True)
    web = APP_DIR / "web"
    version = BUILD.web_version()
    for page in ("index.html", "status.html", "mfu.html"):
        BUILD.write_page(web / "static.html", directory / page, version)
    for name in PUBLIC_FILES:
        shutil.copyfile(web / name, directory / name)
    (directory / "public.json").write_text(json.dumps(call("public/state"), indent=1))
    (directory / "results.json").write_text(json.dumps(call("public/results"), separators=(",", ":")))
    return directory


def stop_stack(stack: Stack) -> None:
    pid_file = stack.state_dir / "marina.pid"
    if pid_file.exists():
        try:
            os.killpg(int(pid_file.read_text()), 15)
        except ProcessLookupError:
            pass
        pid_file.unlink()
    data = stack.state_dir / "pg"
    if data.exists():
        _pg(stack, "pg_ctl", "-D", str(data), "stop", "-m", "fast", check=False)


# ---------- watcher ----------


def github_json(path: str) -> list | dict:
    out = subprocess.run(["gh", "api", "--paginate", path], check=True, capture_output=True, text=True).stdout
    pages = [json.loads(page) for page in out.replace("][", "]\n[").splitlines()]
    return [item for page in pages for item in page] if isinstance(pages[0], list) else pages[0]


@dataclass(frozen=True)
class GithubApproval:
    """The intake bot's first 🚀 on a #9514 comment, and the first 👍 (the maintainer who approved it)."""

    transferred_at: datetime.datetime
    approved_by: str | None


def approval_time(comment: dict, approvals: dict[str, GithubApproval]) -> datetime.datetime | None:
    """When the intake bot transferred ``comment``, if that was after the cutover; else None."""
    approval = approvals.get(str(comment["id"]))
    if approval is None or approval.transferred_at <= CONFIG.github_approval_cutover:
        return None
    return approval.transferred_at


def intake_comments(approvals: dict[str, GithubApproval]) -> list[dict]:
    """Comments on the intake issue. One the intake bot transferred is approved upstream; one it transferred after
    the cutover also carries ``approved_at`` and ``approved_by``, which queue it. ``approvals`` caches reactions."""
    comments = []
    for c in github_json(f"repos/{REPOSITORY}/issues/{INTAKE_ISSUE}/comments?per_page=100"):
        if c["user"]["type"] == "Bot" or c["body"].startswith(BOT_MARK):
            continue  # bot and agent comments (tests, archived descriptions) are not suggestions
        handled = c["reactions"]["rocket"] > 0
        cid = str(c["id"])
        if handled and cid not in approvals:
            reactions = github_json(f"repos/{REPOSITORY}/issues/comments/{cid}/reactions?per_page=100")
            rockets = sorted(r["created_at"] for r in reactions if r["content"] == "rocket")
            thumbs = sorted((r["created_at"], r["user"]["login"]) for r in reactions if r["content"] == "+1")
            if rockets:
                approvals[cid] = GithubApproval(
                    datetime.datetime.fromisoformat(rockets[0]), thumbs[0][1] if thumbs else None
                )
        at = approval_time(c, approvals)
        comments.append(
            {
                "comment_id": cid,
                "author": c["user"]["login"],
                "body": c["body"],
                "url": c["html_url"],
                "posted_at": c["created_at"],
                "approved_upstream": handled,
                "approved_at": at.isoformat() if at else None,
                "approved_by": approvals[cid].approved_by if at else None,
            }
        )
    return comments


def _run_state(coord: str, train: str | None) -> str:
    if coord in ("pending", "running") and train is not None:
        return train
    if coord == "running":
        return "pending"  # the coordinator is up; the training job is not placed yet
    return coord


def registry_results(registry_dir: Path) -> tuple[str, dict]:
    """The registry checkout's HEAD and its ``registry build`` output (the public page's results data)."""
    revision = subprocess.run(
        ["git", "-C", str(registry_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    built = subprocess.run(
        [
            sys.executable,
            str(registry_dir / "experiments/belay/registry.py"),
            "build",
            "--placeholder-bpb-ratio",
            PLACEHOLDER_BPB_RATIO,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, json.loads(built)


class Watcher:
    def __init__(self, agent_dir: Path, registry_dir: Path, token: str) -> None:
        self.agent_dir = agent_dir
        self.registry_dir = registry_dir
        self.token = token
        self.final: dict[str, dict] = {}
        self.storage_at = 0.0
        self.wandb = wandb.Api()
        self.approvals: dict[str, GithubApproval] = {}  # #9514 comment id -> its GitHub approval

    def _wandb(self, name: str) -> dict:
        try:
            run = self.wandb.run(f"{WANDB_PATH}/{name}")
        except (wandb.errors.CommError, ValueError):
            return {}
        loss = run.summary.get("eval/paloma/macro_loss")
        return {
            "step": run.summary.get("_step"),
            "loss": float(loss) if loss is not None else None,
            "wandb_url": run.url,
            "finished": run.state == "finished",
        }

    def runs(self) -> list[dict]:
        self.wandb = wandb.Api()  # a fresh client per tick: Api caches each Run it has loaded
        coords, trains, submitted = {}, {}, {}
        for job in iris_jobs():
            parts = job.job_id.strip("/").split("/")
            if len(parts) == 2 and parts[1].endswith("-coord"):
                name = parts[1].removesuffix("-coord")
                coords[name] = (job.job_id, job.state)
                submitted[name] = job.submitted_at
            elif len(parts) == 3 and parts[1].endswith("-coord"):
                trains[parts[1].removesuffix("-coord")] = job.state
        reports = []
        for name, (job_id, coord_state) in coords.items():
            state = _run_state(coord_state, trains.get(name))
            report = {
                "name": name,
                "job_id": job_id,
                "state": state,
                "nodes": 1,
                "total_steps": None,
                "submitted_at": submitted.get(name),
            }
            if name in self.final:
                report.update(self.final[name])
            elif state != "pending":
                progress = self._wandb(name)
                report.update({k: v for k, v in progress.items() if k != "finished"})
                if state == "succeeded" and progress.get("finished") and progress.get("loss") is not None:
                    self.final[name] = {k: v for k, v in progress.items() if k != "finished"}
            reports.append(report)
        return reports

    def storage(self) -> tuple[int, int] | None:
        if time.monotonic() - self.storage_at < STORAGE_INTERVAL and self.storage_at:
            return None
        fs = s3fs.S3FileSystem(
            key=os.environ["CW_KEY_ID"],
            secret=os.environ["CW_KEY_SECRET"],
            endpoint_url=CW_ENDPOINT,
            config_kwargs={"s3": {"addressing_style": "virtual"}},
        )
        prefix = f"{STORAGE_BUCKET}/{CONFIG.output_root.rsplit('/', 1)[0]}/"
        files = fs.find(prefix, detail=True)
        self.storage_at = time.monotonic()
        return sum(f["size"] for f in files.values()), len(files)

    def upload_registry(self) -> None:
        """Rebuild and upload the registry's results when the registry checkout moved."""
        head = subprocess.run(
            ["git", "-C", str(self.registry_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if call("agent/registry", token=self.token)["revision"] == head:
            return
        revision, data = registry_results(self.registry_dir)
        call("agent/registry", {"revision": revision, "data": data}, token=self.token)
        logger.info("uploaded registry results at %s (%d ideas)", revision[:10], len(data["ideas"]))

    def tick(self) -> None:
        error = None
        try:
            call("agent/sync/intake", {"comments": intake_comments(self.approvals)}, token=self.token)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            error = f"intake: {e}"
        try:
            self.upload_registry()
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            error = f"registry: {e}"
        reports = self.runs()
        call("agent/sync/runs", {"runs": reports}, token=self.token)
        storage = None
        try:
            storage = self.storage()
        except (OSError, KeyError) as e:
            error = f"storage: {e}"
        nodes = sum(r["nodes"] for r in reports if r["state"] == "running")
        body = {"nodes_in_use": nodes, "error": error}
        if storage is not None:
            body.update(storage_bytes=storage[0], storage_objects=storage[1])
        call("agent/tick", body, token=self.token)
        logger.info("tick: %d runs, %d nodes running%s", len(reports), nodes, f", {error}" if error else "")


# ---------- sessions ----------

ALLOWED_TOOLS = (
    "Read",
    "Glob",
    "Grep",
    "Bash(cat:*)",
    "Bash(ls:*)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(grep:*)",
    "Bash(git status:*)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(git show:*)",
    "Bash(uv run python -m experiments.belay:*)",
    # Code changes (runbook section 1b): edits only where code-change --finish allows them, and the revert it prints.
    "Edit(experiments/grug/fast_track/**)",
    "Write(experiments/grug/fast_track/**)",
    "Edit(tests/grug/test_fast_track_*)",
    "Write(tests/grug/test_fast_track_*)",
    "Bash(git checkout -- .)",
    "Bash(git clean -fd experiments/grug/fast_track tests/grug)",
    # Read-only GitHub, in every session: an idea may point at a PR or issue whose change it asks to port.
    "Bash(gh pr view:*)",
    "Bash(gh pr diff:*)",
    "Bash(gh issue view:*)",
    "Bash(uv run iris --cluster marin job list:*)",
    "Bash(uv run iris --cluster marin job logs:*)",
)
RUNBOOK = REPO / "experiments/belay/RUNBOOK.md"


def session_prompt(session_id: str, events_file: Path) -> str:
    return f"""You are Belay session {session_id}, started {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M %Z}.
Read RUNBOOK.md (in your system prompt). Identify the event type. Run its steps exactly. If anything doesn't match,
post to the issue and stop: `uv run python -m experiments.belay escalate --title "..." --body "..."`.
Edit files only in section 1b, and only under experiments/grug/fast_track/ and tests/grug/test_fast_track_*.py.

Your events file: {events_file}
Your first command: `uv run python -m experiments.belay start --events {events_file}`
"""


# The only variables a session inherits: no storage keys (CW_*, AWS_*), no other tokens. Training jobs get the
# cluster's iris-task-env credentials, not these.
SESSION_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "TERM", "SSH_AUTH_SOCK", "WANDB_API_KEY")
SESSION_ENV_PREFIXES = ("UV_", "XDG_")
BOT_TOKEN_ENV = "BELAY_INTAKE_TOKEN"


def session_env() -> dict[str, str]:
    """The allowlisted part of this process's environment, for a session."""
    keep = {*SESSION_ENV, BOT_TOKEN_ENV}
    return {k: v for k, v in os.environ.items() if k in keep or k.startswith(SESSION_ENV_PREFIXES)}


def run_session(
    session_id: str,
    events: list[dict],
    agent_dir: Path,
    registry_dir: Path,
    state_dir: Path,
    token: str,
) -> None:
    log_path = state_dir / "sessions" / f"{session_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    events_file = state_dir / "sessions" / f"{session_id}.events.json"
    events_file.write_text(json.dumps(events, indent=2))
    command = [
        "claude",
        "-p",
        session_prompt(session_id, events_file),
        "--append-system-prompt",
        RUNBOOK.read_text(),
        "--output-format",
        "stream-json",
        "--verbose",
        "--add-dir",
        str(registry_dir),
        "--allowedTools",
        *ALLOWED_TOOLS,
        "--max-turns",
        str(SESSION_MAX_TURNS),
    ]
    logger.info("session %s: starting claude with %d event(s)", session_id, len(events))
    with open(log_path, "w") as log:
        process = subprocess.run(
            command,
            cwd=agent_dir,
            env={**session_env(), TOKEN_ENV: token, SESSION_ID_ENV: session_id},
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    transcript, summary, cost, turns = _read_transcript(log_path)
    state = "succeeded" if process.returncode == 0 and summary else "failed"
    if not summary:
        summary = f"claude exited with code {process.returncode} without a final message; see the transcript."
    call(
        f"agent/sessions/{session_id}/finish",
        {"state": state, "summary": summary, "transcript": transcript, "cost_usd": cost, "turns": turns},
        token=token,
    )
    logger.info("session %s: %s ($%.2f, %s turns)", session_id, state, cost or 0, turns)


def _read_transcript(log_path: Path) -> tuple[str, str, float | None, int | None]:
    """A readable transcript from Claude Code's stream-json log, plus the final message, cost and turn count."""
    lines, summary, cost, turns = [], "", None, None
    for raw in log_path.read_text().splitlines():
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            lines.append(raw)
            continue
        kind = message.get("type")
        if kind == "assistant":
            for block in message["message"].get("content", []):
                if block.get("type") == "text":
                    lines.append(f"\n[assistant]\n{block['text']}")
                elif block.get("type") == "tool_use":
                    lines.append(f"\n[tool: {block['name']}] {json.dumps(block['input'])[:2000]}")
        elif kind == "user":
            for block in message["message"].get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content")
                    text = content if isinstance(content, str) else json.dumps(content)
                    lines.append(f"[result] {text[:3000]}")
        elif kind == "result":
            summary = message.get("result") or ""
            cost = message.get("total_cost_usd")
            turns = message.get("num_turns")
    return "\n".join(lines), summary.strip(), cost, turns


class Dispatcher:
    def __init__(self, agent_dir: Path, registry_dir: Path, state_dir: Path, token: str) -> None:
        self.token = token
        self.agent_dir = agent_dir
        self.registry_dir = registry_dir
        self.state_dir = state_dir
        self.thread: threading.Thread | None = None

    def dispatch(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        session_id = f"s{datetime.datetime.now():%m%d-%H%M%S}"
        claim = call("agent/dispatch/claim", {"session_id": session_id}, token=self.token)
        if not claim["launch"]:
            return
        logger.info("dispatch: session %s for %s", session_id, "; ".join(e["summary"] for e in claim["events"]))
        self.thread = threading.Thread(
            target=self._run, args=(session_id, claim["events"]), name=f"session-{session_id}", daemon=True
        )
        self.thread.start()

    def _run(self, session_id: str, events: list[dict]) -> None:
        try:
            run_session(session_id, events, self.agent_dir, self.registry_dir, self.state_dir, self.token)
        except Exception as e:
            logger.exception("session %s crashed", session_id)
            call(
                f"agent/sessions/{session_id}/finish",
                {"state": "failed", "summary": f"runner error: {e}"},
                token=self.token,
            )


def watch(stack: Stack, agent_dir: Path, registry_dir: Path, interval: int, token: str) -> None:
    watcher = Watcher(agent_dir, registry_dir, token)
    dispatcher = Dispatcher(agent_dir, registry_dir, stack.state_dir, token)
    last_tick = 0.0
    while True:
        if time.monotonic() - last_tick >= interval:
            try:
                watcher.tick()
            except Exception:
                logger.exception("watcher tick failed")
            last_tick = time.monotonic()
        try:
            dispatcher.dispatch()
        except Exception:
            logger.exception("dispatch failed")
        time.sleep(DISPATCH_INTERVAL)


# ---------- the session's page client ----------


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("up", "down", "tick", "release"):
        p = sub.add_parser(name)
        p.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
        p.add_argument("--marina-checkout", type=Path, default=DEFAULT_MARINA_CHECKOUT)
        p.add_argument("--pg-bin", type=Path, default=DEFAULT_PG_BIN)
        p.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
        p.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
        p.add_argument("--interval", type=int, default=WATCH_INTERVAL, help="seconds between watcher ticks")
        p.add_argument("--no-release", action="store_true", help="keep the currently released app revision")
        p.add_argument(
            "--demo-approver",
            default=f"{getpass.getuser()}@local",
            help="identity an unauthenticated local browser acts as in the admin view (local previews only)",
        )
    export = sub.add_parser("export-public", help="write the public page and a JSON snapshot for static hosting")
    export.add_argument("directory", type=Path)
    args = parser.parse_args(argv)

    if args.command == "export-public":
        print("wrote", export_public(args.directory))
        return
    args.state_dir.mkdir(parents=True, exist_ok=True)
    stack = Stack(args.state_dir, args.marina_checkout, args.pg_bin)
    if args.command == "down":
        stop_stack(stack)
        print("stopped Marina and Postgres; state kept in", args.state_dir)
        return
    if args.command == "release":
        print("released belay revision", release_app(stack))
        return
    if args.command == "tick":
        Watcher(args.agent_dir, args.registry_dir, bootstrap_access(stack, args.demo_approver)).tick()
        return
    start_postgres(stack)
    start_marina(stack)
    if not args.no_release:
        logger.info("released belay revision %d", release_app(stack))
    token = bootstrap_access(stack, args.demo_approver)
    print(
        f"\n  Belay:  {stack.page_url}\n  Status: {stack.page_url}status\n"
        f"  Admin:  {stack.page_url}admin (local browser acts as "
        f"{args.demo_approver})\n  Watcher every {args.interval}s; Ctrl-C stops the loop "
        f"(`demo down` stops Marina and Postgres).\n",
        flush=True,
    )
    try:
        watch(stack, args.agent_dir, args.registry_dir, args.interval, token)
    except KeyboardInterrupt:
        print("watcher stopped; Marina and Postgres keep running (`demo down` stops them)")


if __name__ == "__main__":
    main(sys.argv[1:])
