# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local end-to-end Belay demo: the Marina ``belay`` app, its watcher and dispatcher, and agent sessions.

    uv run python -m experiments.belay.demo up        # Postgres + local Marina + release the app + watch/dispatch
    uv run python -m experiments.belay.demo down      # stop Marina and Postgres (state is kept)
    uv run python -m experiments.belay.demo api state # the full state, as a session reads it (needs the token)
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
  ``experiments/belay/AGENTS.md`` appended to its system prompt, the queued events in its prompt, and a tool
  allowlist that admits only Belay's own commands. Its transcript and summary go back to the app.
"""

import argparse
import datetime
import getpass
import hashlib
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

from experiments.belay.submit import (
    NODES_BY_SIZE,
    OUTPUT_ROOT,
    RUN_PREFIX,
    iris_jobs,
    load_spec,
)
from experiments.grug.fast_track.belay_ckpt1 import LADDER_RUNGS

logger = logging.getLogger("belay.demo")

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "infra/marina/apps/belay"
APP_ID = "10534466-6e92-4305-a354-217d0e4f5038"
HOME = Path.home()
DEFAULT_STATE_DIR = HOME / ".belay_demo"
DEFAULT_MARINA_CHECKOUT = HOME / "marin_wt_marina_main"
DEFAULT_AGENT_DIR = HOME / "marin_wt_loomdemo_agent"
DEFAULT_REGISTRY_DIR = HOME / "marin_wt_registry_demo"
DEFAULT_PG_BIN = Path("/opt/homebrew/opt/postgresql@17/bin")
PG_PORT = 55433
MARINA_PORT = 8091
WATCH_INTERVAL = 120
DISPATCH_INTERVAL = 5
STORAGE_INTERVAL = 3600
INTAKE_ISSUE = 9514
REPOSITORY = "marin-community/marin"
WANDB_PATH = "marin-community/marin_moe"
STORAGE_BUCKET = "marin-us-east-02a"
CW_ENDPOINT = "https://cwobject.com"
BOT_MARK = "🤖"
SESSION_MAX_TURNS = 150
TOKEN_HEADER = "X-Belay-Token"
# How a session's `demo api` calls receive the service token (Loom would inject it as a profile secret).
TOKEN_ENV = "BELAY_SERVICE_TOKEN"
APP_SCHEMA = "belay"
PUBLIC_FILES = ("common.js", "public.js", "style.css")
SHARED_CSS = ("tokens.css", "parts.css")


def api_base(port: int = MARINA_PORT) -> str:
    return f"http://127.0.0.1:{port}/belay/api"


def call(path: str, body: dict | None = None, *, token: str | None = None) -> dict:
    """GET (no body) or POST JSON to the Belay app; ``token`` is required for its ``agent/`` routes."""
    url = f"{api_base()}/{path}"
    data = None if body is None else json.dumps(body).encode()
    headers = {"content-type": "application/json"}
    if token is not None:
        headers[TOKEN_HEADER] = token
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


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
    """Write the public page and a JSON snapshot of ``api/public/state`` for static hosting (GitHub Pages)."""
    directory.mkdir(parents=True, exist_ok=True)
    web = APP_DIR / "web"
    shutil.copyfile(web / "public.html", directory / "index.html")
    for name in PUBLIC_FILES:
        shutil.copyfile(web / name, directory / name)
    for name in SHARED_CSS:
        shutil.copyfile(REPO / "infra/marina/web" / name, directory / name)
    (directory / "public.json").write_text(json.dumps(call("public/state"), indent=1))
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


def intake_comments() -> list[dict]:
    """Comments on the intake issue, flagged as already approved upstream when the intake bot handled them."""
    out = subprocess.run(
        ["gh", "api", "--paginate", f"repos/{REPOSITORY}/issues/{INTAKE_ISSUE}/comments?per_page=100"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    comments = []
    for page in out.replace("][", "]\n[").splitlines():
        for c in json.loads(page):
            handled = c["reactions"]["rocket"] > 0 or c["user"]["type"] == "Bot" or c["body"].startswith(BOT_MARK)
            comments.append(
                {
                    "comment_id": str(c["id"]),
                    "author": c["user"]["login"],
                    "body": c["body"],
                    "url": c["html_url"],
                    "posted_at": c["created_at"],
                    "approved_upstream": handled,
                }
            )
    return comments


def _run_state(coord: str, train: str | None) -> str:
    if coord in ("pending", "running") and train is not None:
        return train
    if coord == "running":
        return "pending"  # the coordinator is up; the training job is not placed yet
    return coord


class Watcher:
    def __init__(self, agent_dir: Path, token: str) -> None:
        self.agent_dir = agent_dir
        self.token = token
        self.final: dict[str, dict] = {}
        self.storage_at = 0.0
        self.wandb = wandb.Api()

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
        for job_id, state, submitted_at in iris_jobs(f"/{getpass.getuser()}/{RUN_PREFIX}"):
            parts = job_id.strip("/").split("/")
            if len(parts) == 2 and parts[1].endswith("-coord"):
                name = parts[1].removesuffix("-coord")
                coords[name] = (job_id, state)
                submitted[name] = submitted_at
            elif len(parts) == 3 and parts[1].endswith("-coord"):
                trains[parts[1].removesuffix("-coord")] = state
        reports = []
        for name, (job_id, coord_state) in coords.items():
            state = _run_state(coord_state, trains.get(name))
            experiment, _, run = name.removeprefix(RUN_PREFIX).partition("-")
            nodes, total = 1, None
            try:
                spec = load_spec(experiment, self.agent_dir / "experiments/belay/runs")
                size = spec.runs[run].size
                nodes = NODES_BY_SIZE[size]
                total = LADDER_RUNGS[size][1] if spec.launcher == "ckpt1" and size in LADDER_RUNGS else None
            except (FileNotFoundError, KeyError):
                pass
            report = {
                "name": name,
                "job_id": job_id,
                "state": state,
                "nodes": nodes,
                "total_steps": total,
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
        prefix = f"{STORAGE_BUCKET}/{OUTPUT_ROOT.rsplit('/', 1)[0]}/"
        files = fs.find(prefix, detail=True)
        self.storage_at = time.monotonic()
        return sum(f["size"] for f in files.values()), len(files)

    def tick(self) -> None:
        error = None
        try:
            call("agent/sync/intake", {"comments": intake_comments()}, token=self.token)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            error = f"intake: {e}"
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
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "TodoWrite",
    "Bash(cd:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(grep:*)",
    "Bash(wc:*)",
    "Bash(git status:*)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(git show:*)",
    "Bash(git rev-parse:*)",
    "Bash(git add:*)",
    "Bash(git commit:*)",
    "Bash(git fetch:*)",
    "Bash(git pull:*)",
    "Bash(./infra/pre-commit.py:*)",
    "Bash(uv run python -m experiments.belay.submit:*)",
    "Bash(uv run python -m experiments.belay.results:*)",
    "Bash(uv run python -m experiments.belay.demo api:*)",
    "Bash(uv run iris --cluster marin job list:*)",
    "Bash(uv run iris --cluster marin job logs:*)",
)


def session_prompt(session_id: str, events: list[dict], agent_dir: Path, registry_dir: Path, page_url: str) -> str:
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=agent_dir, check=True, capture_output=True, text=True
    ).stdout.strip()
    registry_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=registry_dir, check=True, capture_output=True, text=True
    ).stdout.strip()
    python = agent_dir / ".venv/bin/python"
    return f"""You are Belay session {session_id}, started {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M %Z}.
Follow the Belay session guidelines in your system prompt.

- Experiment checkout (your working directory): {agent_dir}, branch `{branch}`. Push with `git push origin {branch}`.
- Registry checkout: {registry_dir}, branch `{registry_branch}` (a demo copy of the registry). Run registry commands as
  `{python} {registry_dir}/experiments/belay/registry.py ...` (e.g. `new`, `validate`). Use git there only as
  `git -C {registry_dir} <add|commit|push origin {registry_branch}|pull --ff-only|status|log>`; do not `cd` into it.
  Registry file URLs:
  https://github.com/{REPOSITORY}/blob/{registry_branch}/experiments/belay/registry/<file>
- Belay page: {page_url} (use it as --comment-url for submit).
- Iris: `uv run iris --cluster marin job list --prefix /{getpass.getuser()}/belay-`; runs go to cw-us-east-02a.

Queued events for this session:
```json
{json.dumps(events, indent=2)}
```
"""


def run_session(
    session_id: str,
    events: list[dict],
    agent_dir: Path,
    registry_dir: Path,
    state_dir: Path,
    page_url: str,
    token: str,
) -> None:
    log_path = state_dir / "sessions" / f"{session_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    guidelines = (REPO / "experiments/belay/AGENTS.md").read_text()
    python = agent_dir / ".venv/bin/python"
    agent_branch, registry_branch = _branches(agent_dir, registry_dir)
    registry_git = f"git -C {registry_dir}"
    allowed = [
        *ALLOWED_TOOLS,
        f"Bash({python} {registry_dir}/experiments/belay/registry.py:*)",
        f"Bash(git push origin {agent_branch}:*)",
        *(f"Bash({registry_git} {verb}:*)" for verb in ("add", "commit", "status", "log", "diff", "pull --ff-only")),
        f"Bash({registry_git} push origin {registry_branch}:*)",
    ]
    command = [
        "claude",
        "-p",
        session_prompt(session_id, events, agent_dir, registry_dir, page_url),
        "--append-system-prompt",
        guidelines,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
        "--add-dir",
        str(registry_dir),
        "--allowedTools",
        *allowed,
        "--max-turns",
        str(SESSION_MAX_TURNS),
    ]
    logger.info("session %s: starting claude with %d event(s)", session_id, len(events))
    with open(log_path, "w") as log:
        process = subprocess.run(
            command,
            cwd=agent_dir,
            env={**os.environ, TOKEN_ENV: token},
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


def _branches(agent_dir: Path, registry_dir: Path) -> list[str]:
    return [
        subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=d, check=True, capture_output=True, text=True
        ).stdout.strip()
        for d in (agent_dir, registry_dir)
    ]


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
    def __init__(self, agent_dir: Path, registry_dir: Path, state_dir: Path, page_url: str, token: str) -> None:
        self.token = token
        self.agent_dir = agent_dir
        self.registry_dir = registry_dir
        self.state_dir = state_dir
        self.page_url = page_url
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
            run_session(session_id, events, self.agent_dir, self.registry_dir, self.state_dir, self.page_url, self.token)
        except Exception as e:
            logger.exception("session %s crashed", session_id)
            call(
                f"agent/sessions/{session_id}/finish",
                {"state": "failed", "summary": f"runner error: {e}"},
                token=self.token,
            )


def watch(stack: Stack, agent_dir: Path, registry_dir: Path, interval: int, token: str) -> None:
    watcher = Watcher(agent_dir, token)
    dispatcher = Dispatcher(agent_dir, registry_dir, stack.state_dir, stack.page_url, token)
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


def api_command(args: argparse.Namespace) -> None:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"{TOKEN_ENV} is not set; sessions receive the Belay service token from the dispatcher")
    if args.api == "state":
        print(json.dumps(call("agent/state", token=token), indent=2))
    elif args.api == "report":
        report = json.loads(Path(args.file).read_text())
        print(json.dumps(call(f"agent/ideas/{args.idea_id}/report", report, token=token)))
    else:
        body = Path(args.body_file).read_text() if args.body_file else ""
        github_url = None
        if call("agent/state", token=token)["settings"]["github_posting"]:
            if args.github_issue is None:
                raise SystemExit("GitHub posting is on: pass --github-issue (the open belay-weekly issue)")
            github_url = subprocess.run(
                [
                    "gh",
                    "issue",
                    "comment",
                    str(args.github_issue),
                    "--repo",
                    REPOSITORY,
                    "--body",
                    f"{BOT_MARK} **{args.title}**\n\n{body}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        print(
            json.dumps(
                call(
                    "agent/activity",
                    {
                        "kind": args.kind,
                        "title": args.title,
                        "body": body,
                        "registry_id": args.registry_id,
                        "idea_id": args.idea_id,
                        "github_url": github_url,
                    },
                    token=token,
                )
            )
        )


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
    api = sub.add_parser("api", help="the session's client for the Belay page")
    api_sub = api.add_subparsers(dest="api", required=True)
    api_sub.add_parser("state")
    report = api_sub.add_parser("report")
    report.add_argument("idea_id", type=int)
    report.add_argument("--file", required=True)
    post = api_sub.add_parser("post")
    post.add_argument("--kind", required=True, choices=["plan", "result", "escalation", "note"])
    post.add_argument("--title", required=True)
    post.add_argument("--body-file")
    post.add_argument("--registry-id")
    post.add_argument("--idea-id", type=int)
    post.add_argument("--github-issue", type=int, help="used only when GitHub posting is turned on")
    args = parser.parse_args(argv)

    if args.command == "api":
        api_command(args)
        return
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
        Watcher(args.agent_dir, bootstrap_access(stack, args.demo_approver)).tick()
        return
    start_postgres(stack)
    start_marina(stack)
    if not args.no_release:
        logger.info("released belay revision %d", release_app(stack))
    token = bootstrap_access(stack, args.demo_approver)
    print(
        f"\n  Public page: {stack.page_url}\n  Admin page:  {stack.page_url}admin (local browser acts as "
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
