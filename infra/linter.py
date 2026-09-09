# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic lint-review runner (lanes + composer) invoked by `pre-commit.py --review`.

Fans out one headless agent per structured rule lane under ``infra/lint/`` over the branch's
changes and merges the per-lane findings with deterministic dedupe-and-concat. An
explicit option adds a composer agent. Each lane is handed the changed-file inventory (`git diff --stat`)
and read-only git access, and probes each file itself rather than reading a pasted
diff — so it can follow the change into other files and skip binary/oversized files
on its own. This subsystem is self-contained and used only by the `--review` path of
pre-commit.py.
"""

import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import click

ROOT_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from infra.codehealth import complexity as complexity_leads  # noqa: E402
from infra.lint.catalog import LintLane, catalog_sha, load_catalog, render_lane  # noqa: E402

LINT_DIR = ROOT_DIR / "infra/lint"
LINT_CATALOG = load_catalog(LINT_DIR)


CLAUDE_LINT_MODEL = "claude-haiku-4-5-20251001"
LINT_REVIEW_AGENT_DEFAULT = f"claude -p --model {CLAUDE_LINT_MODEL} --effort low --output-format json"

LINT_REVIEW_TIMEOUT = 600

CODEX_SANDBOX_BYPASS_FLAGS = frozenset(("--dangerously-bypass-approvals-and-sandbox", "--yolo"))
CODEX_READ_ONLY_SANDBOX = "read-only"

# The lint review runs fully-headless agents over the working tree. They are REVIEWERS:
# the only job is to READ the change and emit advisory findings on stdout — never to modify
# a file or touch git/PR state. That contract is enforced two ways: the prompt mandate
# (READ_ONLY_MANDATE, stated to every agent) and a CLI-specific hard permission lockdown
# (_with_readonly_access).

# Built-in tools the headless `claude` agent may have AT ALL. Edit/Write/NotebookEdit are
# absent, so the agent cannot modify a file regardless of what any inherited settings.json
# grants — a tool that does not exist cannot be permitted.
LINT_REVIEW_BUILTIN_TOOLS = "Bash,Read,Grep,Glob"

# Read-only git subcommands the lanes pre-approve to probe the change themselves (we hand
# them a `git diff --stat`, not a pasted diff). Pre-approved via `--allowedTools`.
LINT_REVIEW_READONLY_GIT = (
    "git status",
    "git diff",
    "git show",
    "git log",
    "git ls-files",
    "git rev-parse",
    "git merge-base",
    "git cat-file",
    "git blame",
    "git grep",
)

# State-changing git/gh commands explicitly DENIED via `--disallowedTools`. `deny` beats
# `allow` at every scope, so this holds even if the developer's own settings.json broadly
# allows `Bash(git:*)` or `Bash(gh:*)` — closing the hole that once let a lane commit, push,
# and open a PR by itself.
LINT_REVIEW_DENIED_COMMANDS = (
    "git add",
    "git commit",
    "git push",
    "git pull",
    "git fetch",
    "git reset",
    "git rebase",
    "git checkout",
    "git switch",
    "git merge",
    "git restore",
    "git stash",
    "git tag",
    "git clean",
    "git apply",
    "git am",
    "git rm",
    "git mv",
    "git cherry-pick",
    "git revert",
    "git update-ref",
    "git config",
    "git remote",
    "git branch",
    "git worktree",
    "gh",
)

# Raw per-arm and combined output from each review run is written under here for debugging a
# slow or broken lint cycle; the run's directory is printed at the end of the review.
LINT_REVIEW_LOG_ROOT = pathlib.Path("/tmp/marin-linter")


# Prepended to every lane and composer prompt. The `claude` tool lockdown already makes
# mutation impossible; this states the same contract in plain language for every agent CLI
# (including Codex, which manages its own permissions) and orients the model on its job.
READ_ONLY_MANDATE = (
    "## Your role: READ-ONLY reviewer\n\n"
    "You are a code REVIEWER running non-interactively. Your ONLY output is advisory lint "
    "findings on stdout, in the Output format defined below. You are inspecting someone "
    "else's in-progress branch and have NO mandate to change it. You MUST NOT, under any "
    "circumstances:\n"
    "- edit, create, move, or delete any file;\n"
    "- run any state-changing git command (add, commit, push, pull, fetch, reset, rebase, "
    "checkout, switch, merge, restore, stash, tag, branch, worktree, config, …) — use git "
    "ONLY to READ the change (diff, show, log, status, ls-files, rev-parse, merge-base, "
    "cat-file, blame, grep);\n"
    "- run `gh`, open or comment on a pull request, or take any action beyond reading code;\n"
    "- try to 'fix' anything. A fix is a finding to report, never an edit to make.\n\n"
    "Committing, pushing, or opening a PR on the author's behalf is a serious error, not "
    "helpfulness. If you cannot complete the review read-only, emit nothing and stop."
)


# Coarse lanes — one headless agent each, derived from the structured catalog.
LINT_LANES = LINT_CATALOG.lanes

# The composer merges the lanes' outputs. Authored to never silently drop a real
# finding — it may only collapse true duplicates and drop overlap-precedence losers.
COMPOSER_INSTRUCTIONS = (
    "You are the COMPOSER. Several specialist lanes each scanned the SAME branch diff against "
    "their slice of the catalog and emitted findings in the Output format above — including one "
    "holistic 'meta' lane that reasons over the whole change rather than single hunks, so its "
    "findings anchor on different lines than a local finding for the same underlying issue. "
    "Their labelled raw outputs and the changed-file inventory follow below. "
    "You are an EDITOR, not a reviewer: merge them into the single final findings list, reasoned "
    "— not a blind concat. You do NOT re-scan for new issues or invent findings. Use read-only "
    "git and Read only to adjudicate a duplicate/precedence call or sanity-check that a cited "
    "line exists.\n\n"
    "PRIME DIRECTIVE — KEEP EVERY DISTINCT FINDING. Default to KEEP. Trust the lanes: a finding "
    "survives even if you would not have raised it, even if its confidence is low, even if its "
    'rule isn\'t "yours." The Self-evaluation / "when uncertain, suppress" guidance above governs '
    "the LANES, NOT you — never drop, soften, or re-judge a finding's substance, and never trim "
    "for brevity. Silently losing a real finding is your one unforgivable error.\n\n"
    "The ONLY findings you may remove:\n"
    "1. DUPLICATES — two findings sharing (path, line, underlying issue): same path:line describing "
    "the same concrete defect, even across lanes, different ml- codes, or different wording. "
    "Collapse to ONE; keep the higher-confidence finding's code and message (tie → crisper "
    "message), use the max confidence. You may lightly tighten the kept message but must never "
    "weaken or narrow its claim. Keep it ≤200 chars.\n"
    "2. PRECEDENCE LOSERS — two findings on the SAME line that conflict and the Overlap precedence "
    "section above names a winner (more-specific rule wins). Emit the winner alone; drop the loser "
    "(do not merge).\n\n"
    "NOT removable — keep BOTH: same line but two genuinely different defects; same code on "
    "different lines; compatible claims worded differently. When unsure whether two findings are "
    "the same issue, they are NOT — keep both. Different path:line is always distinct.\n\n"
    "Reason privately — cluster by path then ascending line, apply precedence, decide each "
    "near-collision — but this reasoning must NEVER appear in the output.\n\n"
    "Output ONLY the canonical lines from the Output format above — "
    "`<path>:<line>: <code> (<confidence>) <message>`, two-decimal confidence, message ≤200 chars "
    "— grouped by file, ascending line, no blank lines. No preamble, summary, counts, reasoning, "
    "markdown, or fences; a regex parser consumes this. Your first character is the first finding's, "
    "or the output is empty. If zero findings survive, emit nothing."
)

# Env vars that identify or authorize the calling agent session. Stripped before
# exec so the sub-agent cannot re-bind to or act as its parent session.
LINT_REVIEW_STRIPPED_ENV = (
    "ANTHROPIC_API_KEY",  # force subscription auth, not metered API billing
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_SSE_PORT",
    "CODEX_THREAD_ID",
    "LOOM_SESSION_ID",
    "LOOM_TOKEN",
    "WEAVER_BRANCH",
)


# Output format the agent emits, per infra/lint/catalog.yaml "Output format":
#   <path>:<line>: <code> (<confidence>) <message>
_FINDING_RE = re.compile(r"^(?P<path>[^:\s]+):(?P<line>\d+): (?P<code>ml-[\w-]+) \((?P<conf>[\d.]+)\) (?P<msg>.*)$")


def _parse_findings(stdout: str) -> list[list]:
    rows: list[list] = []
    for line in stdout.splitlines():
        m = _FINDING_RE.match(line.strip())
        if not m:
            continue
        try:
            line_no = int(m["line"])
            conf = float(m["conf"])
        except ValueError:
            continue
        rows.append([m["path"], line_no, m["code"], conf, m["msg"][:200]])
    return rows


def _diff_numstat(merge_base: str) -> tuple[int, int, int]:
    """`(files, added, removed)` for the branch vs `merge_base`, from `git diff --numstat`.

    Binary files (numstat renders their counts as `-`) count as one changed file with
    zero line deltas. Used for telemetry and the meta lane's diff-size gate.
    """
    out = subprocess.run(
        ["git", "diff", "--numstat", merge_base],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    files = added = removed = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        files += 1
        if parts[0].isdigit():
            added += int(parts[0])
        if parts[1].isdigit():
            removed += int(parts[1])
    return files, added, removed


def _git(args: list[str]) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or None
    except Exception:
        return None


# A CI runner checks out the synthetic pull-request merge ref, so local git
# state describes the merge commit rather than the branch under review. The
# review harness supplies the real identity through the environment. A
# developer's machine sets none of them and falls back to local git.
REVIEW_TRIGGER_ENV = "MARIN_REVIEW_TRIGGER"
REVIEW_PR_NUMBER_ENV = "MARIN_REVIEW_PR_NUMBER"
REVIEW_HEAD_SHA_ENV = "MARIN_REVIEW_HEAD_SHA"


def _review_trigger() -> str:
    return os.environ.get(REVIEW_TRIGGER_ENV) or "local"


def _review_pr_number() -> int | None:
    raw = (os.environ.get(REVIEW_PR_NUMBER_ENV) or "").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def _review_head_sha() -> str | None:
    """The reviewed commit: the harness-supplied head SHA, else local HEAD."""
    return (os.environ.get(REVIEW_HEAD_SHA_ENV) or "").strip() or _git(["rev-parse", "HEAD"])


def _ship_review_stats(event: dict, log_dir: pathlib.Path | None) -> None:
    """Fire-and-forget: hand the event off to infra/codehealth/log_stats.py via
    `uv run`. Detached so the Finelog connect and write never block the dev.

    The child's stderr goes to `stats.log` in the run's log directory, so a
    failed ship (no uv, no Finelog credentials, no network) is diagnosable
    afterwards rather than silently dropped. Without a log directory there is
    nowhere to put it and the output is discarded.
    """
    if not shutil.which("uv"):
        return
    stats_log = None
    try:
        if log_dir is not None:
            stats_log = (log_dir / "stats.log").open("wb")
        proc = subprocess.Popen(
            # --no-sync: a bare `uv run` re-resolves the root workspace and
            # can start installing into it from this detached process.
            ["uv", "run", "--quiet", "--no-sync", str(ROOT_DIR / "infra" / "codehealth" / "log_stats.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stats_log or subprocess.DEVNULL,
            cwd=ROOT_DIR,
            start_new_session=True,
        )
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(event).encode())
        proc.stdin.close()
    except (OSError, ValueError) as e:
        click.echo(f"  ⚠ Lint review: could not ship review stats: {e}")
    finally:
        # The child holds its own dup of the descriptor.
        if stats_log is not None:
            stats_log.close()


@dataclass(frozen=True)
class AgentSpec:
    vendor: str
    model: str
    effort: str


@dataclass(frozen=True)
class AgentUsage:
    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    reported_total_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.reported_total_tokens is not None:
            return self.reported_total_tokens
        values = (
            self.input_tokens,
            self.cache_creation_input_tokens,
            self.cache_read_input_tokens,
            self.output_tokens,
        )
        present = [value for value in values if value is not None]
        return sum(present) if present else None


@dataclass(frozen=True)
class AgentResult:
    stdout: str
    stderr: str
    returncode: int
    usage: AgentUsage


@dataclass(frozen=True)
class LaneResult:
    name: str
    stdout: str
    stderr: str
    returncode: int | None  # None means the lane agent timed out
    elapsed: float  # wall-clock seconds the lane's agent ran
    prompt: str  # the exact prompt fed to the lane (logged for debugging)
    usage: AgentUsage = AgentUsage()


@dataclass(frozen=True)
class ComposerRun:
    """The composer agent's raw run, captured for logging."""

    prompt: str
    stdout: str
    stderr: str
    returncode: int | None  # None means the composer timed out
    elapsed: float
    usage: AgentUsage = AgentUsage()


@dataclass(frozen=True)
class MergeOutcome:
    """Result of merging the lanes: the final findings plus how they were produced."""

    findings_text: str
    mode: str
    composer_rc: int
    timed_out: bool
    composer: ComposerRun | None  # None when lanes were concatenated (no composer ran)


@dataclass(frozen=True)
class ReviewRun:
    """One review run's facts, gathered for logging (the summary and per-arm dumps)."""

    lane_results: list[LaneResult]
    outcome: MergeOutcome
    merge_base: str
    diff_stats: tuple[int, int, int]
    elapsed_total: float
    branch: str | None
    agent_spec: AgentSpec
    agent_command: tuple[str, ...]


def _lint_review_env() -> dict[str, str]:
    # Strip session/auth markers so the headless agent runs as a fresh session
    # rather than nesting under the calling agent's transcript or being billed
    # via metered API auth.
    return {k: v for k, v in os.environ.items() if k not in LINT_REVIEW_STRIPPED_ENV}


def _readonly_agent_flags() -> list[str]:
    """Flags that lock a headless `claude` agent to read-only review.

    Three layers: `--tools` removes the edit tools entirely (no Edit/Write/NotebookEdit),
    `--allowedTools` pre-approves only read-only git plus Read/Grep/Glob, and
    `--disallowedTools` hard-denies every mutating git/gh command so an over-permissive
    inherited settings.json (`Bash(git:*)`, `Bash(gh:*)`, …) cannot re-open the hole —
    `deny` wins over `allow` at every scope.
    """
    allow = ["Read", "Grep", "Glob", *(f"Bash({c}:*)" for c in LINT_REVIEW_READONLY_GIT)]
    deny = [f"Bash({c}:*)" for c in LINT_REVIEW_DENIED_COMMANDS]
    return [
        "--tools",
        LINT_REVIEW_BUILTIN_TOOLS,
        "--allowedTools",
        ",".join(allow),
        "--disallowedTools",
        ",".join(deny),
    ]


def _option_value(command: list[str], names: tuple[str, ...]) -> str | None:
    for index, argument in enumerate(command):
        for name in names:
            if argument == name:
                return command[index + 1] if index + 1 < len(command) else None
            if argument.startswith(f"{name}="):
                return argument.split("=", maxsplit=1)[1]
    return None


def _codex_effort(command: list[str]) -> str | None:
    for index, argument in enumerate(command):
        if argument in {"--config", "-c"} and index + 1 < len(command):
            setting = command[index + 1]
        elif argument.startswith(("--config=", "-c=")):
            setting = argument.split("=", maxsplit=1)[1]
        else:
            continue
        if setting.startswith("model_reasoning_effort="):
            return setting.split("=", maxsplit=1)[1].strip("\"'")
    return None


def _agent_spec(agent_cmd: list[str]) -> AgentSpec:
    """Return the explicit vendor, model, and effort for a recursive agent command."""
    agent_name = os.path.basename(agent_cmd[0])
    model = _option_value(agent_cmd, ("--model", "-m"))
    effort = _codex_effort(agent_cmd) if agent_name == "codex" else _option_value(agent_cmd, ("--effort",))
    if not model or not effort:
        raise ValueError(
            "agent commands must select an explicit model and effort; "
            "use '--model <model> --effort <level>' for Claude-compatible CLIs or "
            "'--model <model> --config model_reasoning_effort=<level>' for Codex"
        )
    return AgentSpec(vendor=agent_name, model=model, effort=effort)


def _int_field(data: dict, key: str) -> int | None:
    value = data.get(key)
    return int(value) if isinstance(value, int | float) else None


def _float_field(data: dict, key: str) -> float | None:
    value = data.get(key)
    return float(value) if isinstance(value, int | float) else None


def _claude_output(stdout: str) -> tuple[str, AgentUsage]:
    try:
        event = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip(), AgentUsage()
    if not isinstance(event, dict) or event.get("type") != "result":
        return stdout.strip(), AgentUsage()
    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    return str(event.get("result") or "").strip(), AgentUsage(
        input_tokens=_int_field(usage, "input_tokens"),
        cache_creation_input_tokens=_int_field(usage, "cache_creation_input_tokens"),
        cache_read_input_tokens=_int_field(usage, "cache_read_input_tokens"),
        output_tokens=_int_field(usage, "output_tokens"),
        cost_usd=_float_field(event, "total_cost_usd"),
    )


def _codex_output(stdout: str) -> tuple[str, AgentUsage]:
    messages: list[str] = []
    usage = AgentUsage()
    parsed_event = False
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        parsed_event = True
        item = event.get("item")
        if event.get("type") == "item.completed" and isinstance(item, dict):
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                messages.append(item["text"])
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            raw_usage = event["usage"]
            input_tokens = _int_field(raw_usage, "input_tokens")
            output_tokens = _int_field(raw_usage, "output_tokens")
            usage = AgentUsage(
                input_tokens=input_tokens,
                cache_read_input_tokens=_int_field(raw_usage, "cached_input_tokens"),
                output_tokens=output_tokens,
                reported_total_tokens=(
                    input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
                ),
            )
    if not parsed_event:
        return stdout.strip(), AgentUsage()
    return "\n".join(messages).strip(), usage


def _agent_output(spec: AgentSpec, stdout: str) -> tuple[str, AgentUsage]:
    if spec.vendor == "claude":
        return _claude_output(stdout)
    if spec.vendor == "codex":
        return _codex_output(stdout)
    return stdout.strip(), AgentUsage()


def _with_usage_output(agent_cmd: list[str], spec: AgentSpec) -> list[str]:
    command = list(agent_cmd)
    if spec.vendor == "claude":
        output_format = _option_value(command, ("--output-format",))
        if output_format is None:
            command.extend(("--output-format", "json"))
        elif output_format != "json":
            raise ValueError("Claude lint commands must use '--output-format json' for usage telemetry")
    elif spec.vendor == "codex" and "--json" not in command:
        command.append("--json")
    return command


def _with_readonly_access(agent_cmd: list[str]) -> list[str]:
    """Return an agent command that enforces the review's read-only contract."""
    agent_name = os.path.basename(agent_cmd[0])
    if agent_name == "codex" and len(agent_cmd) > 1 and agent_cmd[1] in {"exec", "e"}:
        command = [arg for arg in agent_cmd if arg not in CODEX_SANDBOX_BYPASS_FLAGS]
        if "--ephemeral" not in command:
            command.append("--ephemeral")
        sandbox_assignment = next(
            (index for index, arg in enumerate(command) if arg.startswith(("--sandbox=", "-s="))), None
        )
        if sandbox_assignment is not None:
            flag = command[sandbox_assignment].split("=", maxsplit=1)[0]
            command[sandbox_assignment] = f"{flag}={CODEX_READ_ONLY_SANDBOX}"
        else:
            for flag in ("--sandbox", "-s"):
                if flag in command:
                    sandbox_index = command.index(flag) + 1
                    if sandbox_index == len(command):
                        command.append(CODEX_READ_ONLY_SANDBOX)
                    else:
                        command[sandbox_index] = CODEX_READ_ONLY_SANDBOX
                    break
            else:
                command.extend(("--sandbox", CODEX_READ_ONLY_SANDBOX))
        return command
    if agent_name != "claude" or "--allowedTools" in agent_cmd:
        return agent_cmd
    return [*agent_cmd, *_readonly_agent_flags()]


def _run_agent(agent_cmd: list[str], spec: AgentSpec, env: dict[str, str], prompt: str) -> AgentResult | None:
    """Run one headless agent over `prompt`; None if it times out."""
    try:
        completed = subprocess.run(
            agent_cmd,
            input=prompt,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            env=env,
            timeout=LINT_REVIEW_TIMEOUT,
        )
        stdout, usage = _agent_output(spec, completed.stdout)
        return AgentResult(stdout, completed.stderr.strip(), completed.returncode, usage)
    except subprocess.TimeoutExpired:
        return None


def _timed_agent(
    agent_cmd: list[str], spec: AgentSpec, env: dict[str, str], prompt: str
) -> tuple[AgentResult | None, float]:
    """`_run_agent` plus the wall-clock seconds it took (measured inside the worker thread)."""
    start = time.time()
    cp = _run_agent(agent_cmd, spec, env, prompt)
    return cp, time.time() - start


def _changed_py_files(merge_base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", merge_base, "--name-only", "--", "*.py"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    ).stdout
    return [p for p in out.splitlines() if p.strip()]


def _read_worktree(rel_path: str) -> str | None:
    try:
        return (ROOT_DIR / rel_path).read_text(encoding="utf-8")
    except OSError:
        return None


def _change_context(merge_base: str, stat: str) -> str:
    """The concrete change handed to each lane/composer: the merge-base SHA and the
    `git diff --stat` inventory. How to inspect it — probe per file, what to skip — lives
    in the catalog's "Inputs" section, which is in the same prompt, so it is not repeated here.
    """
    return (
        "## The change\n\n"
        f"Merge base: `{merge_base}`. Inspect each changed file below per the catalog's "
        '"Inputs" section (`git diff` or `Read` it; skip what it says to skip).\n\n'
        f"```\n{stat}\n```"
    )


def _lane_prompt(shared_text: str, lane: LintLane, merge_base: str, stat: str, leads: str) -> str:
    parts = [READ_ONLY_MANDATE, shared_text, render_lane(LINT_CATALOG, lane.name)]
    if lane.include_complexity_leads and leads:
        parts.append(leads)
    parts.append(_change_context(merge_base, stat))
    return "\n\n".join(parts) + "\n"


def _run_lanes(
    lanes: list[LintLane],
    shared_text: str,
    merge_base: str,
    stat: str,
    leads: str,
    agent_cmd: list[str],
    spec: AgentSpec,
    env: dict[str, str],
) -> list[LaneResult]:
    prompts = {lane.name: _lane_prompt(shared_text, lane, merge_base, stat, leads) for lane in lanes}
    results: dict[str, LaneResult] = {}
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = {pool.submit(_timed_agent, agent_cmd, spec, env, prompts[lane.name]): lane for lane in lanes}
        for future in as_completed(futures):
            lane = futures[future]
            cp, elapsed = future.result()
            prompt = prompts[lane.name]
            if cp is None:
                results[lane.name] = LaneResult(lane.name, "", "", None, elapsed, prompt)
            else:
                results[lane.name] = LaneResult(
                    lane.name, cp.stdout, cp.stderr, cp.returncode, elapsed, prompt, cp.usage
                )
    return [results[lane.name] for lane in lanes]


def _lane_body(result: LaneResult) -> str:
    if result.returncode is None:
        return "(lane timed out — no findings)"
    if result.returncode != 0:
        return "(lane errored — no findings)"
    return result.stdout or "(no findings)"


def _composer_prompt(lane_results: list[LaneResult], shared_text: str, merge_base: str, stat: str) -> str:
    labelled = "\n\n".join(f"=== Lane: {r.name} ===\n{_lane_body(r)}" for r in lane_results)
    parts = [READ_ONLY_MANDATE, shared_text, COMPOSER_INSTRUCTIONS, labelled, _change_context(merge_base, stat)]
    return "\n\n".join(parts) + "\n"


def _compose(
    lane_results: list[LaneResult],
    shared_text: str,
    merge_base: str,
    stat: str,
    agent_cmd: list[str],
    spec: AgentSpec,
    env: dict[str, str],
) -> ComposerRun:
    prompt = _composer_prompt(lane_results, shared_text, merge_base, stat)
    cp, elapsed = _timed_agent(agent_cmd, spec, env, prompt)
    if cp is None:
        return ComposerRun(prompt, "", "", None, elapsed)
    return ComposerRun(prompt, cp.stdout, cp.stderr, cp.returncode, elapsed, cp.usage)


def _concat_findings(lane_results: list[LaneResult]) -> str:
    """Deterministic merge: dedupe by (path, line, code), keep max confidence, sort."""
    best: dict[tuple, list] = {}
    for r in lane_results:
        for row in _parse_findings(r.stdout):
            key = (row[0], row[1], row[2])
            if key not in best or row[3] > best[key][3]:
                best[key] = row
    ordered = sorted(best.values(), key=lambda x: (x[0], x[1], x[2]))
    return "\n".join(f"{p}:{ln}: {code} ({conf:.2f}) {msg}" for p, ln, code, conf, msg in ordered)


def _resolve_review_stat() -> tuple[str, str] | None:
    """Resolve the merge-base with origin/main and a `git diff --stat` of the branch.

    Returns `(merge_base, stat)`, or None (after echoing the reason) when the
    merge-base can't be resolved or the branch has no changes — both advisory
    no-ops for the caller. Lanes get this changed-file inventory (every file, any
    language), not a pasted diff, and probe each file themselves.
    """
    base = subprocess.run(["git", "merge-base", "origin/main", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True)
    if base.returncode != 0:
        click.echo("  ⚠ Lint review skipped: could not resolve merge-base with origin/main")
        click.echo(f"    (run `git fetch origin main` first; git said: {base.stderr.strip()})")
        return None
    merge_base = base.stdout.strip()
    # Stat the working tree against the merge-base: covers all branch work, committed and
    # uncommitted, so the review runs whether or not the author has committed before the
    # pre-push checklist.
    stat = subprocess.run(
        ["git", "diff", "--stat", merge_base],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not stat.strip():
        click.echo("Lint review: no changes on this branch.")
        return None
    return merge_base, stat


def _merge_lane_results(
    lane_results: list[LaneResult],
    lanes: list[LintLane],
    shared_text: str,
    merge_base: str,
    stat: str,
    agent_cmd: list[str],
    spec: AgentSpec,
    env: dict[str, str],
    compose: bool,
) -> MergeOutcome:
    """Merge lane outputs deterministically, or use the optional composer."""
    timed_out = any(r.returncode is None for r in lane_results)
    if compose and len(lanes) > 1:
        run = _compose(lane_results, shared_text, merge_base, stat, agent_cmd, spec, env)
        if run.returncode is None:
            click.echo("  ⚠ Lint composer timed out; falling back to concat")
            return MergeOutcome(_concat_findings(lane_results), "compose", -1, True, run)
        if run.returncode != 0:
            click.echo(f"  ⚠ Lint composer exited {run.returncode}; falling back to concat")
            return MergeOutcome(_concat_findings(lane_results), "compose", run.returncode, timed_out, run)
        return MergeOutcome(run.stdout, "compose", 0, timed_out, run)
    mode = "concat" if len(lanes) > 1 else f"lane:{lanes[0].name}"
    return MergeOutcome(_concat_findings(lane_results), mode, 0, timed_out, None)


def _write_arm_log(
    arm_dir: pathlib.Path, prompt: str, stdout: str, stderr: str, returncode: int | None, elapsed: float
) -> None:
    """Write one arm's exact prompt and raw output (stdout/stderr/status) under `arm_dir`."""
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "prompt.md").write_text(prompt)
    status = "timed out" if returncode is None else f"exit {returncode}"
    body = (
        f"# {arm_dir.name} — lint review arm\n\n"
        f"- status: {status}\n"
        f"- elapsed: {elapsed:.2f}s\n\n"
        f"## stdout\n\n{stdout}\n\n"
        f"## stderr\n\n{stderr}\n"
    )
    (arm_dir / "output.md").write_text(body)


def _summary_md(log_dir: pathlib.Path, run: ReviewRun) -> str:
    files, added, removed = run.diff_stats
    outcome = run.outcome
    n_findings = len(_parse_findings(outcome.findings_text)) if outcome.findings_text else 0
    lines = [
        f"# Marin lint review — {log_dir.name}",
        "",
        f"- log dir: `{log_dir}`",
        f"- branch: `{run.branch or '?'}`",
        f"- merge base: `{run.merge_base}`",
        f"- diff: {files} files, +{added} -{removed}",
        f"- agent: {run.agent_spec.vendor} / {run.agent_spec.model} / {run.agent_spec.effort}",
        f"- merge mode: {outcome.mode}",
        f"- composer exit: {outcome.composer_rc}",
        f"- timed out: {str(outcome.timed_out).lower()}",
        f"- total elapsed: {run.elapsed_total:.2f}s",
        f"- findings: {n_findings} (see `combined.md`)",
        "",
        "## Arms",
        "",
        "| arm | status | elapsed | stdout lines | tokens | cost USD |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    arms: list[tuple[str, int | None, float, str, AgentUsage]] = [
        (r.name, r.returncode, r.elapsed, r.stdout, r.usage) for r in run.lane_results
    ]
    if outcome.composer is not None:
        c = outcome.composer
        arms.append(("composer", c.returncode, c.elapsed, c.stdout, c.usage))
    for name, rc, elapsed, stdout, usage in arms:
        status = "timed out" if rc is None else f"exit {rc}"
        tokens = str(usage.total_tokens) if usage.total_tokens is not None else "?"
        cost = f"{usage.cost_usd:.6f}" if usage.cost_usd is not None else "?"
        lines.append(f"| {name} | {status} | {elapsed:.2f}s | {len(stdout.splitlines())} | {tokens} | {cost} |")
    return "\n".join(lines) + "\n"


def _review_log_dir(branch: str | None, started: float) -> pathlib.Path:
    """Create and return this run's log directory.

    Runs are namespaced by branch so concurrent worktrees on one host don't interleave,
    and the leaf is `mkdtemp`-unique so two runs starting in the same second don't
    clobber each other: `/tmp/marin-linter/<sanitized-branch>/<timestamp>-<uniq>/`.
    """
    branch_root = LINT_REVIEW_LOG_ROOT / re.sub(r"[^\w.-]+", "-", branch or "detached")
    branch_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(started))
    return pathlib.Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=branch_root))


def _write_review_log(log_dir: pathlib.Path, run: ReviewRun) -> None:
    """Persist the raw per-arm prompts/outputs, the combined findings, and a run summary.

    Layout under `log_dir` (one directory per review run):
        <arm>/prompt.md, <arm>/output.md  — one per lane plus `composer/`
        combined.md                       — the final merged findings (what is printed)
        summary.md                        — run metadata and a per-arm status/timing table
    """
    for r in run.lane_results:
        _write_arm_log(log_dir / r.name, r.prompt, r.stdout, r.stderr, r.returncode, r.elapsed)
    if run.outcome.composer is not None:
        c = run.outcome.composer
        _write_arm_log(log_dir / "composer", c.prompt, c.stdout, c.stderr, c.returncode, c.elapsed)
    (log_dir / "combined.md").write_text((run.outcome.findings_text or "") + "\n")
    (log_dir / "summary.md").write_text(_summary_md(log_dir, run))


def _sum_complete_int(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _sum_complete_float(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _review_arms(run: ReviewRun) -> list[tuple[str, int | None, float, AgentUsage]]:
    arms = [(result.name, result.returncode, result.elapsed, result.usage) for result in run.lane_results]
    if run.outcome.composer is not None:
        composer = run.outcome.composer
        arms.append(("composer", composer.returncode, composer.elapsed, composer.usage))
    return arms


def _aggregate_usage(run: ReviewRun) -> AgentUsage:
    usages = [usage for _, _, _, usage in _review_arms(run)]
    return AgentUsage(
        input_tokens=_sum_complete_int([usage.input_tokens for usage in usages]),
        cache_creation_input_tokens=_sum_complete_int([usage.cache_creation_input_tokens for usage in usages]),
        cache_read_input_tokens=_sum_complete_int([usage.cache_read_input_tokens for usage in usages]),
        output_tokens=_sum_complete_int([usage.output_tokens for usage in usages]),
        cost_usd=_sum_complete_float([usage.cost_usd for usage in usages]),
        reported_total_tokens=_sum_complete_int([usage.total_tokens for usage in usages]),
    )


def _ship_review_event(
    run: ReviewRun,
    started: float,
    findings: list[list],
    log_dir: pathlib.Path | None,
) -> None:
    """Assemble and ship the review's telemetry event (see infra/codehealth/log_stats.py)."""
    diff_files, diff_added, diff_removed = run.diff_stats
    usage = _aggregate_usage(run)
    arms = _review_arms(run)
    arm_usage = [
        {
            "name": name,
            "exit_code": returncode,
            "elapsed": elapsed,
            "input_tokens": arm.input_tokens,
            "cache_creation_input_tokens": arm.cache_creation_input_tokens,
            "cache_read_input_tokens": arm.cache_read_input_tokens,
            "output_tokens": arm.output_tokens,
            "total_tokens": arm.total_tokens,
            "cost_usd": arm.cost_usd,
        }
        for name, returncode, elapsed, arm in arms
    ]
    _ship_review_stats(
        {
            "invocation_id": str(uuid.uuid4()),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "tool": "pre-commit-review",
            "invocation": {
                "variant": run.outcome.mode,
                "trigger": _review_trigger(),
                "agent_cli": run.agent_command[0],
                "agent_vendor": run.agent_spec.vendor,
                "agent_model": run.agent_spec.model,
                "agent_effort": run.agent_spec.effort,
                "agent_calls": len(arms),
                "agent_calls_json": json.dumps(arm_usage, separators=(",", ":")),
                "input_tokens": usage.input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": usage.cost_usd,
                "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
                "merge_base_sha": run.merge_base,
                "head_sha": _review_head_sha(),
                "pr_number": _review_pr_number(),
                "marin_user": _git(["config", "user.email"]),
                "lint_catalog_sha": catalog_sha(LINT_CATALOG),
                "diff_files": diff_files,
                "diff_added_lines": diff_added,
                "diff_removed_lines": diff_removed,
                "elapsed": run.elapsed_total,
                "agent_exit_code": run.outcome.composer_rc,
                "timed_out": run.outcome.timed_out,
            },
            "findings": findings,
        },
        log_dir,
    )


def run_lint_review(agent_command: str, lane_names: list[str] | None = None, compose: bool = False) -> int:
    """Run the advisory `infra/lint/` catalog over the branch's changes via headless agents.

    Fans out one agent per lane (see `LINT_LANES`); each lane is handed the changed-file
    inventory (`git diff --stat`) plus read-only git access and probes the files itself.
    The complexity lane is also fed static-complexity leads. Their outputs use deterministic
    dedupe-and-concat by default; `compose=True` adds a composer agent.
    `lane_names` restricts the run to a subset of lanes for debugging.

    `agent_command` is the headless CLI invocation each lane/composer agent reads
    its prompt from on stdin. It must select a model and effort explicitly.

    Findings are advisory and never block. Returns 0 for every outcome that fits
    that contract (no findings, findings emitted, agent unavailable, merge-base
    unresolved, lane/composer timeout). Returns 1 only on a usage error (unknown
    lane) or when every lane's agent failed to run, which indicates a broken
    agent CLI worth surfacing.
    """
    lanes = list(LINT_LANES)
    if lane_names:
        known = {lane.name for lane in LINT_LANES}
        unknown = [n for n in lane_names if n not in known]
        if unknown:
            click.echo(f"Error: unknown lint lane(s): {', '.join(unknown)}. Valid: {', '.join(sorted(known))}", err=True)
            return 1
        lanes = [lane for lane in LINT_LANES if lane.name in lane_names]

    agent_cmd = shlex.split(agent_command)
    if not agent_cmd:
        click.echo("Error: agent command is empty", err=True)
        return 1
    try:
        spec = _agent_spec(agent_cmd)
        agent_cmd = _with_usage_output(agent_cmd, spec)
    except ValueError as error:
        click.echo(f"Error: {error}", err=True)
        return 1
    if shutil.which(agent_cmd[0]) is None:
        click.echo(f"  ⚠ Lint review skipped: agent '{agent_cmd[0]}' not found on PATH")
        return 0
    agent_cmd = _with_readonly_access(agent_cmd)

    resolved = _resolve_review_stat()
    if resolved is None:
        return 0
    merge_base, stat = resolved

    # Drop lanes whose diff-size floor the change doesn't clear (only the holistic meta lane
    # sets one). Do this before computing leads / running agents so neither pays for a lane
    # that will not run.
    diff_stats = _diff_numstat(merge_base)
    changed_lines = diff_stats[1] + diff_stats[2]
    runnable = []
    for lane in lanes:
        if changed_lines > lane.min_diff_lines:
            runnable.append(lane)
        else:
            click.echo(
                f"  Lint review: '{lane.name}' lane skipped (diff {changed_lines} ≤ {lane.min_diff_lines}-line floor)"
            )
    lanes = runnable
    if not lanes:
        return 0

    shared_text = LINT_CATALOG.shared_prompt
    leads = ""
    if any(lane.include_complexity_leads for lane in lanes):
        leads = complexity_leads.compute_leads(_read_worktree, _changed_py_files(merge_base))
    env = _lint_review_env()
    started = time.time()

    lane_results = _run_lanes(lanes, shared_text, merge_base, stat, leads, agent_cmd, spec, env)
    for r in lane_results:
        if r.returncode is None:
            click.echo(f"  ⚠ Lint lane '{r.name}' timed out after {LINT_REVIEW_TIMEOUT}s")
        elif r.returncode != 0:
            detail = r.stderr.splitlines()[0] if r.stderr else ""
            click.echo(f"  ⚠ Lint lane '{r.name}' exited {r.returncode}: {detail}")

    outcome = _merge_lane_results(lane_results, lanes, shared_text, merge_base, stat, agent_cmd, spec, env, compose)
    parsed = _parse_findings(outcome.findings_text) if outcome.findings_text else []
    elapsed = time.time() - started

    # Persist raw per-arm + combined output for debugging a slow/broken cycle. Log I/O is a
    # side channel: a write failure (e.g. /tmp not writable) must not fail the advisory review.
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    run = ReviewRun(lane_results, outcome, merge_base, diff_stats, elapsed, branch, spec, tuple(agent_cmd))
    log_dir: pathlib.Path | None = None
    try:
        log_dir = _review_log_dir(branch, started)
        _write_review_log(log_dir, run)
        click.echo(f"  Lint review logs: {log_dir}")
    except OSError as e:
        log_dir = None
        click.echo(f"  ⚠ Lint review: could not write logs under {LINT_REVIEW_LOG_ROOT}: {e}")

    # Ships after the log dir exists so a failed write lands in stats.log beside
    # the run it belongs to instead of vanishing.
    _ship_review_event(run, started, parsed, log_dir)

    if all(r.returncode != 0 for r in lane_results):
        click.echo("  ⚠ Lint review: every lane failed to run (is the agent CLI working?)")
        return 1

    if not outcome.findings_text:
        click.echo("Lint review: no findings.")
        return 0

    click.echo("Lint review findings (advisory — search infra/lint/ for each ml-... code):\n")
    click.echo(outcome.findings_text)
    return 0
