# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic lint-review runner (lanes + composer) invoked by `pre-commit.py --review`.

Fans out one headless agent per "lane" (rules in infra/lint/*.md) over the branch
diff and merges the per-lane findings with a composer agent (or a deterministic
dedupe-and-concat). This subsystem is self-contained and used only by the
`--review` path of pre-commit.py.
"""

import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import click

# codehealth/ holds stdlib-only helpers imported by path (it is not an installed package).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "codehealth"))
import complexity as complexity_leads

ROOT_DIR = pathlib.Path(__file__).parent.parent
LINT_DIR = ROOT_DIR / "infra/lint"
LINT_SHARED = LINT_DIR / "shared.md"


LINT_REVIEW_AGENT_DEFAULT = "claude -p"

LINT_REVIEW_TIMEOUT = 600


@dataclass(frozen=True)
class LintLane:
    """One fan-out lane of the lint review: a rule file under infra/lint/."""

    name: str
    include_complexity_leads: bool


# Coarse lanes — one headless agent each. Keep this aligned with infra/lint/*.md.
LINT_LANES = (
    LintLane("complexity", True),
    LintLane("interfaces", False),
    LintLane("robustness", False),
    LintLane("cruft", False),
    LintLane("prose", False),
)

LINT_LANE_INSTRUCTIONS = (
    "You are ONE lane of the review. Apply ONLY the rules in the lane catalog above "
    'to the branch diff below. Follow the shared "Detector usage" and "Output format" '
    "exactly: emit one finding per line in the format it specifies, and emit nothing at "
    "all when there are no findings. Resolve overlap precedence within your lane; leave "
    "cross-lane duplicates to the composer. Work from the diff as given; do not re-derive it."
)

# The composer merges the lanes' outputs. Authored to never silently drop a real
# finding — it may only collapse true duplicates and drop overlap-precedence losers.
COMPOSER_INSTRUCTIONS = (
    "You are the COMPOSER. Five specialist lanes (complexity, interfaces, robustness, cruft, "
    "prose) each scanned the SAME branch diff against their slice of the catalog and emitted "
    "findings in the Output format above. Their labelled raw outputs and the diff follow below. "
    "You are an EDITOR, not a reviewer: merge them into the single final findings list, reasoned "
    "— not a blind concat. You do NOT re-scan for new issues, re-derive the diff, or invent "
    "findings. Read the diff only to adjudicate a duplicate/precedence call or sanity-check that "
    "a cited line exists.\n\n"
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

# Env vars that mark a Claude Code session and would re-bind the spawned headless
# agent to its parent's transcript / session. Stripped before exec so the
# sub-agent runs as a fresh, isolated session.
LINT_REVIEW_STRIPPED_ENV = (
    "ANTHROPIC_API_KEY",  # force subscription auth, not metered API billing
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_SSE_PORT",
)


# Output format the agent emits, per infra/lint/shared.md "Output format":
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


def _diff_stats(diff: str) -> tuple[int, int, int]:
    files = added = removed = 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return files, added, removed


def _git(args: list[str]) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or None
    except Exception:
        return None


def _lint_catalog_sha() -> str | None:
    """Fingerprint the multi-file lint catalog: sha1 over the sorted lane files.

    Mirrors infra/codehealth/log_stats.py so the `lint_catalog_sha` stat is
    comparable across the local review and the stats aggregator.
    """
    files = sorted(LINT_DIR.glob("*.md"))
    if not files:
        return None
    h = hashlib.sha1()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()


def _ship_review_stats(event: dict) -> None:
    """Fire-and-forget: hand the event off to infra/codehealth/log_stats.py via
    `uv run`. Detached so the W&B init/network cost never blocks the dev.
    Silent on every failure mode (no uv, no wandb, no auth, no network).
    """
    if not shutil.which("uv"):
        return
    try:
        proc = subprocess.Popen(
            ["uv", "run", "--quiet", str(ROOT_DIR / "infra" / "codehealth" / "log_stats.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=ROOT_DIR,
            start_new_session=True,
        )
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(event).encode())
        proc.stdin.close()
    except Exception:
        pass


@dataclass(frozen=True)
class LaneResult:
    name: str
    stdout: str
    stderr: str
    returncode: int | None  # None means the lane agent timed out


def _lint_review_env() -> dict[str, str]:
    # Strip session/auth markers so the headless agent runs as a fresh session
    # rather than nesting under the calling agent's transcript or being billed
    # via metered API auth.
    return {k: v for k, v in os.environ.items() if k not in LINT_REVIEW_STRIPPED_ENV}


def _run_agent(agent_cmd: list[str], env: dict[str, str], prompt: str) -> subprocess.CompletedProcess | None:
    """Run one headless agent over `prompt`; None if it times out."""
    try:
        return subprocess.run(
            agent_cmd,
            input=prompt,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            env=env,
            timeout=LINT_REVIEW_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None


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


def _lane_prompt(shared_text: str, lane: LintLane, diff: str, leads: str) -> str:
    parts = [shared_text, (LINT_DIR / f"{lane.name}.md").read_text()]
    if lane.include_complexity_leads and leads:
        parts.append(leads)
    parts.append(LINT_LANE_INSTRUCTIONS)
    parts.append(f"```diff\n{diff}\n```")
    return "\n\n".join(parts) + "\n"


def _run_lanes(
    lanes: list[LintLane], shared_text: str, diff: str, leads: str, agent_cmd: list[str], env: dict[str, str]
) -> list[LaneResult]:
    results: dict[str, LaneResult] = {}
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = {
            pool.submit(_run_agent, agent_cmd, env, _lane_prompt(shared_text, lane, diff, leads)): lane for lane in lanes
        }
        for future in as_completed(futures):
            lane = futures[future]
            cp = future.result()
            if cp is None:
                results[lane.name] = LaneResult(lane.name, "", "", None)
            else:
                results[lane.name] = LaneResult(lane.name, cp.stdout.strip(), cp.stderr.strip(), cp.returncode)
    return [results[lane.name] for lane in lanes]


def _lane_body(result: LaneResult) -> str:
    if result.returncode is None:
        return "(lane timed out — no findings)"
    if result.returncode != 0:
        return "(lane errored — no findings)"
    return result.stdout or "(no findings)"


def _compose(
    lane_results: list[LaneResult], shared_text: str, diff: str, agent_cmd: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess | None:
    labelled = "\n\n".join(f"=== Lane: {r.name} ===\n{_lane_body(r)}" for r in lane_results)
    prompt = "\n\n".join([shared_text, COMPOSER_INSTRUCTIONS, labelled, f"```diff\n{diff}\n```"]) + "\n"
    return _run_agent(agent_cmd, env, prompt)


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


def _resolve_review_diff() -> tuple[str, str] | None:
    """Resolve the merge-base with origin/main and the branch diff.

    Returns `(merge_base, diff)`, or None (after echoing the reason) when the
    merge-base can't be resolved or there are no Python/proto changes — both
    advisory no-ops for the caller.
    """
    base = subprocess.run(["git", "merge-base", "origin/main", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True)
    if base.returncode != 0:
        click.echo("  ⚠ Lint review skipped: could not resolve merge-base with origin/main")
        click.echo(f"    (run `git fetch origin main` first; git said: {base.stderr.strip()})")
        return None
    merge_base = base.stdout.strip()
    # Diff the working tree against the merge-base: covers all branch work,
    # committed and uncommitted, so the review runs whether or not the author
    # has committed before reaching the pre-push checklist.
    diff = subprocess.run(
        ["git", "diff", merge_base, "-U15", "--", "*.py", "*.proto"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not diff.strip():
        click.echo("Lint review: no Python/proto changes on this branch.")
        return None
    return merge_base, diff


def _merge_lane_results(
    lane_results: list[LaneResult],
    lanes: list[LintLane],
    shared_text: str,
    diff: str,
    agent_cmd: list[str],
    env: dict[str, str],
    compose: bool,
) -> tuple[str, str, int, bool]:
    """Merge lane outputs via the composer (default) or a deterministic concat.

    Returns `(findings_text, mode, composer_exit_code, timed_out)`.
    """
    timed_out = any(r.returncode is None for r in lane_results)
    if compose and len(lanes) > 1:
        cp = _compose(lane_results, shared_text, diff, agent_cmd, env)
        if cp is None:
            click.echo("  ⚠ Lint composer timed out; falling back to concat")
            return _concat_findings(lane_results), "compose", -1, True
        if cp.returncode != 0:
            click.echo(f"  ⚠ Lint composer exited {cp.returncode}; falling back to concat")
            return _concat_findings(lane_results), "compose", cp.returncode, timed_out
        return cp.stdout.strip(), "compose", 0, timed_out
    mode = "concat" if len(lanes) > 1 else f"lane:{lanes[0].name}"
    return _concat_findings(lane_results), mode, 0, timed_out


def _ship_review_event(
    mode: str,
    agent_cmd: list[str],
    merge_base: str,
    diff: str,
    started: float,
    composer_rc: int,
    timed_out: bool,
    findings: list[list],
) -> None:
    """Assemble and ship the review's telemetry event (see infra/codehealth/log_stats.py)."""
    diff_files, diff_added, diff_removed = _diff_stats(diff)
    _ship_review_stats(
        {
            "invocation_id": str(uuid.uuid4()),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "tool": "pre-commit-review",
            "invocation": {
                "variant": mode,
                "trigger": "local",
                "agent_cli": agent_cmd[0],
                "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
                "merge_base_sha": merge_base,
                "head_sha": _git(["rev-parse", "HEAD"]),
                "pr_number": None,
                "marin_user": _git(["config", "user.email"]),
                "lint_catalog_sha": _lint_catalog_sha(),
                "diff_files": diff_files,
                "diff_added_lines": diff_added,
                "diff_removed_lines": diff_removed,
                "elapsed": time.time() - started,
                "agent_exit_code": composer_rc,
                "timed_out": timed_out,
            },
            "findings": findings,
        }
    )


def run_lint_review(agent_command: str, lane_names: list[str] | None = None, compose: bool = True) -> int:
    """Run the advisory `infra/lint/` catalog over the branch diff via headless agents.

    Fans out one agent per lane (see `LINT_LANES`); the complexity lane is also
    fed static-complexity leads. Their outputs are merged by a composer agent
    (`compose=True`) or a deterministic dedupe-and-concat (`compose=False`).
    `lane_names` restricts the run to a subset of lanes for debugging.

    `agent_command` is the headless CLI invocation each lane/composer agent reads
    its prompt from on stdin (e.g. `claude -p`, `codex exec`). Callers should pass
    the command for the agent they are themselves running.

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
    if not agent_cmd or shutil.which(agent_cmd[0]) is None:
        agent_name = agent_cmd[0] if agent_cmd else "(empty)"
        click.echo(f"  ⚠ Lint review skipped: agent '{agent_name}' not found on PATH")
        return 0

    resolved = _resolve_review_diff()
    if resolved is None:
        return 0
    merge_base, diff = resolved

    shared_text = LINT_SHARED.read_text()
    leads = ""
    if any(lane.include_complexity_leads for lane in lanes):
        leads = complexity_leads.compute_leads(_read_worktree, _changed_py_files(merge_base))
    env = _lint_review_env()
    started = time.time()

    lane_results = _run_lanes(lanes, shared_text, diff, leads, agent_cmd, env)
    for r in lane_results:
        if r.returncode is None:
            click.echo(f"  ⚠ Lint lane '{r.name}' timed out after {LINT_REVIEW_TIMEOUT}s")
        elif r.returncode != 0:
            detail = r.stderr.splitlines()[0] if r.stderr else ""
            click.echo(f"  ⚠ Lint lane '{r.name}' exited {r.returncode}: {detail}")

    findings_text, mode, composer_rc, timed_out = _merge_lane_results(
        lane_results, lanes, shared_text, diff, agent_cmd, env, compose
    )
    parsed = _parse_findings(findings_text) if findings_text else []
    _ship_review_event(mode, agent_cmd, merge_base, diff, started, composer_rc, timed_out, parsed)

    if all(r.returncode != 0 for r in lane_results):
        click.echo("  ⚠ Lint review: every lane failed to run (is the agent CLI working?)")
        return 1

    if not findings_text:
        click.echo("Lint review: no findings.")
        return 0

    click.echo("Lint review findings (advisory — search infra/lint/ for each ml-... code):\n")
    click.echo(findings_text)
    return 0
