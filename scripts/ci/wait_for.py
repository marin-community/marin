#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Block until the first of several events fires — a ``select`` over shell
predicates and GitHub PR state.

Each event is ``<kind> <arg>``. The general kind is an arbitrary shell predicate;
the ``github.*`` kinds are built-in conveniences so common PR waits need no shell:

    poll <shell command>    fires when the command exits 0
    github.ci <PR>          fires the moment any check fails, else when all checks pass
    github.pr_comment <PR>  fires on a new comment that raises a real code concern
    github.review <PR>      fires on a decisive review, or one whose body raises a concern

``github.pr_comment`` and ``github.review`` skip low-signal chatter by default so the
caller is not woken for nothing. A catalog of rules (``COMMENT_RULES``) names, per bot,
the mundane shapes that bot emits: the in-progress placeholder it posts the moment a PR
opens and edits in place once done, its "no issues found" verdict, and the
automated-review wrapper whose findings arrive as separate inline comments. Only the
automation a rule names is ever suppressed, so a comment from a human — or from a bot the
catalog does not cover — always fires. Comments are keyed on content, so a placeholder a
bot later edits into a real finding re-surfaces as new activity. Pass
``--comment-filter all`` to fire on every new comment instead.

`poll` is the escape hatch for anything without a built-in: compose the predicate
with the shell (``| grep -q``, ``| jq -e``, ``test``). For example, wait for the
PR to close with ``poll 'test "$(gh pr view 1234 --json state --jq .state)" != OPEN'``.
Specs come from argv (one quoted token each) or stdin (one per line; ``#`` comments):

    uv run scripts/ci/wait_for.py --timeout 12h \\
      'poll loom session poll weaver/foo --quiet | grep -q done' \\
      'github.ci 1234' 'github.pr_comment 1234'

    uv run scripts/ci/wait_for.py --timeout 12h <<'HERE'
    poll loom session poll weaver/foo --quiet | grep -q done
    github.ci 1234
    github.pr_comment 1234
    HERE

Prints one JSON object naming the arm that fired and its payload. Exit ``0`` an arm
fired, ``2`` --timeout elapsed, ``1`` error. Exit ``0`` means an arm fired, not that
CI passed — read ``result.conclusion``.
"""

import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import NamedTuple

import click
from rigging.timing import ExponentialBackoff

# `gh` calls are quick metadata reads; bound them so a hung call cannot wedge the
# select loop. `poll` commands get their own, larger budget (--poll-timeout).
GH_TIMEOUT = 60.0
# Give a flaky source several backoff rounds before declaring the wait unworkable.
MAX_SOURCE_ERRORS = 5

# `gh pr checks --json` reports one bucket per check, already deduped to the latest
# run (the same view as the UI / `gh pr checks` exit code), so superseded reruns do
# not leak through. We fire the instant anything is failing/cancelled — without
# waiting on the slower checks — so the caller can react to the failure; with nothing
# failing we wait until nothing is pending and report success.
_CI_PENDING_BUCKET = "pending"
_CI_FAILING_BUCKETS = {"fail", "cancel"}

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$")
_DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


class GhError(RuntimeError):
    """A `gh` invocation failed (non-zero exit or unparseable output)."""


class EventKind(StrEnum):
    GITHUB_CI = "github.ci"
    GITHUB_PR_COMMENT = "github.pr_comment"
    GITHUB_REVIEW = "github.review"
    POLL = "poll"


@dataclass(frozen=True)
class EventSpec:
    kind: EventKind
    arg: str
    raw: str


def parse_duration(text: str) -> float:
    """Parse a duration like ``90``, ``30m``, ``4h`` into seconds (bare number = seconds)."""
    match = _DURATION_RE.match(text)
    if not match:
        raise click.BadParameter(f"invalid duration {text!r}; use e.g. 90, 30m, 4h")
    return float(match.group(1)) * _DURATION_UNITS[match.group(2) or "s"]


def parse_spec(line: str) -> EventSpec:
    """Parse one ``<kind> <arg...>`` spec. The first token is the kind; the rest is the arg."""
    text = line.strip()
    kind_token, _, arg = text.partition(" ")
    arg = arg.strip()
    try:
        kind = EventKind(kind_token)
    except ValueError:
        valid = ", ".join(k.value for k in EventKind)
        raise click.BadParameter(f"unknown event kind {kind_token!r} in {line!r}; valid kinds: {valid}") from None
    if not arg:
        raise click.BadParameter(f"event {kind_token!r} requires an argument: {line!r}")
    return EventSpec(kind=kind, arg=arg, raw=text)


# --------------------------------------------------------------------------- IO


def _gh(args: list[str], *, timeout: float = GH_TIMEOUT) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise GhError(f"`gh {' '.join(args)}` failed (exit {proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def gh_json(args: list[str], *, timeout: float = GH_TIMEOUT) -> object:
    out = _gh(args, timeout=timeout)
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise GhError(f"`gh {' '.join(args)}` returned unparseable JSON: {exc}") from exc


def gh_pr_checks(pr: str, repo: str) -> list[dict]:
    """Return gh's check rows for the PR head (empty until any check registers)."""
    # `gh pr checks --json` prints the rows and exits 0 even while checks are pending.
    # We read stdout directly instead of via gh_json because a PR with no checks
    # registered yet returns an empty body — which means "nothing to judge yet, keep
    # polling", but which gh_json would raise on (json.loads("") fails / non-zero exit).
    proc = subprocess.run(
        ["gh", "pr", "checks", pr, "--repo", repo, "--json", "name,bucket,state"],
        capture_output=True,
        text=True,
        timeout=GH_TIMEOUT,
    )
    out = proc.stdout.strip()
    if not out:
        return []
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise GhError(f"`gh pr checks {pr}` returned unparseable JSON: {exc}") from exc


@dataclass(frozen=True)
class GhRecord:
    """A comment or review from the GitHub API — the fields the activity sources need."""

    id: int
    author: str
    body: str
    url: str
    state: str | None
    kind: str


_RECORD_JQ = '.[]|{id:.id,author:(.user.login//""),body:(.body//""),url:(.html_url//""),state:(.state//null)}'


def gh_api_list(repo: str, path: str, *, kind: str) -> list[GhRecord]:
    """Return every record in a paginated GitHub collection, tagged with ``kind``."""
    # `gh --paginate` with `--jq` applies the filter per page, so the output is one
    # compact JSON object per element per line (JSONL) across all pages.
    out = _gh(["api", "--paginate", f"repos/{repo}/{path}", "--jq", _RECORD_JQ])
    return [GhRecord(**json.loads(line), kind=kind) for line in out.splitlines() if line.strip()]


def resolve_repo(repo: str | None) -> str:
    if repo:
        return repo
    data = gh_json(["repo", "view", "--json", "nameWithOwner"])
    return data["nameWithOwner"]  # pyrefly: ignore  # gh JSON shape


def authenticated_user() -> str:
    return _gh(["api", "user", "--jq", ".login"]).strip()


class PollResult(NamedTuple):
    exit_code: int | None  # None on timeout
    stdout: str
    stderr: str


def run_poll(command: str, *, timeout: float) -> PollResult:
    """Run a shell predicate; ``exit_code`` is None on timeout."""
    try:
        proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return PollResult(None, "", "")
    return PollResult(proc.returncode, proc.stdout, proc.stderr)


# -------------------------------------------------------------------- pure logic


@dataclass(frozen=True)
class CiOutcome:
    done: bool
    observed: int
    conclusion: str | None
    failing: tuple[str, ...]
    pending: tuple[str, ...]
    checks: tuple[dict, ...]


def evaluate_ci(rows: Iterable[dict]) -> CiOutcome:
    """Decide whether the ``github.ci`` arm should fire from `gh pr checks` rows.

    Fire the moment any check fails so the caller can start addressing the failure
    without waiting for the slower checks to finish; with nothing failing, fire only
    once every check has settled. Empty rows (no checks registered yet) or
    still-pending checks with none failing ⇒ keep waiting.
    """
    rows = list(rows)
    failing = tuple(r["name"] for r in rows if r["bucket"] in _CI_FAILING_BUCKETS)
    pending = tuple(r["name"] for r in rows if r["bucket"] == _CI_PENDING_BUCKET)
    if not rows or (pending and not failing):
        return CiOutcome(done=False, observed=len(rows), conclusion=None, failing=(), pending=pending, checks=())
    checks = tuple({"name": r["name"], "bucket": r["bucket"]} for r in rows)
    return CiOutcome(
        done=True,
        observed=len(rows),
        conclusion="failure" if failing else "success",
        failing=failing,
        pending=pending,
        checks=checks,
    )


def _fingerprint(record: GhRecord) -> str:
    """A content signature for a record, so an edited comment counts as new activity."""
    return f"{record.state or ''}\x00{record.body}"


def select_new(records: list[GhRecord], baseline: dict[int, str], ignore_authors: set[str]) -> list[GhRecord]:
    """Records that are new or edited since the baseline snapshot, excluding ignored authors.

    Keying on content rather than id alone lets a comment that a review bot posts as an
    in-progress placeholder and later edits in place re-surface once its real content lands.
    """
    return [r for r in records if r.author not in ignore_authors and baseline.get(r.id) != _fingerprint(r)]


class Significance(StrEnum):
    """How much a PR comment warrants waking the monitoring agent."""

    CONCERN = "concern"  # raises a real code concern — worth firing on
    PROGRESS = "progress"  # an in-progress placeholder, edited in place once the bot is done
    CLEAN = "clean"  # an explicit "nothing to address" verdict
    WRAPPER = "wrapper"  # an automated-review container; its findings arrive separately


class CommentFilter(StrEnum):
    """Which new comments a PR-activity arm fires on."""

    SIGNIFICANT = "significant"  # only comments classified CONCERN
    ALL = "all"  # every new comment


# Bot logins as the REST API reports them (`.user.login`, which carries the `[bot]` suffix —
# unlike the GraphQL login, which does not).
CLAUDE_BOT = "claude[bot]"
CODEX_BOT = "chatgpt-codex-connector[bot]"

# --- body shapes -------------------------------------------------------------------
#
# The review bots announce themselves the moment a PR opens with a placeholder — a heading,
# a task checklist, a spinner image, a link to the job — and then edit it in place into the
# real review. The heading wording tracks whatever prompt the bot is running ("Code review in
# progress", "Reviewing PR #123", "PR Review: <title>"), and the spinner is an opaque
# user-attachments URL, so neither is worth matching on. What holds across every variant is
# the structure: strip the scaffolding and a placeholder has no prose left.
#
# Structure alone is not enough to suppress — a reviewer may legitimately enumerate required
# fixes as an all-unchecked task list — so a placeholder must both look like one (a checklist
# or a "working…" label) and carry no substantive text.
_TASK_ITEM_RE = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s.*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s.*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_HRULE_RE = re.compile(r"^\s*(?:-{3,}|={3,}|\*{3,})\s*$", re.MULTILINE)
_JOB_LINK_RE = re.compile(r"\[[^\]]*view (?:job|run)[^\]]*\]\([^)]*\)", re.IGNORECASE)
_WORKING_LABEL = r"working(?:\.\.\.|…)"
_WORKING_RE = re.compile(rf"{_WORKING_LABEL}|⏳|🔄", re.IGNORECASE)
_PLACEHOLDER_FILLER_RE = re.compile(rf"i'?ll analyze this and get back to you\.?|{_WORKING_LABEL}", re.IGNORECASE)
# Whitespace and markdown decoration carry no prose, so they do not count as residue.
_DECORATION_RE = re.compile("[\\s*_`>#~|.\\-\u2014\u2013]+")  # \u2014\u2013: the em/en dashes the bots rule off with
# Across 200 PRs of review-bot comments the two populations are cleanly separated: a
# placeholder leaves at most 50 characters of residue (a progress line such as "Running a
# multi-agent correctness review…"), while the shortest real review leaves 121. Nothing lands
# in between, so the cutoff sits in the empty band.
_PLACEHOLDER_RESIDUE_MAX = 80

# An explicit "nothing to address" verdict. The bots qualify the noun they cleared ("No code
# issues found", "No correctness bugs found", "No blocking issues"), so allow a couple of
# words between the negation and it.
_CLEAN_RE = re.compile(
    r"\bno\s+(?:\w+[\s-]+){0,2}(?:issues?|bugs?|concerns?|problems?|blockers?|findings?)\b"
    r"|\blgtm\b|\blooks good(?: to me)?\b|\bship it\b",
    re.IGNORECASE,
)
_CLEAN_HEAD = 400  # a clean verdict counts only if it leads the comment, not buried below a concern

# A qualified verdict clears one axis, not the review: the bots write "No correctness bugs
# found" and then report a compliance finding ("#### Findings (2 …)", "One hard-rule
# violation …") further down. So a verdict only counts as clean when nothing else in the body
# reports anything.
_FINDING_RE = re.compile(
    r"^\s*#{1,6}\s*findings?\b"
    r"|\b(?:one|two|three|four|five|\d+)\s+(?:finding|issue|bug|problem|violation|concern)s?\b"
    r"|\bviolat(?:es|ion|ions)\b|\bmust (?:be )?fix|\bneeds? fixing\b",
    re.IGNORECASE | re.MULTILINE,
)

# An automated-review summary whose actionable findings are posted as separate inline
# comments (e.g. Codex's top-level review body).
_WRAPPER_RE = re.compile(
    r"automated review suggestions|<summary>[^<]*About Codex|#+\s*💡?\s*Codex Review", re.IGNORECASE
)


def _placeholder_residue(body: str) -> str:
    """The prose a body carries once in-progress-placeholder scaffolding is stripped."""
    for pattern in (_HTML_TAG_RE, _TASK_ITEM_RE, _HEADING_RE, _HRULE_RE, _JOB_LINK_RE, _PLACEHOLDER_FILLER_RE):
        body = pattern.sub(" ", body)
    return _DECORATION_RE.sub("", body)


def is_progress_placeholder(body: str) -> bool:
    """Whether a body is a bot's in-progress placeholder rather than a report."""
    if not _TASK_ITEM_RE.search(body) and not _WORKING_RE.search(body):
        return False
    return len(_placeholder_residue(body)) <= _PLACEHOLDER_RESIDUE_MAX


def is_clean_verdict(body: str) -> bool:
    """Whether a body leads with a verdict that clears the whole review."""
    verdict = _CLEAN_RE.search(body[:_CLEAN_HEAD])
    if not verdict:
        return False
    # Drop the verdict itself ("no issues") and the checklist scaffolding ("- [x] Validate
    # findings") before asking whether anything left over reports a finding.
    rest = body[: verdict.start()] + body[verdict.end() :]
    return not _FINDING_RE.search(_TASK_ITEM_RE.sub(" ", rest))


def is_review_wrapper(body: str) -> bool:
    return bool(_WRAPPER_RE.search(body))


@dataclass(frozen=True)
class CommentRule:
    """One entry in the noise catalog: whose comments it judges, and which shape it suppresses."""

    author: str
    matches: Callable[[str], bool]
    significance: Significance


# The catalog of mundane comment shapes, keyed on author and body shape. A rule only ever
# applies to the one bot it names, so a comment from anyone else — every human, and every bot
# we have not catalogued — wakes the agent. Extend by adding a rule, not by loosening one:
# suppressing a real review comment costs far more than an extra wake-up.
COMMENT_RULES: tuple[CommentRule, ...] = (
    CommentRule(CLAUDE_BOT, is_progress_placeholder, Significance.PROGRESS),
    CommentRule(CLAUDE_BOT, is_clean_verdict, Significance.CLEAN),
    CommentRule(CODEX_BOT, is_review_wrapper, Significance.WRAPPER),
)


def classify_significance(body: str, author: str) -> Significance:
    """Classify a comment body as one of the ``Significance`` levels.

    An empty body is a container for inline comments that fire on their own. Otherwise the body
    is judged against the rules naming ``author``; anything left over is a ``CONCERN``, so an
    uncatalogued author always fires.
    """
    text = body.strip()
    if not text:
        return Significance.WRAPPER
    for rule in COMMENT_RULES:
        if author == rule.author and rule.matches(text):
            return rule.significance
    return Significance.CONCERN


# Reviews that decide the merge always fire, regardless of body — the state is the signal.
_DECISIVE_REVIEW_STATES = {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}


# ----------------------------------------------------------------------- sources


class Source:
    """One arm of the select. Subclasses implement ``check`` (return a payload or None)."""

    def __init__(self, spec: EventSpec):
        self.kind = spec.kind.value
        self.arg = spec.arg
        self.label = spec.raw
        self.last_status = "not yet checked"

    def check(self) -> dict | None:
        raise NotImplementedError


def _parse_pr(arg: str) -> str:
    try:
        return str(int(arg))
    except ValueError:
        raise click.BadParameter(f"expected a PR number, got {arg!r}") from None


class CiSource(Source):
    """Fires the moment any check fails, else once every check passes, via gh's deduped view."""

    def __init__(self, spec: EventSpec, repo: str):
        super().__init__(spec)
        self.repo = repo
        self.pr = _parse_pr(spec.arg)

    def check(self) -> dict | None:
        outcome = evaluate_ci(gh_pr_checks(self.pr, self.repo))
        if not outcome.done:
            self.last_status = f"{outcome.observed} checks, none failing, not all done"
            return None
        if outcome.pending:
            self.last_status = f"failing early: {', '.join(outcome.failing)} ({len(outcome.pending)} still pending)"
        else:
            self.last_status = f"done: {outcome.conclusion} ({outcome.observed} checks)"
        return {
            "conclusion": outcome.conclusion,
            "failing": list(outcome.failing),
            "pending": list(outcome.pending),
            "observed_checks": outcome.observed,
            "checks": list(outcome.checks),
        }


class PrActivitySource(Source):
    """Fires on a new comment or review since launch, diffed against a content baseline.

    Subclasses supply the fetch (which endpoints), the significance test (which new
    records are worth firing on), and the fired payload; the baseline-snapshot / diff /
    absorb scaffold is shared.
    """

    noun = "records"

    def __init__(self, spec: EventSpec, repo: str, ignore_authors: set[str], comment_filter: CommentFilter):
        super().__init__(spec)
        self.repo = repo
        self.pr = _parse_pr(spec.arg)
        self.ignore_authors = ignore_authors
        self.comment_filter = comment_filter
        self.baseline: dict[int, str] | None = None

    def _fetch(self) -> list[GhRecord]:
        raise NotImplementedError

    def _is_significant(self, record: GhRecord) -> bool:
        """Whether this new/edited record raises a real concern worth firing on."""
        return classify_significance(record.body, record.author) is Significance.CONCERN

    def _payload(self, new: list[GhRecord]) -> dict:
        raise NotImplementedError

    def check(self) -> dict | None:
        records = self._fetch()
        fingerprints = {r.id: _fingerprint(r) for r in records}
        if self.baseline is None:
            self.baseline = fingerprints
            self.last_status = f"baseline {len(records)} {self.noun}"
            return None
        changed = select_new(records, self.baseline, self.ignore_authors)
        self.baseline.update(fingerprints)  # absorb current content so ignored/noise records never re-fire
        fired = changed if self.comment_filter is CommentFilter.ALL else [r for r in changed if self._is_significant(r)]
        self.last_status = f"{len(records)} {self.noun}" + (
            f"; {len(changed)} new/edited, {len(fired)} significant" if changed else ""
        )
        return self._payload(fired) if fired else None


class CommentSource(PrActivitySource):
    """Fires on a new issue-comment or review-comment."""

    noun = "comments"
    ENDPOINTS = (("issue_comment", "issues/{pr}/comments"), ("review_comment", "pulls/{pr}/comments"))

    def _fetch(self) -> list[GhRecord]:
        out: list[GhRecord] = []
        for kind, path in self.ENDPOINTS:
            out += gh_api_list(self.repo, path.format(pr=self.pr), kind=kind)
        return out

    def _payload(self, new: list[GhRecord]) -> dict:
        return {
            "comments": [
                {
                    "author": r.author,
                    "body": r.body,
                    "url": r.url,
                    "kind": r.kind,
                    "significance": classify_significance(r.body, r.author).value,
                }
                for r in new
            ]
        }


class ReviewSource(PrActivitySource):
    """Fires on a decisive review (approve / changes-requested / dismissed) or one whose body raises a concern."""

    noun = "reviews"

    def _fetch(self) -> list[GhRecord]:
        # PENDING reviews are unsubmitted drafts; they are not events.
        return [r for r in gh_api_list(self.repo, f"pulls/{self.pr}/reviews", kind="review") if r.state != "PENDING"]

    def _is_significant(self, record: GhRecord) -> bool:
        # A merge-deciding state is the signal even with an empty body; a bare COMMENTED
        # review is usually just a wrapper for inline comments that fire on their own.
        return record.state in _DECISIVE_REVIEW_STATES or super()._is_significant(record)

    def _payload(self, new: list[GhRecord]) -> dict:
        return {"reviews": [{"author": r.author, "state": r.state, "url": r.url} for r in new]}


class PollSource(Source):
    def __init__(self, spec: EventSpec, poll_timeout: float):
        super().__init__(spec)
        self.command = spec.arg
        self.poll_timeout = poll_timeout

    def check(self) -> dict | None:
        poll = run_poll(self.command, timeout=self.poll_timeout)
        if poll.exit_code is None:
            self.last_status = f"timed out after {self.poll_timeout:g}s"
            return None
        self.last_status = f"exit {poll.exit_code}"
        if poll.exit_code != 0:
            return None
        return {
            "exit_code": 0,
            "command": self.command,
            "stdout_tail": _tail(poll.stdout),
            "stderr_tail": _tail(poll.stderr),
        }


def build_source(
    spec: EventSpec, *, repo: str, ignore_authors: set[str], poll_timeout: float, comment_filter: CommentFilter
) -> Source:
    if spec.kind is EventKind.GITHUB_CI:
        return CiSource(spec, repo)
    if spec.kind is EventKind.GITHUB_PR_COMMENT:
        return CommentSource(spec, repo, ignore_authors, comment_filter)
    if spec.kind is EventKind.GITHUB_REVIEW:
        return ReviewSource(spec, repo, ignore_authors, comment_filter)
    if spec.kind is EventKind.POLL:
        return PollSource(spec, poll_timeout)
    raise click.BadParameter(f"unsupported event kind {spec.kind!r}")  # pragma: no cover


# --------------------------------------------------------------------- scheduler


@dataclass
class _Armed:
    source: Source
    backoff: ExponentialBackoff
    due_at: float
    errors: int = 0


@dataclass(frozen=True)
class BackoffConfig:
    initial: float
    maximum: float
    factor: float
    jitter: float


def _tail(text: str, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    text = text.strip()
    if not text:
        return ""
    return "\n".join(text.splitlines()[-max_lines:])[-max_chars:]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timeout_result(sources: list[Source]) -> dict:
    return {
        "event": None,
        "status": "timeout",
        "sources": [{"label": s.label, "last_status": s.last_status} for s in sources],
    }


def select_loop(sources: list[Source], *, deadline: float | None, backoff: BackoffConfig) -> dict:
    """Poll each source on its own backoff; return the first fired event, or a timeout result."""
    now = time.monotonic()
    armed = [
        _Armed(s, ExponentialBackoff(backoff.initial, backoff.maximum, backoff.factor, backoff.jitter), now)
        for s in sources
    ]
    while True:
        next_arm = min(armed, key=lambda a: a.due_at)
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            return _timeout_result(sources)
        wait = next_arm.due_at - now
        if deadline is not None:
            wait = min(wait, deadline - now)
        if wait > 0:
            time.sleep(wait)
            if deadline is not None and time.monotonic() >= deadline:
                return _timeout_result(sources)
        try:
            result = next_arm.source.check()
        except (GhError, OSError, subprocess.SubprocessError) as exc:
            next_arm.errors += 1
            click.echo(
                f"[wait_for] {next_arm.source.label}: {exc} (error {next_arm.errors}/{MAX_SOURCE_ERRORS})", err=True
            )
            if next_arm.errors >= MAX_SOURCE_ERRORS:
                raise
            next_arm.due_at = time.monotonic() + next_arm.backoff.next_interval()
            continue
        next_arm.errors = 0
        if result is not None:
            return {
                "event": next_arm.source.kind,
                "arg": next_arm.source.arg,
                "label": next_arm.source.label,
                "fired_at": _now_iso(),
                "result": result,
            }
        next_arm.due_at = time.monotonic() + next_arm.backoff.next_interval()


# ----------------------------------------------------------------------- CLI glue


def read_specs(argv_specs: tuple[str, ...], *, use_stdin: bool | None) -> list[EventSpec]:
    raw = list(argv_specs)
    # Explicit --stdin always merges stdin in; otherwise only auto-read it when no
    # argv specs were given, so the argv form never blocks on or consumes stdin.
    if use_stdin or (use_stdin is None and not raw and not sys.stdin.isatty()):
        raw += [line for line in sys.stdin.read().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not raw:
        raise click.UsageError("no events given; pass specs as arguments or on stdin")
    return [parse_spec(s) for s in raw]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("specs", nargs=-1)
@click.option(
    "--stdin/--no-stdin",
    "use_stdin",
    default=None,
    help="Read specs from stdin (default: auto when no argv specs and stdin is not a TTY).",
)
@click.option("--timeout", default=None, help="Overall deadline, e.g. 90, 30m, 4h. Default: wait indefinitely.")
@click.option("--poll-timeout", default="120", help="Per-attempt timeout for `poll` commands.")
@click.option("--initial-interval", default="10", help="First backoff interval per source.")
@click.option("--max-interval", default="120", help="Backoff ceiling per source.")
@click.option("--factor", default=2.0, type=float, help="Backoff growth factor.")
@click.option("--jitter", default=0.1, type=float, help="Backoff jitter fraction in [0, 1).")
@click.option("--repo", default=None, help="OWNER/NAME (default: gh auto-detect from cwd).")
@click.option("--ignore-author", "ignore_authors", multiple=True, help="Comment/review author to ignore (repeatable).")
@click.option("--include-self", is_flag=True, help="Do not ignore the authenticated user's own comments.")
@click.option(
    "--comment-filter",
    type=click.Choice([f.value for f in CommentFilter]),
    default=CommentFilter.SIGNIFICANT.value,
    help="Which new comments fire github.pr_comment/github.review: 'significant' skips the review bots' "
    "in-progress placeholders, clean verdicts, and review wrappers; 'all' fires on every new comment.",
)
@click.option("--quiet", is_flag=True, help="Print only the fired event kind, not the JSON payload.")
def main(
    specs: tuple[str, ...],
    use_stdin: bool | None,
    timeout: str | None,
    poll_timeout: str,
    initial_interval: str,
    max_interval: str,
    factor: float,
    jitter: float,
    repo: str | None,
    ignore_authors: tuple[str, ...],
    include_self: bool,
    comment_filter: str,
    quiet: bool,
) -> None:
    """Block until the first armed event fires; print which one as JSON."""
    parsed = read_specs(specs, use_stdin=use_stdin)
    needs_github = any(s.kind is not EventKind.POLL for s in parsed)
    needs_authors = any(s.kind in (EventKind.GITHUB_PR_COMMENT, EventKind.GITHUB_REVIEW) for s in parsed)

    try:
        resolved_repo = resolve_repo(repo) if needs_github else ""
        ignored = set(ignore_authors)
        if needs_authors and not include_self:
            ignored.add(authenticated_user())
        sources = [
            build_source(
                s,
                repo=resolved_repo,
                ignore_authors=ignored,
                poll_timeout=parse_duration(poll_timeout),
                comment_filter=CommentFilter(comment_filter),
            )
            for s in parsed
        ]
        deadline = None if timeout is None else time.monotonic() + parse_duration(timeout)
        backoff = BackoffConfig(
            initial=parse_duration(initial_interval),
            maximum=parse_duration(max_interval),
            factor=factor,
            jitter=jitter,
        )
        result = select_loop(sources, deadline=deadline, backoff=backoff)
    except GhError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        click.echo("[wait_for] interrupted", err=True)
        raise SystemExit(130) from None

    if result.get("status") == "timeout":
        click.echo("timeout" if quiet else json.dumps(result))
        raise SystemExit(2)
    click.echo(result["event"] if quiet else json.dumps(result))


if __name__ == "__main__":
    main()
