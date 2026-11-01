# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Belay session's commands, one per section of ``experiments/belay/RUNBOOK.md``.

    uv run python -m experiments.belay start --events <file>                      # section 0
    uv run python -m experiments.belay new-idea --idea-id N [--category ...]      # section 1
    uv run python -m experiments.belay code-change X#### --start | --finish       # section 1b
    uv run python -m experiments.belay run-finished X####                         # section 2
    uv run python -m experiments.belay analyze X####                              # section 2b
    uv run python -m experiments.belay run-failed X#### --run <run>               # section 3
    uv run python -m experiments.belay weekly-rollover                            # section 4
    uv run python -m experiments.belay recipe-change                              # section 5
    uv run python -m experiments.belay escalate --title "..." --body "..."        # section 6
    uv run python -m experiments.belay tracks [--json]                            # section 9: every launch

Each command checks its inputs, does its step, and prints a ``next:`` line (the next command, or ``stop``); a
refusal says what is wrong and what to run instead. ``--dry-run`` runs every read-only check and prints what would
be written, committed, pushed, launched or posted, without doing it.
"""

import argparse
import dataclasses
import datetime
import importlib.util
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from experiments.belay.analysis import Header, fetch_series, report_dir, report_url, write_report
from experiments.belay.branch import (
    current_branch,
    cut_week_branch,
    git,
    head_contains,
    named_checkout,
    repo_root,
    require_clean_pushed_head,
    require_week_branch,
    run_git,
    temporary_worktree,
)
from experiments.belay.code_change import (
    changed_paths,
    checkout_python,
    flag_commits,
    new_flags,
    probe,
    refuse_flag_off_change,
    refuse_leak,
    refuse_outside_allowed,
    run_precommit,
    run_tests,
)
from experiments.belay.config import (
    CHECKPOINT_TAG_PREFIX,
    CONFIG,
    CONTROL_RUN,
    TRACKS,
    WEEKLY_BRANCH,
    Baseline,
    Comparison,
    Track,
)
from experiments.belay.page import (
    Idea,
    PageRun,
    PageState,
    Post,
    comment_times,
    page_state,
    post_activity,
    publish_comment,
    read_comment,
    record_activity,
    report_idea,
    require_unedited,
    visible_text,
)
from experiments.belay.posts import (
    Comment,
    Row,
    control_label,
    footer,
    hardware,
    render,
    rule_line,
    stage_section,
    title,
    updated,
)
from experiments.belay.registry_entry import (
    ENTRIES,
    add_trial,
    add_trial_run,
    comment_url_of,
    commit_and_push,
    create_entry,
    entry_for_intake,
    entry_path,
    entry_url,
    entry_with_results,
    load_entry,
    require_registry_checkout,
)
from experiments.belay.results import Measure, RunResult, Verdict, compared, measure, overall_verdict
from experiments.belay.submit import (
    DEFAULT_TRACK,
    PROBE_SKIP,
    RUN_PREFIX,
    ExperimentSpec,
    RunSpec,
    Stage,
    check_launch,
    check_variant,
    iris_jobs,
    latest_stage,
    launch_command,
    load_spec,
    new_spec_text,
    next_stage_runs,
    parse_spec,
    parse_track,
    run_name,
    run_suffix,
    setting_fields,
    settings_of,
    spec_path,
    spec_with_runs,
    stage_runs,
    track_command,
    track_of_comment,
    variant_settings,
    with_launched,
)

BELAY = "uv run python -m experiments.belay"
ESCALATED = "Escalated:"
PAGE_RESULT_KIND = {Verdict.WIN: "better", Verdict.LOSS: "worse", Verdict.NOISE: "neutral"}


class EventKind(StrEnum):
    APPROVED = "approved"
    WORK_PENDING = "work_pending"
    RUN_FINISHED = "run_finished"
    NOTE = "note"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    idea_id: int | None
    experiment: str | None


def say_next(command: str) -> None:
    print(f"next: {command}")


def read_events(path: Path) -> list[Event]:
    events = json.loads(path.read_text())
    return [Event(EventKind(e["kind"]), e["payload"].get("idea_id"), e["payload"].get("experiment")) for e in events]


# ---------- 0. start ----------


def idea_step(idea: Idea) -> str:
    """What section 1 needs for ``idea``: a ``skip:`` line or its first command."""
    if (idea.note or "").startswith(ESCALATED):
        return f"skip: idea #{idea.id} is escalated and waits for a maintainer"
    if idea.state != "approved":
        return f"skip: idea #{idea.id} is {idea.state}"
    return f"section 1: {BELAY} new-idea --idea-id {idea.id}"


def finished_run_step(experiment: str, runs: list[PageRun]) -> str:
    """What a finished run of ``experiment`` needs: a ``skip:`` line, section 2, or section 3."""
    if not spec_path(experiment).exists():
        return f"skip: no run spec {experiment}.yaml on this branch, so no session launched it"
    failed = [r for r in runs if r.terminal and r.state != "succeeded"]
    if failed:
        return f"section 3: {BELAY} run-failed {experiment} --run {failed[0].run}"
    return f"section 2: {BELAY} run-finished {experiment}"


def start(events_file: Path, today: datetime.date) -> None:
    repo = repo_root()
    week = CONFIG.week_branch(today)
    print(f"recipe: {CONFIG.recipe_name} ({CONFIG.recipe_branch}); this week's branch: {week}")
    if current_branch(repo) != week or not head_contains(repo, f"origin/{CONFIG.base_branch}"):
        print(f"event: weekly rollover (the checkout is on {current_branch(repo)})")
        say_next(f"{BELAY} weekly-rollover   (section 4), then run start again")
        return
    require_registry_checkout()
    state = page_state()
    todo = []
    for event in read_events(events_file):
        if event.kind in (EventKind.APPROVED, EventKind.WORK_PENDING):
            line = f"[{event.kind} idea #{event.idea_id}] {idea_step(state.idea(event.idea_id))}"
        elif event.kind is EventKind.RUN_FINISHED:
            step = finished_run_step(event.experiment, state.runs_of(event.experiment))
            line = f"[run_finished {event.experiment}] {step}"
        else:
            line = f'[note] section 6: {BELAY} escalate --title "Note for a maintainer" --body "<the note>"'
        if line not in todo:
            todo.append(line)
    if not todo:
        print("no events")
        say_next("stop")
        return
    print("do these in order:")
    for i, line in enumerate(todo, 1):
        print(f"  {i}. {line}")
    say_next("the command in item 1 (a 'skip:' item needs nothing; go on to the next item)")


# ---------- 1. new idea ----------


def categories() -> list[str]:
    return json.loads((CONFIG.registry_checkout / ENTRIES / "categories.json").read_text())


def approved_idea(idea: Idea) -> tuple[Idea, list[str]]:
    """The idea with only the text that was approved and is visible: refuses a comment edited after approval, and
    strips hidden HTML and invisible characters (returned as notes for the plan)."""
    if idea.url and "issuecomment-" in idea.url:
        updated, transferred = comment_times(idea.url)
        approvals = [t for t in (transferred, idea.decided_at and datetime.datetime.fromisoformat(idea.decided_at)) if t]
        approved = min(approvals) if approvals else None  # the first approval, on either path, froze the text
        if approved is None:
            raise SystemExit(f"refusing: idea #{idea.id} has no approval time.\nnext: {BELAY} escalate ...")
        try:
            require_unedited(updated, approved)
        except SystemExit as refusal:  # add the escalation as the next step
            raise SystemExit(
                f"{refusal}\nnext: {BELAY} escalate --idea-id {idea.id} --title "
                f'"Idea #{idea.id} was edited after approval" --body "It needs re-approval of the edited text."'
            ) from None
    body, notes = visible_text(idea.body)
    return dataclasses.replace(idea, body=body), notes


def idea_track(idea: Idea) -> tuple[Track, bool]:
    """The idea's Track and whether it defaulted to speedrun; refuses an unknown one with the escalation to run."""
    try:
        return track_of_comment(idea.body)
    except SystemExit as refusal:  # add the escalation as the next step
        raise SystemExit(
            f"{refusal}\nnext: {BELAY} escalate --idea-id {idea.id} "
            f'--title "Idea #{idea.id} has an unknown Track" --body "Its Track: line names no track in RUNBOOK.md."'
        ) from None


def runnable(idea: Idea, track: Track) -> Baseline:
    """The track's baseline; refuses a track no session runs (custom, hero-mfu) with the escalation to run."""
    baseline = TRACKS[track]
    if baseline.launch is None:
        raise SystemExit(
            f"refusing: track {track} ({baseline.description}) is run by a maintainer.\n"
            f'next: {BELAY} escalate --idea-id {idea.id} --title "Idea #{idea.id} is on track {track}" '
            '--body "Sessions do not run this track; a maintainer runs it."'
        )
    return baseline


def track_label(track: Track, defaulted: bool) -> str:
    return f"Track: {track}{' (default)' if defaulted else ''}"


def print_idea_brief(idea: Idea, track: Track, defaulted: bool) -> None:
    baseline = TRACKS[track]
    print(f"idea #{idea.id}: {idea.name}\n\n{idea.body.strip()}\n")
    label = track_label(track, defaulted)
    print(f"{label}: {baseline.description}; judged on {baseline.metric}, noise {baseline.noise}.\n")
    if (idea.note or "").startswith(MAINTAINER_NOTE):
        print(f"MAINTAINER NOTE (follow it; it settles the checks it addresses):\n{idea.note}\n")
    print(
        "Check 1, house rules: if the idea asks for anything in RUNBOOK.md section 8, run section 6 and stop, unless "
        "the maintainer note says how to meet the rule; then follow the note."
    )
    print(f"Check 2, category (the best match; 'Unsorted' if none): {', '.join(categories())}")
    print("Check 3, settings: the fields below that make the change. If none does, the idea needs code (section 1b).")
    for line in setting_fields():
        print(f"  {line}")
    say_next(
        f'{BELAY} new-idea --idea-id {idea.id} --category "<category>" --variant "<what changes, at most 60 '
        'characters>" --opt-set <name>=<value>   (one --opt-set or --model-set per setting)\n'
        f'   or, if the idea needs code: {BELAY} new-idea --idea-id {idea.id} --category "<category>"'
    )


def write_spec(repo: Path, branch: str, text: str, *, dry_run: bool, replace: bool = False) -> ExperimentSpec:
    """Commit and push the run spec ``text`` on the week's branch; reuse it if the same spec is already there.
    ``replace`` overwrites an existing spec: the next stage adds its runs to it."""
    spec = parse_spec(text)
    path = spec_path(spec.experiment, repo / SPECS)
    if path.exists() and not (replace and path.read_text() != text):
        if path.read_text() != text:
            raise SystemExit(f"refusing: {path.name} already exists with different runs; a maintainer decides")
        print(f"{path.name} is already committed")
        return spec
    print(f"{'would write' if dry_run else 'writing'} {path.relative_to(repo)}:\n{text}")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)  # the first spec on a fresh weekly branch
        path.write_text(text)
    run_git(repo, ["add", str(path.relative_to(repo))], dry_run=dry_run)
    message = f"Belay {spec.experiment}: {'next stage' if replace else 'run spec'}"
    run_git(repo, ["commit", "--quiet", "-m", message], dry_run=dry_run)
    run_git(repo, ["push", "--quiet", "origin", branch], dry_run=dry_run)
    return spec


MAX_NAME = 50
RETEST = "retest on current recipe"
MAINTAINER_NOTE = "Maintainer "  # the app's prefix for a re-queue note
SPECS = Path("experiments/belay/runs")


def prior_trials_hold(entry: dict, note: str | None) -> str | None:
    """The prior results that hold an idea back (escalated to the admin page), unless a maintainer re-queued it with
    a note asking for a retest on the current recipe. None when it may run."""
    trials = entry.get("trials") or []
    if not trials or RETEST in (note or "").lower():
        return None
    lines = [
        f"{t.get('id')}: {t.get('variant')} at {t.get('rung')} ({t.get('date')}), Δ {t.get('delta', 'not recorded')}, "
        f"baseline {t.get('baseline', '?')}"
        for t in trials
    ]
    return "\n".join(lines)


def request_go_ahead(idea: Idea, registry_id: str, track: Track, flags: list[str], *, dry_run: bool) -> None:
    """Escalate a Track whose launches a maintainer approves, with the exact commands at a pinned commit."""
    baseline = TRACKS[track]
    commit = git(repo_root(), "ls-remote", "origin", f"refs/heads/{baseline.code}").split()[0]
    control = " ".join(track_command(track, f"{RUN_PREFIX}{registry_id}-control", (), "uv run python"))
    variant = " ".join(
        track_command(track, f"{RUN_PREFIX}{registry_id}-seed{CONFIG.seed}", tuple(flags), "uv run python")
    )
    seed = ""
    if baseline.seed_run:
        seed_command = track_command(track, f"{RUN_PREFIX}{registry_id}-seedrun", baseline.seed_run, "uv run python")
        seed = f"First, if this program has no autotune cache yet:\n\n```\n{' '.join(seed_command)}\n```\n\n"
    body = seed + (
        f"{registry_id} is on track {track} ({baseline.description}): {baseline.nodes} node(s) of {baseline.hardware} "
        f"on {baseline.cluster} at {baseline.band} priority, so a maintainer approves the launch. From a checkout of "
        f"`{baseline.code}` at `{commit[:10]}`, run the control and the variant on the same rack:\n\n"
        f"```\n{control}\n{variant}\n```\n\nRule: {baseline.metric}, {baseline.rule}, noise {baseline.noise}."
    )
    escalate(f"Go-ahead needed: {registry_id} on {track}", body, idea.id, registry_id, dry_run=dry_run)


def new_idea(
    idea_id: int,
    name: str | None,
    category: str | None,
    variant: str | None,
    settings: list[tuple[str, str]],
    defaults: str | None,
    *,
    dry_run: bool,
) -> None:
    """Register an approved idea, commit its run spec, post the plan and launch it; resumes where it stopped."""
    repo = repo_root()
    branch = require_week_branch(repo, datetime.date.today())
    require_registry_checkout()
    state = page_state()
    idea = state.idea(idea_id)
    if idea.state != "approved" or (idea.note or "").startswith(ESCALATED):
        raise SystemExit(f"refusing: idea #{idea_id} is {idea.state} ({idea.note or 'no note'}).\nnext: stop")
    idea, hidden = approved_idea(idea)
    if hidden:
        print(f"note: the comment had text a reviewer would not see: {', '.join(hidden)}")
    track, defaulted = idea_track(idea)
    runnable(idea, track)
    registry_id = idea.registry_id or (entry_for_intake(idea.url) if idea.url else None)
    if registry_id is not None:
        held = prior_trials_hold(load_entry(registry_id), idea.note)
        if held is not None:
            escalate(
                f"Idea #{idea.id} was already tested as {registry_id}",
                f"{registry_id} has trials:\n{held}\nRe-queue with the note '{RETEST}' to run it again.",
                idea.id,
                registry_id,
                dry_run=dry_run,
            )
            return
    if category is None:
        print_idea_brief(idea, track, defaulted)
        return
    if category not in categories():
        raise SystemExit(f"refusing: category {category!r} is not in the list printed by new-idea --idea-id {idea_id}")
    if variant is not None or settings:
        check_variant(variant or "", settings)
        if not defaults:
            raise SystemExit(
                "refusing: give --defaults: each setting the idea left open and the value chosen, or 'none'"
            )
    require_clean_pushed_head(repo, branch)  # before any write
    if registry_id is None:
        if not name or len(name) > MAX_NAME:
            raise SystemExit(f"refusing: give --name, the idea's short name (at most {MAX_NAME} characters)")
        registry_id, _ = create_entry(idea, name, category, track, dry_run=dry_run)
    else:
        print(f"idea #{idea.id} is {registry_id}")
    report_idea(
        idea.id, {"registry_id": registry_id, "registry_url": entry_url(entry_path(registry_id))}, dry_run=dry_run
    )
    if variant is None or not settings:
        print(f"{registry_id} is registered; its change needs code (section 1b)")
        say_next(f"{BELAY} code-change {registry_id} --start")
        return
    if TRACKS[track].needs_go_ahead:
        request_go_ahead(idea, registry_id, track, [x for setting in settings for x in setting], dry_run=dry_run)
        return
    if spec_path(registry_id, repo / SPECS).exists():
        text = spec_path(registry_id, repo / SPECS).read_text()  # resuming: the committed spec decides
    else:
        text = new_spec_text(registry_id, track, variant, settings, run_suffix(registry_id, iris_jobs()))
    spec = write_spec(repo, branch, text, dry_run=dry_run)
    commit = require_clean_pushed_head(repo, branch)
    check_launch(spec, iris_jobs())
    print(f"checks passed: {branch} at {commit[:10]}, track {spec.track}, {len(spec.runs)} runs fit the cap")
    entry = load_entry(registry_id) if not dry_run or idea.registry_id else {"name": name or registry_id}
    defaults_text = defaults + (f" (hidden text removed from the comment: {', '.join(hidden)})" if hidden else "")
    if defaulted:
        defaults_text = f"track speedrun (no Track: line); {defaults_text}"
    url = comment_url_of(entry)
    number = len([t for t in entry.get("trials") or [] if t.get("comment_url") == url]) + 1 if url else 1
    data_runs = stage_runs(spec, Stage.DATA)
    variant_key = next(r.name for r in data_runs if not r.control)
    rows = plan_rows(spec, data_runs, commit)
    section = stage_section(
        number, comparison_text(spec.track, Stage.DATA), rows, reproduce(registry_id, commit, variant_key)
    )
    if url is None:
        body = render(
            Comment(
                title(registry_id, entry["name"], spec.track, "plan"),
                {number: section},
                post_footer(idea, registry_id, spec.track, defaults_text, commit),
            )
        )
    else:  # a retest: one more section in the idea's comment
        body = updated(read_comment(url), title(registry_id, entry["name"], spec.track, "plan"), number, section)
    url = publish_comment(url, body, github_posting=state.github_posting, dry_run=dry_run)
    plan = {"summary": entry["name"], "commit": commit, "branch": branch, "track": spec.track.value, "comment": url}
    report_idea(idea.id, {"state": "planned", "plan": plan}, dry_run=dry_run)
    if dry_run:
        launch_stage(repo, branch, spec, data_runs, None, dry_run=True)
        return
    trial_id, path = add_trial(
        registry_id, spec.track, "equal_steps", variant, commit, settings_of(variant_settings(spec)), url or ""
    )
    commit_and_push([path], f"Belay registry: {registry_id} {trial_id} planned", dry_run=False)
    launch_stage(repo, branch, spec, data_runs, trial_id, dry_run=False)
    body = updated(read_comment(url), title(registry_id, entry["name"], spec.track, "running"), number, section)
    publish_comment(url, body, github_posting=state.github_posting, dry_run=False)
    running = title(registry_id, entry["name"], spec.track, "running")
    record_activity(Post("plan", running, body, registry_id, idea.id), url, dry_run=False)
    say_next("stop (the watcher starts a new session when the runs finish)")


def post_footer(idea: Idea, registry_id: str, track: Track, defaults: str, commit: str) -> list[str]:
    source = f"[#9514 comment]({idea.url}) by @{idea.author}" if idea.url else f"proposal by {idea.author}"
    approved = f" · approved by @{idea.decided_by}" if idea.decided_by else ""
    return footer(defaults, track, f"{source}{approved}", commit, entry_url(entry_path(registry_id)))


def run_steps(spec: ExperimentSpec, run: RunSpec) -> int:
    """The run's training steps: its ``--num-steps`` flag, else the Track's baseline steps."""
    flags = list(run.flags)
    if "--num-steps" in flags:
        return int(flags[flags.index("--num-steps") + 1])
    steps = TRACKS[spec.track].baseline_steps
    assert steps is not None, spec.track
    return steps


def plan_rows(spec: ExperimentSpec, runs: list[RunSpec], commit: str) -> list[Row]:
    """Table rows for planned ``runs``: steps, hardware and, where the Track fixes it, the expected time."""
    rows = []
    for run in sorted(runs, key=lambda r: not r.control):
        name = run_name(spec.experiment, run.name)
        label = control_label(spec.track, commit) if run.control else run.variant
        steps = run_steps(spec, run)
        seconds = TRACKS[spec.track].train_seconds
        timed = run.control or run.stage in (Stage.TIME, Stage.CONFIRM)
        expected = seconds / steps if seconds and timed else None  # equal time: every timed arm trains `seconds`
        rows.append(Row(name, label, run.seed, hardware(spec.track), steps, expected))
    return rows


def comparison_text(track: Track, stage: Stage, tokens: float | None = None) -> str:
    baseline = TRACKS[track]
    if stage in (Stage.TIME, Stage.CONFIRM):
        seed = " at seed 1 (confirmation)" if stage is Stage.CONFIRM else ""
        return f"equal time{seed}, {baseline.train_seconds:g} s of training per arm"
    data = f"{baseline.baseline_steps:,} steps" + (f" / {tokens / 1e9:.2f}B tokens" if tokens else "")
    wall = "" if baseline.comparison is Comparison.EQUAL_DATA_THEN_TIME else "; wall time shown"
    return f"equal data, {data} per arm{wall}"


def launch_stage(
    repo: Path, branch: str, spec: ExperimentSpec, runs: list[RunSpec], trial_id: str | None, *, dry_run: bool
) -> dict[str, list[str]]:
    """The one launch path (``new-idea``, ``run-finished`` and ``launch``): submit ``runs`` (controls first) from
    the spec's clean, pushed commit, add variant runs to ``trial_id``, then record each submitted command under the
    spec's ``launched`` and commit that. Returns the commands."""
    require_clean_pushed_head(repo, branch)
    check_launch(ExperimentSpec(spec.experiment, spec.track, {r.name: r for r in runs}), iris_jobs())
    commands = {}
    for run in sorted(runs, key=lambda r: not r.control):
        run_id = run_name(spec.experiment, run.name)
        commands[run.name] = launch_command(spec, run)
        print(f"{'would launch' if dry_run else 'launching'} {run_id}:\n  {shlex.join(commands[run.name])}")
        if dry_run:
            continue
        subprocess.run(commands[run.name], cwd=repo, check=True)
        if trial_id is not None and not run.control:
            path = add_trial_run(spec.experiment, trial_id, run_id)
            commit_and_push([path], f"Belay registry: {spec.experiment} {run.name} launched", dry_run=False)
    path = spec_path(spec.experiment, repo / SPECS)
    if not dry_run:
        path.write_text(with_launched(path.read_text(), commands))
        run_git(repo, ["add", str(path.relative_to(repo))], dry_run=False)
        run_git(repo, ["commit", "--quiet", "-m", f"Belay {spec.experiment}: record launched commands"], dry_run=False)
        run_git(repo, ["push", "--quiet", "origin", branch], dry_run=False)
    return commands


def launch(registry_id: str, run: str, *, print_only: bool, dry_run: bool) -> None:
    """Show (``--print``) or submit one run of the committed spec, through the same path as every other launch."""
    repo = repo_root()
    spec = load_spec(registry_id, repo / SPECS)
    if run not in spec.runs:
        raise SystemExit(f"refusing: {registry_id} has no run {run!r}; runs: {', '.join(spec.runs)}")
    if print_only:
        print(shlex.join(launch_command(spec, spec.runs[run])))
        return
    branch = require_week_branch(repo, datetime.date.today())
    variant = spec.runs[run].variant
    trials = [t for t in load_entry(registry_id).get("trials") or [] if t.get("variant") == variant]
    launch_stage(repo, branch, spec, [spec.runs[run]], trials[-1]["id"] if trials else None, dry_run=dry_run)
    say_next("stop (the watcher starts a new session when the run finishes)")


def reproduce(registry_id: str, commit: str, run: str) -> str:
    spec = f"https://github.com/{CONFIG.repository}/blob/{commit}/experiments/belay/runs/{registry_id}.yaml"
    command = f"git checkout {commit[:10]} && uv run python -m experiments.belay launch {registry_id} --run {run}"
    return f"Reproduce: [spec]({spec}) · `{command}`"


# ---------- 1b. code change ----------

CODE_RULES = """Rules for the change (code-change --finish checks every one; passing them is the only gate):
  1. Edit only experiments/grug/fast_track/ and tests/grug/test_fast_track_*.py.
  2. Add a new field to GrugModelConfig (model.py) or GrugMoeMuonHConfig (optimizer.py) whose default is off
     (False, None, 0 or 0.0), and put every behavior change behind it. Change no existing default.
  3. With the flag off, the tiny model's loss and every gradient must equal the commit before the change, bit for bit.
  4. With the flag on (--model-set), changing future tokens must not move past hidden states (no leakage).
  5. Add tests/grug/test_fast_track_<flag>.py that turns the flag on; it must pass on CPU.
  6. ./infra/pre-commit.py must pass on the changed files."""


def code_checkout(registry_id: str) -> tuple[Path, str]:
    """The checkout and branch an idea's code goes on: this week's branch, or its Track's own branch."""
    track = parse_track(load_entry(registry_id).get("track") or DEFAULT_TRACK.value)
    repo = repo_root()
    code = TRACKS[track].code
    if code == WEEKLY_BRANCH:
        return repo, require_week_branch(repo, datetime.date.today())
    return named_checkout(repo, code), code


def code_change_start(registry_id: str) -> None:
    repo, branch = code_checkout(registry_id)
    require_clean_pushed_head(repo, branch)
    print(f"{registry_id}: edit {repo} on {branch} (clean, at origin/{branch}).\n{CODE_RULES}")
    say_next(f"{BELAY} code-change {registry_id} --finish --model-set <flag>=<on value>   (or --opt-set)")


def code_change_finish(registry_id: str, settings: list[tuple[str, str]], *, dry_run: bool) -> None:
    """Run every check on the uncommitted change, then commit and push it to its code branch."""
    repo, branch = code_checkout(registry_id)
    paths = changed_paths(repo)
    if not paths:
        raise SystemExit(f"refusing: nothing changed.\nnext: {BELAY} code-change {registry_id} --start")
    revert = f"next: git -C {repo} checkout -- . && git -C {repo} clean -fd experiments/grug/fast_track tests/grug"
    refuse_outside_allowed(paths)
    print(f"1. paths ok: {paths}")
    python = checkout_python(repo)  # the checkout's own environment, also for its pre-change copy
    with temporary_worktree(repo, "HEAD") as before_dir:
        before = probe(before_dir, [], python)
    after = probe(repo, [], python)
    flags = new_flags(before.defaults, after.defaults)
    print(f"2. new default-off flags: {flags}")
    refuse_flag_off_change(before, after)
    print("3. flag off: tiny-model loss and gradients bit-identical to HEAD")
    if not settings:
        raise SystemExit(f"refusing: give the flag's on value, e.g. --model-set <flag>=true.\n{revert}")
    names = {f.split(".")[1] for f in flags}
    unknown = [s for _, s in settings if s.split("=")[0] not in names]
    if unknown:
        raise SystemExit(f"refusing: {unknown} set fields that are not this change's new flags {sorted(names)}")
    model_settings = [s for option, s in settings if option == "--model-set"]
    if model_settings:
        refuse_leak(probe(repo, model_settings, python))
        print(f"4. flag on ({', '.join(model_settings)}): no future leakage")
    else:
        print("4. optimizer flag only: the leakage check does not apply")
    run_tests(repo, paths)
    print("5. the change's tests pass on CPU")
    run_precommit(repo, paths)
    print("6. pre-commit passes")
    message = f"{registry_id}: {', '.join(sorted(names))} (default off)"
    run_git(repo, ["add", "--", *paths], dry_run=dry_run)
    run_git(repo, ["commit", "--quiet", "-m", message], dry_run=dry_run)
    run_git(repo, ["push", "--quiet", "origin", branch], dry_run=dry_run)
    idea = page_state().idea_for(registry_id)
    idea_id = "<N>" if idea is None else idea.id
    flags_text = " ".join(f"{option} {s}" for option, s in settings)
    say_next(f'{BELAY} new-idea --idea-id {idea_id} --category "<category>" --variant "<what changes>" {flags_text}')


def report_flags(repo: Path) -> None:
    """List the flag commits since the base branch, for a maintainer to delete the ones that never won."""
    commits = flag_commits(repo, f"origin/{CONFIG.base_branch}")
    print(f"flags added since {CONFIG.base_branch} ({len(commits)}); a maintainer deletes those that never won:")
    for line in commits:
        print(f"  {line}")


# ---------- 2. run finished ----------


def measured_row(spec: ExperimentSpec, run: RunSpec, m: Measure, control: Measure | None, commit: str) -> Row:
    label = control_label(spec.track, commit) if run.control else run.variant
    gpus = hardware(spec.track)
    if control is None:
        return Row(m.name, label, run.seed, gpus, m.steps, m.s_step, None, m.loss, None, m.bpb)
    time_change = (m.steps * m.s_step) / (control.steps * control.s_step) - 1
    delta, _ = compared(control, m, TRACKS[spec.track])
    return Row(m.name, label, run.seed, gpus, m.steps, m.s_step, time_change, m.loss, delta, m.bpb)


def stage_pairs(spec: ExperimentSpec, stage: Stage) -> list[tuple[RunSpec, RunSpec]]:
    """(control, variant) pairs a finished stage compares: equal time reuses the stage-1 control."""
    variants = [r for r in stage_runs(spec, stage) if not r.control]
    controls = stage_runs(spec, Stage.CONFIRM if stage is Stage.CONFIRM else Stage.DATA)
    return [(next(c for c in controls if c.control and c.seed == v.seed), v) for v in variants]


def run_finished(registry_id: str, *, dry_run: bool) -> None:
    """Record a finished stage, update the idea's #9451 comment in place, and start the next stage the Track's
    comparison calls for (equal data, then equal time and a seed-1 confirmation for a speedrun win)."""
    repo = repo_root()
    branch = require_week_branch(repo, datetime.date.today())
    spec = load_spec(registry_id)
    require_registry_checkout()
    state = page_state()
    names = {run_name(registry_id, r) for r in spec.runs}
    runs = [r for r in state.runs_of(registry_id) if r.name in names]
    live = [r.name for r in runs if not r.terminal]
    if live:
        print(f"still running: {live}; the last one to finish starts a new session")
        say_next("stop")
        return
    failed = [r for r in runs if r.state != "succeeded"]
    if failed:
        print(f"{failed[0].name} is {failed[0].state}")
        say_next(f"{BELAY} run-failed {registry_id} --run {failed[0].run}   (section 3)")
        return
    baseline = TRACKS[spec.track]
    entry = load_entry(registry_id)
    url = comment_url_of(entry)
    idea = state.idea_for(registry_id)
    stage = latest_stage(spec)
    trials = [t for t in entry.get("trials") or [] if t.get("comment_url") == url] if url else entry["trials"]
    if stage is Stage.PROBE:
        probe = next(r for r in stage_runs(spec, Stage.PROBE))
        s_step = measure(run_name(registry_id, probe.name), baseline.metric, PROBE_SKIP).s_step
        steps = round(baseline.train_seconds / s_step)
        print(f"timing probe: {s_step:.4f} s/step, so {steps:,} steps make {baseline.train_seconds:g} s")
        note = f"Timing probe: {s_step:.4f} s/step, so the variant trains {steps:,} steps."
        advance(Next(repo, branch, spec, registry_id, Stage.TIME, steps, url, entry, state, note), dry_run=dry_run)
        return
    trial = next(t for t in trials if run_name(registry_id, stage_pairs(spec, stage)[0][1].name) in t["wandb"])
    number = trials.index(trial) + 1
    rows: dict[str, Row] = {}
    results: list[RunResult] = []
    verdicts: list[Verdict] = []
    for control_run, variant_run in stage_pairs(spec, stage):
        control = measure(run_name(registry_id, control_run.name), baseline.metric)
        variant = measure(run_name(registry_id, variant_run.name), baseline.metric)
        delta, v = compared(control, variant, baseline)
        verdicts.append(v)
        rows[control.name] = measured_row(spec, control_run, control, None, trial["commit"])
        rows[variant.name] = measured_row(spec, variant_run, variant, control, trial["commit"])
        results.append(as_result(control_run, control, None, None))
        results.append(as_result(variant_run, variant, delta, v))
    overall = overall_verdict(verdicts)
    path, text = entry_with_results(registry_id, results)
    print(f"{'would write' if dry_run else 'writing'} {path.name}:\n{text}")
    if not dry_run:
        path.write_text(text)
    report = write_analysis(
        registry_id, spec.track, results, overall, idea, trial["id"], trial["commit"], dry_run=dry_run
    )
    commit_and_push([path, *report], f"Belay registry: {registry_id} {trial['id']} recorded", dry_run=dry_run)
    pairs = stage_pairs(spec, stage)
    stage_rows = [rows[run_name(registry_id, r.name)] for pair in pairs for r in pair]
    tokens = measure(stage_rows[0].name, baseline.metric).tokens if stage is Stage.DATA else None
    variant_key = pairs[0][1].name
    note = reproduce(registry_id, trial["commit"], variant_key)
    section = stage_section(number, comparison_text(spec.track, stage, tokens), stage_rows, note)
    staged = baseline.comparison is Comparison.EQUAL_DATA_THEN_TIME
    where = {Stage.DATA: " at equal data", Stage.TIME: " at equal time", Stage.CONFIRM: " at equal time"}[stage]
    status = f"{overall}{where if staged else ''}"
    next_stage = {Stage.DATA: Stage.PROBE, Stage.TIME: Stage.CONFIRM}.get(stage) if staged else None
    if stage is Stage.CONFIRM:
        status = f"win at equal time, {'confirmed' if overall is Verdict.WIN else 'not confirmed'} at seed 1"
    elif next_stage is not None and overall is Verdict.WIN:
        status += " · " + ("equal-time run pending" if next_stage is Stage.PROBE else "seed-1 confirm pending")
    analysis = (f"{registry_id} {trial['id']}", report_url(registry_id, trial["id"]))
    new_title = title(registry_id, entry["name"], spec.track, status)
    body = updated(read_comment(url), new_title, number, section, analysis)
    publish_comment(url, body, github_posting=state.github_posting, dry_run=dry_run)
    if idea is not None:
        result = {"rows": [asdict(r) for r in results], "verdict": overall, "kind": PAGE_RESULT_KIND[overall]}
        report_idea(idea.id, {"state": "recorded", "result": {**result, "summary": status}}, dry_run=dry_run)
    idea_id = None if idea is None else idea.id
    record_activity(Post("result", new_title, body, registry_id, idea_id), url, dry_run=dry_run)
    if next_stage is not None and overall is Verdict.WIN:
        steps = None if next_stage is Stage.PROBE else run_steps(spec, pairs[0][1])
        advance(Next(repo, branch, spec, registry_id, next_stage, steps, url, entry, state), dry_run=dry_run)
        return
    if overall is Verdict.WIN:
        adoption = f"See {url}. A maintainer decides on adoption into {CONFIG.recipe_name}."
        escalate(f"{registry_id}: {status}", adoption, idea_id, registry_id, dry_run=dry_run)
        return
    say_next("stop")


@dataclass(frozen=True)
class Next:
    """The stage ``run-finished`` starts next, and where it is recorded."""

    repo: Path
    branch: str
    spec: ExperimentSpec
    registry_id: str
    stage: Stage
    steps: int | None
    url: str | None
    entry: dict
    state: PageState
    note: str = ""


STATUS_RUNNING = {
    Stage.PROBE: "win at equal data · equal-time timing probe running",
    Stage.TIME: "win at equal data · equal-time run running",
    Stage.CONFIRM: "win at equal time · seed-1 confirm running",
}


def as_result(run: RunSpec, m: Measure, delta: float | None, v: Verdict | None) -> RunResult:
    return RunResult(m.name, run.variant, run.seed, run.control, m.steps, m.loss, delta, v, m.bpb)


def advance(nxt: Next, *, dry_run: bool) -> None:
    """Start the next stage: add its runs to the spec, commit and push it, launch them, and show them in the
    comment."""
    repo, branch, spec, registry_id, stage = nxt.repo, nxt.branch, nxt.spec, nxt.registry_id, nxt.stage
    jobs = iris_jobs()
    runs = next_stage_runs(spec, stage, run_suffix(registry_id, jobs), nxt.steps)
    text = spec_with_runs(spec_path(registry_id, repo / SPECS).read_text(), runs)
    new_spec = write_spec(repo, branch, text, dry_run=dry_run, replace=True)
    commit = "dry-run" if dry_run else require_clean_pushed_head(repo, branch)
    started = [r for r in new_spec.runs.values() if r.name in runs]
    check_launch(ExperimentSpec(new_spec.experiment, new_spec.track, {r.name: r for r in started}), jobs)
    number = len([t for t in nxt.entry.get("trials") or [] if t.get("comment_url") == nxt.url]) + 1
    if stage is Stage.PROBE:
        pending = "Pending: a timing probe measures the variant's s/step."
        section = stage_section(number, comparison_text(spec.track, Stage.TIME), [], pending)
    else:
        pair = stage_pairs(new_spec, stage)[0]
        planned = plan_rows(new_spec, [*pair], commit)
        notes = "\n\n".join(n for n in (nxt.note, reproduce(registry_id, commit, pair[1].name)) if n)
        section = stage_section(number, comparison_text(spec.track, stage), planned, notes)
    new_title = title(registry_id, nxt.entry["name"], spec.track, STATUS_RUNNING[stage])
    body = updated(read_comment(nxt.url), new_title, number, section)
    publish_comment(nxt.url, body, github_posting=nxt.state.github_posting, dry_run=dry_run)
    if dry_run:
        launch_stage(repo, branch, new_spec, started, None, dry_run=True)
        return
    trial_id = None
    if stage is not Stage.PROBE:
        variant = next(r for r in started if not r.control)
        settings = settings_of(variant.flags)
        trial_id, path = add_trial(
            registry_id, spec.track, "fixed_time", variant.variant, commit, settings, nxt.url or ""
        )
        commit_and_push([path], f"Belay registry: {registry_id} {trial_id} planned ({stage})", dry_run=False)
    launch_stage(repo, branch, new_spec, started, trial_id, dry_run=False)
    say_next("stop (the watcher starts a new session when the runs finish)")


def analyze(registry_id: str, *, dry_run: bool) -> None:
    """Rewrite the analysis report of every finished stage of the experiment, and commit and push them."""
    spec = load_spec(registry_id)
    require_registry_checkout()
    baseline = TRACKS[spec.track]
    entry = load_entry(registry_id)
    idea = page_state().idea_for(registry_id)
    reports: list[Path] = []
    for stage in (Stage.DATA, Stage.TIME, Stage.CONFIRM):
        for control_run, variant_run in stage_pairs(spec, stage):
            variant_name = run_name(registry_id, variant_run.name)
            trial = next((t for t in entry["trials"] if variant_name in (t.get("wandb") or [])), None)
            if trial is None:
                continue
            control = measure(run_name(registry_id, control_run.name), baseline.metric)
            variant = measure(variant_name, baseline.metric)
            delta, v = compared(control, variant, baseline)
            results = [as_result(control_run, control, None, None), as_result(variant_run, variant, delta, v)]
            reports += write_analysis(
                registry_id, spec.track, results, v, idea, trial["id"], trial["commit"], dry_run=dry_run
            )
    commit_and_push(reports, f"Belay registry: {registry_id} analysis", dry_run=dry_run)
    say_next("stop")


def write_analysis(
    registry_id: str,
    track: Track,
    results: list[RunResult],
    overall: Verdict,
    idea: Idea | None,
    trial: str,
    commit: str,
    *,
    dry_run: bool,
) -> list[Path]:
    """The trial's analysis report and plots in the registry checkout, for the results commit."""
    out_dir = report_dir(CONFIG.registry_checkout, registry_id)
    if dry_run:
        print(f"would write the analysis {out_dir / trial}.md and its plots")
        return []
    control = next(r for r in results if r.control)
    variant = next(r for r in results if not r.control)
    header = Header(
        registry_id=registry_id,
        trial=trial,
        title=load_entry(registry_id)["name"],
        track=track.value,
        verdict=str(overall),
        commit=commit,
        approver=None if idea is None else idea.decided_by,
        idea_url=None if idea is None else idea.url,
        rule=rule_line(track),
    )
    final = {r.name: (r.loss, r.delta, r.bpb) for r in sorted(results, key=lambda r: not r.control)}
    report = write_report(out_dir, header, fetch_series(control.name), fetch_series(variant.name), final)
    print(f"wrote the analysis {report} ({report_url(registry_id, trial)})")
    return [report, report.with_name(f"{trial}_loss.png"), report.with_name(f"{trial}_delta.png")]


# ---------- 3. run failed ----------


def run_failed(registry_id: str, run: str, *, dry_run: bool) -> None:
    """Escalate a failed run with its state and log command; never relaunch it."""
    state = page_state()
    name = run_name(registry_id, run)
    rows = [r for r in state.runs if r.name == name]
    if not rows:
        raise SystemExit(f"refusing: no run {name} on the Belay page")
    row = rows[0]
    if not row.terminal or row.state == "succeeded":
        raise SystemExit(f"refusing: {name} is {row.state}, not failed.\nnext: {BELAY} run-finished {registry_id}")
    body = (
        f"`{name}` ended `{row.state}` at step {row.step} (job `{row.job_id}`). Nothing was relaunched. "
        f"Logs: `uv run iris --cluster marin job logs {row.job_id}`."
    )
    idea = state.idea_for(registry_id)
    escalate(f"{name} {row.state}", body, None if idea is None else idea.id, registry_id, dry_run=dry_run)


# ---------- 4. weekly rollover ----------


def weekly_rollover(today: datetime.date, *, fresh: bool, dry_run: bool) -> None:
    repo = repo_root()
    target = CONFIG.week_branch(today)
    if not cut_week_branch(repo, today, fresh=fresh, dry_run=dry_run):
        print(f"merging {CONFIG.base_branch} into {target} conflicts; aborted, nothing pushed")
        say_next(
            f'{BELAY} escalate --title "{target} conflicts with {CONFIG.base_branch}" '
            '--body "A maintainer resolves the merge, or runs weekly-rollover --fresh."'
        )
        return
    print(f"on {target}, which contains origin/{CONFIG.base_branch}")
    report_flags(repo)
    say_next(f"{BELAY} start --events <the same events file>")


# ---------- 5. recipe change ----------


def recipe_change(source: str, today: datetime.date, *, dry_run: bool) -> None:
    """Maintainers only: merge ``source`` into the recipe branch and tag the new tip ``belay-ckpt-<date>``."""
    repo = repo_root()
    missing = [t.launch[0] for t in TRACKS.values() if t.launch and importlib.util.find_spec(t.launch[0]) is None]
    if missing:
        raise SystemExit(f"refusing: launch modules {missing} (config.TRACKS) do not exist")
    git(repo, "fetch", "--quiet", "origin", CONFIG.recipe_branch, source)
    tag = f"{CHECKPOINT_TAG_PREFIX}{today.isoformat()}"
    with temporary_worktree(repo, f"origin/{CONFIG.recipe_branch}") as recipe:
        print(f"merging origin/{source} into {CONFIG.recipe_branch}")
        run_git(recipe, ["merge", "--ff", "--no-edit", f"origin/{source}"], dry_run=dry_run)
        name = f"Belay Ckpt {today:%b %d %Y}"
        run_git(recipe, ["tag", "--annotate", tag, "--message", name], dry_run=dry_run)
        run_git(recipe, ["push", "--quiet", "origin", f"HEAD:{CONFIG.recipe_branch}", tag], dry_run=dry_run)
    report_flags(repo)
    say_next(
        f'set recipe_tag="{tag}" in experiments/belay/config.py and commit it; at the next weekly rollover, '
        f"{BELAY} weekly-rollover --fresh"
    )


# ---------- 6. escalate ----------


def escalate(headline: str, body: str, idea_id: int | None, registry_id: str | None, *, dry_run: bool) -> None:
    """Record a question for a maintainer on the admin page (never #9451), with this session's transcript, and mark
    the idea so no session takes it up again until a maintainer re-queues or declines it."""
    post = Post("escalation", headline, body, registry_id=registry_id, idea_id=idea_id)
    post_activity(post, github_posting=False, dry_run=dry_run)
    if idea_id is not None:
        report_idea(idea_id, {"note": f"{ESCALATED} {headline}"}, dry_run=dry_run)
    say_next("stop")


# ---------- 9. tracks ----------


def track_rows() -> list[dict[str, object]]:
    """Every Track as printed by ``tracks``: the one definition of each launch, from ``config.TRACKS``."""
    rows = []
    for track, baseline in TRACKS.items():
        code = f"{CONFIG.week_branch_prefix}<monday>" if baseline.code == WEEKLY_BRANCH else baseline.code
        command = seed_run = None
        if baseline.launch is not None:
            command = " ".join(track_command(track, f"{RUN_PREFIX}<X####>-<run>", ("<flags>",), "uv run python"))
        if baseline.seed_run:
            seed_run = " ".join(track_command(track, f"{RUN_PREFIX}<X####>-seedrun", baseline.seed_run, "uv run python"))
        rows.append(
            {
                "track": track.value,
                "baseline": baseline.description,
                "code": code,
                "command": command,
                "seed_run": seed_run,
                "hardware": (
                    f"{baseline.nodes} node(s) of {baseline.hardware} on {baseline.cluster}" if baseline.nodes else ""
                ),
                "priority": baseline.band,
                "control": CONTROL_RUN,
                "metric": baseline.metric,
                "rule": f"{baseline.rule}, noise {baseline.noise}",
                "go_ahead": baseline.needs_go_ahead,
            }
        )
    return rows


def tracks(as_json: bool) -> None:
    rows = track_rows()
    if as_json:
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        return
    for row in rows:
        print(f"{row['track']}: {row['baseline']}")
        for key in ("code", "command", "hardware", "priority", "control", "metric", "rule"):
            print(f"  {key}: {row[key] or '(none: a maintainer runs it)'}")
        if row["seed_run"]:
            print(f"  seed run (once per program, before the first timed run): {row['seed_run']}")
        if row["go_ahead"]:
            print("  go-ahead: a maintainer approves each launch; sessions escalate with the exact command")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true", help="check and print; write, push, launch and post nothing")
        return p

    p = command("start")
    p.add_argument("--events", type=Path, required=True, help="the session's events file, named in its prompt")
    p.add_argument("--today", type=datetime.date.fromisoformat, help="pretend date (with --dry-run only)")
    p = command("new-idea")
    p.add_argument("--idea-id", type=int, required=True)
    p.add_argument("--name", help=f"short name for a new registry entry, at most {MAX_NAME} characters")
    p.add_argument("--defaults", help="each setting the idea left open and the value chosen, or 'none'")
    p.add_argument("--category")
    p.add_argument("--variant")
    p.add_argument("--opt-set", action="append", default=[])
    p.add_argument("--model-set", action="append", default=[])
    p = command("launch")
    p.add_argument("registry_id")
    p.add_argument("--run", required=True, help="the run's key in experiments/belay/runs/X####.yaml")
    p.add_argument("--print", action="store_true", dest="print_only", help="print the full command, submit nothing")
    p = command("code-change")
    p.add_argument("registry_id")
    step = p.add_mutually_exclusive_group(required=True)
    step.add_argument("--start", action="store_true")
    step.add_argument("--finish", action="store_true")
    p.add_argument("--opt-set", action="append", default=[], help="with --finish: the new flag's on value")
    p.add_argument("--model-set", action="append", default=[], help="with --finish: the new flag's on value")
    command("run-finished").add_argument("registry_id")
    command("analyze").add_argument("registry_id")
    p = command("run-failed")
    p.add_argument("registry_id")
    p.add_argument("--run", required=True)
    p = command("weekly-rollover")
    p.add_argument("--fresh", action="store_true", help="cut from the base branch instead of last week's branch")
    p.add_argument("--today", type=datetime.date.fromisoformat, help="pretend date (with --dry-run only)")
    p = command("recipe-change")
    p.add_argument("--from", dest="source", required=True, help="the branch whose tip becomes the new recipe")
    command("tracks").add_argument("--json", action="store_true", help="the table as JSON (web/tracks.json)")
    p = command("escalate")
    p.add_argument("--title", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--idea-id", type=int)
    p.add_argument("--registry-id")
    args = parser.parse_args(argv)

    today = datetime.date.today()
    if getattr(args, "today", None) is not None:
        if not args.dry_run:
            raise SystemExit("refusing: --today is for dry runs only")
        today = args.today

    if args.command == "start":
        start(args.events, today)
    elif args.command == "new-idea":
        settings = [("--opt-set", s) for s in args.opt_set] + [("--model-set", s) for s in args.model_set]
        new_idea(args.idea_id, args.name, args.category, args.variant, settings, args.defaults, dry_run=args.dry_run)
    elif args.command == "launch":
        launch(args.registry_id, args.run, print_only=args.print_only, dry_run=args.dry_run)
    elif args.command == "code-change" and args.start:
        code_change_start(args.registry_id)
    elif args.command == "code-change":
        settings = [("--opt-set", s) for s in args.opt_set] + [("--model-set", s) for s in args.model_set]
        code_change_finish(args.registry_id, settings, dry_run=args.dry_run)
    elif args.command == "run-finished":
        run_finished(args.registry_id, dry_run=args.dry_run)
    elif args.command == "analyze":
        analyze(args.registry_id, dry_run=args.dry_run)
    elif args.command == "run-failed":
        run_failed(args.registry_id, args.run, dry_run=args.dry_run)
    elif args.command == "weekly-rollover":
        weekly_rollover(today, fresh=args.fresh, dry_run=args.dry_run)
    elif args.command == "recipe-change":
        recipe_change(args.source, today, dry_run=args.dry_run)
    elif args.command == "tracks":
        tracks(args.json)
    else:
        escalate(args.title, args.body, args.idea_id, args.registry_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])
