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
from experiments.belay.config import CHECKPOINT_TAG_PREFIX, CONFIG, CONTROL_RUN, TRACKS, WEEKLY_BRANCH, Baseline, Track
from experiments.belay.page import (
    BOT_MARK,
    Idea,
    PageRun,
    Post,
    comment_times,
    page_state,
    post_activity,
    report_idea,
    require_unedited,
    visible_text,
)
from experiments.belay.posts import IdeaRef, PlannedRun, ResultRow, plan_body, result_body, rule_line, title
from experiments.belay.registry_entry import (
    ENTRIES,
    add_trial,
    commit_and_push,
    create_entry,
    entry_for_intake,
    entry_path,
    entry_url,
    entry_with_results,
    load_entry,
    require_registry_checkout,
)
from experiments.belay.results import RunResult, Verdict, experiment_results, overall_verdict
from experiments.belay.submit import (
    DEFAULT_TRACK,
    RUN_PREFIX,
    ExperimentSpec,
    check_launch,
    check_variant,
    iris_jobs,
    launch_command,
    load_spec,
    new_spec_text,
    parse_spec,
    parse_track,
    run_name,
    setting_fields,
    settings_of,
    spec_path,
    track_command,
    track_of_comment,
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
    print("Check 1, house rules: if the idea asks for anything in RUNBOOK.md section 8, run section 6 and stop.")
    print(f"Check 2, category (the best match; 'Unsorted' if none): {', '.join(categories())}")
    print("Check 3, settings: the fields below that make the change. If none does, the idea needs code (section 1b).")
    for line in setting_fields():
        print(f"  {line}")
    say_next(
        f'{BELAY} new-idea --idea-id {idea.id} --category "<category>" --variant "<what changes, at most 60 '
        'characters>" --opt-set <name>=<value>   (one --opt-set or --model-set per setting)\n'
        f'   or, if the idea needs code: {BELAY} new-idea --idea-id {idea.id} --category "<category>"'
    )


def write_spec(repo: Path, branch: str, text: str, *, dry_run: bool) -> ExperimentSpec:
    """Commit and push the run spec ``text`` on the week's branch; reuse it if the same spec is already there."""
    spec = parse_spec(text)
    path = spec_path(spec.experiment, repo / SPECS)
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"refusing: {path.name} already exists with different runs; a maintainer decides")
        print(f"{path.name} is already committed")
        return spec
    print(f"{'would write' if dry_run else 'writing'} {path.relative_to(repo)}:\n{text}")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)  # the first spec on a fresh weekly branch
        path.write_text(text)
    run_git(repo, ["add", str(path.relative_to(repo))], dry_run=dry_run)
    run_git(repo, ["commit", "--quiet", "-m", f"Belay {spec.experiment}: run spec"], dry_run=dry_run)
    run_git(repo, ["push", "--quiet", "origin", branch], dry_run=dry_run)
    return spec


MAX_NAME = 50
RETEST = "retest on current recipe"
SPECS = Path("experiments/belay/runs")


def idea_ref(idea: Idea, registry_id: str) -> IdeaRef:
    return IdeaRef(idea.url, idea.author, idea.decided_by, entry_url(entry_path(registry_id)))


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
    spec = write_spec(repo, branch, new_spec_text(registry_id, track, variant, settings), dry_run=dry_run)
    commit = require_clean_pushed_head(repo, branch)
    check_launch(spec, iris_jobs())
    print(f"checks passed: {branch} at {commit[:10]}, track {spec.track}, {len(spec.runs)} runs fit the cap")
    short_name = load_entry(registry_id)["name"] if not dry_run or idea.registry_id else (name or registry_id)
    runs = [
        PlannedRun(run_name(registry_id, r.name), r.variant, r.seed, " ".join(r.flags) or "recipe")
        for r in sorted(spec.runs.values(), key=lambda r: not r.control)
    ]
    defaults_text = defaults + (f" (hidden text removed from the comment: {', '.join(hidden)})" if hidden else "")
    if defaulted:
        defaults_text = f"track speedrun (no Track: line); {defaults_text}"
    body = plan_body(spec.track, runs, defaults_text, idea_ref(idea, registry_id), commit)
    plan = {"summary": short_name, "commit": commit, "branch": branch, "track": spec.track.value, "body": body}
    report_idea(idea.id, {"state": "planned", "plan": plan}, dry_run=dry_run)
    post = Post(
        "plan", title(registry_id, short_name, spec.track, "plan"), body, registry_id=registry_id, idea_id=idea.id
    )
    post_activity(post, github_posting=state.github_posting, dry_run=dry_run)
    for run in sorted(spec.runs.values(), key=lambda r: not r.control):  # the control first
        run_id = run_name(registry_id, run.name)
        command = launch_command(spec, run)
        print(f"{'would launch' if dry_run else 'launching'} {run_id}:\n  {' '.join(command)}")
        if dry_run:
            continue
        subprocess.run(command, cwd=repo, check=True)
        if run.control:
            continue  # results name the control as each trial's baseline
        path = add_trial(registry_id, spec.track, run.variant, run_id, commit, settings_of(run.flags))
        commit_and_push([path], f"Belay registry: {registry_id} {run.name} launched", dry_run=False)
    say_next("stop (the watcher starts a new session when the runs finish)")


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


def trial_of(entry: dict, run: str) -> tuple[str, str]:
    """The id and commit of the entry's trial that ran ``run``."""
    for t in entry.get("trials") or []:
        if run in (t.get("wandb") or []):
            return t["id"], t.get("commit", "")
    raise SystemExit(f"refusing: no trial of {entry['id']} lists the run {run}")


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
    final = {r.name: (r.loss, r.delta) for r in sorted(results, key=lambda r: not r.control)}
    report = write_report(out_dir, header, fetch_series(control.name), fetch_series(variant.name), final)
    print(f"wrote the analysis {report} ({report_url(registry_id, trial)})")
    return [report, report.with_name(f"{trial}_loss.png"), report.with_name(f"{trial}_delta.png")]


def result_post(
    registry_id: str,
    track: Track,
    results: list[RunResult],
    overall: Verdict,
    idea: Idea | None,
    trial: str,
    commit: str,
) -> Post:
    rows = [
        ResultRow(r.name, "control" if r.control else r.variant, r.seed, r.loss, r.delta)
        for r in sorted(results, key=lambda r: not r.control)
    ]
    entry_link = entry_url(entry_path(registry_id))
    ref = idea_ref(idea, registry_id) if idea is not None else IdeaRef(None, "maintainer", None, entry_link)
    body = result_body(track, rows, ref, commit, (f"{registry_id} {trial}", report_url(registry_id, trial)))
    name = load_entry(registry_id)["name"]
    return Post(
        "result",
        title(registry_id, name, track, str(overall)),
        body,
        registry_id=registry_id,
        idea_id=None if idea is None else idea.id,
    )


def analyze(registry_id: str, *, dry_run: bool) -> None:
    """Write (or rewrite) the analysis of an experiment's recorded trial, commit and push it, and print the result
    post it belongs in."""
    spec = load_spec(registry_id)
    require_registry_checkout()
    results = experiment_results(spec)
    variants = [r for r in results if not r.control]
    overall = overall_verdict([r.verdict for r in variants])
    idea = page_state().idea_for(registry_id)
    trial, commit = trial_of(load_entry(registry_id), variants[0].name)
    report = write_analysis(registry_id, spec.track, results, overall, idea, trial, commit, dry_run=dry_run)
    commit_and_push(report, f"Belay registry: {registry_id} analysis {trial}", dry_run=dry_run)
    post = result_post(registry_id, spec.track, results, overall, idea, trial, commit)
    print(f"result post:\n{BOT_MARK} **{post.title}**\n\n{post.body}")
    say_next("stop")


def run_finished(registry_id: str, *, dry_run: bool) -> None:
    """Record a finished experiment: losses and verdict into the registry, the result onto the page."""
    spec = load_spec(registry_id)
    require_registry_checkout()
    state = page_state()
    runs = state.runs_of(registry_id)
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
    results = experiment_results(spec)
    variants = [r for r in results if not r.control]
    overall = overall_verdict([r.verdict for r in variants])
    idea = state.idea_for(registry_id)
    trial, commit = trial_of(load_entry(registry_id), variants[0].name)
    path, text = entry_with_results(registry_id, results)
    print(f"{'would write' if dry_run else 'writing'} {path.name}:\n{text}")
    if not dry_run:
        path.write_text(text)
    report = write_analysis(registry_id, spec.track, results, overall, idea, trial, commit, dry_run=dry_run)
    commit_and_push(
        [path, *report], f"Belay registry: {registry_id} results recorded, analysis {trial}", dry_run=dry_run
    )
    summary = "; ".join(f"{r.variant}: Δ {r.delta:+.4f} = {r.verdict}" for r in variants)
    if idea is not None:
        result = {"rows": [asdict(r) for r in results], "verdict": overall, "kind": PAGE_RESULT_KIND[overall]}
        report_idea(idea.id, {"state": "recorded", "result": {**result, "summary": summary}}, dry_run=dry_run)
    post = result_post(registry_id, spec.track, results, overall, idea, trial, commit)
    post_activity(post, github_posting=state.github_posting, dry_run=dry_run)
    if overall is not Verdict.WIN:
        say_next("stop")
        return
    escalate(
        f"{registry_id} beat the control beyond noise",
        f"{summary}. A maintainer decides on a seed-1 confirmation or adoption into {CONFIG.recipe_name}.",
        None if idea is None else idea.id,
        registry_id,
        dry_run=dry_run,
    )


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
