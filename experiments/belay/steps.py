# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Marin Belay session's commands: one per step of ``experiments/belay/RUNBOOK.md``.

    uv run python -m experiments.belay.steps start --events <file>     # which runbook section each event needs
    uv run python -m experiments.belay.steps week                      # weekly rollover
    uv run python -m experiments.belay.steps new-idea --idea-id N --category "..."
    uv run python -m experiments.belay.steps flags                     # the settings a run spec may change
    uv run python -m experiments.belay.steps spec X0222 --variant "..." --opt-set name=value
    uv run python -m experiments.belay.steps launch X0222
    uv run python -m experiments.belay.steps record X0222
    uv run python -m experiments.belay.steps fail X0222 --run seed0
    uv run python -m experiments.belay.steps escalate --title "..." --body "..." [--idea-id N]

Every command checks its preconditions, refuses with the reason when one fails, and ends by printing a ``next:``
line: the exact command to run next, or ``stop``. ``--dry-run`` runs every read-only check and prints what would be
written, committed, pushed, submitted or posted, without doing it. Fixed values come from
``experiments.belay.config.CONFIG``.
"""

import argparse
import dataclasses
import datetime
import json
import re
import subprocess
import sys
import typing
from pathlib import Path

import yaml

from experiments.belay.config import CONFIG
from experiments.belay.demo import call, post_activity, report_idea, service_token
from experiments.belay.results import Verdict, experiment_results, record
from experiments.belay.submit import (
    MAX_RUN_KEY,
    RUNS_DIR,
    _git,
    load_spec,
    require_registry_checkout,
    require_week_branch,
    run_name,
    submit_runs,
)
from experiments.grug.fast_track.launch import _parse_as
from experiments.grug.fast_track.model import GrugModelConfig
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

STEPS = "uv run python -m experiments.belay.steps"
SETTINGS = {"--model-set": GrugModelConfig, "--opt-set": GrugMoeMuonHConfig}
TERMINAL_RUN_STATES = ("succeeded", "failed", "killed", "worker_failed", "unschedulable")
ESCALATED = "Escalated:"
MAX_VARIANT = 60
MAX_SUMMARY = 400
PAGE_RESULT_KIND = {Verdict.WIN: "better", Verdict.LOSS: "worse", Verdict.NOISE: "neutral"}


def say_next(command: str) -> None:
    print(f"next: {command}")


def repo_root() -> Path:
    return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))


def page_state() -> dict:
    return call("agent/state", token=service_token())


def idea_by_id(state: dict, idea_id: int) -> dict:
    for idea in state["ideas"]:
        if idea["id"] == idea_id:
            return idea
    raise SystemExit(f"no idea #{idea_id} on the Belay page")


def idea_by_registry_id(state: dict, registry_id: str) -> dict:
    ideas = [i for i in state["ideas"] if i["registry_id"] == registry_id]
    if len(ideas) != 1:
        raise SystemExit(f"expected one idea with registry id {registry_id} on the Belay page, found {len(ideas)}")
    return ideas[0]


def registry_entry_path(registry_id: str) -> Path:
    paths = list((CONFIG.registry_checkout / "experiments/belay/registry").glob(f"{registry_id}-*.yaml"))
    if len(paths) != 1:
        raise SystemExit(f"registry: expected one entry for {registry_id}, found {[p.name for p in paths]}")
    return paths[0]


def registry_url(path: Path) -> str:
    return f"https://github.com/{CONFIG.repository}/blob/{CONFIG.registry_branch}/experiments/belay/registry/{path.name}"


def slugify(text: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return "-".join(words[:6])


def first_paragraph(text: str) -> str:
    paragraph = " ".join(text.strip().split("\n\n")[0].split())
    return paragraph if len(paragraph) <= MAX_SUMMARY else paragraph[: MAX_SUMMARY - 1].rsplit(" ", 1)[0] + "…"


def remote_branch_exists(repo: Path, branch: str) -> bool:
    out = _git(repo, "ls-remote", "--heads", "origin", branch)
    return bool(out)


def run_git(repo: Path, args: list[str], *, dry_run: bool) -> None:
    print(f"{'would run' if dry_run else 'run'}: git {' '.join(args)}")
    if not dry_run:
        _git(repo, *args)


# ---------- start ----------


def idea_next_step(idea: dict, launched: set[str]) -> str:
    """The runbook step an approved idea is at, from its page state, registry entry and run spec."""
    if idea["state"] in ("planned", "running", "recorded", "declined"):
        return f"skip: idea #{idea['id']} is already {idea['state']}"
    if (idea["note"] or "").startswith(ESCALATED):
        return f"skip: idea #{idea['id']} is escalated and waits for a maintainer"
    if idea["state"] != "approved":
        return f"skip: idea #{idea['id']} is {idea['state']}, not approved"
    rid = idea["registry_id"]
    if rid is None:
        return f'section 1, step 1: {STEPS} new-idea --idea-id {idea["id"]} --category "<category>"'
    if not (RUNS_DIR / f"{rid}.yaml").exists():
        return f"section 1, step 2: {STEPS} flags   (then step 3: {STEPS} spec {rid} ...)"
    if rid in launched:
        return f"skip: {rid} already has runs on Iris"
    return f"section 1, step 4: {STEPS} launch {rid}"


def start(events_file: Path | None, today: datetime.date) -> None:
    repo = repo_root()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    week = CONFIG.week_branch(today)
    print(f"recipe: {CONFIG.recipe_name} ({CONFIG.recipe_branch}); this week's branch: {week}")
    if branch != week:
        print(f"event: weekly rollover (checkout is on {branch}). Do section 4 first, then run start again.")
        say_next(f"{STEPS} week")
        return
    require_week_branch(repo, today)
    require_registry_checkout(CONFIG.registry_checkout)
    _git(CONFIG.registry_checkout, "pull", "--ff-only", "--quiet", "origin", CONFIG.registry_branch)
    events = json.loads(events_file.read_text()) if events_file else []
    if not events:
        print("no events")
        say_next("stop")
        return
    state = page_state()
    launched = {r["registry_id"] for r in state["runs"]}
    todo: list[str] = []
    finished: set[str] = set()
    for event in events:
        kind, payload = event["kind"], event["payload"]
        if kind in ("approved", "work_pending"):
            step = idea_next_step(idea_by_id(state, payload["idea_id"]), launched)
            todo.append(f"[{kind} idea #{payload['idea_id']}] {step}")
        elif kind == "run_finished":
            finished.add(payload["experiment"])
        else:
            todo.append(f'[{kind}] section 6: {STEPS} escalate --title "Note needs a maintainer" --body "<the note>"')
    runs_by_id: dict[str, list[dict]] = {}
    for run in state["runs"]:
        runs_by_id.setdefault(run["registry_id"], []).append(run)
    for rid in finished:
        failed = [r for r in runs_by_id.get(rid, []) if r["state"] in TERMINAL_RUN_STATES and r["state"] != "succeeded"]
        if failed:
            for r in failed:
                todo.append(f"[run_finished {r['name']} {r['state']}] section 3: {STEPS} fail {rid} --run {r['run']}")
        else:
            todo.append(f"[run_finished {rid}] section 2: {STEPS} record {rid}")
    print("do these in order (one runbook section each):")
    for i, line in enumerate(todo, 1):
        print(f"  {i}. {line}")
    say_next("the command in item 1 (a 'skip' item needs nothing; go to the next item)")


# ---------- week ----------


def week(fresh: bool, today: datetime.date, *, dry_run: bool) -> None:
    repo = repo_root()
    target = CONFIG.week_branch(today)
    last = CONFIG.week_branch(today - datetime.timedelta(days=7))
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"refusing: {repo} has uncommitted changes to tracked files")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _git(repo, "fetch", "--quiet", "origin", CONFIG.base_branch)
    if branch != target:
        if remote_branch_exists(repo, target):
            print(f"{target} exists on origin; checking it out")
            run_git(repo, ["fetch", "--quiet", "origin", target], dry_run=dry_run)
            run_git(repo, ["checkout", "-B", target, f"origin/{target}"], dry_run=dry_run)
        else:
            source = last if not fresh and remote_branch_exists(repo, last) else CONFIG.base_branch
            print(f"cutting {target} from origin/{source}")
            run_git(repo, ["fetch", "--quiet", "origin", source], dry_run=dry_run)
            run_git(repo, ["checkout", "-b", target, f"origin/{source}"], dry_run=dry_run)
    contains = subprocess.run(
        ["git", "merge-base", "--is-ancestor", f"origin/{CONFIG.base_branch}", "HEAD"], cwd=repo
    ).returncode
    if dry_run and branch != target:
        print(f"would run: git merge --no-edit origin/{CONFIG.base_branch} (if {target} lacks it)")
    elif contains != 0:
        print(f"{target} lacks origin/{CONFIG.base_branch}; merging it in")
        merge = subprocess.run(["git", "merge", "--no-edit", f"origin/{CONFIG.base_branch}"], cwd=repo)
        if merge.returncode != 0:
            subprocess.run(["git", "merge", "--abort"], cwd=repo, check=True)
            print(f"merge of {CONFIG.base_branch} into {target} conflicts; aborted, nothing pushed")
            say_next(
                f'{STEPS} escalate --title "Weekly branch {target} conflicts with {CONFIG.base_branch}" '
                '--body "A maintainer resolves the merge or reruns week with --fresh."'
            )
            return
    run_git(repo, ["push", "--quiet", "-u", "origin", target], dry_run=dry_run)
    print(f"on {target}, which contains origin/{CONFIG.base_branch}")
    say_next(f"{STEPS} start --events <the same events file>")


# ---------- new idea ----------


def new_idea(idea_id: int, category: str, slug: str | None, *, dry_run: bool) -> None:
    state = page_state()
    idea = idea_by_id(state, idea_id)
    if idea["state"] != "approved" or idea["registry_id"]:
        raise SystemExit(f"refusing: idea #{idea_id} is {idea['state']} with registry id {idea['registry_id']}")
    require_registry_checkout(CONFIG.registry_checkout)
    name = (idea["title"] or idea["body"].strip().splitlines()[0])[:90]
    command = [
        sys.executable,
        "experiments/belay/registry.py",
        "new",
        "--slug",
        slug or slugify(name),
        "--name",
        name,
        "--category",
        category,
        "--summary",
        first_paragraph(idea["body"]),
        "--source",
        "intake" if idea["origin"] == "intake" else "maintainer",
    ]
    if idea["url"]:
        command += ["--intake-url", idea["url"], "--intake-author", idea["author"]]
        command += ["--approved-by", idea["decided_by"]]
    if dry_run:
        command.append("--dry-run")
    out = subprocess.run(command, cwd=CONFIG.registry_checkout, check=True, capture_output=True, text=True).stdout
    print(out.rstrip())
    if dry_run:
        rid = re.search(r"(X\d{4})-", out).group(1)
        print(f"would commit and push the entry on {CONFIG.registry_branch}")
        report_idea(idea_id, {"registry_id": rid}, dry_run=True)
    else:
        path = Path(out.strip().splitlines()[-1])
        rid = path.name.split("-")[0]
        _git(CONFIG.registry_checkout, "add", str(path))
        _git(CONFIG.registry_checkout, "commit", "--quiet", "-m", f"Belay registry: {rid} {name} (idea #{idea_id})")
        _git(CONFIG.registry_checkout, "push", "--quiet", "origin", CONFIG.registry_branch)
        report_idea(idea_id, {"registry_id": rid, "registry_url": registry_url(path)}, dry_run=False)
    print(f"idea #{idea_id} is {rid}")
    say_next(f"{STEPS} flags   (section 1, step 2)")


# ---------- flags ----------


def flags() -> None:
    for option, cls in SETTINGS.items():
        hints = typing.get_type_hints(cls)
        print(f"{option} <name>=<value>   ({cls.__name__})")
        for f in dataclasses.fields(cls):
            if f.init:
                default = f.default if f.default is not dataclasses.MISSING else "(required)"
                print(f"  {f.name}: {getattr(hints[f.name], '__name__', hints[f.name])} = {default!r}")
    print("Defaults shown are the dataclass defaults; the recipe overrides some of them.")
    say_next(f'{STEPS} spec <X####> --variant "<what changes>" --opt-set <name>=<value>   (section 1, step 3)')


# ---------- spec ----------


def check_setting(option: str, setting: str) -> None:
    cls = SETTINGS[option]
    name, sep, value = setting.partition("=")
    if not sep:
        raise SystemExit(f"refusing: {option} {setting!r} is not name=value")
    hints = typing.get_type_hints(cls)
    if name not in {f.name for f in dataclasses.fields(cls)}:
        raise SystemExit(f"refusing: {cls.__name__} has no field {name!r} (see `{STEPS} flags`)")
    _parse_as(hints[name], value, name)


def spec(registry_id: str, variant: str, settings: list[tuple[str, str]], *, dry_run: bool) -> None:
    repo = repo_root()
    branch = require_week_branch(repo, datetime.date.today())
    entry = yaml.safe_load(registry_entry_path(registry_id).read_text())
    if entry["status"] != "approved":
        raise SystemExit(f"refusing: {registry_id} is {entry['status']}, not approved")
    path = RUNS_DIR / f"{registry_id}.yaml"
    if path.exists():
        raise SystemExit(f"refusing: {path} already exists.\nnext: {STEPS} launch {registry_id}")
    if not settings:
        raise SystemExit("refusing: give at least one --model-set or --opt-set setting")
    if len(variant) > MAX_VARIANT:
        raise SystemExit(f"refusing: --variant is longer than {MAX_VARIANT} characters")
    for option, setting in settings:
        check_setting(option, setting)
    run_flags = [part for pair in settings for part in pair]
    data = {
        "experiment": registry_id,
        "launcher": CONFIG.launcher,
        "protocol": "equal_steps",
        "runs": {
            f"seed{CONFIG.seed}": {"size": CONFIG.rung, "seed": CONFIG.seed, "variant": variant, "flags": run_flags},
            "control": {
                "size": CONFIG.rung,
                "seed": CONFIG.seed,
                "variant": f"{CONFIG.recipe_name} control",
                "control": True,
            },
        },
    }
    assert all(len(k) <= MAX_RUN_KEY for k in data["runs"])
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)
    print(f"{'would write' if dry_run else 'wrote'} {path.relative_to(repo)}:\n{text}")
    if not dry_run:
        path.write_text(text)
    run_git(repo, ["add", str(path.relative_to(repo))], dry_run=dry_run)
    run_git(repo, ["commit", "--quiet", "-m", f"Belay {registry_id}: run spec ({variant})"], dry_run=dry_run)
    run_git(repo, ["push", "--quiet", "origin", branch], dry_run=dry_run)
    say_next(f"{STEPS} launch {registry_id}   (section 1, step 4)")


# ---------- launch ----------


def plan_text(registry_id: str, entry: dict, runs: list[dict], branch: str, commit: str, cost: str) -> str:
    rows = "\n".join(f"| `{r['name']}` | {r['variant']} | {r['seed']} | {'yes' if r['control'] else ''} |" for r in runs)
    return (
        f"**{registry_id} {entry['name']}**: {entry['summary']}\n\n"
        f"| Run | Variant | Seed | Control |\n|---|---|---|---|\n{rows}\n\n"
        f"Recipe {CONFIG.recipe_name}, rung {CONFIG.rung} at the recipe's step count, ranked by `{CONFIG.metric}` "
        f"against the same-commit control; |Δ| < {CONFIG.noise} is noise. Code: `{branch}` at `{commit[:10]}`. "
        f"Cost: {cost}."
    )


def launch(registry_id: str, *, dry_run: bool) -> None:
    repo = repo_root()
    experiment = load_spec(registry_id)
    state = page_state()
    idea = idea_by_registry_id(state, registry_id)
    if idea["state"] != "approved":
        raise SystemExit(f"refusing: idea #{idea['id']} is {idea['state']}, not approved")
    entry = yaml.safe_load(registry_entry_path(registry_id).read_text())
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    order = sorted(experiment.runs, key=lambda r: not experiment.runs[r].control)  # controls first
    submit_runs(repo, experiment, order, commit, CONFIG.page_url, dry_run=True)
    runs = [
        {"run": r, "name": run_name(registry_id, r), "variant": s.variant, "seed": s.seed, "control": s.control}
        for r, s in experiment.runs.items()
    ]
    nodes = sum(s.nodes for s in experiment.runs.values())
    node_hours = nodes * CONFIG.rung_minutes / 60
    cost = f"{len(runs)} runs x 1 node (8xH100) x ~{CONFIG.rung_minutes} min, {node_hours:.1f} node-hours"
    plan = {
        "summary": entry["summary"],
        "protocol": f"equal_steps, {CONFIG.rung}, seed {CONFIG.seed}; {CONFIG.metric}, noise {CONFIG.noise}",
        "cost": cost,
        "commit": commit,
        "branch": branch,
        "spec_url": f"https://github.com/{CONFIG.repository}/blob/{commit}/experiments/belay/runs/{registry_id}.yaml",
        "runs": runs,
    }
    report_idea(idea["id"], {"state": "planned", "plan": plan}, dry_run=dry_run)
    post_activity(
        "plan",
        f"Plan: {registry_id} {entry['name']}",
        plan_text(registry_id, entry, runs, branch, commit, cost),
        registry_id=registry_id,
        idea_id=idea["id"],
        dry_run=dry_run,
    )
    if not dry_run:
        submit_runs(repo, experiment, order, commit, CONFIG.page_url, dry_run=False)
    say_next("stop (the watcher starts a session when the runs finish)")


# ---------- record ----------


def record_step(registry_id: str, *, dry_run: bool) -> None:
    state = page_state()
    idea = idea_by_registry_id(state, registry_id)
    runs = [r for r in state["runs"] if r["registry_id"] == registry_id]
    live = [r["name"] for r in runs if r["state"] not in TERMINAL_RUN_STATES]
    if live:
        print(f"still running: {live}; a later session records {registry_id}")
        say_next("stop")
        return
    failed = [r for r in runs if r["state"] != "succeeded"]
    if failed:
        for r in failed:
            print(f"{r['name']} is {r['state']}")
        say_next(f"{STEPS} fail {registry_id} --run {failed[0]['run']}   (section 3)")
        return
    results = experiment_results(registry_id)
    path = record(registry_id, results, CONFIG.registry_checkout, dry_run=dry_run)
    control = {r.seed: r for r in results if r.control}
    variants = [r for r in results if not r.control]
    verdicts = {r.verdict for r in variants}
    overall = Verdict.WIN if Verdict.WIN in verdicts else Verdict.LOSS if verdicts == {Verdict.LOSS} else Verdict.NOISE
    entry_name = yaml.safe_load(path.read_text())["name"]
    rows = [
        {"name": r.name, "variant": r.variant, "seed": r.seed, "loss": r.loss, "delta": r.delta, "control": r.control}
        for r in results
    ]
    table = "\n".join(
        f"| `{r.name}` | {r.variant} | {r.seed} | {r.loss:.4f} | {'' if r.delta is None else f'{r.delta:+.4f}'} "
        f"| {r.verdict or 'control'} |"
        for r in sorted(results, key=lambda r: not r.control)
    )
    summary = "; ".join(
        f"{r.variant}: {r.loss:.4f} vs control {control[r.seed].loss:.4f}, Δ {r.delta:+.4f} = {r.verdict}"
        for r in variants
    )
    body = (
        f"**{registry_id} {entry_name}**, `{CONFIG.metric}` at {CONFIG.rung} (rule: Δ ≤ -{CONFIG.noise} win, "
        f"Δ ≥ +{CONFIG.noise} loss, otherwise noise).\n\n"
        f"| Run | Variant | Seed | Loss | Δ | Verdict |\n|---|---|---|---|---|---|\n{table}\n\n"
        f"Verdict: **{overall}**. Registry: {registry_url(path)}"
    )
    report_idea(
        idea["id"],
        {
            "state": "recorded",
            "result": {"rows": rows, "verdict": str(overall), "kind": PAGE_RESULT_KIND[overall], "summary": summary},
        },
        dry_run=dry_run,
    )
    post_activity(
        "result", f"Result: {registry_id} {entry_name}: {overall}", body, registry_id=registry_id,
        idea_id=idea["id"], dry_run=dry_run,
    )  # fmt: skip
    if overall is Verdict.WIN:
        post_activity(
            "escalation",
            f"{registry_id} beat the control beyond noise",
            f"{summary}. A maintainer decides on a seed-1 confirmation or adoption into {CONFIG.recipe_name}.",
            registry_id=registry_id,
            idea_id=idea["id"],
            dry_run=dry_run,
        )
    say_next("stop" if overall is not Verdict.WIN else "stop (a maintainer decides what follows the win)")


# ---------- fail and escalate ----------


def escalate(title: str, body: str, idea_id: int | None, registry_id: str | None, *, dry_run: bool) -> None:
    post_activity("escalation", title, body, registry_id=registry_id, idea_id=idea_id, dry_run=dry_run)
    if idea_id is not None:
        report_idea(idea_id, {"note": f"{ESCALATED} {title}"}, dry_run=dry_run)
    say_next("stop")


def fail(registry_id: str, run: str, *, dry_run: bool) -> None:
    state = page_state()
    idea = idea_by_registry_id(state, registry_id)
    name = run_name(registry_id, run)
    rows = [r for r in state["runs"] if r["name"] == name]
    if not rows:
        raise SystemExit(f"no run {name} on the Belay page")
    row = rows[0]
    if row["state"] not in TERMINAL_RUN_STATES or row["state"] == "succeeded":
        raise SystemExit(f"refusing: {name} is {row['state']}, not failed.\nnext: {STEPS} record {registry_id}")
    body = (
        f"`{name}` ended `{row['state']}` at step {row['step']} (job `{row['job_id']}`). A non-preemption failure is a "
        f"bug: nothing was rerun. Logs: `uv run iris --cluster marin job logs {row['job_id']}`."
    )
    escalate(f"{name} {row['state']}", body, idea["id"], registry_id, dry_run=dry_run)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true", help="check and print; write, push, submit and post nothing")
        return p

    p = command("start")
    p.add_argument("--events", type=Path, help="the session's events file (in your prompt)")
    p.add_argument("--today", type=datetime.date.fromisoformat, help="pretend date, with --dry-run only")
    p = command("week")
    p.add_argument("--fresh", action="store_true", help="cut from the base branch, not last week's")
    p.add_argument("--today", type=datetime.date.fromisoformat, help="pretend date, with --dry-run only")
    p = command("new-idea")
    p.add_argument("--idea-id", type=int, required=True)
    p.add_argument("--category", required=True, help="one of the registry's categories.json; 'Unsorted' if unsure")
    p.add_argument("--slug", help="default: the first six words of the idea's title")
    command("flags")
    p = command("spec")
    p.add_argument("registry_id")
    p.add_argument("--variant", required=True, help=f"what the variant changes, at most {MAX_VARIANT} characters")
    for option in SETTINGS:
        p.add_argument(option, action="append", default=[], dest=option.strip("-").replace("-", "_"))
    command("launch").add_argument("registry_id")
    command("record").add_argument("registry_id")
    p = command("fail")
    p.add_argument("registry_id")
    p.add_argument("--run", required=True)
    p = command("escalate")
    p.add_argument("--title", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--idea-id", type=int)
    p.add_argument("--registry-id")
    args = parser.parse_args(argv)
    today = getattr(args, "today", None)
    if today is not None and not args.dry_run:
        raise SystemExit("refusing: --today is for dry runs only")
    today = today or datetime.date.today()

    if args.command == "start":
        start(args.events, today)
    elif args.command == "week":
        week(args.fresh, today, dry_run=args.dry_run)
    elif args.command == "new-idea":
        new_idea(args.idea_id, args.category, args.slug, dry_run=args.dry_run)
    elif args.command == "flags":
        flags()
    elif args.command == "spec":
        settings = [("--model-set", s) for s in args.model_set] + [("--opt-set", s) for s in args.opt_set]
        spec(args.registry_id, args.variant, settings, dry_run=args.dry_run)
    elif args.command == "launch":
        launch(args.registry_id, dry_run=args.dry_run)
    elif args.command == "record":
        record_step(args.registry_id, dry_run=args.dry_run)
    elif args.command == "fail":
        fail(args.registry_id, args.run, dry_run=args.dry_run)
    else:
        escalate(args.title, args.body, args.idea_id, args.registry_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])
