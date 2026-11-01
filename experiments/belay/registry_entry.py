# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Writes to the experiment registry checkout (``CONFIG.registry_checkout``): new entries, trials and results.

The registry's own format and validator, ``experiments/belay/registry.py``, live on the registry branch; this module
calls it there and edits ``experiments/belay/registry/X####-<slug>.yaml`` files, then commits and pushes.
"""

import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.belay.branch import current_branch, git
from experiments.belay.config import CONFIG, Track
from experiments.belay.page import Idea
from experiments.belay.results import RunResult

ENTRIES = "experiments/belay/registry"
MAX_SUMMARY = 400


def require_registry_checkout() -> None:
    branch = current_branch(CONFIG.registry_checkout)
    if branch != CONFIG.registry_branch:
        raise SystemExit(f"refusing: {CONFIG.registry_checkout} is on {branch}, not {CONFIG.registry_branch}")
    git(CONFIG.registry_checkout, "pull", "--ff-only", "--quiet", "origin", CONFIG.registry_branch)


def entry_path(registry_id: str) -> Path:
    paths = list((CONFIG.registry_checkout / ENTRIES).glob(f"{registry_id}-*.yaml"))
    if len(paths) != 1:
        raise SystemExit(f"registry: expected one entry for {registry_id}, found {[p.name for p in paths]}")
    return paths[0]


def entry_url(path: Path) -> str:
    return f"https://github.com/{CONFIG.repository}/blob/{CONFIG.registry_branch}/{ENTRIES}/{path.name}"


def load_entry(registry_id: str) -> dict:
    return yaml.safe_load(entry_path(registry_id).read_text())


def slugify(text: str) -> str:
    return "-".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split()[:6])


def first_paragraph(text: str) -> str:
    paragraph = " ".join(text.strip().split("\n\n")[0].split())
    return paragraph if len(paragraph) <= MAX_SUMMARY else paragraph[: MAX_SUMMARY - 1].rsplit(" ", 1)[0] + "…"


def entry_for_intake(url: str) -> str | None:
    """The id of the entry already made from the intake comment at ``url``, if any."""
    for path in sorted((CONFIG.registry_checkout / ENTRIES).glob("X*.yaml")):
        entry = yaml.safe_load(path.read_text())
        if (entry.get("intake") or {}).get("url") == url:
            return entry["id"]
    return None


def create_entry(idea: Idea, name: str, category: str, track: Track, *, dry_run: bool) -> tuple[str, Path]:
    """Allocate the next ``X####`` id for ``idea`` and write its ``approved`` entry under the short ``name``;
    returns the id and path."""
    command = [
        sys.executable,
        "experiments/belay/registry.py",
        "new",
        "--slug",
        slugify(name),
        "--name",
        name,
        "--category",
        category,
        "--summary",
        first_paragraph(idea.body),
        "--source",
        "intake" if idea.origin == "intake" else "maintainer",
        "--track",
        track.value,
    ]
    if idea.url:
        command += ["--intake-url", idea.url, "--intake-author", idea.author, "--approved-by", idea.decided_by or ""]
    if dry_run:
        command.append("--dry-run")
    out = subprocess.run(command, cwd=CONFIG.registry_checkout, check=True, capture_output=True, text=True).stdout
    print(out.rstrip())
    match = re.search(r"(X\d{4})-[^/\s]*\.yaml", out)
    if match is None:
        raise SystemExit(f"registry.py new printed no entry path:\n{out}")
    path = CONFIG.registry_checkout / ENTRIES / match.group(0)
    commit_and_push([path], f"Belay registry: {match.group(1)} {name} (idea #{idea.id})", dry_run=dry_run)
    return match.group(1), path


def add_trial(
    registry_id: str, track: Track, protocol: str, variant: str, commit: str, settings: dict[str, str], comment_url: str
) -> tuple[str, Path]:
    """A new trial of the entry (``protocol`` is ``equal_steps`` or ``fixed_time``); returns its id and the path."""
    path = entry_path(registry_id)
    entry = yaml.safe_load(path.read_text())
    trials = entry.setdefault("trials", [])
    trial_id = f"t{len(trials) + 1:03d}"
    trial = {
        "id": trial_id,
        "variant": variant,
        "rung": track.value,
        "protocol": protocol,
        "batch": registry_id,
        "date": datetime.date.today().isoformat(),
        "comment_url": comment_url,
        "wandb": [],
        "settings": settings,
        "commit": commit[:10],
    }
    trials.append(trial)
    entry["status"] = "running"
    path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120))
    return trial_id, path


def add_trial_run(registry_id: str, trial_id: str, run_name: str) -> Path:
    path = entry_path(registry_id)
    entry = yaml.safe_load(path.read_text())
    trial = next(t for t in entry["trials"] if t["id"] == trial_id)
    if run_name not in trial["wandb"]:
        trial["wandb"].append(run_name)
    path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120))
    return path


def comment_url_of(entry: dict) -> str | None:
    """The idea's one #9451 comment, recorded on its Track trials. Trials from before Tracks (rung ``d512`` and so
    on) link the batch reports they were posted in, which are not the idea's comment and are never edited."""
    tracks = {t.value for t in Track}
    for trial in entry.get("trials") or []:
        url = trial.get("comment_url") or ""
        if trial.get("rung") in tracks and "issuecomment-" in url:
            return url
    return None


def entry_with_results(registry_id: str, results: list[RunResult]) -> tuple[Path, str]:
    """The entry with each trial's loss, delta, steps and same-seed control filled in (by W&B run name), as YAML."""
    path = entry_path(registry_id)
    entry = yaml.safe_load(path.read_text())
    by_name = {r.name: r for r in results}
    controls = {r.seed: r for r in results if r.control}
    for trial in entry.get("trials", []):
        runs = [by_name[n] for n in trial.get("wandb", []) if n in by_name]
        if not runs:
            continue
        trial["loss"] = [round(r.loss, 4) for r in runs]
        trial["delta"] = [r.delta for r in runs]
        trial["steps"] = max(r.step for r in runs)
        trial["baseline"] = ", ".join(
            f"{controls[r.seed].name} {controls[r.seed].loss:.4f} (same commit, seed {r.seed})" for r in runs
        )
    entry["status"] = "done"
    return path, yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120)


def with_step_time(text: str, trial_id: str, time_change: float) -> str:
    """The entry YAML ``text`` with the trial's variant train time relative to its control (``+31%``)."""
    entry = yaml.safe_load(text)
    trial = next(t for t in entry["trials"] if t["id"] == trial_id)
    trial["step_time_change"] = f"{time_change:+.0%}" if round(time_change * 100) else "0%"
    return yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120)


def potential_of(entry: dict, score: int, rationale: str, model: str) -> dict:
    """The entry's ``potential``: a session's 0-10 score, tied by digest to the results it was given."""
    inputs = {k: entry.get(k) for k in ("status", "trials", "events")}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {
        "score": score,
        "rationale": rationale,
        "model": model,
        "date": datetime.date.today().isoformat(),
        "inputs_digest": digest,
    }


def commit_and_push(paths: list[Path], message: str, *, dry_run: bool) -> None:
    """Validate the registry, then commit ``paths`` together and push them on the registry branch; refuses if the
    push did not land."""
    if dry_run:
        print(f"would commit and push {', '.join(p.name for p in paths)} on {CONFIG.registry_branch}")
        return
    validate = [sys.executable, "experiments/belay/registry.py", "validate"]
    subprocess.run(validate, cwd=CONFIG.registry_checkout, check=True)
    git(CONFIG.registry_checkout, "add", *(str(p) for p in paths))
    if not git(CONFIG.registry_checkout, "status", "--porcelain", *(str(p) for p in paths)):
        print(f"{', '.join(p.name for p in paths)} unchanged; nothing to push")
        return
    git(CONFIG.registry_checkout, "commit", "--quiet", "-m", message)
    git(CONFIG.registry_checkout, "push", "--quiet", "origin", CONFIG.registry_branch)
    head = git(CONFIG.registry_checkout, "rev-parse", "HEAD")
    git(CONFIG.registry_checkout, "fetch", "--quiet", "origin", CONFIG.registry_branch)
    remote = git(CONFIG.registry_checkout, "rev-parse", f"origin/{CONFIG.registry_branch}")
    if remote != head:
        raise SystemExit(
            f"refusing: the registry commit {head[:10]} is not on origin/{CONFIG.registry_branch} ({remote[:10]})"
        )
    print(f"pushed {', '.join(p.name for p in paths)} as {head[:10]} to origin/{CONFIG.registry_branch}")
