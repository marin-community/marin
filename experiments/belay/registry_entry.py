# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Writes to the experiment registry checkout (``CONFIG.registry_checkout``): new entries, trials and results.

The registry's own format and validator, ``experiments/belay/registry.py``, live on the registry branch; this module
calls it there and edits ``experiments/belay/registry/X####-<slug>.yaml`` files, then commits and pushes.
"""

import datetime
import re
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.belay.branch import current_branch, git
from experiments.belay.config import CONFIG
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


def create_entry(idea: Idea, category: str, *, dry_run: bool) -> tuple[str, Path]:
    """Allocate the next ``X####`` id for ``idea`` and write its ``approved`` entry; returns the id and path."""
    command = [
        sys.executable,
        "experiments/belay/registry.py",
        "new",
        "--slug",
        slugify(idea.name),
        "--name",
        idea.name,
        "--category",
        category,
        "--summary",
        first_paragraph(idea.body),
        "--source",
        "intake" if idea.origin == "intake" else "maintainer",
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
    commit_and_push(path, f"Belay registry: {match.group(1)} {idea.name} (idea #{idea.id})", dry_run=dry_run)
    return match.group(1), path


def add_trial(registry_id: str, variant: str, run_name: str, commit: str, settings: dict[str, str]) -> Path:
    """Add ``run_name`` as a new trial of the entry (or to its existing trial for the same variant and commit)."""
    path = entry_path(registry_id)
    entry = yaml.safe_load(path.read_text())
    trials = entry.setdefault("trials", [])
    trial = next((t for t in trials if t["variant"] == variant and t.get("commit") == commit[:10]), None)
    if trial is None:
        trial = {
            "id": f"t{len(trials) + 1:03d}",
            "variant": variant,
            "rung": CONFIG.rung,
            "protocol": "equal_steps",
            "batch": registry_id,
            "date": datetime.date.today().isoformat(),
            "comment_url": CONFIG.page_url,
            "wandb": [],
            "settings": settings,
            "commit": commit[:10],
        }
        trials.append(trial)
    if run_name not in trial["wandb"]:
        trial["wandb"].append(run_name)
    entry["status"] = "running"
    path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120))
    return path


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


def commit_and_push(path: Path, message: str, *, dry_run: bool) -> None:
    """Validate the registry, then commit and push ``path`` on the registry branch."""
    if dry_run:
        print(f"would commit and push {path.name} on {CONFIG.registry_branch}")
        return
    validate = [sys.executable, "experiments/belay/registry.py", "validate"]
    subprocess.run(validate, cwd=CONFIG.registry_checkout, check=True)
    git(CONFIG.registry_checkout, "add", str(path))
    git(CONFIG.registry_checkout, "commit", "--quiet", "-m", message)
    git(CONFIG.registry_checkout, "push", "--quiet", "origin", CONFIG.registry_branch)
