# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select pilot agentic trajectories and the held-out task split for the Snowball v0 recipe sweep.

Order of operations, all with fixed seeds:
1. Eligibility. TMax: every row of the success split except trajectories whose bash commands mention a
   hidden verifier asset of their environment (tmax_leaks). Kimi SWE-smith: result == "1.0", a single
   SWE-smith repo parsed from the verifier output, repo not in SWE-bench Verified, and no command
   that reads pre-bug code from git history (the packages leave the clean commit and a local
   main branch in the agent's container, so these passes may copy the fix). Every
   trajectory must fit the 32,768-token context after rendering; nothing is truncated.
2. Kimi only: stratified sample of KIMI_TASKS tasks, allocated across repos in proportion to
   eligible tasks (largest remainder).
3. Held-out: round(5%) of each family's tasks, drawn from the eligible (and sampled) task list.
   SWE-smith held-out tasks are drawn only from tasks with no trial in allkimi's training list
   (v0 carries a graft of that SFT), so validation loss is not read on allkimi-seen tasks.
   Held-out tasks never train; each gets one fixed validation trajectory.
"""

import collections
import random
import re
from typing import NotRequired, TypedDict

from experiments.post_training.sweep_data.render import kimi_messages, render, tmax_messages
from experiments.post_training.sweep_data.tmax_leaks import references_hidden_asset

CONTEXT = 32768
SEED = 20261016
KIMI_TASKS = 2100
HELDOUT_FRACTION = 0.05
SWEBENCH_VERIFIED_REPOS = frozenset(
    {
        "astropy/astropy",
        "django/django",
        "matplotlib/matplotlib",
        "mwaskom/seaborn",
        "pallets/flask",
        "psf/requests",
        "pydata/xarray",
        "pylint-dev/pylint",
        "pytest-dev/pytest",
        "scikit-learn/scikit-learn",
        "sphinx-doc/sphinx",
        "sympy/sympy",
    }
)


class TrainingRecord(TypedDict):
    """One rendered trajectory: token ids and the per-token assistant mask, plus its identity."""

    family: str
    task: str
    trajectory: str
    ids: list[int]
    mask: list[int]
    repo: NotRequired[str]  # SWE-smith rows only
    position: NotRequired[dict]  # replay rows only: source file, position and raw-bytes sha256


class Selection(TypedDict):
    """What select() returns: per-family train/validation records, held-out ids and selection counts."""

    allkimi_overlap: dict[str, int]
    tmax_source_tasks: int
    train: dict[str, list[TrainingRecord]]
    validation: dict[str, list[TrainingRecord]]
    heldout: dict[str, list[str]]
    dropped: dict[str, int]
    swesmith_allocation: dict[str, int]
    eligible_tasks: dict[str, int]


SWESMITH_INSTANCE = re.compile(r"\b([A-Za-z0-9_.-]+?)__([A-Za-z0-9_.-]+?)\.([0-9a-f]{8})\b")


KEYSTROKES = re.compile(r'"keystrokes":\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
# git commands that read code from another commit, branch or the reflog.
GIT_HISTORY_READ = re.compile(
    r"git\s+(diff|show)\s+[^\n|;&]*(HEAD~|HEAD\^|\bmain\b|\bmaster\b|origin/|\b[0-9a-f]{7,40}\b)"
    r"|git\s+(checkout|restore)\s+[^\n|;&]*(\bmain\b|\bmaster\b|HEAD~|HEAD\^|--source)"
    r"|git\s+log\s+[^\n|;&]*(-p\b|--patch|--stat)"
    r"|git\s+(stash\s+show|cherry-pick|revert)"
)


def reads_git_history(row: dict) -> bool:
    """True when any terminus-2 command in a Kimi trajectory reads code from git history."""
    return any(
        GIT_HISTORY_READ.search(command)
        for message in row["conversations"]
        if message["role"] == "assistant"
        for command in KEYSTROKES.findall(message["content"])
    )


def kimi_repo(row: dict) -> str | None:
    repos = {f"{owner}/{name}" for owner, name, _ in SWESMITH_INSTANCE.findall(row["verifier_output"] or "")}
    return repos.pop() if len(repos) == 1 else None


def rendered(tokenizer, family: str, row: dict) -> TrainingRecord:
    if family == "tmax":
        messages, tools = tmax_messages(row)
        task, trajectory = row["metadata"]["task"], row["metadata"]["trial_name"]
    else:
        messages, tools = kimi_messages(row), None
        task, trajectory = row["task"], row["trial_name"]
    ids, mask = render(tokenizer, messages, tools)
    return {"family": family, "task": task, "trajectory": trajectory, "ids": ids, "mask": mask}


def proportional(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` across strata in proportion to `counts`."""
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"Cannot sample {total} of {available}")
    quotas = {key: total * count / available for key, count in counts.items()}
    allocation = {key: int(quota) for key, quota in quotas.items()}
    for key in sorted(quotas, key=lambda key: (allocation[key] - quotas[key], key))[: total - sum(allocation.values())]:
        allocation[key] += 1
    return allocation


def select(
    tokenizer,
    tmax_rows: list[dict],
    kimi_rows: list[dict],
    allkimi_trials: frozenset[str],
    tmax_hidden: dict[str, list[str]],
) -> Selection:
    rng = random.Random(SEED)
    dropped = collections.Counter()
    records: dict[str, list[TrainingRecord]] = {"tmax": [], "swesmith": []}
    for row in tmax_rows:
        if references_hidden_asset(row, tmax_hidden.get(row["metadata"]["task"], [])):
            dropped["tmax_references_hidden_verifier_asset"] += 1
            continue
        record = rendered(tokenizer, "tmax", row)
        if len(record["ids"]) > CONTEXT:
            dropped["tmax_over_context"] += 1
            continue
        records["tmax"].append(record)
    repo_of = {}
    for row in kimi_rows:
        if row["result"] != "1.0":
            dropped["swesmith_not_passing"] += 1
            continue
        if reads_git_history(row):
            dropped["swesmith_reads_git_history"] += 1
            continue
        repo = kimi_repo(row)
        if repo is None:
            dropped["swesmith_no_repo"] += 1
            continue
        if repo in SWEBENCH_VERIFIED_REPOS:
            dropped["swesmith_verified_repo"] += 1
            continue
        record = rendered(tokenizer, "swesmith", row)
        if len(record["ids"]) > CONTEXT:
            dropped["swesmith_over_context"] += 1
            continue
        record["repo"] = repo
        repo_of[record["task"]] = repo
        records["swesmith"].append(record)
    if len(repo_of) != len(records["swesmith"]):
        raise ValueError("Expected one Kimi trajectory per task")

    by_repo = collections.defaultdict(list)
    for task, repo in sorted(repo_of.items()):
        by_repo[repo].append(task)
    allocation = proportional({repo: len(tasks) for repo, tasks in by_repo.items()}, KIMI_TASKS)
    sampled = {task for repo in sorted(by_repo) for task in rng.sample(by_repo[repo], allocation[repo])}
    records["swesmith"] = [record for record in records["swesmith"] if record["task"] in sampled]

    # Held-out exclusion is by task identity: a trial name is "<task>__<suffix>".
    allkimi_tasks = {trial.rsplit("__", 1)[0] for trial in allkimi_trials}
    heldout, validation, train = {}, {}, {}
    for family in ("tmax", "swesmith"):
        tasks = sorted({record["task"] for record in records[family]})
        candidates = sorted(set(tasks) - allkimi_tasks)
        heldout[family] = sorted(rng.sample(candidates, round(HELDOUT_FRACTION * len(tasks))))
        held = set(heldout[family])
        chosen = {}
        for record in sorted(records[family], key=lambda record: record["trajectory"]):
            if record["task"] in held:
                chosen.setdefault(record["task"], record)
        validation[family] = [chosen[task] for task in heldout[family]]
        train[family] = [record for record in records[family] if record["task"] not in held]
    trained = {record["task"] for record in train["swesmith"]}
    passing = {row["task"] for row in kimi_rows if row["result"] == "1.0"}
    return {
        "allkimi_overlap": {
            "allkimi_tasks": len(allkimi_tasks),
            "allkimi_tasks_passing_in_this_dataset": len(allkimi_tasks & passing),
            "train_tasks": len(trained),
            "train_tasks_in_allkimi": len(trained & allkimi_tasks),
            "heldout_tasks_in_allkimi": len(set(heldout["swesmith"]) & allkimi_tasks),
        },
        "tmax_source_tasks": len({row["metadata"]["task"] for row in tmax_rows}),
        "train": train,
        "validation": validation,
        "heldout": heldout,
        "dropped": dict(dropped),
        "swesmith_allocation": allocation,
        "eligible_tasks": {family: len({r["task"] for r in records[family]}) for family in records},
    }
