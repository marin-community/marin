# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Outcome-blind task selection and held-out split for pool P.

SWE-smith: drop packages whose repo is a SWE-bench Verified repo, then allocate SWESMITH_TASKS across repos in
proportion to the remaining packages (largest remainder) and draw at random within each repo with SEED. No model's
pass/fail is read. The only fixed inclusions are the LR-sweep set's held-out SWE-smith tasks, so the sweep's
validation tasks stay held out in P; they count toward their repo's allocation.

Held-out: round(5%) of each family's pool tasks. The LR-sweep set's held-out ids are kept; the rest are drawn with SEED
from pool tasks that have no trial in allkimi's training list (laion/calibforge-relay-traces kimi_trials.txt), because
v0 carries a graft of that SFT.
"""

import random
import re

SEED = 20261016
SWESMITH_TASKS = 2100
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


def upstream_repo(config_repo: str) -> str:
    """Map a SWE-smith mirror name ('swesmith/<owner>__<name>.<sha8>') to its upstream 'owner/name'."""
    match = re.fullmatch(r"swesmith/(.+?)__(.+)\.[0-9a-f]{8}", config_repo)
    if match is None:
        raise ValueError(f"Unexpected SWE-smith repo name: {config_repo}")
    return f"{match[1]}/{match[2]}"


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


def sample_swesmith(repo_of: dict[str, str], forced: set[str]) -> tuple[list[str], dict[str, int], int]:
    """Return (sampled task ids, per-repo allocation, Verified-repo drops) from {task: upstream 'owner/name'}."""
    if not forced <= set(repo_of):
        raise ValueError(f"Forced tasks missing from the packages: {sorted(forced - set(repo_of))[:5]}")
    kept = {task: repo for task, repo in repo_of.items() if repo not in SWEBENCH_VERIFIED_REPOS}
    if not forced <= set(kept):
        raise ValueError("A forced task is in a SWE-bench Verified repo")
    by_repo: dict[str, list[str]] = {}
    for task, repo in sorted(kept.items()):
        by_repo.setdefault(repo, []).append(task)
    allocation = proportional({repo: len(tasks) for repo, tasks in by_repo.items()}, SWESMITH_TASKS)
    rng = random.Random(SEED)
    sampled = []
    for repo in sorted(by_repo):
        pinned = [task for task in by_repo[repo] if task in forced]
        if len(pinned) > allocation[repo]:
            raise ValueError(f"{repo}: {len(pinned)} forced tasks exceed its allocation {allocation[repo]}")
        rest = [task for task in by_repo[repo] if task not in forced]
        sampled += pinned + rng.sample(rest, allocation[repo] - len(pinned))
    return sorted(sampled), allocation, len(repo_of) - len(kept)


def heldout(pool: list[str], carried: set[str], excluded: set[str]) -> list[str]:
    """round(5%) of `pool`: every carried id, topped up at random from pool tasks not carried and not excluded."""
    target = round(HELDOUT_FRACTION * len(pool))
    if not carried <= set(pool):
        raise ValueError(f"Carried held-out ids missing from the pool: {sorted(carried - set(pool))[:5]}")
    if len(carried) > target:
        raise ValueError(f"{len(carried)} carried held-out ids exceed the {target} target")
    candidates = sorted(set(pool) - carried - excluded)
    return sorted(carried | set(random.Random(SEED).sample(candidates, target - len(carried))))
