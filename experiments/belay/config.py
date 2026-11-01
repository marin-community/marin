# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fixed Marin Belay constants: branches, cluster, budget, metric and the win/noise/loss rule.

Every Belay script reads ``CONFIG`` from here and nowhere else. Change a value here (and only here) when a
maintainer changes the rules; ``experiments/belay/RUNBOOK.md`` section 5 lists the steps for a new recipe.
"""

import datetime
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class BelayConfig:
    recipe_name: str
    """The accepted recipe every experiment is measured against."""
    recipe_branch: str
    """The accepted recipe's code; read-only for sessions."""
    base_branch: str
    """Every weekly experiment branch must contain this branch's tip."""
    week_branch_prefix: str
    """The weekly experiment branch is ``<prefix><monday as YYYY-MM-DD>``."""
    registry_branch: str
    registry_checkout: Path
    """Local checkout of ``registry_branch`` that the commands write to."""
    launcher: str
    """The recipe's name in run specs (``launcher:``); a spec naming anything else is refused at launch."""
    launch_module: tuple[str, ...]
    """``python -m`` arguments that launch the recipe; ``--size``, the spec's flags and run options follow."""
    rung: str
    seed: int
    rung_minutes: int
    """Wall time of one run at ``rung`` on one 8xH100 node, for the cost line of a plan."""
    cluster: str
    band: str
    node_cap: int
    """Soft cap on active Belay runs; every run at ``rung`` takes one 8xH100 node."""
    output_root: str
    metric: str
    noise: float
    """The win/noise/loss threshold on Δ = variant - control (same seed and commit): Δ ≤ -noise is a win,
    Δ ≥ +noise a loss, anything between is noise. 0.006 is the d512 seed-to-seed spread."""
    wandb_project: str
    repository: str
    batch_issue: int
    """Where plans and results are posted when GitHub posting is on."""
    page_url: str

    def week_branch(self, day: datetime.date) -> str:
        monday = day - datetime.timedelta(days=day.weekday())
        return f"{self.week_branch_prefix}{monday.isoformat()}"


PRODUCTION = BelayConfig(
    recipe_name="Belay Ckpt Oct 20 2026",
    recipe_branch="marin_belay_v1_clean_s7",
    base_branch="marin_belay_v1_clean_s7",
    week_branch_prefix="marin_belay_v1_w",
    registry_branch="marin_belay_v1_registry",
    registry_checkout=Path.home() / "marin_wt_registry",
    launcher="s7n",
    launch_module=("experiments.grug.fast_track.belay_s7", "--ladder", "s7n"),
    rung="d512",
    seed=0,
    rung_minutes=38,
    cluster="cw-us-east-02a",
    band="batch",
    node_cap=8,
    output_root="tmp/ttl=30d/belay/runs",
    metric="eval/paloma/macro_loss",
    noise=0.006,
    wandb_project="marin-community/marin_moe",
    repository="marin-community/marin",
    batch_issue=9451,
    page_url="https://marina.oa.dev/belay/",
)

# The local demo: same rules, but its weekly branches are cut from the demo code branch (which carries the Belay
# tooling on top of the recipe), and its registry is a demo copy.
DEMO = replace(
    PRODUCTION,
    base_branch="marin_belay_v1_loomdemo",
    week_branch_prefix="marin_belay_v1_loomdemo_w",
    registry_branch="marin_belay_v1_registry_demo",
    registry_checkout=Path.home() / "marin_wt_registry_demo",
    page_url="http://127.0.0.1:8091/belay/",
)

CONFIG = DEMO
