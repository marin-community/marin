# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fixed Marin Belay constants: branches, cluster, budget, and each Track's baseline, metric and noise band.

Every Belay script reads ``CONFIG`` from here and nowhere else. Change a value here (and only here) when a
maintainer changes the rules; ``experiments/belay/RUNBOOK.md`` section 5 lists the steps for a new recipe.
"""

import datetime
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

CHECKPOINT_TAG_PREFIX = "belay-ckpt-"


@dataclass(frozen=True)
class BelayConfig:
    recipe_branch: str
    """The adopted recipe's code. Only ``recipe-change`` (run by a maintainer) moves it."""
    recipe_tag: str
    """The annotated tag ``belay-ckpt-<YYYY-MM-DD>`` that ``recipe-change`` put at the recipe's tip."""
    base_branch: str
    """Every weekly experiment branch must contain this branch's tip."""
    week_branch_prefix: str
    """The weekly experiment branch is ``<prefix><monday as YYYY-MM-DD>``."""
    registry_branch: str
    registry_checkout: Path
    """Local checkout of ``registry_branch`` that the commands write to."""
    seed: int
    node_cap: int
    """Soft cap on 8xH100 nodes held by active Belay runs."""
    output_root: str
    wandb_project: str
    repository: str
    batch_issue: int
    """Where plans and results are posted when GitHub posting is on."""
    page_url: str
    github_approval_cutover: datetime.datetime
    """A #9514 comment the intake bot transfers (🚀) after this queues the idea by itself; earlier ones need the
    admin page's Queue, so the backlog does not flood the queue."""
    default_author: str
    """GitHub login credited for ideas with no #9514 comment and no recorded proposer (internal proposals)."""

    @property
    def recipe_name(self) -> str:
        """``Belay Ckpt <Mon DD YYYY>``, from the recipe tag's date."""
        day = datetime.date.fromisoformat(self.recipe_tag.removeprefix(CHECKPOINT_TAG_PREFIX))
        return f"Belay Ckpt {day:%b %d %Y}"

    def week_branch(self, day: datetime.date) -> str:
        monday = day - datetime.timedelta(days=day.weekday())
        return f"{self.week_branch_prefix}{monday.isoformat()}"


PRODUCTION = BelayConfig(
    recipe_branch="marin_belay_v1_recipe",
    recipe_tag="belay-ckpt-2026-10-20",
    base_branch="marin_belay_v1_recipe",
    week_branch_prefix="marin_belay_v1_w",
    registry_branch="marin_belay_v1_registry",
    registry_checkout=Path.home() / "marin_wt_registry",
    seed=0,
    node_cap=8,
    output_root="tmp/ttl=30d/belay/runs",
    wandb_project="marin-community/marin_moe",
    repository="marin-community/marin",
    batch_issue=9451,
    page_url="https://marina.oa.dev/belay/",
    github_approval_cutover=datetime.datetime(2026, 11, 1, 8, 26, tzinfo=datetime.UTC),
    default_author="ClassicLarry",
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


PALOMA_MACRO = "eval/paloma/macro_loss"
# The noise band on Paloma macro loss (maintainer setting, 2026-11-01), used at every loss rung.
LOSS_NOISE = 0.005
S7 = "experiments.grug.fast_track.belay_s7"
TRACKS_MODULE = "experiments.grug.fast_track.belay_tracks"


class Track(StrEnum):
    """The baseline an idea is built on and compared against, named on its intake comment's ``Track:`` line."""

    VANILLA_D512 = "vanilla-d512"
    VANILLA_D768 = "vanilla-d768"
    VANILLA_D1024 = "vanilla-d1024"
    VANILLA_D1280 = "vanilla-d1280"
    MOE_D512 = "moe-d512"
    MOE_D768 = "moe-d768"
    MOE_D1024 = "moe-d1024"
    MOE_D1280 = "moe-d1280"
    COMPETITIVE_D512 = "competitive-d512"
    COMPETITIVE_D768 = "competitive-d768"
    COMPETITIVE_D1024 = "competitive-d1024"
    COMPETITIVE_D1280L16 = "competitive-d1280L16"
    SPEEDRUN = "speedrun"
    HERO_MFU = "hero-mfu"
    CUSTOM = "custom"


class Comparison(StrEnum):
    """What a Track holds equal between the variant and its control; every post states it above its table."""

    EQUAL_DATA = "equal data"
    """Both arms train on the same tokens (the rung's steps); wall time is shown."""
    EQUAL_DATA_THEN_TIME = "equal data, then equal time"
    """Stage 1 at equal data; then, by ``results.after_equal_data``, stage 2 at equal training time
    (``train_seconds``), whose verdict is final."""
    FIXED_CONFIG = "fixed config"
    """Same config and hardware; compared on step time."""


class Rule(StrEnum):
    """How Δ is measured; lower is better for every metric."""

    DIFFERENCE = "difference"
    """Δ = variant - control."""
    RELATIVE = "relative"
    """Δ = (variant - control) / control."""


H100_CLUSTER = "cw-us-east-02a"
GB200_CLUSTER = "cw-us-east-08a"
# The weekly experiment branch holds the code for every H100 track.
WEEKLY_BRANCH = "weekly"
HERO_MFU_BRANCH = "marin_belay_v1_hero_mfu"
# On HERO_MFU_BRANCH: the launcher that renders the measured mfl-mp-qall stack (c6d473c243; PGLE mfl-mp-m7g.pb, slop
# 112, the S7 settings plus the qall extras) for one GB200 NVL72 rack; `--plan` prints it, `--accept <run>` checks a run.
HERO_MFU_LAUNCH = ("experiments.grug.fast_track.belay_hero_mfu", "--stack", "mfl-mp-qall")
HERO_MFU_NODES = 16  # replicas of 4xGB200, as mfl-mp-qall ran
CONTROL_RUN = "belay-<X####>-control"


@dataclass(frozen=True)
class Baseline:
    """Everything about one Track's runs: where its code lives, the exact launch, the hardware and the rule."""

    description: str
    code: str
    """The branch the runs' code is on: ``WEEKLY_BRANCH`` (this week's branch) or a named branch."""
    launch: tuple[str, ...] | None
    """``python -m`` arguments for the baseline run; the variant adds its flags. None: no launch recipe."""
    cluster: str
    nodes: int
    hardware: str
    band: str
    metric: str
    rule: Rule
    noise: float
    """Δ ≤ -noise is a win, Δ ≥ +noise a loss, anything between is noise."""
    needs_go_ahead: bool
    """A maintainer approves each launch (it shares capacity with production runs); sessions escalate it."""
    control_name: str = ""
    """How a post names the control: "speedrun d512", "Basic MoE d768", "hero MFU stack mfl-mp-qall"."""
    comparison: Comparison = Comparison.EQUAL_DATA
    train_seconds: float | None = None
    """Stage 2 of ``EQUAL_DATA_THEN_TIME``: seconds of training per arm (the control's steps x its s/step)."""
    baseline_steps: int | None = None
    """The control's steps (stage 1 for both arms)."""
    step_time_tolerance: float = 0.01
    """Within this relative step-time difference, equal data already is equal time (no stage 2)."""
    output_path: bool = True
    """The launcher takes ``--output-path`` (Belay writes under ``CONFIG.output_root``)."""
    seed_run: tuple[str, ...] = ()
    """Extra arguments of a one-off run that must precede the first timed run of a program (an autotune seed)."""


H100 = "8xH100"
# Steps per rung, as the launchers set them (belay_tracks.RUNGS and belay_s7.RUNGS); posts show them in every table.
TRACK_STEPS = {
    "vanilla": {"d512": 720, "d768": 2161, "d1024": 2881, "d1280": 5251},
    "moe": {"d512": 1971, "d768": 5547, "d1024": 7334, "d1280": 13053},
}
COMPETITIVE_STEPS = {"d512": 10767, "d768": 30389, "d1024": 39210, "d1280L16": 20023}


def _vanilla_or_moe(track: str, size: str) -> Baseline:
    name = "Vanilla Track" if track == "vanilla" else "Basic MoE"
    return Baseline(
        f"{name} at {size}",
        WEEKLY_BRANCH,
        (TRACKS_MODULE, "--track", track, "--size", size),
        H100_CLUSTER,
        1,
        H100,
        "batch",
        PALOMA_MACRO,
        Rule.DIFFERENCE,
        LOSS_NOISE,
        needs_go_ahead=False,
        control_name=f"{name} {size}",
        baseline_steps=TRACK_STEPS[track][size],
    )


def _competitive(size: str, nodes: int) -> Baseline:
    return Baseline(
        f"{PRODUCTION.recipe_name} (s7n) at {size}",
        WEEKLY_BRANCH,
        (S7, "--ladder", "s7n", "--size", size),
        H100_CLUSTER,
        nodes,
        H100,
        "batch",
        PALOMA_MACRO,
        Rule.DIFFERENCE,
        LOSS_NOISE,
        needs_go_ahead=False,
        control_name=f"competitive {size}",
        baseline_steps=COMPETITIVE_STEPS[size],
    )


# The one place every Track's launch is defined. `uv run python -m experiments.belay tracks` prints it; RUNBOOK.md
# section 9 and the page's track cards (web/tracks.json, generated by `tracks --json`) quote it.
TRACKS: dict[Track, Baseline] = {
    Track.VANILLA_D512: _vanilla_or_moe("vanilla", "d512"),
    Track.VANILLA_D768: _vanilla_or_moe("vanilla", "d768"),
    Track.VANILLA_D1024: _vanilla_or_moe("vanilla", "d1024"),
    Track.VANILLA_D1280: _vanilla_or_moe("vanilla", "d1280"),
    Track.MOE_D512: _vanilla_or_moe("moe", "d512"),
    Track.MOE_D768: _vanilla_or_moe("moe", "d768"),
    Track.MOE_D1024: _vanilla_or_moe("moe", "d1024"),
    Track.MOE_D1280: _vanilla_or_moe("moe", "d1280"),
    Track.COMPETITIVE_D512: _competitive("d512", 1),
    Track.COMPETITIVE_D768: _competitive("d768", 1),
    Track.COMPETITIVE_D1024: _competitive("d1024", 1),
    Track.COMPETITIVE_D1280L16: _competitive("d1280L16", 4),
    Track.SPEEDRUN: Baseline(
        "8-minute d512 run of the current recipe; baseline belay-8min-ckpt-oct20: 3,030 steps, Paloma macro 3.0441, "
        "BPB 1.2590 (measured at 0dd84c79f9)",
        WEEKLY_BRANCH,
        (S7, "--ladder", "s7n", "--size", "d512", "--num-steps", "3030"),
        H100_CLUSTER,
        1,
        H100,
        "batch",
        PALOMA_MACRO,
        Rule.DIFFERENCE,
        LOSS_NOISE,
        needs_go_ahead=False,
        comparison=Comparison.EQUAL_DATA_THEN_TIME,
        train_seconds=480.0,
        baseline_steps=3030,
        control_name="speedrun d512",
    ),
    # Rack-to-rack step-time noise is about 20-35 ms (0.2-0.3%). A result counts only against a control on the same
    # rack, at the same commit, with the same PGLE profile; a win is a step at least 0.5% faster.
    Track.HERO_MFU: Baseline(
        "One-rack GB200 MFU stack mfl-mp-qall: 11.465 s/step, 36.8% MFU",
        HERO_MFU_BRANCH,
        HERO_MFU_LAUNCH,
        GB200_CLUSTER,
        HERO_MFU_NODES,
        "4xGB200 (one NVL72 rack)",
        "production",
        "s/step",
        Rule.RELATIVE,
        0.005,
        needs_go_ahead=True,
        comparison=Comparison.FIXED_CONFIG,
        control_name="hero MFU stack mfl-mp-qall",
        baseline_steps=30,
        output_path=False,
        seed_run=("--seed-xla-autotune-cache",),
    ),
    Track.CUSTOM: Baseline(
        "The submitter states the baseline",
        WEEKLY_BRANCH,
        None,
        "",
        0,
        "",
        "",
        "stated by the submitter",
        Rule.DIFFERENCE,
        0,
        needs_go_ahead=True,
    ),
}
