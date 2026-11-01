# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two public #9451 post formats, plan and result. Every other post (escalations) stays on the admin page.

Each post's title is ``X#### · <short name> · <track> · <status>`` (``post_activity`` renders it as the bold first
line after the bot mark), so the thread scans by registry id.
"""

from dataclasses import dataclass

from experiments.belay.config import CONFIG, TRACKS, Rule, Track

MINUS = "\u2212"  # the typographic minus of the public format


@dataclass(frozen=True)
class IdeaRef:
    """Where an idea came from, for the last line of every post."""

    url: str | None
    author: str
    approved_by: str | None
    registry_url: str


@dataclass(frozen=True)
class PlannedRun:
    name: str
    variant: str
    seed: int
    flags: str


@dataclass(frozen=True)
class ResultRow:
    name: str
    variant: str
    seed: int
    value: float
    delta: float | None


def title(registry_id: str, short_name: str, track: Track, status: str) -> str:
    return f"{registry_id} · {short_name} · {track} · {status}"


def wandb_link(run: str) -> str:
    return f"[{run}](https://wandb.ai/{CONFIG.wandb_project}/runs/{run})"


def rule_line(track: Track) -> str:
    baseline = TRACKS[track]
    unit = "%" if baseline.rule is Rule.RELATIVE else ""
    noise = f"{baseline.noise * 100:g}" if baseline.rule is Rule.RELATIVE else f"{baseline.noise:g}"
    return f"Rule: Δ ≤ {MINUS}{noise}{unit} win · ≥ +{noise}{unit} loss · otherwise noise"


def idea_line(ref: IdeaRef, commit: str) -> str:
    source = f"[#9514 comment]({ref.url}) by @{ref.author}" if ref.url else f"proposal by {ref.author}"
    approved = f" · approved by @{ref.approved_by}" if ref.approved_by else ""
    return f"Idea: {source}{approved} · commit `{commit[:10]}` · [registry]({ref.registry_url})"


def delta_text(delta: float | None) -> str:
    return "" if delta is None else f"{delta:+.4f}"


def plan_body(track: Track, runs: list[PlannedRun], defaults: str, ref: IdeaRef, commit: str) -> str:
    rows = "\n".join(f"| {wandb_link(r.name)} | {r.variant} | {r.seed} | {r.flags} |" for r in runs)
    return (
        f"| Run | Variant | Seed | Settings |\n|---|---|---|---|\n{rows}\n\n"
        f"Baseline: {TRACKS[track].description}\n"
        f"Defaults chosen: {defaults}\n"
        f"{rule_line(track)}\n"
        f"{idea_line(ref, commit)}"
    )


def result_body(track: Track, rows: list[ResultRow], ref: IdeaRef, commit: str, analysis: tuple[str, str]) -> str:
    """``analysis`` is the report's (label, url), e.g. ("X0217 t005", <GitHub blob link>)."""
    metric = "Paloma macro loss" if TRACKS[track].metric.startswith("eval/paloma") else TRACKS[track].metric
    lines = "\n".join(
        f"| {wandb_link(r.name)} | {r.variant} | {r.seed} | {r.value:.4f} | {delta_text(r.delta)} |" for r in rows
    )
    return (
        f"| Run | Variant | Seed | {metric} | Δ vs control |\n|---|---|---|---|---|\n{lines}\n\n"
        f"{rule_line(track)}\n"
        f"{idea_line(ref, commit)}\n"
        f"Analysis: [{analysis[0]}]({analysis[1]})"
    )
