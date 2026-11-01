# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The one #9451 comment per idea: built here, posted once, then edited in place as the idea moves on.

    🤖 **X#### · <name> · <track> · <status>**

    **Stage 1. Comparison: <what is held equal>.**
    <table>

    **Stage 2. ...**            (later stages and retests add sections; an existing section is replaced)

    Defaults chosen: ...
    Rule: ...
    Idea: ...
    Analysis: [X#### t001](...) · [X#### t002](...)

Every table has the same columns, Steps always among them, so each comparison can be checked from the comment.
"""

import re
from dataclasses import dataclass

from experiments.belay.config import CONFIG, HERO_MFU_BRANCH, TRACKS, Rule, Track

BOT_MARK = "🤖"
MINUS = chr(0x2212)  # the typographic minus of the public format
TIMES = chr(0x00D7)
COLUMNS = (
    "Run",
    "Variant",
    "Seed",
    "Hardware",
    "Steps",
    "s/step",
    "Time",
    "Paloma macro loss",
    "Δ vs control",
    "Paloma macro BPB",
)
STAGE_HEADING = re.compile(r"^\*\*Stage (\d+)\.")
FOOTER_START = "Defaults chosen:"


@dataclass(frozen=True)
class Row:
    """One run in a table. A plan row knows only its steps; a result row has the measurements too."""

    name: str
    variant: str
    seed: int
    hardware: str
    steps: int
    s_step: float | None = None
    """Measured median s/step, or for a plan the expected s/step when the Track knows it."""
    time_change: float | None = None
    """Train time relative to the control, e.g. 0.31 for +31%; None for the control and for plans."""
    loss: float | None = None
    delta: float | None = None
    bpb: float | None = None
    """Byte-weighted Paloma macro BPB, shown for reference; verdicts stay on loss."""


@dataclass(frozen=True)
class Comment:
    title: str
    stages: dict[int, str]
    footer: list[str]


def control_label(track: Track, commit: str) -> str:
    """The control row's variant cell: what the control is, linked to its exact code."""
    baseline = TRACKS[track]
    repo = f"https://github.com/{CONFIG.repository}"
    if track.value.startswith(("vanilla-", "moe-")):
        return f"control: [{baseline.control_name}]({repo}/blob/{commit}/experiments/grug/fast_track/belay_tracks.py)"
    if baseline.code == HERO_MFU_BRANCH:
        return f"control: [{baseline.control_name}]({repo}/tree/{commit})"
    recipe = f"[{CONFIG.recipe_name}]({repo}/tree/{CONFIG.recipe_tag})"
    return f"control: current recipe, {recipe} ({baseline.control_name})"


def hardware(track: Track) -> str:
    """The table's Hardware cell for one run of ``track``: its GPUs, e.g. 8xH100 or 64xGB200 (with a times sign)."""
    baseline = TRACKS[track]
    if not baseline.hardware:
        return "—"
    per_node, gpu = baseline.hardware.split("(")[0].strip().split("x", 1)
    return f"{baseline.nodes * int(per_node)}{TIMES}{gpu}"


def title(registry_id: str, short_name: str, track: Track, status: str) -> str:
    return f"{registry_id} · {short_name} · {track} · {status}"


def signed(value: float, digits: int = 4) -> str:
    text = f"{value:+.{digits}f}"
    return text.replace("-", MINUS)


def rule_line(track: Track) -> str:
    baseline = TRACKS[track]
    if baseline.rule is Rule.RELATIVE:
        noise = f"{baseline.noise * 100:g}%"
    else:
        noise = f"{baseline.noise:g}"
    line = f"Rule: Δ ≤ {MINUS}{noise} win · ≥ +{noise} loss · otherwise noise."
    if baseline.metric.startswith("eval/paloma"):
        line += " Verdict on loss; BPB (byte-weighted) shown for reference."
    if baseline.train_seconds:
        line += " The final speedrun verdict is the equal-time verdict."
    return line


def table(rows: list[Row]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        link = f"[{r.name}](https://wandb.ai/{CONFIG.wandb_project}/runs/{r.name})"
        train = "—"
        if r.s_step is not None:
            train = f"{r.steps * r.s_step:.0f} s" if r.loss is not None else f"~{r.steps * r.s_step:.0f} s"
            if r.time_change is not None:
                train += f" ({signed(r.time_change * 100, 0)}%)"
        cells = [
            link,
            r.variant,
            str(r.seed),
            r.hardware,
            f"{r.steps:,}",
            "" if r.s_step is None else f"{r.s_step:.4f}",
            train,
            "" if r.loss is None else f"{r.loss:.4f}",
            "" if r.delta is None else signed(r.delta),
            "" if r.bpb is None else f"{r.bpb:.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stage_section(number: int, comparison: str, rows: list[Row], note: str = "") -> str:
    section = f"**Stage {number}. Comparison: {comparison}.**"
    if rows:
        section += "\n\n" + table(rows)
    if note:
        section += "\n\n" + note
    return section


def render(comment: Comment) -> str:
    parts = [f"{BOT_MARK} **{comment.title}**", *(comment.stages[n] for n in sorted(comment.stages))]
    return "\n\n".join(parts) + "\n\n" + "\n".join(comment.footer)


def parse(body: str) -> Comment:
    """A posted comment back into its parts, so an edit replaces one stage and keeps the rest."""
    lines = body.strip().splitlines()
    head = lines[0].removeprefix(BOT_MARK).strip().strip("*")
    stages: dict[int, list[str]] = {}
    footer: list[str] = []
    current: list[str] | None = None
    for line in lines[1:]:
        match = STAGE_HEADING.match(line)
        if match:
            current = stages.setdefault(int(match.group(1)), [])
        elif line.startswith(FOOTER_START) or footer:
            footer.append(line)
            current = None
            continue
        if current is not None:
            current.append(line)
    return Comment(head, {n: "\n".join(block).strip() for n, block in stages.items()}, footer)


def with_reproduce(body: str, lines: dict[int, str]) -> str:
    """``body`` with a Reproduce line added to each stage section in ``lines`` that has none yet."""
    comment = parse(body)
    stages = dict(comment.stages)
    for number, line in lines.items():
        if number in stages and "Reproduce:" not in stages[number]:
            stages[number] += "\n\n" + line
    return render(Comment(comment.title, stages, comment.footer))


def updated(body: str, new_title: str, number: int, section: str, analysis: tuple[str, str] | None = None) -> str:
    """``body`` with its title replaced, stage ``number`` set to ``section``, and an analysis link added."""
    comment = parse(body)
    footer = list(comment.footer)
    if analysis is not None:
        link = f"[{analysis[0]}]({analysis[1]})"
        index = next((i for i, line in enumerate(footer) if line.startswith("Analysis:")), None)
        if index is None:
            footer.append(f"Analysis: {link}")
        elif link not in footer[index]:
            footer[index] += f" · {link}"
    stages = {n: text.replace("| Train time |", "| Time |") for n, text in comment.stages.items()}
    return render(Comment(new_title, {**stages, number: section}, footer))


def footer(defaults: str, track: Track, idea: str, commit: str, registry_url: str) -> list[str]:
    return [
        f"Defaults chosen: {defaults}",
        rule_line(track),
        f"Idea: {idea} · commit `{commit[:10]}` · [registry]({registry_url})",
    ]
