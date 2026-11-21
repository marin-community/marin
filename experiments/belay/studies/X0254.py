# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""X0254: a same-commit rerun of the recipe at d1280 against the cross-commit s7n baseline.

Every competitive d1280 rung is judged against the finished run ``s7n-d1280L16`` (Paloma macro 2.2706), trained on
another commit. The ladder projection weights the d1280 difference by about 0.980, so an offset between this rerun
and that baseline passes almost fully into every ladder's projected verdict. An offset of +0.003 or more means every
ladder's d1280 difference is biased toward a loss by that amount; one within 0.001 rules a bias out.

The report reads the rerun's logged Paloma macro loss from its W&B history; it reads no checkpoint.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

PALOMA_MACRO = "eval/paloma/macro_loss"
BASELINE_RUN = "s7n-d1280L16"
BASELINE_LOSS = 2.2706
# First-order weight of the d1280 difference in the ladder projection to hero compute.
D1280_PROJECTION_WEIGHT = 0.980
BIAS_THRESHOLD = 0.003
MATCH_THRESHOLD = 0.001
FIGURE = "paloma_macro.png"


def paloma_curve(history: list[dict]) -> list[tuple[int, float]]:
    """(step, Paloma macro loss) for every logged evaluation, in step order."""
    points = [(int(row["_step"]), float(row[PALOMA_MACRO])) for row in history if row.get(PALOMA_MACRO) is not None]
    return sorted(points)


def verdict(offset: float) -> str:
    if abs(offset) <= MATCH_THRESHOLD:
        return f"within {MATCH_THRESHOLD} of the baseline: no d1280 baseline bias"
    if offset >= BIAS_THRESHOLD:
        return (
            f"{offset:+.4f} above the baseline: every ladder's d1280 difference is biased toward a loss by about "
            "this much"
        )
    if offset <= -BIAS_THRESHOLD:
        return f"{offset:+.4f} below the baseline: every ladder's d1280 difference is biased toward a win"
    return f"{offset:+.4f}: between the match and bias thresholds; inconclusive"


def plot(curve: list[tuple[int, float]], out_dir: Path) -> None:
    steps, losses = np.array(curve).T
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.plot(steps, losses, marker="o", label="rerun")
    ax.axhline(BASELINE_LOSS, color="gray", linestyle="--", label=f"{BASELINE_RUN} final")
    ax.set_xlabel("step")
    ax.set_ylabel("Paloma macro loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / FIGURE)


def analyze(checkpoints: Mapping[int, Mapping[str, np.ndarray]], history: list[dict], out_dir: Path) -> None:
    curve = paloma_curve(history)
    lines = [
        "# X0254: same-commit recipe rerun at d1280",
        "",
        f"Baseline: `{BASELINE_RUN}`, final Paloma macro loss {BASELINE_LOSS:.4f} (another commit).",
        "",
    ]
    if not curve:
        lines.append(f"The run logged no `{PALOMA_MACRO}`; no comparison is possible.")
        (out_dir / "report.md").write_text("\n".join(lines) + "\n")
        return
    final_step, final_loss = curve[-1]
    offset = final_loss - BASELINE_LOSS
    lines += [
        "| | Value |",
        "|---|---|",
        f"| Rerun final Paloma macro (step {final_step}) | {final_loss:.4f} |",
        f"| Rerun - baseline | {offset:+.4f} |",
        f"| Implied shift of every ladder projection (-{D1280_PROJECTION_WEIGHT} * offset) | "
        f"{-D1280_PROJECTION_WEIGHT * offset:+.4f} |",
        "",
        f"Result: {verdict(offset)}.",
        "",
        "Measured against this rerun instead of the baseline, each ladder's d1280 difference moves by "
        f"{-offset:+.4f} and its projected difference by {-D1280_PROJECTION_WEIGHT * offset:+.4f}.",
        "",
        f"![Paloma macro loss over training]({FIGURE})",
    ]
    plot(curve, out_dir)
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
