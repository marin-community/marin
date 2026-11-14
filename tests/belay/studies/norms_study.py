# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A study script in the contract ``study_checkpoint`` runs: each parameter array's RMS at every saved step, plus the
logged loss, as a report and a figure."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def analyze(checkpoints: Mapping[int, dict[str, np.ndarray]], history: list[dict], out_dir: Path) -> None:
    rms = {
        step: {path: float(np.sqrt(np.mean(np.square(a.astype(np.float64))))) for path, a in params.items()}
        for step, params in checkpoints.items()
    }
    last = max(rms)
    figure = Figure(figsize=(6, 3))
    axis = figure.subplots()
    for path in rms[last]:
        axis.plot(list(rms), [rms[step][path] for step in rms])
    axis.set_xlabel("step")
    axis.set_ylabel("RMS")
    figure.savefig(out_dir / "rms.png")
    losses = [row["train/loss"] for row in history if "train/loss" in row]
    rows = "\n".join(f"| `{path}` | {value:.4g} |" for path, value in sorted(rms[last].items()))
    (out_dir / "report.md").write_text(
        f"# Parameter RMS\n\n{len(rms[last])} arrays at {len(rms)} steps; "
        f"logged loss {losses[0]:.3f} to {losses[-1]:.3f}."
        f"\n\n![RMS per array over training](rms.png)\n\n| Array | RMS at step {last} |\n|---|---|\n{rows}\n"
    )
