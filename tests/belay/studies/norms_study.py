# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A study script in the contract ``study_checkpoint`` runs: each parameter array's RMS, as a report and a figure."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def analyze(params: dict[str, np.ndarray], out_dir: Path) -> None:
    rms = {path: float(np.sqrt(np.mean(np.square(array.astype(np.float64))))) for path, array in params.items()}
    figure = Figure(figsize=(6, 3))
    axis = figure.subplots()
    axis.bar(range(len(rms)), list(rms.values()))
    axis.set_xlabel("array")
    axis.set_ylabel("RMS")
    figure.savefig(out_dir / "rms.png")
    rows = "\n".join(f"| `{path}` | {value:.4g} |" for path, value in sorted(rms.items()))
    (out_dir / "report.md").write_text(
        f"# Parameter RMS\n\n{len(rms)} arrays.\n\n![RMS per array](rms.png)\n\n| Array | RMS |\n|---|---|\n{rows}\n"
    )
