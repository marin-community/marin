# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot observed mixture weights for the strongest upward and downward rank movers."""

from __future__ import annotations

from pathlib import Path

from experiments.domain_phase_mix.exploratory.two_phase_many.plot_run00125_vs_run00018_weights import (
    COLOR_MAP,
    build_two_run_weights_plot,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "run0050_vs_run0092_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / "run0050_vs_run0092_weights.csv"
RUN_NAMES = ("run_00050", "run_00092")
RUN_COLORS = {
    "run_00050": COLOR_MAP(0.12),
    "run_00092": COLOR_MAP(0.88),
}


def main() -> None:
    build_two_run_weights_plot(
        run_names=RUN_NAMES,
        output_png=PLOT_PNG,
        output_csv=WEIGHTS_CSV,
        run_colors=RUN_COLORS,
    )


if __name__ == "__main__":
    main()
