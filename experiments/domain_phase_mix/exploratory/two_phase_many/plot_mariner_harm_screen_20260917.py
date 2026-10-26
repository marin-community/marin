# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the anchor screen with equal weight for each panel/objective group."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_harm_screen_models_20260917 as forms


def render(report: Path) -> None:
    table = pd.read_csv(report / "screen_by_panel_objective.csv")
    ranking = pd.read_csv(report / "screen_ranking.csv")
    columns = [
        (panel, target)
        for panel in ("delphi_3e18_39bucket", "300m_39bucket", "60m_39bucket")
        for target in ("uncheatable", "table9")
    ]
    matrix = table.pivot(index="model", columns=["panel", "target"], values="delta_spearman")
    matrix = -matrix.reindex(index=ranking.model, columns=columns)
    values = matrix.to_numpy(dtype=float)
    labels = {spec.model_id: spec.label for spec in forms.candidate_specs()}
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "pdf.fonttype": 42,
            "text.usetex": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.grid": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(10.4, 7.6), layout="constrained")
        limit = max(float(np.abs(matrix.to_numpy()).max()), 0.005)
        cells = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_yticks(range(len(matrix)), [labels[name] for name in matrix.index])
        ax.set_xticks(
            range(6),
            [
                "Qwen\nUnch. anchors",
                "Qwen\nSuite anchors",
                "Llama 200M\nUnch. anchors",
                "Llama 200M\nSuite anchors",
                "Llama 160M\nUnch. anchors",
                "Llama 160M\nSuite anchors",
            ],
        )
        ax.tick_params(length=0)
        for row in range(len(matrix)):
            for column in range(6):
                value = float(values[row, column])
                ax.text(
                    column,
                    row,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if abs(value) > 0.75 * limit else "#17212b",
                )
        ax.set_title(
            "Harm family screen: loss of held-out rank correlation relative to MARINER\n"
            "Six anchor components per swarm; negative values are better",
            loc="left",
            pad=15,
            fontsize=12,
        )
        fig.colorbar(cells, ax=ax, shrink=0.6, label="Spearman loss")
        for suffix in ("png", "pdf"):
            fig.savefig(report / f"screen_heatmap.{suffix}", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    render(parser.parse_args().report)


if __name__ == "__main__":
    main()
