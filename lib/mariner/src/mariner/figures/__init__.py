# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The paper's main-text figures, redrawn in a plain layout from a mariner-data snapshot.

Each redraw reads only the snapshot's tables, and fits this package's models and, for Figure 1's surrogate panel,
LightGBM's boosted trees. It shows the paper figure's data, fitted curves and comparisons in the shared layout of
``mariner.figures.layout``, and returns every series it plots. ``write_figures`` writes them as PDFs, as ``mariner
figures`` does. The package needs matplotlib and LightGBM, which marin-mariner's figures extra installs.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from mariner.figures import analysis, epochs, pipeline, results, surrogate
from mariner.figures.layout import FigureData, save, style

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FigureSpec:
    """A redraw: the paper's figure label, the PDF's file stem, the dataset configs it reads, and how to draw it.

    ``tables`` names configs as the snapshot's ``figures_tables.csv`` does, without its ``constant`` and ``deferred``
    markers.
    """

    label: str
    stem: str
    tables: tuple[str, ...]
    draw: Callable[[Path], FigureData]


FIGURES: dict[str, FigureSpec] = {
    spec.label: spec
    for spec in (
        FigureSpec("fig:pipeline", "f1_pipeline", ("starcoder", "ladder", "ladder_accuracy"), pipeline.draw_pipeline),
        FigureSpec("fig:epoch-matching", "epoch_matching_tpp10", ("starcoder",), epochs.draw_epoch_matching),
        FigureSpec("fig:b5", "b5_matched_compute_scaling", ("starcoder",), epochs.draw_matched_compute_scaling),
        FigureSpec("fig:b6", "b6_olmix_capacity_failure", ("starcoder",), surrogate.draw_olmix_capacity_failure),
        FigureSpec("fig:mariner-terms", "m_mariner_terms", (), surrogate.draw_mariner_terms),
        FigureSpec("fig:r6", "r6_learning_curve_per_fold_ratio", ("fits",), surrogate.draw_learning_curves),
        FigureSpec(
            "fig:proposal-predictions",
            "r_proposal_predictions",
            ("validation_3e18", "fits"),
            surrogate.draw_proposal_predictions,
        ),
        FigureSpec("fig:accuracy", "r_accuracy_vs_bpb", ("ladder_accuracy",), results.draw_accuracy_vs_bpb),
        # Figures 9 and 10 read the ladder alone: its 3e18 rung holds the runs of the other configs their rows of
        # figures_tables.csv name, the main swarm's proportional repeats and the validation launches' runs.
        FigureSpec(
            "fig:r1",
            "r1_scaling_curves_placeholder",
            ("ladder", "validation_3e18", "swarm_qwen3_360m_1p6b"),
            results.draw_scaling_curves,
        ),
        FigureSpec(
            "fig:off-objective", "r_on_off_objective", ("ladder", "validation_3e18"), results.draw_on_off_objective
        ),
        FigureSpec(
            "fig:component-tradeoffs",
            "r8_component_tradeoffs_4panel",
            ("validation_3e18", "fits"),
            results.draw_component_tradeoffs,
        ),
        FigureSpec("fig:domain-repetition", "figure4_cap_policy", ("domain_sweeps",), analysis.draw_domain_repetition),
        FigureSpec(
            "fig:mixture-epochs", "r_mixture_epochs", ("validation_3e18", "partition"), analysis.draw_mixture_epochs
        ),
        FigureSpec("fig:r7", "r7_r2_vs_snr", ("fits",), analysis.draw_r2_vs_snr),
    )
}


def write_figures(root: Path, out: Path, labels: Sequence[str] | None = None) -> list[Path]:
    """Draw the figures ``labels`` (every figure when None) from the snapshot at ``root`` as ``<stem>.pdf`` in ``out``.

    Raises ``ValueError`` naming the labels that no figure has before it draws anything.
    """
    selected = list(FIGURES) if labels is None else list(dict.fromkeys(labels))
    unknown = [label for label in selected if label not in FIGURES]
    if unknown:
        raise ValueError(f"no figure is labelled {', '.join(unknown)}; the figures are {', '.join(FIGURES)}")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for label in selected:
        spec = FIGURES[label]
        path = out / f"{spec.stem}.pdf"
        with style():
            save(spec.draw(root).figure, path)
        logger.info("wrote %s (%s)", path, label)
        paths.append(path)
    return paths
