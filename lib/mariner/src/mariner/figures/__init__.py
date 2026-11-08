# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The paper's figures, main text and appendix, redrawn in a plain layout from a mariner-data snapshot.

Each redraw reads only the snapshot's tables, and fits this package's models and, for Figure 1's surrogate panel,
LightGBM's boosted trees. It shows the paper figure's data, fitted curves and comparisons in the shared layout of
``mariner.figures.layout``, and returns every series it plots. ``write_figures`` writes them as PDFs, as ``mariner
figures`` does. The package needs matplotlib and LightGBM, which marin-mariner's figures extra installs.

A spec's ``draw`` leaves the fonts, sizes and PDF settings to its caller: ``write_figure`` and ``write_figures`` draw
inside ``layout.style()``. To draw and save one figure in the shared style, call ``write_figure(FIGURES[label], root,
out)``; to draw it without saving, call its ``draw`` inside ``with layout.style():``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from mariner.figures import (
    analysis,
    baselines,
    epochs,
    heldout,
    pipeline,
    repetition,
    results,
    schedules,
    surrogate,
    swarm,
)
from mariner.figures.layout import FigureData, save, style

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FigureSpec:
    """A redraw: the paper's figure label, the PDF's file stem, and how to draw it from a snapshot.

    A redraw reads only the configs that its row of the snapshot's ``figures_tables.csv`` names; the tests draw each
    figure from a snapshot that holds nothing else.
    """

    label: str
    stem: str
    draw: Callable[[Path], FigureData]

    def path(self, out: Path) -> Path:
        """The PDF's path in the directory ``out``."""
        return out / f"{self.stem}.pdf"


FIGURES: dict[str, FigureSpec] = {
    spec.label: spec
    for spec in (
        FigureSpec("fig:pipeline", "f1_pipeline", pipeline.draw_pipeline),
        FigureSpec("fig:epoch-matching", "epoch_matching_tpp10", epochs.draw_epoch_matching),
        FigureSpec("fig:b5", "b5_matched_compute_scaling", epochs.draw_matched_compute_scaling),
        FigureSpec("fig:b6", "b6_olmix_capacity_failure", surrogate.draw_olmix_capacity_failure),
        FigureSpec("fig:mariner-terms", "m_mariner_terms", surrogate.draw_mariner_terms),
        FigureSpec("fig:r6", "r6_learning_curve_per_fold_ratio", surrogate.draw_learning_curves),
        FigureSpec("fig:proposal-predictions", "r_proposal_predictions", surrogate.draw_proposal_predictions),
        FigureSpec("fig:accuracy", "r_accuracy_vs_bpb", results.draw_accuracy_vs_bpb),
        FigureSpec("fig:r1", "r1_scaling_curves", results.draw_scaling_curves),
        FigureSpec("fig:off-objective", "r_on_off_objective", results.draw_on_off_objective),
        FigureSpec("fig:component-tradeoffs", "r8_component_tradeoffs_4panel", results.draw_component_tradeoffs),
        FigureSpec("fig:domain-repetition", "figure4_cap_policy", analysis.draw_domain_repetition),
        FigureSpec("fig:mixture-epochs", "r_mixture_epochs", analysis.draw_mixture_epochs),
        FigureSpec("fig:r3", "r3_transfer_main", heldout.draw_scale_transfer),
        FigureSpec("fig:r7", "r7_r2_vs_snr", analysis.draw_r2_vs_snr),
        FigureSpec("fig:analysis-floor", "r_analysis_frontier", heldout.draw_frontier),
        FigureSpec("fig:analysis-extrapolation", "r_extrapolation_residuals", heldout.draw_extrapolation_residuals),
        FigureSpec("fig:a-learning-ablations", "a_learning_curve_ablations", baselines.draw_learning_ablations),
        FigureSpec("fig:a-learning-baselines", "a_learning_curve_baselines", baselines.draw_learning_baselines),
        FigureSpec("fig:a-comparator-paths", "a_comparator_paths_full", baselines.draw_comparator_paths),
        FigureSpec("fig:b5a", "b5_downsampling_fixed_budget", repetition.draw_downsampling_by_budget),
        FigureSpec("fig:a-downsampling", "a_replay_burden_ladder", repetition.draw_downsampling_by_repetition),
        FigureSpec("fig:a-deletion", "a_deletion_matrix_full", swarm.draw_deletion_matrix),
        FigureSpec("fig:a-support", "a_swarm_support", swarm.draw_swarm_support),
        FigureSpec("fig:a-curricula-sweeps", "a_curricula_sweeps", schedules.draw_curricula_sweeps),
        FigureSpec("fig:a-switch-smoothing", "a_switch_smoothing", schedules.draw_switch_smoothing),
        FigureSpec("fig:a-ramp-schedules", "a_ramp_schedules", schedules.draw_ramp_schedules),
    )
}


def write_figure(spec: FigureSpec, root: Path, out: Path) -> FigureData:
    """Draw ``spec`` from the snapshot at ``root`` in the shared style and save it as ``spec.path(out)``, creating
    ``out`` if needed; returns the drawn figure and every series it plots."""
    out.mkdir(parents=True, exist_ok=True)
    with style():
        drawn = spec.draw(root)
        save(drawn.figure, spec.path(out))
    return drawn


def write_figures(root: Path, out: Path, labels: Sequence[str] | None = None) -> list[Path]:
    """Draw the figures ``labels`` (every figure when None) from the snapshot at ``root`` as ``<stem>.pdf`` in ``out``.

    Raises ``ValueError`` naming the labels that no figure has before it draws anything.
    """
    selected = list(FIGURES) if labels is None else list(dict.fromkeys(labels))
    unknown = [label for label in selected if label not in FIGURES]
    if unknown:
        raise ValueError(f"no figure is labelled {', '.join(unknown)}; the figures are {', '.join(FIGURES)}")
    paths = []
    for label in selected:
        spec = FIGURES[label]
        write_figure(spec, root, out)
        logger.info("wrote %s (%s)", spec.path(out), label)
        paths.append(spec.path(out))
    return paths
