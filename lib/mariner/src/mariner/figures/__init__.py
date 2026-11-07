# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The paper's main-text figures, redrawn in a plain layout from a mariner-data snapshot.

Each redraw reads only the snapshot's tables and this package's models. It shows the paper figure's data, fitted
curves and comparisons in the shared layout of ``mariner.figures.layout``, and returns every series it plots.
``write_figures`` writes them as PDFs, as ``mariner figures`` does. The package needs matplotlib, which
marin-mariner's figures extra installs.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from mariner.figures import epochs, surrogate
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
        FigureSpec("fig:b5", "b5_matched_compute_scaling", ("starcoder",), epochs.draw_matched_compute_scaling),
        FigureSpec("fig:b6", "b6_olmix_capacity_failure", ("starcoder",), surrogate.draw_olmix_capacity_failure),
        FigureSpec("fig:mariner-terms", "m_mariner_terms", (), surrogate.draw_mariner_terms),
        FigureSpec("fig:r6", "r6_learning_curve_per_fold_ratio", ("fits",), surrogate.draw_learning_curves),
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
