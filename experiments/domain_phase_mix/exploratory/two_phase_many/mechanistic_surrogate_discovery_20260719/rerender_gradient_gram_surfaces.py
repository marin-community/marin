# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
# ]
# ///

"""Regenerate Round 51/52 surfaces with mechanism-specific titles."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import export_mixture_fit_observatory as observatory
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_logistic_margin_round49 as plot_helpers,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    starcoder_refined_data,
)

OUTPUT_ROOT = Path(__file__).parents[1] / "reference_outputs" / "mechanistic_surrogate_discovery_20260719"
ROUNDS = {
    "round51_gradient_gram_loss_starcoder": "gradient-Gram coupled loss flow",
    "round52_orthogonal_gradient_gram_starcoder": "orthogonal gradient-Gram transport",
}


def main() -> None:
    cosine = observatory.load_cosine_starcoder()
    panels = {
        panel.name: panel
        for panel in (
            starcoder.panel_from_dataset(cosine),
            starcoder.panel_from_dataset(starcoder_refined_data.load_refined_wsd80_starcoder(cosine)),
        )
    }
    for round_name, model_label in ROUNDS.items():
        output_dir = OUTPUT_ROOT / round_name
        for panel_name, panel in panels.items():
            surface_path = output_dir / f"{panel_name}__surface.csv"
            if not surface_path.exists():
                raise FileNotFoundError(surface_path)
            plot_helpers.render_surface(
                panel,
                pd.read_csv(surface_path),
                output_dir / f"{panel_name}__surface.html",
                model_label=model_label,
            )


if __name__ == "__main__":
    main()
