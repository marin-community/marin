# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Compute active-set effective degrees of freedom for phase candidates.

The trace is computed in the fitted link space after freezing nonlinear
hyperparameters. It is therefore an interpretable local complexity diagnostic,
not an exact generalized degrees-of-freedom result for the entire selection
procedure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_nested_support_invariants as support,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_phase_boundary_adaptation as phase,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
DEFAULT_LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "candidate_complexity_audit"
DEFICIT_VARIANT = deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC
CONFIGS = {
    base.DatasetId.DELPHI_3E18_UNCHEATABLE: phase.Config(phase.Mechanism.PHASE_INFORMATION, 0.01),
    base.DatasetId.DELPHI_3E18_TABLE9: phase.Config(phase.Mechanism.PHASE_INFORMATION, 0.1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument("--link-metrics", type=Path, default=DEFAULT_LINK_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def active_set_complexity(
    dataset: Any,
    model: phase.Model,
) -> dict[str, float | int | bool | str]:
    values, names, ridge_multipliers = phase.combined_design(dataset, model.deficit_config, model.config)
    active = model.coefficients > 1e-10
    active_names = np.asarray(names, dtype=object)[active]
    if not np.any(active):
        return {
            "nominal_parameter_count": 1 + len(model.coefficients),
            "active_parameter_count": 1,
            "effective_degrees_of_freedom": 1.0,
            "penalized_condition_number": 1.0,
            "phase_coefficient_active": False,
            "phase_coefficient": 0.0,
            "active_parameters": "",
        }
    x = values[:, active]
    centered = x - x.mean(axis=0)
    gram = centered.T @ centered
    penalty = model.link_config.l2 * np.diag(ridge_multipliers[active])
    penalized = gram + penalty + 1e-12 * np.eye(gram.shape[0])
    hat_trace = float(np.trace(np.linalg.solve(penalized, gram)))
    phase_mask = np.asarray([name.startswith("phase_adaptation:") for name in names])
    phase_coefficient = float(model.coefficients[phase_mask][0]) if np.any(phase_mask) else 0.0
    return {
        "nominal_parameter_count": 1 + len(model.coefficients),
        "active_parameter_count": 1 + int(active.sum()),
        "effective_degrees_of_freedom": 1.0 + hat_trace,
        "penalized_condition_number": float(np.linalg.cond(penalized)),
        "phase_coefficient_active": bool(phase_coefficient > 1e-10),
        "phase_coefficient": phase_coefficient,
        "active_parameters": ";".join(active_names.tolist()),
    }


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.source_metrics)
    gate.assert_sealed_absent(args.link_metrics)
    source = pd.read_csv(args.source_metrics)
    links = pd.read_csv(args.link_metrics)
    rows: list[dict[str, float | int | bool | str]] = []
    for dataset_id, candidate_config in CONFIGS.items():
        dataset = base.load_dataset(dataset_id)
        deficit_config = output_link.selected_deficit_config(dataset_id, DEFICIT_VARIANT, source)
        link_config = support.selected_link_config(dataset_id, links)
        for label, config in (("baseline", None), ("phase_information", candidate_config)):
            model = phase.fit_model(
                dataset,
                deficit_config,
                link_config,
                config,
                np.arange(dataset.n),
            )
            rows.append(
                {
                    "dataset": dataset_id.value,
                    "model": label,
                    "config": "baseline" if config is None else config.key,
                    "link": link_config.link.value,
                    "l2": link_config.l2,
                    "complexity_space": "transformed target / fitted link space",
                    **active_set_complexity(dataset, model),
                }
            )
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "active_set_complexity.csv", index=False)
    report = "\n".join(
        (
            "# Candidate active-set complexity audit",
            "",
            "The reported effective degrees of freedom are `1 + tr[(X_A'X_A + Lambda_A)^-1 X_A'X_A]` "
            "for the active nonnegative coefficients after nonlinear hyperparameters are frozen. The intercept contributes "
            "one degree. This excludes hyperparameter-search degrees of freedom and is therefore a lower bound on total "
            "selection complexity.",
            "",
            frame.drop(columns="active_parameters").to_markdown(index=False, floatfmt=".6f"),
            "",
            "The phase-information extension adds one nominal coefficient but almost no effective complexity when that "
            "coefficient is inactive or heavily shrunk. Parameter-count simplicity therefore does not rescue its failed "
            "cross-swarm identification or raw optimum.",
            "",
        )
    )
    (args.output_dir / "report.md").write_text(report)


if __name__ == "__main__":
    main()
