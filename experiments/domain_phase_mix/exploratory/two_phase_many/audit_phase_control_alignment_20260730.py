# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///
"""Audit whether sampled phase contrasts excite proposed control coordinates.

This is a design-only audit. It never reads evaluation outcomes. The primary
quantity separates transported phase mass from alignment with the tangent
gradient of the independently fitted tied aggregate response:

    m = 0.5 * ||delta||_1
    q = |g(a)^T delta| / (m * (max(g(a)) - min(g(a)))).

The script also evaluates a proposed gradient-orthogonal scalar channel. That
channel is structurally zero in a two-domain panel and is screened here before
it can be fit against outcomes in the 39-bucket panel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_saturating_phase_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    saturating_phase_control_20260730 as saturating,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "phase_control_alignment_20260730"
TOLERANCE = 1e-12
MINIMUM_SIGN_FRACTION = 0.25


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return stable summary quantiles for one nonempty vector."""

    if not len(values):
        raise ValueError(f"{prefix} has no values")
    return {
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_maximum": float(np.max(values)),
    }


def design_summary(
    panel: str,
    weights: np.ndarray,
    aggregate: replay_control.AggregateFitted,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    """Summarize scalar-gradient and gradient-orthogonal design support."""

    tied = replay_control.tied_rows(weights)
    asymmetric = ~tied
    delta = weights[:, 1, :] - weights[:, 0, :]
    mass = 0.5 * np.sum(np.abs(delta), axis=1)
    gradient = saturating.tangent_gradient(weights, aggregate)
    gradient_tangent = gradient - gradient.mean(axis=1, keepdims=True)
    authority = np.ptp(gradient_tangent, axis=1)
    control = np.sum(gradient_tangent * delta, axis=1)

    alignment = np.zeros(len(weights), dtype=float)
    alignment_denominator = mass * authority
    np.divide(
        np.abs(control),
        alignment_denominator,
        out=alignment,
        where=alignment_denominator > TOLERANCE,
    )

    gradient_norm_squared = np.sum(gradient_tangent**2, axis=1)
    projection_scale = np.zeros(len(weights), dtype=float)
    np.divide(
        control,
        gradient_norm_squared,
        out=projection_scale,
        where=gradient_norm_squared > TOLERANCE,
    )
    delta_orthogonal = delta - projection_scale[:, None] * gradient_tangent
    mixture = replay_control.aggregate_mixture(weights, aggregate.geometry)
    mixture_norm = np.linalg.norm(mixture, axis=1)
    gradient_norm = np.sqrt(gradient_norm_squared)
    orthogonal_control = np.zeros(len(weights), dtype=float)
    orthogonal_numerator = np.sum(mixture * delta_orthogonal, axis=1)
    np.divide(
        gradient_norm * orthogonal_numerator,
        mixture_norm,
        out=orthogonal_control,
        where=mixture_norm > TOLERANCE,
    )
    orthogonal_fraction = np.zeros(len(weights), dtype=float)
    np.divide(
        np.abs(orthogonal_control),
        authority,
        out=orthogonal_fraction,
        where=authority > TOLERANCE,
    )

    asym_mass = mass[asymmetric]
    asym_alignment = alignment[asymmetric]
    asym_orthogonal = orthogonal_control[asymmetric]
    asym_orthogonal_fraction = orthogonal_fraction[asymmetric]
    positive = float(np.mean(asym_orthogonal > TOLERANCE))
    negative = float(np.mean(asym_orthogonal < -TOLERANCE))
    p95 = float(np.quantile(asym_orthogonal_fraction, 0.95))
    correlation = (
        float(np.corrcoef(control[asymmetric], asym_orthogonal)[0, 1])
        if np.std(control[asymmetric]) > TOLERANCE and np.std(asym_orthogonal) > TOLERANCE
        else 0.0
    )

    summary: dict[str, float | int | str | bool] = {
        "panel": panel,
        "n_tied": int(tied.sum()),
        "n_asymmetric": int(asymmetric.sum()),
        **quantiles(asym_mass, "transported_mass"),
        **quantiles(asym_alignment, "gradient_alignment"),
        **quantiles(asym_orthogonal_fraction, "orthogonal_control_fraction"),
        "orthogonal_positive_fraction": positive,
        "orthogonal_negative_fraction": negative,
        "control_orthogonal_correlation": correlation,
        "maximum_tied_orthogonal_control": float(np.max(np.abs(orthogonal_control[tied]))),
        "orthogonal_p95_is_descriptive_only": p95,
        "passes_orthogonal_signs": min(positive, negative) >= MINIMUM_SIGN_FRACTION,
    }
    rows = pd.DataFrame(
        {
            "panel": panel,
            "row": np.arange(len(weights)),
            "tied": tied,
            "transported_mass": mass,
            "gradient_control": control,
            "gradient_authority": authority,
            "gradient_alignment": alignment,
            "orthogonal_control": orthogonal_control,
            "orthogonal_control_fraction": orthogonal_fraction,
        }
    )
    return summary, rows


def report(summaries: pd.DataFrame) -> str:
    """Render the design decision as a standalone Markdown report."""

    table = summaries[
        [
            "panel",
            "transported_mass_p50",
            "transported_mass_p95",
            "gradient_alignment_p50",
            "gradient_alignment_p95",
            "orthogonal_control_fraction_p95",
            "orthogonal_positive_fraction",
            "orthogonal_negative_fraction",
        ]
    ].to_markdown(index=False, floatfmt=".6f")
    return f"""# Phase-control alignment audit

This audit uses no evaluation outcomes. It asks whether the sampled phase
contrasts excite two proposed scalar control coordinates before either
coordinate is allowed into a response model.

For aggregate mixture `a`, phase contrast `delta = w1 - w0`, and the tangent
gradient `g(a)` of the tied aggregate model:

```text
transported mass m = 0.5 * ||delta||_1
gradient control u = g(a)^T delta
gradient alignment q = |u| / (m * (max(g) - min(g)))
```

The second channel removes the component of `delta` parallel to `g` and
projects the remainder onto `a`:

```text
delta_perp = delta - (u / ||g||^2) * g
s = ||g|| * a^T delta_perp / ||a||.
```

`|s| / (max(g) - min(g))` is reported descriptively. Unlike `q`, it is not
provably bounded by one, so the scalar-control magnitude gate is not
transferable. The scale-free rejection checks are whether the channel has a
nondegenerate two-domain specialization and whether both signs occur in at
least 25% of asymmetric 300M rows.

{table}

## Decision

The 300M panel has large phase movement: the median policy transports about
half of the simplex mass between phases. The failure of the scalar-gradient
coordinate is therefore not caused by timid schedules. Only about 7% of the
available gradient authority is excited at the median and under 20% at the
95th percentile.

This low alignment is plausibly the generic dimensionality cost of projecting
a 38-dimensional tangent contrast onto one direction, not evidence that the
300M schedules were badly sampled. In WSD80, `q = 1` is an algebraic identity:
the two-domain tangent space is one-dimensional. WSD80 therefore contributes
no evidence about gradient alignment. A permutation-null audit is required
before interpreting the 300M alignment as unusually low or high.

The proposed gradient-orthogonal scalar does not repair the identification
problem. It is identically zero in the two-domain WSD80 geometry and fails the
sign-balance gate on both 300M targets: only 0.8% and 1.3% of values are
positive. Its strong one-sided sign is a property of the design coordinate,
not evidence for a one-sided phase response.

Consequently:

1. Do not modify the nonlinear response `f(u)`; the 300M design cannot identify
   curvature in that scalar coordinate.
2. Do not promote the proposed `a^T delta_perp` channel.
3. The next candidate needs a materially richer mechanistic state that remains
   active on WSD80 and uses the high-dimensional 300M contrast information
   without free per-bucket phase coefficients.
"""


def main() -> None:
    """Run the zero-outcome audit and write durable artifacts."""

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    rows = []

    wsd_panel, wsd_aggregate = benchmark.fit_full_aggregate_wsd()
    summary, panel_rows = design_summary("wsd80", wsd_panel.weights, wsd_aggregate)
    summaries.append(summary)
    rows.append(panel_rows)

    for target in benchmark.benchmark.TARGETS:
        dataset, aggregate = benchmark.fit_full_aggregate_300m(target)
        summary, panel_rows = design_summary(f"300m_{target}", dataset.weights, aggregate)
        summaries.append(summary)
        rows.append(panel_rows)

    summaries_frame = pd.DataFrame(summaries)
    summaries_frame.to_csv(output_dir / "summary.csv", index=False)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "rows.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps({"rows": summaries}, indent=2) + "\n")
    (output_dir / "report.md").write_text(report(summaries_frame))


if __name__ == "__main__":
    main()
