# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize random fixed-aggregate phase schedules around 3e18 frontier anchors.

The panel defines an explicit conditional sampling law q(d | a). For each tied
anchor ``a``, it draws isotropic directions in the 38-dimensional simplex
tangent space and evaluates three fractions of the direction-specific feasible
radius. The realized aggregate mixture remains exactly equal to ``a``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_3e18_frontier_phase_fiber_panel as fiber,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUT_DIR / "delphi_3e18_frontier_random_phase_population_20260720"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_frontier_random_phase_population_20260720"
)
PRIOR_FIBER_DIR = REFERENCE_OUTPUT_DIR / "delphi_3e18_frontier_phase_fiber_20260719"
HPR_PANEL_DIRS = (
    REFERENCE_OUTPUT_DIR / "hpr_300m_to_3e18_optimum_validation_panel_20260720",
    REFERENCE_OUTPUT_DIR / "hpr_3e18_to_3e18_optimum_validation_panel_20260720",
)

N_RANDOM_DIRECTIONS = 48
RADIUS_FRACTIONS = (0.25, 0.50, 0.75)
N_SEED_BLOCKS = 4
EXPECTED_ROWS_PER_ANCHOR = N_SEED_BLOCKS + N_RANDOM_DIRECTIONS * len(RADIUS_FRACTIONS)
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_ANCHOR * len(fiber.ANCHORS)
RUN_ID_BASE = 7_210_000
DATA_SEED_BASE = 7_212_000
RANDOM_DIRECTION_SEED = 20_260_720
AGGREGATE_TOLERANCE = 2e-12
SIMPLEX_TOLERANCE = 2e-12
NOVELTY_TOLERANCE = 1e-10
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    anchor_id: str
    anchor_run_name: str
    anchor_source_run_name: str
    contrast_family: str
    direction_id: str
    direction_label: str
    sign: str
    seed_block: int
    data_seed: int
    trainer_seed: int
    direction_seed: int
    radius_fraction: float
    feasible_radius: float
    realized_radius: float
    direction_vector: np.ndarray
    phase_0_weights: np.ndarray
    phase_1_weights: np.ndarray


def load_policy_anchors(domains: tuple[str, ...]) -> tuple[list[fiber.Anchor], pd.DataFrame]:
    """Load preregistered anchors and rank policy coordinates, not noisy rows."""
    scores = fiber._one_phase_scores()
    coordinate_scores = (
        scores.groupby("mixture_sha256")[["uncheatable_bpb", "table9_macro_bpb"]]
        .agg(["count", "min", "mean", "std"])
        .reset_index()
    )
    for target in ("uncheatable_bpb", "table9_macro_bpb"):
        coordinate_scores[(target, "min_rank")] = coordinate_scores[(target, "min")].rank(method="min")

    anchors = []
    audit_rows = []
    for anchor_id, run_name, expected_mixture_sha256, expected_weight_vector_sha256 in fiber.ANCHORS:
        row = scores.loc[scores["wandb_run_base"].eq(run_name)]
        if len(row) != 1:
            raise ValueError(f"Expected one source row for {run_name}, found {len(row)}")
        source = row.iloc[0]
        if source["mixture_sha256"] != expected_mixture_sha256:
            raise ValueError(f"Mixture hash changed for {run_name}: {source['mixture_sha256']}")
        phase_0 = json.loads(source["phase_0_weights_json"])
        phase_1 = json.loads(source["phase_1_weights_json"])
        anchor_weights = np.asarray([float(phase_0[domain]) for domain in domains])
        late_weights = np.asarray([float(phase_1[domain]) for domain in domains])
        if np.max(np.abs(anchor_weights - late_weights)) > SIMPLEX_TOLERANCE:
            raise ValueError(f"Anchor {run_name} is not phase tied")
        weight_vector_sha256 = fiber._weight_vector_sha256(domains, anchor_weights)
        if weight_vector_sha256 != expected_weight_vector_sha256:
            raise ValueError(f"Realized weight-vector hash changed for {run_name}: {weight_vector_sha256}")

        coordinate = coordinate_scores.loc[coordinate_scores["mixture_sha256"].eq(expected_mixture_sha256)]
        if len(coordinate) != 1:
            raise ValueError(f"Expected one coordinate aggregate for {run_name}, found {len(coordinate)}")
        coordinate_row = coordinate.iloc[0]
        target = f"{anchor_id.removesuffix('_frontier')}_bpb"
        if target == "table9_bpb":
            target = "table9_macro_bpb"
        if float(coordinate_row[(target, "min_rank")]) != 1.0:
            raise ValueError(f"Anchor coordinate {run_name} is no longer the observed {target} frontier")

        anchor = fiber.Anchor(
            anchor_id=anchor_id,
            run_name=run_name,
            source_run_name=run_name,
            mixture_sha256=expected_mixture_sha256,
            weight_vector_sha256=weight_vector_sha256,
            weights=anchor_weights,
            uncheatable_3e18=float(source["uncheatable_bpb"]),
            table9_3e18=float(source["table9_macro_bpb"]),
            uncheatable_one_phase_rank=float(coordinate_row[("uncheatable_bpb", "min_rank")]),
            table9_one_phase_rank=float(coordinate_row[("table9_macro_bpb", "min_rank")]),
            one_phase_policy_count=int(scores["mixture_sha256"].nunique()),
        )
        anchors.append(anchor)
        audit_rows.append(
            {
                "anchor_id": anchor.anchor_id,
                "source_run_name": anchor.source_run_name,
                "mixture_sha256": anchor.mixture_sha256,
                "weight_vector_sha256": anchor.weight_vector_sha256,
                "source_uncheatable_bpb": anchor.uncheatable_3e18,
                "source_table9_macro_bpb": anchor.table9_3e18,
                "coordinate_repeat_count": int(coordinate_row[("uncheatable_bpb", "count")]),
                "coordinate_uncheatable_min": float(coordinate_row[("uncheatable_bpb", "min")]),
                "coordinate_uncheatable_mean": float(coordinate_row[("uncheatable_bpb", "mean")]),
                "coordinate_uncheatable_std": float(coordinate_row[("uncheatable_bpb", "std")]),
                "coordinate_table9_min": float(coordinate_row[("table9_macro_bpb", "min")]),
                "coordinate_table9_mean": float(coordinate_row[("table9_macro_bpb", "mean")]),
                "coordinate_table9_std": float(coordinate_row[("table9_macro_bpb", "std")]),
                "uncheatable_coordinate_min_rank": anchor.uncheatable_one_phase_rank,
                "table9_coordinate_min_rank": anchor.table9_one_phase_rank,
                "one_phase_policy_count": anchor.one_phase_policy_count,
                "min_weight": float(anchor_weights.min()),
                "max_weight": float(anchor_weights.max()),
                "support_size": int(np.count_nonzero(anchor_weights > 0)),
            }
        )
    return anchors, pd.DataFrame(audit_rows)


def _isotropic_tangent_directions(n_domains: int) -> np.ndarray:
    directions = np.asarray(
        [
            np.random.default_rng(RANDOM_DIRECTION_SEED + index).normal(size=n_domains)
            for index in range(N_RANDOM_DIRECTIONS)
        ]
    )
    directions -= directions.mean(axis=1, keepdims=True)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    if np.max(np.abs(directions.sum(axis=1))) > 2e-15:
        raise ValueError("Projected random directions do not conserve simplex mass")
    singular_values = np.linalg.svd(directions, compute_uv=False)
    rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-10))
    if rank != n_domains - 1:
        raise ValueError(f"Random tangent directions have rank {rank}, expected {n_domains - 1}")
    return directions


def _feasible_radius(anchor: np.ndarray, direction: np.ndarray, alpha0: float, alpha1: float) -> float:
    limits = []
    positive = direction > 0
    negative = direction < 0
    limits.extend((anchor[positive] / (alpha1 * direction[positive])).tolist())
    limits.extend((anchor[negative] / (-alpha0 * direction[negative])).tolist())
    if not limits:
        raise ValueError("A tangent direction has no nonzero coordinates")
    radius = float(min(limits))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Invalid feasible radius {radius}")
    return radius


def _phases_from_direction(
    anchor: np.ndarray,
    direction: np.ndarray,
    radius: float,
    alpha0: float,
    alpha1: float,
) -> tuple[np.ndarray, np.ndarray]:
    contrast = radius * direction
    phase_0 = anchor - alpha1 * contrast
    phase_1 = anchor + alpha0 * contrast
    return phase_0, phase_1


def build_candidates(
    anchors: list[fiber.Anchor],
    directions: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> list[Candidate]:
    candidates = []
    block_seeds = [DATA_SEED_BASE + block for block in range(N_SEED_BLOCKS)]
    for anchor_index, anchor in enumerate(anchors):
        for block, data_seed in enumerate(block_seeds):
            candidates.append(
                Candidate(
                    candidate_id=f"random_phase_a{anchor_index}_center_s{block}",
                    anchor_id=anchor.anchor_id,
                    anchor_run_name=anchor.run_name,
                    anchor_source_run_name=anchor.source_run_name,
                    contrast_family="center_control",
                    direction_id="center",
                    direction_label="phase tied",
                    sign="center",
                    seed_block=block,
                    data_seed=data_seed,
                    trainer_seed=0,
                    direction_seed=RANDOM_DIRECTION_SEED,
                    radius_fraction=0.0,
                    feasible_radius=0.0,
                    realized_radius=0.0,
                    direction_vector=np.zeros_like(anchor.weights),
                    phase_0_weights=anchor.weights.copy(),
                    phase_1_weights=anchor.weights.copy(),
                )
            )

        for direction_index, direction in enumerate(directions):
            block = direction_index % N_SEED_BLOCKS
            feasible_radius = _feasible_radius(anchor.weights, direction, alpha0, alpha1)
            for radius_fraction in RADIUS_FRACTIONS:
                realized_radius = radius_fraction * feasible_radius
                phase_0, phase_1 = _phases_from_direction(
                    anchor.weights,
                    direction,
                    realized_radius,
                    alpha0,
                    alpha1,
                )
                radius_label = f"r{round(100 * radius_fraction):02d}"
                candidates.append(
                    Candidate(
                        candidate_id=f"random_phase_a{anchor_index}_d{direction_index:02d}_{radius_label}",
                        anchor_id=anchor.anchor_id,
                        anchor_run_name=anchor.run_name,
                        anchor_source_run_name=anchor.source_run_name,
                        contrast_family="random_isotropic",
                        direction_id=f"random_{direction_index:02d}",
                        direction_label=f"isotropic tangent draw {direction_index:02d}",
                        sign="random",
                        seed_block=block,
                        data_seed=block_seeds[block],
                        trainer_seed=0,
                        direction_seed=RANDOM_DIRECTION_SEED + direction_index,
                        radius_fraction=radius_fraction,
                        feasible_radius=feasible_radius,
                        realized_radius=realized_radius,
                        direction_vector=direction.copy(),
                        phase_0_weights=phase_0,
                        phase_1_weights=phase_1,
                    )
                )
        count = sum(candidate.anchor_id == anchor.anchor_id for candidate in candidates)
        if count != EXPECTED_ROWS_PER_ANCHOR:
            raise ValueError(f"Expected {EXPECTED_ROWS_PER_ANCHOR} rows for {anchor.anchor_id}, found {count}")
    if len(candidates) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} candidates, found {len(candidates)}")
    return candidates


def _weights_from_wide_source(path: Path, domains: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(path)
    return np.asarray(
        [
            [[float(row[f"phase_{phase}_{domain}"]) for domain in domains] for phase in (0, 1)]
            for row in frame.to_dict(orient="records")
        ]
    )


def _weights_from_long_source(path: Path, domains: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(path)
    index_column = "candidate_id" if "candidate_id" in frame.columns else "run_name"
    phase_column = "phase"
    if frame[phase_column].dtype == object:
        frame[phase_column] = frame[phase_column].str.removeprefix("phase_").astype(int)
    pivot = frame.pivot_table(index=index_column, columns=[phase_column, "domain"], values="weight")
    return (
        np.asarray([[pivot[phase].loc[:, list(domains)].to_numpy(dtype=float) for phase in (0, 1)]])
        .transpose(2, 1, 3, 0)
        .squeeze(axis=3)
    )


def _heldout_weights(domains: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(fiber.HELDOUT_PATH)
    rows = []
    for row in frame.to_dict(orient="records"):
        if not isinstance(row.get("phase_0_weights_json"), str) or not isinstance(row.get("phase_1_weights_json"), str):
            continue
        phases = [json.loads(row[f"phase_{phase}_weights_json"]) for phase in (0, 1)]
        rows.append([[float(phases[phase][domain]) for domain in domains] for phase in (0, 1)])
    return np.asarray(rows)


def _reference_policies(domains: tuple[str, ...]) -> np.ndarray:
    references = [fiber._existing_fit_weights(domains), _heldout_weights(domains)]
    prior_fiber_weights = PRIOR_FIBER_DIR / "phase_weights.csv"
    if prior_fiber_weights.exists():
        references.append(_weights_from_long_source(prior_fiber_weights, domains))
    for panel_dir in HPR_PANEL_DIRS:
        source = panel_dir / "launcher_source_panel.csv"
        if source.exists():
            references.append(_weights_from_wide_source(source, domains))
    return np.concatenate(references, axis=0)


def _policy_sha256(domains: tuple[str, ...], weights: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update("\0".join(domains).encode())
    hasher.update(np.asarray(weights, dtype="<f8").tobytes())
    return hasher.hexdigest()


def validate_candidates(
    candidates: list[Candidate],
    anchors: list[fiber.Anchor],
    domains: tuple[str, ...],
    directions: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    anchor_lookup = {anchor.anchor_id: anchor for anchor in anchors}
    weights = np.asarray([[candidate.phase_0_weights, candidate.phase_1_weights] for candidate in candidates])
    if float(weights.min()) < -SIMPLEX_TOLERANCE:
        raise ValueError(f"Panel has negative phase weight {weights.min()}")
    if np.max(np.abs(weights.sum(axis=2) - 1.0)) > SIMPLEX_TOLERANCE:
        raise ValueError("A phase mixture does not sum to one")

    references = _reference_policies(domains)
    min_prior_tv = fiber._weighted_policy_tv(weights, references, alpha0, alpha1).min(axis=1)
    fit_weights = fiber._existing_fit_weights(domains)
    min_fit_tv = fiber._weighted_policy_tv(weights, fit_weights, alpha0, alpha1).min(axis=1)
    rows = []
    for index, candidate in enumerate(candidates):
        anchor = anchor_lookup[candidate.anchor_id]
        aggregate = alpha0 * candidate.phase_0_weights + alpha1 * candidate.phase_1_weights
        aggregate_error = float(np.max(np.abs(aggregate - anchor.weights)))
        if aggregate_error > AGGREGATE_TOLERANCE:
            raise ValueError(f"{candidate.candidate_id} aggregate error is {aggregate_error}")
        aggregate_epochs = np.asarray(
            [
                augmented.SIMULATED_EPOCH_TARGET_BUDGET
                * aggregate[domain_index]
                / augmented.TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain]
                for domain_index, domain in enumerate(domains)
            ]
        )
        rows.append(
            {
                "run_order": index,
                "run_id": RUN_ID_BASE + index,
                "candidate_id": candidate.candidate_id,
                "policy_sha256": _policy_sha256(domains, weights[index]),
                "anchor_id": candidate.anchor_id,
                "anchor_run_name": candidate.anchor_run_name,
                "anchor_source_run_name": candidate.anchor_source_run_name,
                "contrast_family": candidate.contrast_family,
                "direction_id": candidate.direction_id,
                "direction_label": candidate.direction_label,
                "sign": candidate.sign,
                "seed_block": candidate.seed_block,
                "data_seed": candidate.data_seed,
                "trainer_seed": candidate.trainer_seed,
                "direction_seed": candidate.direction_seed,
                "radius_fraction": candidate.radius_fraction,
                "feasible_radius": candidate.feasible_radius,
                "realized_radius": candidate.realized_radius,
                "phase_tv": float(0.5 * np.abs(candidate.phase_1_weights - candidate.phase_0_weights).sum()),
                "phase_information_kl": fiber._phase_information_kl(
                    candidate.phase_0_weights,
                    candidate.phase_1_weights,
                    aggregate,
                    alpha0,
                    alpha1,
                ),
                "aggregate_max_abs_error": aggregate_error,
                "max_weight": float(weights[index].max()),
                "min_weight": float(weights[index].min()),
                "max_simulated_epoch": float(aggregate_epochs.max()),
                "q95_simulated_epoch": float(np.quantile(aggregate_epochs, 0.95)),
                "min_fit_policy_tv": float(min_fit_tv[index]),
                "min_prior_policy_tv": float(min_prior_tv[index]),
            }
        )
    manifest = pd.DataFrame(rows)

    random_mask = manifest["contrast_family"].eq("random_isotropic").to_numpy()
    random_hashes = manifest.loc[random_mask, "policy_sha256"]
    if random_hashes.nunique() != len(random_hashes):
        raise ValueError("Random phase population contains duplicate policies")
    if float(manifest.loc[random_mask, "min_prior_policy_tv"].min()) <= NOVELTY_TOLERANCE:
        duplicate = manifest.loc[random_mask].sort_values("min_prior_policy_tv").iloc[0]
        raise ValueError(
            f"Random policy {duplicate['candidate_id']} aliases prior work at TV {duplicate['min_prior_policy_tv']}"
        )

    expected_radius_counts = {fraction: len(anchors) * N_RANDOM_DIRECTIONS for fraction in RADIUS_FRACTIONS}
    radius_counts = manifest.loc[random_mask].groupby("radius_fraction").size().to_dict()
    if radius_counts != expected_radius_counts:
        raise ValueError(f"Radius strata changed: {radius_counts}")
    seed_counts = manifest.groupby(["anchor_id", "seed_block"]).size()
    expected_seed_rows = 1 + (N_RANDOM_DIRECTIONS // N_SEED_BLOCKS) * len(RADIUS_FRACTIONS)
    if set(seed_counts) != {expected_seed_rows}:
        raise ValueError(f"Seed blocks are not balanced: {seed_counts.to_dict()}")

    singular_values = np.linalg.svd(directions, compute_uv=False)
    nonzero = singular_values[singular_values > singular_values[0] * 1e-10]
    direction_rank = len(nonzero)
    direction_condition = float(nonzero[0] / nonzero[-1])
    if direction_rank != len(domains) - 1:
        raise ValueError(f"Direction rank is {direction_rank}, expected {len(domains) - 1}")
    summary = {
        "panel_rows": len(manifest),
        "anchor_count": len(anchors),
        "random_directions_per_anchor": N_RANDOM_DIRECTIONS,
        "radius_fractions": list(RADIUS_FRACTIONS),
        "rows_per_anchor": manifest.groupby("anchor_id").size().to_dict(),
        "rows_per_contrast_family": manifest.groupby("contrast_family").size().to_dict(),
        "rows_per_radius_fraction": {str(key): int(value) for key, value in radius_counts.items()},
        "rows_per_seed_block": {
            f"{anchor_id}/seed_{seed_block}": int(count) for (anchor_id, seed_block), count in seed_counts.items()
        },
        "realized_phase_fractions": {"phase_0": alpha0, "phase_1": alpha1},
        "sampling_law": ("u=(z-mean(z))/||z-mean(z)|| for z~N(0,I_39); d=rho*r_max(a,u)*u; rho in {0.25,0.50,0.75}"),
        "common_random_directions_across_anchors": True,
        "random_direction_seed": RANDOM_DIRECTION_SEED,
        "direction_rank": direction_rank,
        "direction_condition_number": direction_condition,
        "max_aggregate_error": float(manifest["aggregate_max_abs_error"].max()),
        "min_phase_weight": float(weights.min()),
        "max_phase_weight": float(weights.max()),
        "phase_tv_range": [float(manifest["phase_tv"].min()), float(manifest["phase_tv"].max())],
        "min_fit_policy_tv_range": [
            float(manifest["min_fit_policy_tv"].min()),
            float(manifest["min_fit_policy_tv"].max()),
        ],
        "minimum_random_prior_policy_tv": float(manifest.loc[random_mask, "min_prior_policy_tv"].min()),
        "selection_uses_prior_one_phase_3e18_outcomes": True,
        "selection_uses_two_phase_outcomes": False,
        "native_table9_scheduled": True,
    }
    return manifest, weights, summary


def render_diagnostics(manifest: pd.DataFrame, output_dir: Path) -> None:
    random_rows = manifest.loc[manifest["contrast_family"].eq("random_isotropic")].copy()
    random_rows["radius_stratum"] = random_rows["radius_fraction"].map(lambda value: f"rho={value:g}")
    figure = px.scatter(
        random_rows,
        x="phase_tv",
        y="min_fit_policy_tv",
        color="radius_fraction",
        facet_col="anchor_id",
        hover_name="candidate_id",
        hover_data=[
            "direction_id",
            "seed_block",
            "feasible_radius",
            "phase_information_kl",
            "min_prior_policy_tv",
        ],
        color_continuous_scale="RdYlGn_r",
        title="Random frontier phase population: contrast strength and distance from fit support",
    )
    figure.update_xaxes(title_text="phase TV between phase 0 and phase 1")
    figure.update_yaxes(title_text="nearest weighted policy TV to 280-row two-phase fit panel")
    figure.update_layout(
        width=1500,
        height=760,
        margin={"l": 70, "r": 140, "t": 130, "b": 70},
        coloraxis_colorbar={"title": "radius fraction", "x": 1.03},
    )
    figure.write_html(output_dir / "random_phase_population_diagnostics.html", include_plotlyjs=True, config=PLOT_CONFIG)

    distribution = px.box(
        random_rows,
        x="radius_stratum",
        y="phase_information_kl",
        color="anchor_id",
        points="all",
        hover_name="candidate_id",
        title="Preregistered phase-information distribution by anchor and radius stratum",
        color_discrete_sequence=["#2b6777", "#d95f02"],
    )
    distribution.update_layout(width=1200, height=720)
    distribution.write_html(
        output_dir / "random_phase_population_strata.html",
        include_plotlyjs=True,
        config=PLOT_CONFIG,
    )


def write_report(
    output_dir: Path,
    manifest: pd.DataFrame,
    anchor_audit: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    geometry = (
        manifest.groupby(["anchor_id", "radius_fraction"])[
            ["phase_tv", "phase_information_kl", "min_fit_policy_tv", "min_prior_policy_tv"]
        ]
        .agg(["min", "median", "max"])
        .round(6)
    )
    lines = [
        "# Delphi 3e18 random frontier phase-population panel",
        "",
        "## Scientific estimand",
        "",
        (
            "For each empirical one-phase frontier anchor, estimate the conditional distribution of smooth BPB "
            "under a preregistered random distribution over aggregate-matched two-phase schedules."
        ),
        "",
        "This panel does not sample uniformly from the full feasible policy polytope. Its population is exactly",
        "",
        "$$u=\\frac{z-\\bar z\\mathbf{1}}{\\lVert z-\\bar z\\mathbf{1}\\rVert_2},\\quad "
        "z\\sim\\mathcal N(0,I_{39}),\\quad d=\\rho r_{\\max}(a,u)u,$$",
        "",
        "with $\\rho\\in\\{0.25,0.50,0.75\\}$ and",
        "",
        "$$w^{(0)}=a-\\alpha_1d,\\qquad w^{(1)}=a+\\alpha_0d.$ $".replace("$ $", "$$"),
        "",
        (
            "The direction law is isotropic in the 38-dimensional simplex tangent space. Radius is stratified "
            "relative to the first simplex boundary encountered along each direction, so outcome summaries must "
            "be reported separately by anchor and radius before any pooled distribution or normality test."
        ),
        "",
        "## Frozen design",
        "",
        "- Two phase-tied frontier anchors, one selected by Uncheatable and one by Table-9.",
        "- 48 independent tangent directions shared across anchors as common random numbers.",
        "- Three radius strata per direction: 0.25, 0.50, and 0.75 of the feasible radial limit.",
        "- Four fresh tied controls per anchor, one per seed block.",
        "- 148 checkpoints per anchor and 296 total checkpoints.",
        "- Every checkpoint produces Uncheatable and Marin-native Table-9 BPB.",
        "- No random coordinate aliases the fit panel, append-only heldouts, prior phase-fiber DOE, or HPR panels.",
        "",
        "Forty-eight directions exceed the tangent rank of 38 and provide 48 observations within each "
        "anchor/radius stratum. This is sufficient for empirical CDFs and broad distributional diagnostics, "
        "not for reconstructing an unrestricted 38-dimensional response surface.",
        "",
        "## Anchor audit",
        "",
        anchor_audit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Geometry audit",
        "",
        f"- Panel rows: {summary['panel_rows']}.",
        f"- Direction rank: {summary['direction_rank']}.",
        f"- Direction condition number: {summary['direction_condition_number']:.4f}.",
        f"- Maximum aggregate error: {summary['max_aggregate_error']:.3e}.",
        f"- Minimum random-policy distance to prior coordinates: {summary['minimum_random_prior_policy_tv']:.6g}.",
        "",
        geometry.to_markdown(floatfmt=".6f"),
        "",
        "## Analysis boundary",
        "",
        (
            "After completion, report empirical CDFs, quantiles, skewness, tail probabilities, and the fraction "
            "beating the tied control separately for every anchor/radius stratum. Normality tests are secondary "
            "and only meaningful within a fixed stratum because the pooled design is an intentional mixture "
            "distribution. These outcomes become append-only development heldouts after this analysis."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domains = tuple(augmented.DOMAIN_NAMES)
    alpha0, alpha1 = fiber._realized_phase_fractions()
    anchors, anchor_audit = load_policy_anchors(domains)
    directions = _isotropic_tangent_directions(len(domains))
    candidates = build_candidates(anchors, directions, alpha0, alpha1)
    manifest, weights, summary = validate_candidates(
        candidates,
        anchors,
        domains,
        directions,
        alpha0,
        alpha1,
    )
    manifest_path = args.output_dir / "candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    anchor_audit.to_csv(args.output_dir / "anchor_audit.csv", index=False)
    fiber.write_long_weights(args.output_dir, manifest, weights, domains)
    source_path, source_sha256 = fiber.write_launcher_source_panel(args.output_dir, manifest, weights, domains)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary.update(
        {
            "candidate_manifest_sha256": manifest_sha256,
            "launcher_source_panel_sha256": source_sha256,
            "gcs_launcher_source_panel": f"{args.gcs_output_dir}/source/launcher_source_panel-{source_sha256[:16]}.csv",
            "gcs_candidate_manifest": f"{args.gcs_output_dir}/source/candidate_manifest-{manifest_sha256[:16]}.csv",
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    render_diagnostics(manifest, args.output_dir)
    write_report(args.output_dir, manifest, anchor_audit, summary)
    if args.upload:
        fiber.upload_artifact(source_path, str(summary["gcs_launcher_source_panel"]))
        fiber.upload_artifact(manifest_path, str(summary["gcs_candidate_manifest"]))
        for name in ("summary.json", "report.md", "phase_weights.csv", "anchor_audit.csv"):
            fiber.upload_artifact(args.output_dir / name, f"{args.gcs_output_dir}/source/{name}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
