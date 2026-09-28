# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize a fixed-aggregate phase-contrast DOE around two 3e18 frontier anchors.

The panel is deliberately exploratory. It holds each anchor's realized aggregate
mixture exactly fixed and varies only phase placement, so every paired contrast
estimates an ordering effect rather than an aggregate-mixture effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUT_DIR / "delphi_3e18_frontier_phase_fiber_20260719"
DEFAULT_GCS_OUTPUT_DIR = "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_frontier_phase_fiber_20260719"
ONE_PHASE_DIR = REFERENCE_OUTPUT_DIR / "delphi_one_phase_augmented_swarm_3e18_20260715"
TWO_PHASE_DIR = REFERENCE_OUTPUT_DIR / "delphi_augmented_swarm_3e18_20260714"
HELDOUT_PATH = REFERENCE_OUTPUT_DIR / "delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"

ANCHORS = (
    (
        "uncheatable_frontier",
        "dphase_unch05_tied_3e18",
        "f748072bd43666191101472b3677f0d49249a5cbd34ad2c298f1be9ea068b05a",
        "f0aa700754b154551a59de1c718ec582f76d61523685fb903f1333e68a6d2ea1",
    ),
    (
        "table9_frontier",
        "dphase_t9b075_tied_3e18",
        "7cabc2721c8877129cacb092b87b4d2223485c63c9c1fe17b779b79f081b2a92",
        "8ff1297263e59665ae6c703fa6fe159dd6eef518d425857e50e0e0e1eadc4573",
    ),
)
EXPECTED_ROWS_PER_ANCHOR = 100
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_ANCHOR * len(ANCHORS)
N_SEED_BLOCKS = 4
N_PAIRWISE_DIRECTIONS = 9
PAIRWISE_FEASIBILITY_FRACTION = 0.6
RUN_ID_BASE = 7_190_000
DATA_SEED_BASE = 7_192_000
AGGREGATE_TOLERANCE = 2e-12
SIMPLEX_TOLERANCE = 2e-12
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    run_name: str
    source_run_name: str
    mixture_sha256: str
    weight_vector_sha256: str
    weights: np.ndarray
    uncheatable_3e18: float
    table9_3e18: float
    uncheatable_one_phase_rank: float
    table9_one_phase_rank: float
    one_phase_policy_count: int


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
    direction_scale: float
    phase_0_weights: np.ndarray
    phase_1_weights: np.ndarray


def _realized_phase_fractions() -> tuple[float, float]:
    manifest = pd.read_csv(ONE_PHASE_DIR / "training_manifest.csv")
    alpha0_values = manifest["phase_0_fraction"].drop_duplicates().to_numpy(dtype=float)
    alpha1_values = manifest["phase_1_fraction"].drop_duplicates().to_numpy(dtype=float)
    if len(alpha0_values) != 1 or len(alpha1_values) != 1:
        raise ValueError("One-phase panel does not have one realized phase split")
    alpha0 = float(alpha0_values[0])
    alpha1 = float(alpha1_values[0])
    if abs(alpha0 + alpha1 - 1.0) > 1e-12:
        raise ValueError(f"Realized phase fractions sum to {alpha0 + alpha1}")
    return alpha0, alpha1


def _weight_vector_sha256(domains: tuple[str, ...], weights: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update("\0".join(domains).encode())
    hasher.update(np.asarray(weights, dtype="<f8").tobytes())
    return hasher.hexdigest()


def _one_phase_scores() -> pd.DataFrame:
    heldout = pd.read_csv(HELDOUT_PATH)
    scores = heldout.loc[heldout["policy_class"].eq("single_phase_tied")].copy()
    if scores[["uncheatable_bpb", "table9_macro_bpb"]].isna().any().any():
        raise ValueError("One-phase heldout registry has missing target values")
    scores["uncheatable_one_phase_rank"] = scores["uncheatable_bpb"].rank(method="min")
    scores["table9_one_phase_rank"] = scores["table9_macro_bpb"].rank(method="min")
    return scores


def load_anchors(domains: tuple[str, ...]) -> tuple[list[Anchor], pd.DataFrame]:
    scores = _one_phase_scores()
    anchors = []
    audit_rows = []
    for anchor_id, run_name, expected_mixture_sha256, expected_weight_vector_sha256 in ANCHORS:
        row = scores.loc[scores["wandb_run_base"].eq(run_name)]
        if len(row) != 1:
            raise ValueError(f"Expected one completed score row for {run_name}, found {len(row)}")
        score = row.iloc[0]
        if score["mixture_sha256"] != expected_mixture_sha256:
            raise ValueError(f"Mixture hash changed for {run_name}: {score['mixture_sha256']}")
        phase_0 = json.loads(score["phase_0_weights_json"])
        phase_1 = json.loads(score["phase_1_weights_json"])
        anchor_weights = np.asarray([float(phase_0[domain]) for domain in domains])
        late_weights = np.asarray([float(phase_1[domain]) for domain in domains])
        if np.max(np.abs(anchor_weights - late_weights)) > SIMPLEX_TOLERANCE:
            raise ValueError(f"Anchor {run_name} is not phase tied")
        if np.any(anchor_weights <= 0):
            raise ValueError(f"Anchor {run_name} must have full support for all 39 phase directions")
        weight_vector_sha256 = _weight_vector_sha256(domains, anchor_weights)
        if weight_vector_sha256 != expected_weight_vector_sha256:
            raise ValueError(f"Realized weight-vector hash changed for {run_name}: {weight_vector_sha256}")
        target = anchor_id.removesuffix("_frontier")
        if float(score[f"{target}_one_phase_rank"]) != 1.0:
            raise ValueError(f"Anchor {run_name} is no longer the observed one-phase {target} frontier")
        anchor = Anchor(
            anchor_id=anchor_id,
            run_name=run_name,
            source_run_name=run_name,
            mixture_sha256=expected_mixture_sha256,
            weight_vector_sha256=weight_vector_sha256,
            weights=anchor_weights,
            uncheatable_3e18=float(score["uncheatable_bpb"]),
            table9_3e18=float(score["table9_macro_bpb"]),
            uncheatable_one_phase_rank=float(score["uncheatable_one_phase_rank"]),
            table9_one_phase_rank=float(score["table9_one_phase_rank"]),
            one_phase_policy_count=len(scores),
        )
        anchors.append(anchor)
        audit_rows.append(
            {
                **{field: getattr(anchor, field) for field in anchor.__dataclass_fields__ if field != "weights"},
                "min_weight": float(anchor_weights.min()),
                "max_weight": float(anchor_weights.max()),
                "support_size": int(np.count_nonzero(anchor_weights > 0)),
            }
        )
    anchor_audit = pd.DataFrame(audit_rows)
    return anchors, anchor_audit


def _domain_vs_rest_direction(anchor: np.ndarray, domain_index: int) -> np.ndarray:
    mass = float(anchor[domain_index])
    if not 0 < mass < 1:
        raise ValueError(f"Domain mass must be in (0, 1), found {mass}")
    direction = -anchor * mass / (1.0 - mass)
    direction[domain_index] = mass
    if abs(float(direction.sum())) > 1e-12:
        raise ValueError("Domain-vs-rest direction does not conserve mass")
    return direction


def _high_mass_pair_directions(
    anchor: np.ndarray,
    domains: tuple[str, ...],
    alpha0: float,
) -> list[tuple[str, str, str, np.ndarray, float]]:
    top_indices = np.argsort(anchor)[::-1][: N_PAIRWISE_DIRECTIONS + 1]
    directions = []
    for pair_index, (left, right) in enumerate(pairwise(top_indices)):
        symmetric_limit = min(anchor[left], anchor[right]) / alpha0
        scale = PAIRWISE_FEASIBILITY_FRACTION * symmetric_limit
        direction = np.zeros_like(anchor)
        direction[left] = scale
        direction[right] = -scale
        directions.append(
            (
                f"pair_{pair_index:02d}",
                domains[left],
                domains[right],
                direction,
                scale,
            )
        )
    return directions


def _phases_from_direction(
    anchor: np.ndarray,
    direction: np.ndarray,
    sign: float,
    alpha0: float,
    alpha1: float,
) -> tuple[np.ndarray, np.ndarray]:
    signed_direction = sign * direction
    phase_0 = anchor - alpha1 * signed_direction
    phase_1 = anchor + alpha0 * signed_direction
    return phase_0, phase_1


def build_candidates(
    anchors: list[Anchor],
    domains: tuple[str, ...],
    alpha0: float,
    alpha1: float,
) -> list[Candidate]:
    candidates = []
    for anchor_index, anchor in enumerate(anchors):
        seed_blocks = [DATA_SEED_BASE + 10 * anchor_index + block for block in range(N_SEED_BLOCKS)]
        for block, data_seed in enumerate(seed_blocks):
            candidates.append(
                Candidate(
                    candidate_id=f"fiber_{anchor_index}_center_s{block}",
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
                    direction_scale=0.0,
                    phase_0_weights=anchor.weights.copy(),
                    phase_1_weights=anchor.weights.copy(),
                )
            )

        directions: list[tuple[str, str, str, np.ndarray, float]] = []
        for domain_index, domain in enumerate(domains):
            direction = _domain_vs_rest_direction(anchor.weights, domain_index)
            directions.append((f"domain_{domain_index:02d}", domain, "all other buckets", direction, 1.0))
        directions.extend(_high_mass_pair_directions(anchor.weights, domains, alpha0))
        if len(directions) != 48:
            raise ValueError(f"Expected 48 paired directions, found {len(directions)}")

        for direction_index, (direction_id, late_label, early_label, direction, scale) in enumerate(directions):
            block = direction_index % N_SEED_BLOCKS
            for sign_name, sign_value in (("plus", 1.0), ("minus", -1.0)):
                phase_0, phase_1 = _phases_from_direction(
                    anchor.weights,
                    direction,
                    sign_value,
                    alpha0,
                    alpha1,
                )
                label = f"{late_label} later than {early_label}"
                if sign_name == "minus":
                    label = f"{early_label} later than {late_label}"
                candidates.append(
                    Candidate(
                        candidate_id=f"fiber_{anchor_index}_{direction_id}_{sign_name}",
                        anchor_id=anchor.anchor_id,
                        anchor_run_name=anchor.run_name,
                        anchor_source_run_name=anchor.source_run_name,
                        contrast_family=("domain_vs_rest" if direction_id.startswith("domain_") else "high_mass_pair"),
                        direction_id=direction_id,
                        direction_label=label,
                        sign=sign_name,
                        seed_block=block,
                        data_seed=seed_blocks[block],
                        trainer_seed=0,
                        direction_scale=scale,
                        phase_0_weights=phase_0,
                        phase_1_weights=phase_1,
                    )
                )
        anchor_count = sum(candidate.anchor_id == anchor.anchor_id for candidate in candidates)
        if anchor_count != EXPECTED_ROWS_PER_ANCHOR:
            raise ValueError(f"Expected {EXPECTED_ROWS_PER_ANCHOR} rows for {anchor.anchor_id}, found {anchor_count}")
    if len(candidates) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} candidates, found {len(candidates)}")
    return candidates


def _existing_fit_weights(domains: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(TWO_PHASE_DIR / "phase_weights.csv")
    pivot = frame.pivot_table(index="run_name", columns=["phase", "domain"], values="weight")
    return np.stack(
        [
            pivot["phase_0"].loc[:, list(domains)].to_numpy(dtype=float),
            pivot["phase_1"].loc[:, list(domains)].to_numpy(dtype=float),
        ],
        axis=1,
    )


def _weighted_policy_tv(
    candidates: np.ndarray,
    references: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    delta = np.abs(candidates[:, None] - references[None, :])
    return 0.5 * (alpha0 * delta[:, :, 0].sum(axis=2) + alpha1 * delta[:, :, 1].sum(axis=2))


def validate_candidates(
    candidates: list[Candidate],
    anchors: list[Anchor],
    domains: tuple[str, ...],
    alpha0: float,
    alpha1: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    anchor_lookup = {anchor.anchor_id: anchor for anchor in anchors}
    weights = np.asarray([[candidate.phase_0_weights, candidate.phase_1_weights] for candidate in candidates])
    if np.min(weights) < -SIMPLEX_TOLERANCE:
        index = np.unravel_index(np.argmin(weights), weights.shape)
        raise ValueError(f"Negative phase weight {weights[index]} at {index}")
    phase_sums = weights.sum(axis=2)
    if np.max(np.abs(phase_sums - 1.0)) > SIMPLEX_TOLERANCE:
        raise ValueError("A phase mixture does not sum to one")

    rows = []
    fit_weights = _existing_fit_weights(domains)
    min_fit_tv = _weighted_policy_tv(weights, fit_weights, alpha0, alpha1).min(axis=1)
    for index, candidate in enumerate(candidates):
        anchor = anchor_lookup[candidate.anchor_id]
        aggregate = alpha0 * candidate.phase_0_weights + alpha1 * candidate.phase_1_weights
        aggregate_error = float(np.max(np.abs(aggregate - anchor.weights)))
        if aggregate_error > AGGREGATE_TOLERANCE:
            raise ValueError(f"{candidate.candidate_id} aggregate error is {aggregate_error}")
        phase_tv = float(0.5 * np.abs(candidate.phase_1_weights - candidate.phase_0_weights).sum())
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
                "direction_scale": candidate.direction_scale,
                "phase_tv": phase_tv,
                "phase_information_kl": _phase_information_kl(
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
            }
        )
    manifest = pd.DataFrame(rows)
    _validate_pairing(manifest, weights, alpha0, alpha1)
    direction_rank = {}
    direction_condition = {}
    normalized_direction_condition = {}
    for anchor in anchors:
        mask = manifest["anchor_id"].eq(anchor.anchor_id) & manifest["sign"].eq("plus")
        plus = weights[mask.to_numpy()]
        directions = plus[:, 1, :] - plus[:, 0, :]
        singular_values = np.linalg.svd(directions, compute_uv=False)
        nonzero = singular_values[singular_values > singular_values[0] * 1e-10]
        direction_rank[anchor.anchor_id] = len(nonzero)
        direction_condition[anchor.anchor_id] = float(nonzero[0] / nonzero[-1])
        normalized_directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        normalized_singular_values = np.linalg.svd(normalized_directions, compute_uv=False)
        normalized_nonzero = normalized_singular_values[
            normalized_singular_values > normalized_singular_values[0] * 1e-10
        ]
        normalized_direction_condition[anchor.anchor_id] = float(normalized_nonzero[0] / normalized_nonzero[-1])
        if len(nonzero) != len(domains) - 1:
            raise ValueError(f"{anchor.anchor_id} direction rank is {len(nonzero)}, expected {len(domains) - 1}")
    summary = {
        "panel_rows": len(manifest),
        "anchor_count": len(anchors),
        "rows_per_anchor": manifest.groupby("anchor_id").size().to_dict(),
        "rows_per_contrast_family": manifest.groupby("contrast_family").size().to_dict(),
        "rows_per_seed_block": {
            f"{anchor_id}/seed_{seed_block}": int(count)
            for (anchor_id, seed_block), count in manifest.groupby(["anchor_id", "seed_block"]).size().items()
        },
        "realized_phase_fractions": {"phase_0": alpha0, "phase_1": alpha1},
        "max_aggregate_error": float(manifest["aggregate_max_abs_error"].max()),
        "min_phase_weight": float(weights.min()),
        "max_phase_weight": float(weights.max()),
        "phase_tv_range": [float(manifest["phase_tv"].min()), float(manifest["phase_tv"].max())],
        "min_fit_policy_tv_range": [
            float(manifest["min_fit_policy_tv"].min()),
            float(manifest["min_fit_policy_tv"].max()),
        ],
        "direction_rank": direction_rank,
        "raw_direction_condition_number": direction_condition,
        "normalized_direction_condition_number": normalized_direction_condition,
        "selection_uses_prior_one_phase_3e18_outcomes": True,
        "selection_uses_two_phase_outcomes": False,
        "native_table9_scheduled": True,
    }
    return manifest, weights, summary


def _phase_information_kl(
    phase_0: np.ndarray,
    phase_1: np.ndarray,
    aggregate: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> float:
    value = 0.0
    for alpha, phase in ((alpha0, phase_0), (alpha1, phase_1)):
        positive = phase > 0
        value += alpha * float(np.sum(phase[positive] * np.log(phase[positive] / aggregate[positive])))
    return value


def _validate_pairing(manifest: pd.DataFrame, weights: np.ndarray, alpha0: float, alpha1: float) -> None:
    for (anchor_id, direction_id), group in manifest.loc[~manifest["sign"].eq("center")].groupby(
        ["anchor_id", "direction_id"]
    ):
        if set(group["sign"]) != {"plus", "minus"} or len(group) != 2:
            raise ValueError(f"{anchor_id}/{direction_id} is not an exact +/- pair")
        if group["data_seed"].nunique() != 1:
            raise ValueError(f"{anchor_id}/{direction_id} does not share a data seed")
        indices = group.index.to_numpy(dtype=int)
        midpoint = weights[indices].mean(axis=0)
        aggregate = alpha0 * midpoint[0] + alpha1 * midpoint[1]
        anchor_rows = manifest.loc[
            manifest["anchor_id"].eq(anchor_id) & manifest["contrast_family"].eq("center_control")
        ]
        anchor_index = int(anchor_rows.index[0])
        if np.max(np.abs(midpoint - weights[anchor_index])) > 2e-12:
            raise ValueError(f"{anchor_id}/{direction_id} pair midpoint is not the tied anchor")
        center_aggregate = alpha0 * weights[anchor_index, 0] + alpha1 * weights[anchor_index, 1]
        if np.max(np.abs(aggregate - center_aggregate)) > AGGREGATE_TOLERANCE:
            raise ValueError(f"{anchor_id}/{direction_id} midpoint aggregate changed")


def write_launcher_source_panel(
    output_dir: Path,
    manifest: pd.DataFrame,
    weights: np.ndarray,
    domains: tuple[str, ...],
) -> tuple[Path, str]:
    rows = []
    for row, candidate_weights in zip(manifest.to_dict(orient="records"), weights, strict=True):
        source_row = dict(row)
        for phase_index in (0, 1):
            for domain, weight in zip(domains, candidate_weights[phase_index], strict=True):
                source_row[f"phase_{phase_index}_{domain}"] = float(weight)
        rows.append(source_row)
    source_path = output_dir / "launcher_source_panel.csv"
    pd.DataFrame(rows).to_csv(source_path, index=False)
    return source_path, hashlib.sha256(source_path.read_bytes()).hexdigest()


def write_long_weights(
    output_dir: Path,
    manifest: pd.DataFrame,
    weights: np.ndarray,
    domains: tuple[str, ...],
) -> None:
    rows = []
    for row, candidate_weights in zip(manifest.to_dict(orient="records"), weights, strict=True):
        for phase_index in (0, 1):
            for domain, weight in zip(domains, candidate_weights[phase_index], strict=True):
                rows.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "anchor_id": row["anchor_id"],
                        "contrast_family": row["contrast_family"],
                        "direction_id": row["direction_id"],
                        "sign": row["sign"],
                        "phase": phase_index,
                        "domain": domain,
                        "weight": float(weight),
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "phase_weights.csv", index=False)


def render_diagnostics(manifest: pd.DataFrame, anchor_audit: pd.DataFrame, output_dir: Path) -> None:
    plot = manifest.loc[~manifest["contrast_family"].eq("center_control")].copy()
    figure = px.scatter(
        plot,
        x="phase_tv",
        y="min_fit_policy_tv",
        color="phase_information_kl",
        symbol="contrast_family",
        facet_col="anchor_id",
        hover_name="candidate_id",
        hover_data=["direction_label", "sign", "seed_block", "max_simulated_epoch"],
        color_continuous_scale="RdYlGn_r",
        title="Frontier phase-fiber DOE: contrast strength and distance from existing 3e18 support",
    )
    figure.update_xaxes(title_text="phase TV between phase 0 and phase 1")
    figure.update_yaxes(title_text="nearest weighted policy TV to 280-row two-phase fit panel")
    figure.update_layout(
        width=1500,
        height=760,
        margin={"l": 70, "r": 140, "t": 130, "b": 70},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.07, "yanchor": "bottom"},
        coloraxis_colorbar={"title": "phase-information KL", "x": 1.03},
    )
    figure.write_html(output_dir / "phase_fiber_diagnostics.html", include_plotlyjs=True, config=PLOT_CONFIG)

    anchor_plot = px.scatter(
        anchor_audit,
        x="uncheatable_one_phase_rank",
        y="table9_one_phase_rank",
        color="anchor_id",
        size="max_weight",
        hover_name="source_run_name",
        hover_data=["uncheatable_3e18", "table9_3e18", "mixture_sha256"],
        title="Empirical one-phase frontier anchors in the append-only 3e18 registry",
    )
    anchor_plot.update_xaxes(title_text="Uncheatable rank among prior one-phase 3e18 checkpoints")
    anchor_plot.update_yaxes(title_text="Table-9 rank among prior one-phase 3e18 checkpoints")
    anchor_plot.update_layout(width=900, height=650)
    anchor_plot.write_html(output_dir / "anchor_audit.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    manifest: pd.DataFrame,
    anchor_audit: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    composition = manifest.groupby(["anchor_id", "contrast_family"]).size().rename("rows").reset_index()
    geometry = (
        manifest.groupby(["anchor_id", "contrast_family"])[
            ["phase_tv", "phase_information_kl", "min_fit_policy_tv", "max_simulated_epoch"]
        ]
        .agg(["min", "median", "max"])
        .round(6)
    )
    lines = [
        "# Delphi 3e18 frontier phase-fiber DOE",
        "",
        "## Scientific question",
        "",
        (
            "At a fixed aggregate mixture, does phase placement produce a systematic performance distribution, "
            "and which buckets benefit from early versus late placement near robust one-phase frontier mixtures?"
        ),
        "",
        "This is an exploratory causal DOE, not evidence that tied-first acquisition is already more sample-efficient "
        "than the one-stage two-phase swarm. The prior fixed-budget audit failed that stronger gate.",
        "",
        "## Parameterization",
        "",
        "For realized phase fractions $\\alpha_0$ and $\\alpha_1$, each schedule uses",
        "",
        "$$w^{(0)}=a-\\alpha_1d,\\qquad w^{(1)}=a+\\alpha_0d,\\qquad \\mathbf{1}^Td=0.$$",
        "",
        "Therefore $\\alpha_0w^{(0)}+\\alpha_1w^{(1)}=a$ exactly. The two signs of every direction share a data seed.",
        "",
        "## Anchor selection",
        "",
        (
            "Anchors are the best observed full-support phase-tied policies for each target in the append-only 3e18 "
            "registry, not optima selected from a fitted two-phase surrogate. Selection therefore uses prior one-phase "
            "outcomes and makes this panel exploratory. It does not use any prior two-phase outcome. Four fresh tied "
            "controls per anchor test whether the selected score was a winner's-curse draw."
        ),
        "",
        anchor_audit.drop(columns=[]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Per-anchor design",
        "",
        "- 4 fresh phase-tied controls, one for each data-seed block.",
        (
            "- 39 one-vs-rest directions in both signs. Positive sign places the named bucket later; "
            "negative sign places it earlier."
        ),
        (
            "- 9 sparse high-mass bucket-to-bucket directions in both signs, connecting adjacent buckets "
            "among the anchor's top 10 masses."
        ),
        "- 48 paired directions are balanced across four seed blocks (12 pairs per block).",
        "- Every checkpoint produces both Uncheatable and native Table-9, regardless of the anchor-selection label.",
        "",
        composition.to_markdown(index=False),
        "",
        "## Local construction audit",
        "",
        f"- Panel rows: {summary['panel_rows']}.",
        f"- Maximum aggregate-coordinate error: {summary['max_aggregate_error']:.3e}.",
        f"- Minimum phase weight: {summary['min_phase_weight']:.6g}.",
        f"- Phase-TV range: {summary['phase_tv_range']}.",
        f"- Direction rank: {summary['direction_rank']} (expected 38 per anchor).",
        f"- Unit-direction condition number: {summary['normalized_direction_condition_number']}.",
        (
            "- Raw direction condition numbers are recorded in `summary.json`; their larger spread reflects "
            "intentionally different bucket masses, not angular collinearity."
        ),
        (
            "- The nearest-policy TV range is deliberately large: these empirical frontier aggregates are outside "
            "the original random two-phase fit panel. This is an extrapolative causal probe, not an interpolation test."
        ),
        "",
        geometry.to_markdown(floatfmt=".6f"),
        "",
        "## Planned estimands",
        "",
        "For every seed-matched pair, $(Y_+-Y_-)/2$ estimates the odd ordering effect. "
        "$Y_++Y_--2Y_0$ estimates local phase curvature using the tied control from the same seed block. "
        "The four fresh centers also reveal whether an anchor's original low score was a winner's-curse draw.",
        "",
        "## Interpretation boundary",
        "",
        (
            "The 200 outcomes become append-only heldouts for future surrogate work after this preregistered analysis. "
            "They may diagnose model failure and motivate a later model, but they must not be retrospectively "
            "called confirmatory evidence for that model."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def upload_artifact(local_path: Path, remote_path: str) -> None:
    with local_path.open("rb") as source, fsspec.open(remote_path, "wb") as destination:
        destination.write(source.read())


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
    alpha0, alpha1 = _realized_phase_fractions()
    anchors, anchor_audit = load_anchors(domains)
    candidates = build_candidates(anchors, domains, alpha0, alpha1)
    manifest, weights, summary = validate_candidates(candidates, anchors, domains, alpha0, alpha1)
    manifest_path = args.output_dir / "candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    anchor_audit.to_csv(args.output_dir / "anchor_audit.csv", index=False)
    write_long_weights(args.output_dir, manifest, weights, domains)
    source_path, source_sha256 = write_launcher_source_panel(args.output_dir, manifest, weights, domains)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary.update(
        {
            "candidate_manifest_sha256": manifest_sha256,
            "launcher_source_panel_sha256": source_sha256,
            "gcs_launcher_source_panel": f"{args.gcs_output_dir}/source/launcher_source_panel-{source_sha256[:16]}.csv",
            "gcs_candidate_manifest": f"{args.gcs_output_dir}/source/candidate_manifest-{manifest_sha256[:16]}.csv",
        }
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    render_diagnostics(manifest, anchor_audit, args.output_dir)
    write_report(args.output_dir, manifest, anchor_audit, summary)
    if args.upload:
        upload_artifact(source_path, str(summary["gcs_launcher_source_panel"]))
        upload_artifact(manifest_path, str(summary["gcs_candidate_manifest"]))
        for name in ("summary.json", "report.md", "phase_weights.csv", "anchor_audit.csv"):
            upload_artifact(args.output_dir / name, f"{args.gcs_output_dir}/source/{name}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
