# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline remedies for the WSPU optimism on Table-9, scored by selection value on the development bank.

Every remedy is a transform of the frozen successor's per-component heldout predictions (or of its fitted
per-bucket curves) evaluated on the heldout registry's Table-9 coordinates. Remedies that learn from bank data
(residual calibration, bank-derived reliability weights) are fitted leave-one-source-out, so each coordinate is
scored with a model that never saw its source. Scores are the round-3 selection metrics (regret@1, best-of-5
regret, frontier predicted rank) with a coordinate bootstrap against the uncorrected successor.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_proposals_20260903 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_union_loso_20260903 as loso,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round4_cap_policies_20260903 as policies,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round5_olmix_gap_20260904 as gap,
)

MODEL = "weibull_softplus_unscaled"
DESCRIPTOR_NAMES = (
    "share_cc_high",
    "share_cc_low",
    "share_code",
    "share_curated",
    "share_math",
    "share_synthetic",
    "share_synth_qa",
    "share_olmocr",
    "share_stack",
    "effective_buckets",
    "max_epochs",
    "buckets_beyond_panel",
    "panel_l1_distance",
)
RIDGE_GRID = (0.3, 1.0, 3.0, 10.0)
SHARE_FLOORS = (0.01, 0.02)
KERNEL_BANDWIDTHS = (0.1, 0.2, 0.3)
BOOTSTRAP_DRAWS = 1000
# Table-9 evaluation runs of the matched-seed comparison; coordinates measured by them leave the development bank.
MATCHED_SEED_EVAL_RUNS = ("o518aq9w", "077oz9yd", "esr6cjuw", "22eqjh7q")


@dataclasses.dataclass(frozen=True)
class Curves:
    """Per-component fitted curves of the additive successor, with exact prediction helpers."""

    curves: tuple[proposals.ComponentCurve, ...]

    def component_matrix(
        self, exposures: np.ndarray, share_floor: float = 0.0, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Rows x components predictions; buckets below ``share_floor`` are treated as absent (zero exposure)."""
        values = np.atleast_2d(exposures)
        if share_floor > 0:
            values = np.where(np.atleast_2d(weights) >= share_floor, values, 0.0)
        matrix = np.empty((values.shape[0], len(self.curves)))
        for column, curve in enumerate(self.curves):
            benefit = models.weibull_response(values, curve.shape["rate"], curve.shape["power"])
            harm = models.softplus_harm(values, curve.shape["threshold"])
            contribution = -curve.benefit[None, :] * benefit + curve.harm[None, :] * harm
            matrix[:, column] = curve.intercept + contribution.sum(axis=1)
        return matrix


def descriptors(weights: np.ndarray, panel: harness.BenchPanel, distance: np.ndarray) -> pd.DataFrame:
    buckets = list(panel.buckets)
    types = np.array([policies.bucket_type(bucket) for bucket in buckets])
    exposures = weights * panel.features.inventory[None, :]
    positive = np.where(weights > 0, weights, 1.0)
    table = {f"share_{kind}": weights[:, types == kind].sum(axis=1) for kind in policies.TYPE_ORDER}
    table["share_synth_qa"] = weights[:, buckets.index("dolmino_synth_qa")]
    table["share_olmocr"] = weights[:, buckets.index("dolmino_olmocr_pdfs_hq")]
    table["share_stack"] = weights[:, [buckets.index("dolma3_stack_edu"), buckets.index("dolmino_stack_edu_fim")]].sum(1)
    table["effective_buckets"] = np.exp(-(weights * np.log(positive)).sum(axis=1))
    table["max_epochs"] = exposures.max(axis=1)
    table["buckets_beyond_panel"] = (exposures > panel.features.exposures.max(axis=0)[None, :]).sum(axis=1)
    table["panel_l1_distance"] = distance
    frame = pd.DataFrame(table)
    return frame[list(DESCRIPTOR_NAMES)]


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Standardized ridge regression with an unpenalized intercept; returns (mean, scale, coefficients, intercept)."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - mean) / scale
    centred = y - y.mean()
    coefficients = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ centred)
    return mean, scale, coefficients, float(y.mean())


def ridge_predict(fit: tuple[np.ndarray, np.ndarray, np.ndarray, float], x: np.ndarray) -> np.ndarray:
    mean, scale, coefficients, intercept = fit
    return intercept + ((x - mean) / scale) @ coefficients


def kernel_smooth(
    train_weights: np.ndarray, train_values: np.ndarray, query_weights: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Nadaraya-Watson mean of ``train_values`` under a Gaussian kernel in total-variation distance."""
    distance = 0.5 * np.abs(query_weights[:, None, :] - train_weights[None, :, :]).sum(axis=-1)
    kernel = np.exp(-0.5 * (distance / bandwidth) ** 2)
    return (kernel @ train_values) / np.maximum(kernel.sum(axis=1), 1e-300)


def source_groups(sources: np.ndarray) -> tuple[np.ndarray, list[frozenset[str]]]:
    """Primary group per coordinate (dose, a source with >= MIN_SOURCE members, or the pooled small-source group)."""
    memberships = [frozenset(loso.parse_sources(text)) for text in sources]
    counts = pd.Series([source for sources_ in memberships for source in sources_]).value_counts()
    primary = []
    for text, sources_ in zip(sources, memberships, strict=True):
        first = loso.parse_sources(text)[0]
        if loso.DOSE_SOURCE in sources_:
            primary.append(loso.DOSE_SOURCE)
        elif counts[first] >= loso.MIN_SOURCE:
            primary.append(first)
        else:
            primary.append(loso.OTHER_GROUP)
    return np.array(primary), memberships


def held_out_mask(group: str, primary: np.ndarray, memberships: list[frozenset[str]]) -> np.ndarray:
    """Every coordinate with any membership in the held-out group's sources, not only those pooled under it."""
    if group == loso.DOSE_SOURCE:
        return primary == group
    if group == loso.OTHER_GROUP:
        large = set(primary)
        small = {
            source
            for sources_, label in zip(memberships, primary, strict=True)
            if label == group
            for source in sources_
            if source not in large
        }
        return np.array([bool(sources_ & small) for sources_ in memberships]) & (primary != loso.DOSE_SOURCE)
    return np.array([group in sources_ for sources_ in memberships])


def loso_apply(primary: np.ndarray, memberships: list[frozenset[str]], fit_predict) -> np.ndarray:
    """Out-of-fold values: for each primary group, fit on coordinates with no membership in it, predict its members."""
    out = np.full(len(primary), np.nan)
    for group in sorted(set(primary)):
        held = held_out_mask(group, primary, memberships)
        pooled = primary == group
        out[pooled] = fit_predict(~held, pooled)
    return out


def matched_seed_coordinates(eval_runs: tuple[str, ...]) -> set[str]:
    runs = pd.read_csv(harness.HELDOUT_DIR / "heldout_runs.csv", low_memory=False)
    return set(runs.loc[runs["table9_eval_run_id"].astype(str).isin(eval_runs), "coordinate_id"].astype(str))


def drop_coordinates(bank: selection.Bank, features, drop: set[str]):
    keep = ~np.isin(bank.coordinate_id, sorted(drop))
    trimmed = selection.Bank(
        bank.target,
        bank.coordinate_id[keep],
        bank.measured[keep],
        bank.sources[keep],
        bank.run_count[keep],
        bank.distance[keep],
        bank.tolerance,
    )
    trimmed_features = dataclasses.replace(
        features, exposures=features.exposures[keep], weights=features.weights[keep], label=features.label + "|trimmed"
    )
    return trimmed, trimmed_features, keep


def bank_components(panel: harness.BenchPanel, bank: selection.Bank) -> np.ndarray:
    _coords, components, _hashes = harness.heldout_registry()
    group = panel.group("table9")
    table = components[components["panel"].eq(gap.PANEL) & components["target"].eq("table9")].copy()
    full_name = {gap.short_name(name): name for name in group.components}
    table["component"] = [
        name if name in group.components else full_name[gap.short_name(name)] for name in table["component"]
    ]
    pivot = table.pivot_table(index="coordinate_id", columns="component", values="bpb_mean", aggfunc="first")
    return pivot.reindex(index=bank.coordinate_id, columns=list(group.components)).to_numpy(float)


def shard_matrix(output_dir: Path, panel: harness.BenchPanel, coordinate_id: np.ndarray) -> np.ndarray:
    """Per-component heldout predictions of the frozen successor, rows aligned to ``coordinate_id``."""
    group = panel.group("table9")
    matrix = np.full((len(coordinate_id), len(group.components)), np.nan)
    for index, component in enumerate(group.components):
        payload = harness.load_shard(
            harness.heldout_shard_path(output_dir, MODEL, gap.PANEL, "table9", index, component)
        )
        if payload is None or str(payload["status"].item()) != "ok":
            raise FileNotFoundError(f"missing heldout shard for {component}")
        position = {identifier: row for row, identifier in enumerate(payload["coordinate_id"].astype(str))}
        missing = [identifier for identifier in coordinate_id if identifier not in position]
        if missing:
            raise ValueError(f"shard lacks {len(missing)} registry coordinates")
        matrix[:, index] = payload["prediction"][[position[identifier] for identifier in coordinate_id]]
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument(
        "--shard-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR, help="directory holding heldout_shards"
    )
    parser.add_argument("--output-dir", type=Path, default=gap.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = harness.load_panel(gap.PANEL)
    group = panel.group("table9")
    components = tuple(group.components)
    families = np.array([gap.family(component) for component in components])
    full_bank = selection.load_bank(panel, "table9")
    _frame, full_features = harness.heldout_features(panel, "table9")
    excluded = matched_seed_coordinates(MATCHED_SEED_EVAL_RUNS)
    bank, bank_features, _keep = drop_coordinates(full_bank, full_features, excluded)
    base = shard_matrix(args.shard_dir, panel, bank.coordinate_id)
    observed = bank_components(panel, bank)
    complete = ~np.isnan(observed).any(axis=1)
    macro_base = base.mean(axis=1)
    x = descriptors(bank_features.weights, panel, bank.distance).to_numpy(float)
    groups, memberships = source_groups(bank.sources)
    print(
        f"bank {len(full_bank.measured)} coordinates, {len(excluded)} measured by matched-seed runs dropped, "
        f"{len(bank.measured)} kept, {complete.sum()} with all components; groups {len(set(groups))}"
    )

    with harness.parallel_config(backend="loky", inner_max_num_threads=1):
        fitted = Parallel(n_jobs=args.workers, verbose=5)(
            delayed(proposals.fit_curve)(MODEL, "table9", index, None) for index in range(len(components))
        )
    curves = Curves(tuple(fitted))
    reconstructed = curves.component_matrix(bank_features.exposures)
    curve_gap = float(np.abs(reconstructed - base).max())
    print(f"curve reconstruction vs heldout shards: max abs gap {curve_gap:.2e}")

    predictions: dict[str, np.ndarray] = {"successor": macro_base}
    macro_residual = bank.measured - macro_base

    # R1: residual calibration on mixture descriptors, leave-one-source-out.
    for ridge in RIDGE_GRID:
        predictions[f"residual_calibration@ridge{ridge:g}"] = macro_base + loso_apply(
            groups,
            memberships,
            lambda train, test, r=ridge: ridge_predict(ridge_fit(x[train], macro_residual[train], r), x[test]),
        )
    predictions["residual_mean_shift"] = macro_base + loso_apply(
        groups, memberships, lambda train, test: np.full(test.sum(), macro_residual[train].mean())
    )
    weights_bank = bank_features.weights
    for bandwidth in KERNEL_BANDWIDTHS:
        predictions[f"kernel_residual@tv{bandwidth:g}"] = macro_base + loso_apply(
            groups,
            memberships,
            lambda train, test, h=bandwidth: kernel_smooth(
                weights_bank[train], macro_residual[train], weights_bank[test], h
            ),
        )
        predictions[f"kernel_regression@tv{bandwidth:g}"] = loso_apply(
            groups,
            memberships,
            lambda train, test, h=bandwidth: kernel_smooth(
                weights_bank[train], bank.measured[train], weights_bank[test], h
            ),
        )
    family_names = sorted(set(families))
    family_corrected = np.zeros_like(macro_base)
    for name in family_names:
        columns = families == name
        family_pred = base[:, columns].mean(axis=1)
        family_obs = np.nanmean(observed[:, columns], axis=1)
        residual = family_obs - family_pred

        def fit_family(train, test, residual=residual):
            usable = train & complete
            return ridge_predict(ridge_fit(x[usable], residual[usable], 1.0), x[test])

        family_corrected += (columns.sum() / len(components)) * (
            family_pred + loso_apply(groups, memberships, fit_family)
        )
    predictions["residual_calibration_by_family@ridge1"] = family_corrected

    # R2: reliability-aware aggregation (the observed target stays the unweighted macro).
    repeat_sd = np.array([panel.component_repeat_sd.get(component, np.nan) for component in components])
    inverse_variance = 1.0 / repeat_sd**2
    predictions["reliability_panel_repeat"] = base @ (inverse_variance / inverse_variance.sum())

    def bank_reliability(train, test):
        usable = train & complete
        sd = np.nanstd(observed[usable] - base[usable], axis=0)
        weights = 1.0 / np.maximum(sd, 1e-6) ** 2
        return base[test] @ (weights / weights.sum())

    predictions["reliability_bank_residual"] = loso_apply(groups, memberships, bank_reliability)

    # R3: task-family objectives.
    family_weights = np.array([1.0 / (len(family_names) * (families == name).sum()) for name in families])
    predictions["family_mean_objective"] = base @ family_weights
    non_code = families != "code"
    predictions["macro_excluding_code"] = base[:, non_code].mean(axis=1)

    # R4: extrapolation control on the fitted curves.
    panel_max = panel.features.exposures.max(axis=0)
    panel_p95 = np.quantile(panel.features.exposures, 0.95, axis=0)
    predictions["clamp_exposure_panel_max"] = curves.component_matrix(
        np.minimum(bank_features.exposures, panel_max)
    ).mean(1)
    predictions["clamp_exposure_panel_p95"] = curves.component_matrix(
        np.minimum(bank_features.exposures, panel_p95)
    ).mean(1)
    for floor in SHARE_FLOORS:
        predictions[f"no_credit_below_share{floor:g}"] = curves.component_matrix(
            bank_features.exposures, share_floor=floor, weights=bank_features.weights
        ).mean(axis=1)
    # Combined: calibration on top of the share-floor rule.
    floored = predictions[f"no_credit_below_share{SHARE_FLOORS[-1]:g}"]
    floored_residual = bank.measured - floored
    predictions[f"no_credit_below_share{SHARE_FLOORS[-1]:g}+residual_calibration@ridge1"] = floored + loso_apply(
        groups,
        memberships,
        lambda train, test: ridge_predict(ridge_fit(x[train], floored_residual[train], 1.0), x[test]),
    )

    rows = []
    for name, mask in selection.strata_for(bank):
        for model_id, guess in predictions.items():
            row = {"model": model_id, "stratum": name}
            row.update(selection.selection_row(bank.measured[mask], guess[mask], bank.tolerance))
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "remedies_selection_metrics.csv", index=False)
    boot = []
    for name, mask in selection.strata_for(bank):
        if name in ("pooled", "archive", "dose_response"):
            boot.extend(selection.bootstrap_rows(bank, predictions, "successor", BOOTSTRAP_DRAWS, name, mask))
    bootstrap = pd.DataFrame(boot)
    bootstrap.to_csv(args.output_dir / "remedies_bootstrap.csv", index=False)
    pd.DataFrame(predictions).assign(coordinate_id=bank.coordinate_id, measured=bank.measured, group=groups).to_csv(
        args.output_dir / "remedies_predictions.csv", index=False
    )
    (args.output_dir / "remedies_summary.json").write_text(
        json.dumps(
            {
                "curve_reconstruction_max_abs_gap": curve_gap,
                "descriptors": DESCRIPTOR_NAMES,
                "excluded_matched_seed_coordinates": sorted(excluded),
                "bank_size": len(bank.measured),
            },
            indent=2,
        )
    )
    archive = ~np.array([selection.DOSE_SOURCE in source for source in bank.sources])
    picks = []
    for model_id, guess in predictions.items():
        order = np.where(archive)[0][np.argsort(guess[archive], kind="stable")]
        for rank, index in enumerate(order[:3]):
            picks.append(
                {
                    "model": model_id,
                    "pick": rank + 1,
                    "coordinate_id": bank.coordinate_id[index],
                    "source": bank.sources[index],
                    "measured": float(bank.measured[index]),
                    "predicted": float(guess[index]),
                    "share_synth_qa": float(x[index, DESCRIPTOR_NAMES.index("share_synth_qa")]),
                    "share_stack": float(x[index, DESCRIPTOR_NAMES.index("share_stack")]),
                    "effective_buckets": float(x[index, DESCRIPTOR_NAMES.index("effective_buckets")]),
                }
            )
    pd.DataFrame(picks).to_csv(args.output_dir / "remedies_archive_picks.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    columns = ["regret_at_1", "top5_regret", "frontier_predicted_rank", "spearman", "bias", "rmse", "selected_rank"]
    for name in ("pooled", "archive", "dose_response"):
        print(f"\n== {name} ==")
        print(metrics[metrics["stratum"].eq(name)].set_index("model")[columns].round(4).to_string())
    print("\n== archive top pick per model ==")
    top = pd.DataFrame(picks)
    print(top[top["pick"].eq(1)].drop(columns=["pick", "coordinate_id"]).round(4).to_string(index=False))
    print("\n== bootstrap differences vs successor (negative = better), archive stratum ==")
    view = bootstrap[bootstrap["stratum"].eq("archive") & bootstrap["model"].ne("successor")]
    print(
        view.pivot(index="model", columns="statistic", values=["difference_vs_reference", "share_better_than_reference"])
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
