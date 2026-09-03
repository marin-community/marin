# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Development-data regimes for the Delphi 3e18 bank: fit on the panel plus bank rows, score held-out sources.

Regimes (the canonical 280-run panel is in training unless stated):
  panel_only    train on the panel; test on the archive coordinates (reproduces the frozen heldout stage);
  panel_dose    add the corrected dose-response coordinates; test on the archive coordinates;
  loso          add the dose-response coordinates and every archive source but one; test on that source
                (small sources pool into one held-out group); out-of-fold predictions cover the archive bank;
  dose_holdout  add the archive coordinates; test on the dose-response coordinates;
  dose_only     train on the dose-response coordinates alone (no panel); test on the archive coordinates.
A coordinate can belong to several sources (semicolon-separated); it is held out with every source it belongs to
and its pooled out-of-fold prediction comes from the split that held out its primary (first-listed) source. Bank
rows enter training only when every component of the target is complete for the coordinate; other rows are
test-only. Selection metrics are reported pooled over the test rows and within each held-out source, with a
paired bootstrap over sources against the panel_only regime. Every prediction is development evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_olmix_swarm_single_phase_dsp_20260901 import (  # noqa: E402
    block_labels,
)

PANEL = "delphi_3e18_39bucket"
DOSE_SOURCE = "conditional_epoch_dose_response"
MIN_SOURCE = 5
OTHER_GROUP = "other_archive_sources"
REGIMES = ("panel_only", "panel_dose", "loso", "dose_holdout", "dose_only")
SOURCE_BOOTSTRAP_SEED = 20_260_906


@dataclasses.dataclass(frozen=True)
class Union:
    target: str
    features: object
    outcomes: np.ndarray  # rows x components (NaN on test-only rows)
    aggregate: np.ndarray
    memberships: tuple[frozenset[str], ...]  # sources each row belongs to ({"panel"} for panel rows)
    primary: np.ndarray  # first-listed source, pooled into OTHER_GROUP when small
    coordinate_id: np.ndarray
    distance: np.ndarray
    trainable: np.ndarray  # rows with every component of the target; the others are test-only

    def is_dose(self) -> np.ndarray:
        return np.array([DOSE_SOURCE in membership for membership in self.memberships])

    def is_panel(self) -> np.ndarray:
        return np.array(["panel" in membership for membership in self.memberships])


def parse_sources(text: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(text).split(";") if token.strip())


def build_union(panel: harness.BenchPanel, target: str) -> Union:
    _coords, components, _hashes = harness.heldout_registry()
    bank, features = harness.heldout_features(panel, target)
    group = panel.group(target)
    _count, mean_column = harness.HELDOUT_TARGET_COLUMNS[target]
    table = components[components["panel"].eq(PANEL) & components["target"].eq(target)].copy()
    # Some epoch-cap sources store Table-9 components under their short task names.
    full_name = {name.split("/")[-2] if "/" in name else name: name for name in group.components}
    remapped = [name if name in group.components else full_name.get(name) for name in table["component"]]
    unknown = sorted({str(name) for name, mapped in zip(table["component"], remapped, strict=True) if mapped is None})
    if unknown:
        raise ValueError(f"{target}: component names that map to no benchmark component: {unknown[:5]}")
    table["component"] = remapped
    pivot = table.pivot_table(index="coordinate_id", columns="component", values="bpb_mean", aggfunc="first")
    pivot = pivot.reindex(columns=list(group.components))
    complete = pivot.dropna()
    trainable = bank["coordinate_id"].isin(complete.index).to_numpy()
    bank = bank.reset_index(drop=True)
    outcomes = np.full((len(bank), len(group.components)), np.nan)
    outcomes[trainable] = complete.loc[bank["coordinate_id"][trainable]].to_numpy(float)
    aggregate = bank[mean_column].to_numpy(float)
    reconstructed = outcomes[trainable] @ group.aggregation_weights
    if not np.allclose(reconstructed, aggregate[trainable], atol=2e-4):
        raise ValueError(f"{target}: bank components do not reconstruct the aggregate")
    weights = np.vstack([panel.features.weights, features.weights])
    union_features = dataclasses.replace(
        panel.features,
        exposures=weights * panel.features.inventory[None, :],
        weights=weights,
        label=f"{PANEL}|union|{target}",
    )
    memberships = [parse_sources(text) for text in bank["sources"]]
    counts = pd.Series([source for sources in memberships for source in sources]).value_counts()
    primary = []
    for sources in memberships:
        first = sources[0]
        if DOSE_SOURCE in sources:
            primary.append(DOSE_SOURCE)
        elif counts[first] >= MIN_SOURCE:
            primary.append(first)
        else:
            primary.append(OTHER_GROUP)
    distance = np.abs(features.weights[:, None, :] - panel.features.weights[None, :, :]).sum(-1).min(1)
    return Union(
        target=target,
        features=union_features,
        outcomes=np.vstack([group.outcomes, outcomes]),
        aggregate=np.concatenate([group.aggregate, aggregate]),
        memberships=tuple([frozenset({"panel"})] * panel.rows + [frozenset(sources) for sources in memberships]),
        primary=np.concatenate([np.full(panel.rows, "panel"), np.array(primary)]),
        coordinate_id=np.concatenate(
            [np.array([f"panel:{index}" for index in range(panel.rows)]), bank["coordinate_id"].to_numpy(str)]
        ),
        distance=np.concatenate([np.zeros(panel.rows), distance]),
        trainable=np.concatenate([np.ones(panel.rows, dtype=bool), trainable]),
    )


def inner_folds(features, train: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    labels = block_labels(features.weights[train], harness.INNER_FOLDS, harness.HELDOUT_INNER_SEED)
    return tuple((train[labels != index], train[labels == index]) for index in range(harness.INNER_FOLDS))


def fit_predict(model_id: str, union: Union, component_index: int, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    panel = harness.load_panel(PANEL)
    entry = registry.ENTRY_BY_ID[model_id]
    component = panel.group(union.target).components[component_index]
    features = dataclasses.replace(registry.apply_transform(union.features, entry), component=str(component))
    model = entry.build(features)
    fitted = model.fit(
        features,
        union.outcomes[:, component_index],
        train,
        inner_folds(features, train),
        harness._seed(harness.FitTask(model_id, PANEL, union.target, component_index, component, 0, 0)),
    )
    return np.asarray(model.predict(fitted, features, test), dtype=float)


@dataclasses.dataclass(frozen=True)
class Split:
    held_out: str
    train: np.ndarray
    test: np.ndarray
    pooled: np.ndarray  # test rows whose prediction enters the pooled out-of-fold set


def regime_splits(union: Union, regime: str) -> list[Split]:
    panel_rows = union.is_panel()
    dose_rows = union.is_dose()
    archive_rows = ~panel_rows & ~dose_rows
    rows = np.arange(len(union.aggregate))
    trainable = union.trainable
    if regime == "panel_only":
        return [Split("archive", rows[panel_rows], rows[archive_rows], rows[archive_rows])]
    if regime == "panel_dose":
        return [Split("archive", rows[(panel_rows | dose_rows) & trainable], rows[archive_rows], rows[archive_rows])]
    if regime == "dose_only":
        return [Split("archive", rows[dose_rows & trainable], rows[archive_rows], rows[archive_rows])]
    if regime == "dose_holdout":
        return [Split(DOSE_SOURCE, rows[(panel_rows | archive_rows) & trainable], rows[dose_rows], rows[dose_rows])]
    if regime == "loso":
        splits = []
        for label in sorted(set(union.primary[archive_rows])):
            if label == OTHER_GROUP:
                small = {
                    source
                    for membership, primary in zip(union.memberships, union.primary, strict=True)
                    if primary == OTHER_GROUP
                    for source in membership
                }
                held = (
                    np.array([bool(membership & small) for membership in union.memberships]) & ~panel_rows & ~dose_rows
                )
            else:
                held = np.array([label in membership for membership in union.memberships])
            pooled = held & (union.primary == label)
            splits.append(Split(label, rows[~held & trainable], rows[held], rows[pooled]))
        return splits
    raise ValueError(f"unknown regime {regime}")


def source_metrics(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows = []
    for source, subset in frame.groupby("primary"):
        if len(subset) < MIN_SOURCE:
            continue
        row = {"primary": source}
        row.update(
            selection.selection_row(
                subset["measured_mean_bpb"].to_numpy(float), subset["prediction"].to_numpy(float), tolerance
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir", type=Path, required=True, help="heldout registry directory (use the corrected view)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--targets", default="uncheatable,table9")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tag", default="union")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    model_ids = [token.strip() for token in args.models.split(",") if token.strip()]
    regimes = [token.strip() for token in args.regimes.split(",") if token.strip()]
    targets = [token.strip() for token in args.targets.split(",") if token.strip()]
    prediction_frames = []
    for target in targets:
        union = build_union(panel, target)
        group = panel.group(target)
        print(
            f"{target}: union rows {len(union.aggregate)} "
            f"(panel {panel.rows}, bank {len(union.aggregate) - panel.rows}, "
            f"test-only {int((~union.trainable).sum())}); "
            f"groups {pd.Series(union.primary).value_counts().to_dict()}",
            flush=True,
        )
        for regime in regimes:
            for model_id in model_ids:
                jobs = []
                for split in regime_splits(union, regime):
                    for component_index in range(len(group.components)):
                        jobs.append((split, component_index))
                with harness.parallel_config(backend="loky", inner_max_num_threads=1):
                    results = Parallel(n_jobs=args.workers, verbose=0)(
                        delayed(fit_predict)(model_id, union, component_index, split.train, split.test)
                        for split, component_index in jobs
                    )
                by_split: dict[str, tuple[Split, np.ndarray]] = {}
                for (split, component_index), prediction in zip(jobs, results, strict=True):
                    matrix = by_split.setdefault(
                        split.held_out, (split, np.full((len(split.test), len(group.components)), np.nan))
                    )[1]
                    matrix[:, component_index] = prediction
                for split, matrix in by_split.values():
                    predicted = matrix @ group.aggregation_weights
                    keep = np.isin(split.test, split.pooled)
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "model": model_id,
                                "target": target,
                                "regime": regime,
                                "held_out": split.held_out,
                                "coordinate_id": union.coordinate_id[split.test][keep],
                                "primary": union.primary[split.test][keep],
                                "sources": [";".join(sorted(union.memberships[row])) for row in split.test[keep]],
                                "distance_l1": union.distance[split.test][keep],
                                "prediction": predicted[keep],
                                "measured_mean_bpb": union.aggregate[split.test][keep],
                            }
                        )
                    )
                print(f"  {regime} / {model_id}: {len(jobs)} fits done", flush=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / f"{args.tag}_predictions.csv", index=False)
    manifest_path = args.registry_dir.resolve() / "manifest.json"
    (args.output_dir / f"{args.tag}_inputs.json").write_text(
        json.dumps(
            {
                "registry_dir": str(args.registry_dir.resolve()),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            indent=1,
        )
        + "\n"
    )
    metric_rows, source_rows, paired_rows = [], [], []
    rng = np.random.default_rng(SOURCE_BOOTSTRAP_SEED)
    for (model_id, target, regime), subset in predictions.groupby(["model", "target", "regime"]):
        loss = subset["measured_mean_bpb"].to_numpy(float)
        guess = subset["prediction"].to_numpy(float)
        tolerance = harness.BASIN_TOLERANCE_SD * panel.repeat_sd.get(target, float("nan"))
        row = {"model": model_id, "target": target, "regime": regime, "stratum": "pooled_test_rows"}
        row.update(selection.selection_row(loss, guess, tolerance))
        far = subset["distance_l1"].to_numpy(float) >= 0.5
        if far.sum() >= 5:
            row["bias_far"] = float(np.mean(guess[far] - loss[far]))
            row["rmse_far"] = float(np.sqrt(np.mean((guess[far] - loss[far]) ** 2)))
        metric_rows.append(row)
        per_source = source_metrics(subset, tolerance)
        for _index, source_row in per_source.iterrows():
            source_rows.append({"model": model_id, "target": target, "regime": regime, **source_row.to_dict()})
        if not per_source.empty:
            summary = {
                "model": model_id,
                "target": target,
                "regime": regime,
                "stratum": "within_source_mean",
                "sources": len(per_source),
                "regret_at_1": float(per_source["regret_at_1"].mean()),
                "top5_regret": float(per_source["top5_regret"].mean()),
                "frontier_predicted_rank": float(per_source["frontier_predicted_rank"].mean()),
                "spearman": float(per_source["spearman"].mean()),
                "selection_optimism": float(per_source["selection_optimism"].mean()),
            }
            metric_rows.append(summary)
    metrics = pd.DataFrame(metric_rows)
    sources = pd.DataFrame(source_rows)
    # Paired bootstrap over sources: each regime against panel_only, same resampled sources.
    if not sources.empty:
        for (model_id, target), subset in sources.groupby(["model", "target"]):
            wide = subset.pivot(index="primary", columns="regime", values="regret_at_1")
            ranks = subset.pivot(index="primary", columns="regime", values="frontier_predicted_rank")
            if "panel_only" not in wide.columns:
                continue
            names = wide.index.to_numpy()
            draws = rng.integers(0, len(names), size=(args.bootstrap, len(names)))
            for regime in wide.columns:
                if regime == "panel_only":
                    continue
                pair = wide[[regime, "panel_only"]].dropna()
                rank_pair = ranks[[regime, "panel_only"]].dropna()
                if len(pair) < 3:
                    continue
                differences = pair[regime].to_numpy() - pair["panel_only"].to_numpy()
                rank_differences = rank_pair[regime].to_numpy() - rank_pair["panel_only"].to_numpy()
                sample = np.array([differences[d % len(differences)].mean() for d in draws])
                rank_sample = np.array([rank_differences[d % len(rank_differences)].mean() for d in draws])
                paired_rows.append(
                    {
                        "model": model_id,
                        "target": target,
                        "regime": regime,
                        "sources": len(pair),
                        "regret_difference_vs_panel_only": float(differences.mean()),
                        "regret_ci_low": float(np.quantile(sample, 0.025)),
                        "regret_ci_high": float(np.quantile(sample, 0.975)),
                        "sources_better": int((differences < 0).sum()),
                        "sources_worse": int((differences > 0).sum()),
                        "frontier_rank_difference": float(rank_differences.mean()),
                        "frontier_rank_ci_low": float(np.quantile(rank_sample, 0.025)),
                        "frontier_rank_ci_high": float(np.quantile(rank_sample, 0.975)),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    metrics.to_csv(args.output_dir / f"{args.tag}_metrics.csv", index=False)
    sources.to_csv(args.output_dir / f"{args.tag}_source_metrics.csv", index=False)
    paired.to_csv(args.output_dir / f"{args.tag}_paired_sources.csv", index=False)
    pd.set_option("display.width", 250)
    columns = [
        "model",
        "target",
        "regime",
        "stratum",
        "bank_size",
        "sources",
        "regret_at_1",
        "top5_regret",
        "top10_regret",
        "selected_rank",
        "frontier_predicted_rank",
        "selection_optimism",
        "bias",
        "bias_far",
        "rmse",
        "spearman",
        "spearman_best_quartile",
    ]
    print(
        metrics.loc[:, [column for column in columns if column in metrics.columns]]
        .sort_values(["target", "stratum", "regime", "model"])
        .round(4)
        .to_string(index=False)
    )
    if not paired.empty:
        print("\n=== paired over sources against panel_only (difference < 0 favours the regime)")
        print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
