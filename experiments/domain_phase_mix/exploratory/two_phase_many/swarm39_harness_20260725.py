# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Shared evaluation harness for the 39-bucket swarm at 60M, 300M, and 3e18.

Protocol
--------
Fit on the 280-row two-phase panel of a scale, then evaluate on every
coordinate-disjoint heldout observation for that scale. Heldouts are stratified
by proposal series, policy class, and support distance, because the archive is
intervention-designed rather than IID and pooled metrics hide the failure modes
that matter for deployment.

Model interface
---------------
A model is a callable ``build(panel) -> Design`` producing a nonnegative design
whose head is fitted by nonnegative least squares with an intercept, plus a
``shapes()`` generator for any nonlinear hyperparameters. Nonnegativity is the
mechanistic constraint that benefits reduce loss and penalties increase it; it is
what keeps the fitted response interpretable.

Exposure conventions
--------------------
``E_i = c0_i * p0_i + c1_i * p1_i`` is simulated epochs for bucket ``i``. Because
``c1 = c0 (1 - alpha) / alpha`` in this swarm, ``E`` depends only on the aggregate
mixture, so phase order never changes physical exposure. Roughly half of all
(policy, bucket) cells in the Delphi fit panel exceed one epoch, so unique
coverage and repetition are materially different quantities here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PACKET = REFERENCE_OUTPUTS / "two_phase_surrogate_collaborator_packet_20260721"
CANONICAL = PACKET / "data" / "canonical"
CATALOG = PACKET / "data" / "catalog.json"
SIXTY_M = REFERENCE_OUTPUTS / "60m_39bucket_checkpoint_audit_20260724"
DELPHI_HELDOUTS = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"

SEALED_SERIES_FRAGMENT = "targeted_pairwise"
UNCHEATABLE = "uncheatable_bpb"
TABLE9 = "table9_macro_bpb"

# The proportional policy weights each bucket by its available tokens, so it gives
# every bucket the same epoch count. Both the Delphi 3e18 and 300M swarms
# subsampled their pools so that this shared value is 0.9054 epochs, which is how
# repetition is held constant across scales. Verified against the pipeline's own
# per-bucket `simulated_epochs` and `epoch_multiplier` columns:
# simulated_epochs / (aggregate_weight / proportional) is constant at 0.905353.
# The catalog's c0/c1 are proportional to 1/proportional_weight but carry
# different units per dataset, so they are renormalized to this anchor below.
PROPORTIONAL_POLICY_EPOCHS = 0.905353
EPOCH_ANCHOR_TOLERANCE = 1e-6

SCALE_ALPHA = {"60m": 0.80, "300m": 0.80, "delphi_3e18": 0.7981376787495837}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_sealed_absent(frame: pd.DataFrame, label: str) -> None:
    text = frame.select_dtypes(include=["object", "string"])
    if text.empty:
        return
    hit = text.astype("string").apply(lambda column: column.str.contains(SEALED_SERIES_FRAGMENT, case=False, na=False))
    assert not bool(hit.to_numpy().any()), f"sealed rows present in {label}"


@dataclass(frozen=True)
class Panel:
    """Policies plus outcomes for one scale and split."""

    scale: str
    split: str
    alpha: float
    buckets: tuple[str, ...]
    c0: np.ndarray
    c1: np.ndarray
    family_index: np.ndarray
    family_names: tuple[str, ...]
    phase0: np.ndarray
    phase1: np.ndarray
    targets: dict[str, np.ndarray]
    series: np.ndarray
    policy_class: np.ndarray
    group: np.ndarray
    row_id: np.ndarray

    @property
    def aggregate(self) -> np.ndarray:
        return self.alpha * self.phase0 + (1.0 - self.alpha) * self.phase1

    @property
    def contrast(self) -> np.ndarray:
        return self.phase1 - self.phase0

    @property
    def phase_tv(self) -> np.ndarray:
        return 0.5 * np.abs(self.contrast).sum(axis=1)

    @property
    def epochs(self) -> np.ndarray:
        """Simulated epochs per bucket; a function of the aggregate alone."""
        return self.c0 * self.phase0 + self.c1 * self.phase1

    @property
    def proportional(self) -> np.ndarray:
        """Token-proportional mixture, the policy at which every bucket sees one pass."""
        inverse = 1.0 / self.c0
        return inverse / inverse.sum()

    @property
    def oversampling(self) -> np.ndarray:
        """Aggregate weight relative to proportional; the pipeline's epoch multiplier."""
        return self.aggregate / self.proportional

    def family_pool(self, values: np.ndarray) -> np.ndarray:
        out = np.zeros((values.shape[0], len(self.family_names)))
        for index in range(len(self.family_names)):
            out[:, index] = values[:, self.family_index == index].sum(axis=1)
        return out

    def subset(self, mask: np.ndarray) -> Panel:
        return Panel(
            scale=self.scale,
            split=self.split,
            alpha=self.alpha,
            buckets=self.buckets,
            c0=self.c0,
            c1=self.c1,
            family_index=self.family_index,
            family_names=self.family_names,
            phase0=self.phase0[mask],
            phase1=self.phase1[mask],
            targets={k: v[mask] for k, v in self.targets.items()},
            series=self.series[mask],
            policy_class=self.policy_class[mask],
            group=self.group[mask],
            row_id=self.row_id[mask],
        )

    def __len__(self) -> int:
        return len(self.row_id)


def _catalog_spec(dataset_id: str) -> dict:
    return json.loads(CATALOG.read_text())["datasets"][dataset_id]


def _family_index(domains: tuple[str, ...], families: dict[str, list[str]]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(families)
    lookup = {bucket: names.index(name) for name, members in families.items() for bucket in members}
    return np.asarray([lookup[b] for b in domains], dtype=int), names


def _exposure(dataset_id: str) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Bucket order, epoch conversions anchored to the proportional policy, and families.

    The catalog's exposure multipliers are proportional to the reciprocal of the
    proportional weight but are not in consistent units across datasets, so they
    are rescaled here to make the proportional policy land on
    ``PROPORTIONAL_POLICY_EPOCHS``. That anchor is a property of the swarm design
    rather than of a fitted model, and it makes epochs comparable across scales.

    At the proportional policy both phases run the same mixture, so the anchor is
    ``(c0 + c1) . p_prop`` with no phase weighting.
    """
    spec = _catalog_spec(dataset_id)
    domains = tuple(spec["domains"])
    c0 = np.asarray(spec["c0"], dtype=float)
    c1 = np.asarray(spec["c1"], dtype=float)
    proportional = (1.0 / c0) / (1.0 / c0).sum()
    raw = (c0 + c1) * proportional
    spread = float(raw.max() / raw.min() - 1.0)
    assert spread < EPOCH_ANCHOR_TOLERANCE, f"{dataset_id}: proportional epochs not constant (spread {spread:.2e})"
    scale = PROPORTIONAL_POLICY_EPOCHS / float(raw.mean())
    family_index, family_names = _family_index(domains, spec["families"])
    return domains, c0 * scale, c1 * scale, family_index, family_names


def _panel_from_weight_columns(
    scale: str,
    split: str,
    frame: pd.DataFrame,
    domains: tuple[str, ...],
    c0: np.ndarray,
    c1: np.ndarray,
    family_index: np.ndarray,
    family_names: tuple[str, ...],
    prefix0: str,
    prefix1: str,
    target_columns: dict[str, str],
) -> Panel:
    phase0 = frame[[f"{prefix0}{b}" for b in domains]].to_numpy(float)
    phase1 = frame[[f"{prefix1}{b}" for b in domains]].to_numpy(float)
    targets = {}
    for name, column in target_columns.items():
        targets[name] = frame[column].to_numpy(float) if column in frame else np.full(len(frame), np.nan)
    series = (
        frame["training_series"].astype(str).to_numpy()
        if "training_series" in frame
        else frame.get("source_family", pd.Series(["unknown"] * len(frame))).astype(str).to_numpy()
    )
    policy_class = (
        frame["policy_class"].astype(str).to_numpy() if "policy_class" in frame else np.array(["unknown"] * len(frame))
    )
    for candidate in ("group_id", "policy_hash", "mixture_sha256", "row_id", "heldout_id"):
        if candidate in frame:
            group = frame[candidate].astype(str).to_numpy()
            break
    else:
        group = np.arange(len(frame)).astype(str)
    for candidate in ("row_id", "heldout_id", "observation_id", "policy_hash"):
        if candidate in frame:
            row_id = frame[candidate].astype(str).to_numpy()
            break
    else:
        row_id = np.arange(len(frame)).astype(str)
    return Panel(
        scale=scale,
        split=split,
        alpha=SCALE_ALPHA[scale],
        buckets=domains,
        c0=c0,
        c1=c1,
        family_index=family_index,
        family_names=family_names,
        phase0=phase0,
        phase1=phase1,
        targets=targets,
        series=series,
        policy_class=policy_class,
        group=group,
        row_id=row_id,
    )


def load_scale(scale: str) -> tuple[Panel, Panel]:
    """Return the (fit, heldout) panel pair for a scale."""
    if scale == "delphi_3e18":
        domains, c0, c1, family_index, family_names = _exposure("delphi_3e18_two_phase_fit")
        fit_frame = pd.read_csv(CANONICAL / "delphi_3e18_two_phase_fit.csv")
        assert_sealed_absent(fit_frame, "delphi fit")
        fit = _panel_from_weight_columns(
            scale,
            "fit",
            fit_frame,
            domains,
            c0,
            c1,
            family_index,
            family_names,
            "phase_0_weight::",
            "phase_1_weight::",
            {UNCHEATABLE: UNCHEATABLE, TABLE9: TABLE9},
        )
        held_frame = pd.read_csv(DELPHI_HELDOUTS)
        assert_sealed_absent(held_frame, "delphi heldouts")
        held_frame = held_frame[held_frame["fit_panel_overlap"] == "coordinate_disjoint"].reset_index(drop=True)

        # Heldout weights are stored as bucket-name keyed objects; project them
        # onto the canonical bucket order so every panel shares one coordinate
        # system, and treat an absent bucket as zero weight.
        def to_vector(payload: str) -> np.ndarray:
            mapping = json.loads(payload)
            return np.asarray([float(mapping.get(bucket, 0.0)) for bucket in domains], dtype=float)

        phase0 = np.stack(held_frame["phase_0_weights_json"].map(to_vector).to_numpy())
        phase1 = np.stack(held_frame["phase_1_weights_json"].map(to_vector).to_numpy())
        for name, matrix in (("phase 0", phase0), ("phase 1", phase1)):
            error = float(np.abs(matrix.sum(axis=1) - 1.0).max())
            assert error < 1e-6, f"delphi heldout {name} weights are not normalized (max error {error:.2e})"
        heldout = Panel(
            scale=scale,
            split="heldout",
            alpha=SCALE_ALPHA[scale],
            buckets=domains,
            c0=c0,
            c1=c1,
            family_index=family_index,
            family_names=family_names,
            phase0=phase0,
            phase1=phase1,
            targets={UNCHEATABLE: held_frame[UNCHEATABLE].to_numpy(float), TABLE9: held_frame[TABLE9].to_numpy(float)},
            series=held_frame["training_series"].astype(str).to_numpy(),
            policy_class=held_frame["policy_class"].astype(str).to_numpy(),
            group=held_frame["mixture_sha256"].astype(str).to_numpy(),
            row_id=held_frame["heldout_id"].astype(str).to_numpy(),
        )
        return fit, heldout

    if scale == "300m":
        domains, c0, c1, family_index, family_names = _exposure("300m_two_phase_fit")
        fit_frame = pd.read_csv(CANONICAL / "300m_two_phase_fit.csv")
        held_frame = pd.read_csv(CANONICAL / "300m_heldouts.csv")
        for frame, label in ((fit_frame, "300m fit"), (held_frame, "300m heldouts")):
            assert_sealed_absent(frame, label)
        args = (domains, c0, c1, family_index, family_names, "phase_0_weight::", "phase_1_weight::")
        targets = {UNCHEATABLE: UNCHEATABLE, TABLE9: TABLE9}
        return (
            _panel_from_weight_columns(scale, "fit", fit_frame, *args, targets),
            _panel_from_weight_columns(scale, "heldout", held_frame, *args, targets),
        )

    if scale == "60m":
        domains, c0, c1, family_index, family_names = _exposure("delphi_3e18_two_phase_fit")
        fit_frame = pd.read_csv(SIXTY_M / "fit_two_phase.csv")
        held_frame = pd.read_csv(SIXTY_M / "heldout_observations.csv")
        for frame, label in ((fit_frame, "60m fit"), (held_frame, "60m heldouts")):
            assert_sealed_absent(frame, label)
        args = (domains, c0, c1, family_index, family_names, "phase_0_", "phase_1_")
        targets = {UNCHEATABLE: UNCHEATABLE, TABLE9: TABLE9}
        return (
            _panel_from_weight_columns(scale, "fit", fit_frame, *args, targets),
            _panel_from_weight_columns(scale, "heldout", held_frame, *args, targets),
        )

    raise ValueError(f"unknown scale {scale!r}")


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """Nonnegative design block plus feature names."""

    matrix: np.ndarray
    names: tuple[str, ...]


@dataclass(frozen=True)
class Model:
    """A named design builder with an optional nonlinear shape grid and link.

    ``link`` selects how the response is parameterized. ``"identity"`` fits BPB
    directly, which is what every Observatory baseline does. ``"log_deficit"``
    fits ``log(BPB - floor)`` and predicts ``floor + exp(eta)``, so the response
    is multiplicative in reducible loss and is bounded below by ``floor``. That
    bound is structural rather than fitted: an additive model can predict below
    any entropy floor, which is the mechanism behind severe out-of-support
    optimism, and a multiplicative one cannot.

    The floor is read from ``shape["deficit_floor_fraction"]`` as a fraction of the
    smallest observed target value on the fitting panel.
    """

    name: str
    build: Callable[[Panel, dict], Design]
    shapes: Callable[[], Iterable[dict]] = field(default=lambda: ({},))
    l2_grid: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    # Optional replacement for the module-level nonnegative least-squares head. A model whose
    # definition includes how its coefficients are estimated -- a robust loss, a different column
    # scaling -- must be able to supply that here, or this harness silently fits a different estimator
    # than the one the model is elsewhere benchmarked as.
    head: Callable[[np.ndarray, np.ndarray, float, np.ndarray | None], tuple[float, np.ndarray]] | None = None
    link: str = "identity"
    penalty_scale: Callable[[Panel, dict], np.ndarray] | None = None
    """Optional per-column ridge multipliers.

    A uniform ridge treats every coefficient as exchangeable. Supplying multipliers
    makes the prior hierarchical: penalizing per-bucket deviations more heavily than
    pooled family terms shrinks buckets toward their family mean, which interpolates
    between a flat per-bucket field and a purely pooled one.
    """


@dataclass(frozen=True)
class Fit:
    model: str
    shape: dict
    l2: float
    intercept: float
    coefficients: np.ndarray
    names: tuple[str, ...]
    oof_rmse: float
    floor: float = 0.0

    def predict(self, panel: Panel, model: Model) -> np.ndarray:
        design = model.build(panel, self.shape)
        eta = self.intercept + design.matrix @ self.coefficients
        if model.link == "log_deficit":
            return self.floor + np.exp(np.clip(eta, -30.0, 30.0))
        return eta


def fit_head(
    design: np.ndarray, target: np.ndarray, l2: float, penalty_scale: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Nonnegative least squares on a column-scaled design with a free intercept.

    ``penalty_scale`` multiplies the ridge column by column, which is how a
    hierarchical prior is expressed: large values pull a coefficient toward zero in
    the chosen parameterization, so penalizing residual terms shrinks buckets toward
    their pooled family value.
    """
    scale = np.maximum(np.sqrt((design**2).mean(axis=0)), 1e-12)
    scaled = design / scale
    design_mean = scaled.mean(axis=0, keepdims=True)
    target_mean = float(target.mean())
    centered_design = scaled - design_mean
    centered_target = target - target_mean
    if l2 > 0.0:
        multipliers = np.ones(design.shape[1]) if penalty_scale is None else np.asarray(penalty_scale, dtype=float)
        assert len(multipliers) == design.shape[1], "penalty_scale must have one entry per design column"
        centered_design = np.vstack([centered_design, np.diag(np.sqrt(l2 * multipliers))])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    coefficients, _ = nnls(centered_design, centered_target, maxiter=80 * design.shape[1])
    coefficients = coefficients / scale
    # ``keepdims`` leaves a length-one axis, and NumPy 2.5 made ``float()`` on a non-scalar array an
    # error rather than a warning, so the single element is extracted explicitly.
    intercept = target_mean - float((design.mean(axis=0, keepdims=True) @ coefficients).item())
    return intercept, coefficients


def link_floor(model: Model, shape: dict, observed: np.ndarray) -> float:
    """Lower bound for the log-deficit link, as a fraction of the smallest target."""
    if model.link != "log_deficit":
        return 0.0
    return float(shape["deficit_floor_fraction"]) * float(np.min(observed))


def to_link(model: Model, observed: np.ndarray, floor: float) -> np.ndarray:
    if model.link != "log_deficit":
        return observed
    return np.log(np.maximum(observed - floor, 1e-9))


def from_link(model: Model, eta: np.ndarray, floor: float) -> np.ndarray:
    if model.link != "log_deficit":
        return eta
    return floor + np.exp(np.clip(eta, -30.0, 30.0))


def grouped_splits(panel: Panel, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold over policy groups.

    Uses the same splitter as the collaborator packet so that shape and ridge
    selection is identical between the two implementations; a divergent split
    would make baseline out-of-fold numbers incomparable.
    """
    groups = np.asarray(panel.group)
    rows = len(groups)
    if len(np.unique(groups)) >= n_splits and len(np.unique(groups)) < rows:
        splitter = GroupKFold(n_splits=n_splits)
        indices = list(splitter.split(np.zeros((rows, 1)), None, groups))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        indices = list(splitter.split(np.zeros((rows, 1))))
    folds = []
    for train_index, test_index in indices:
        train = np.zeros(rows, dtype=bool)
        test = np.zeros(rows, dtype=bool)
        train[train_index] = True
        test[test_index] = True
        folds.append((train, test))
    return folds


def mixture_blocked_splits(panel: Panel, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold over contiguous blocks of mixture space rather than over policy groups.

    Grouped splits keep a policy's own replicates together but say nothing about how close two
    *different* policies are, so a held-out fold routinely contains mixtures that sit next to a
    training row. On a densely sampled panel that makes out-of-fold error an optimistic estimate of
    error on genuinely new mixtures, and it rewards extra capacity that will not generalize.

    Blocking by position in mixture space instead gives each fold a region of the simplex that the
    training folds do not cover, which is the structure the coordinate-disjoint heldout panel has.
    Selection made against these folds is therefore comparable to what the heldout panel will show,
    and no heldout information is used to make it.
    """
    coordinates = np.column_stack([panel.phase0, panel.phase1])
    blocks = KMeans(n_clusters=n_splits, n_init=10, random_state=seed).fit_predict(coordinates)
    rows = len(blocks)
    folds = []
    for block in np.unique(blocks):
        test = blocks == block
        if not test.any() or test.all():
            continue
        folds.append((~test, test))
    assert len(folds) >= 2, f"mixture blocking produced {len(folds)} usable folds"
    assert rows == len(blocks), "blocking must label every row"
    return folds


def fit_model(
    panel: Panel,
    model: Model,
    target: str,
    n_splits: int = 5,
    seed: int = 0,
    split_fn: Callable[[Panel, int, int], list[tuple[np.ndarray, np.ndarray]]] = grouped_splits,
) -> Fit:
    """Select shape and ridge by out-of-fold RMSE on ``split_fn`` folds, then refit on everything."""
    observed = panel.targets[target]
    usable = np.isfinite(observed)
    panel = panel.subset(usable)
    observed = panel.targets[target]
    splits = split_fn(panel, n_splits, seed)
    best: tuple[float, dict, float] | None = None
    for shape in model.shapes():
        design = model.build(panel, shape).matrix
        multipliers = None if model.penalty_scale is None else model.penalty_scale(panel, shape)
        floor = link_floor(model, shape, observed)
        response = to_link(model, observed, floor)
        for l2 in model.l2_grid:
            errors = []
            for train, test in splits:
                solve = model.head or fit_head
                intercept, coefficients = solve(design[train], response[train], l2, multipliers)
                predicted = from_link(model, intercept + design[test] @ coefficients, floor)
                errors.append(predicted - observed[test])
            score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
            if best is None or score < best[0]:
                best = (score, shape, l2)
    assert best is not None, f"{model.name}: empty shape grid"
    score, shape, l2 = best
    design = model.build(panel, shape)
    floor = link_floor(model, shape, observed)
    solve = model.head or fit_head
    intercept, coefficients = solve(
        design.matrix,
        to_link(model, observed, floor),
        l2,
        None if model.penalty_scale is None else model.penalty_scale(panel, shape),
    )
    return Fit(
        floor=floor,
        model=model.name,
        shape=shape,
        l2=l2,
        intercept=intercept,
        coefficients=coefficients,
        names=design.names,
        oof_rmse=score,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def support_distance(reference: Panel, query: Panel) -> np.ndarray:
    """Nearest-neighbour L1 distance in aggregate mixture space to the fit panel."""
    a = query.aggregate
    b = reference.aggregate
    out = np.empty(len(a))
    for index in range(len(a)):
        out[index] = np.abs(b - a[index]).sum(axis=1).min()
    return out


def metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = predicted - observed
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.ptp(predicted) > 0 else float("nan")
    ranked = np.argsort(predicted)
    best = float(np.min(observed))
    row: dict[str, float | int] = {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "calibration_slope": slope,
        "optimism_over_0p05": int(np.count_nonzero(observed - predicted > 0.05)),
        "worst_optimism": float(np.max(observed - predicted)),
    }
    for k in (1, 3, 5):
        row[f"regret_at_{k}"] = float(np.min(observed[ranked[: min(k, len(observed))]]) - best)
    tail = ranked[: max(5, math.ceil(0.15 * len(observed)))]
    row["low_tail_rmse"] = float(np.sqrt(np.mean(residual[tail] ** 2)))
    row["low_tail_optimism"] = float(np.mean(np.maximum(observed[tail] - predicted[tail], 0.0)))
    return row


def evaluate(
    fit: Fit,
    model: Model,
    fit_panel: Panel,
    heldout: Panel,
    target: str,
    support: np.ndarray | None = None,
) -> pd.DataFrame:
    """Pooled and stratified heldout metrics for one fitted model and target."""
    usable = np.isfinite(heldout.targets[target])
    panel = heldout.subset(usable)
    observed = panel.targets[target]
    predicted = fit.predict(panel, model)
    # `support` is expected to be aligned to this already-subset panel, since it is
    # cached per (scale, target) by the caller.
    if support is None:
        distance = support_distance(fit_panel, panel)
    else:
        assert len(support) == len(panel), f"support array has {len(support)} rows for {len(panel)} observations"
        distance = support
    rows = [
        {
            "scale": panel.scale,
            "target": target,
            "model": fit.model,
            "stratum_type": "pooled",
            "stratum": "all_coordinate_disjoint",
            "oof_rmse": fit.oof_rmse,
            **metric_row(observed, predicted),
        }
    ]
    for label, values in (("policy_class", panel.policy_class), ("series", panel.series)):
        for value in sorted(set(values.tolist())):
            mask = values == value
            if mask.sum() < 8:
                continue
            rows.append(
                {
                    "scale": panel.scale,
                    "target": target,
                    "model": fit.model,
                    "stratum_type": label,
                    "stratum": str(value),
                    "oof_rmse": fit.oof_rmse,
                    **metric_row(observed[mask], predicted[mask]),
                }
            )
    quartiles = np.quantile(distance, [0.25, 0.5, 0.75])
    for index, name in enumerate(("nearest", "near", "far", "farthest")):
        lower = -np.inf if index == 0 else quartiles[index - 1]
        upper = np.inf if index == 3 else quartiles[index]
        mask = (distance > lower) & (distance <= upper)
        if mask.sum() < 8:
            continue
        rows.append(
            {
                "scale": panel.scale,
                "target": target,
                "model": fit.model,
                "stratum_type": "support_quartile",
                "stratum": name,
                "oof_rmse": fit.oof_rmse,
                **metric_row(observed[mask], predicted[mask]),
            }
        )
    return pd.DataFrame(rows)


def provenance() -> dict[str, str]:
    files = [
        CANONICAL / "delphi_3e18_two_phase_fit.csv",
        CANONICAL / "300m_two_phase_fit.csv",
        CANONICAL / "300m_heldouts.csv",
        DELPHI_HELDOUTS,
        SIXTY_M / "fit_two_phase.csv",
        SIXTY_M / "heldout_observations.csv",
        CATALOG,
    ]
    return {str(p.relative_to(REFERENCE_OUTPUTS)): sha256_of(p) for p in files}


def main() -> None:
    summary = {}
    for scale in ("60m", "300m", "delphi_3e18"):
        fit_panel, heldout = load_scale(scale)
        epochs = fit_panel.epochs
        summary[scale] = {
            "alpha": fit_panel.alpha,
            "fit_rows": len(fit_panel),
            "heldout_rows": len(heldout),
            "heldout_uncheatable": int(np.isfinite(heldout.targets[UNCHEATABLE]).sum()),
            "heldout_table9": int(np.isfinite(heldout.targets[TABLE9]).sum()),
            "heldout_series": len(set(heldout.series.tolist())),
            "heldout_two_phase": int((heldout.policy_class == "two_phase").sum()),
            "fit_epoch_median": float(np.median(epochs)),
            "fit_fraction_cells_over_one_epoch": float((epochs > 1).mean()),
            "fit_max_epoch": float(epochs.max()),
        }
    print(json.dumps({"scales": summary, "provenance_sha256": provenance()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
