# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
"""Frozen paired-contrast spine for conditional aggregate/phase-order modeling.

The spine exposes the phase response as a *paired* quantity rather than raw BPB.
For a two-phase policy with phase-0 fraction ``alpha``:

    a = alpha * p0 + (1 - alpha) * p1        (aggregate exposure coordinate)
    d = p1 - p0                              (phase contrast coordinate)

Every two-phase fit policy in the Delphi 3e18 and 300M swarms has an exactly
aggregate-matched tied counterpart, so

    Delta(a, d) = L(a, d) - L(a, 0)

is observed directly. Delta equals ``O(a, d) + C(a, d)`` for any decomposition
with ``O`` odd and ``C`` even in ``d``, and it carries no aggregate-model error.
The balanced antithetic panel additionally observes ``O`` and ``C`` separately
at two anchors, which makes it a held-out identification test rather than
another fitting set.

The sealed targeted-pairwise panel is never loaded here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PACKET = REFERENCE_OUTPUTS / "two_phase_surrogate_collaborator_packet_20260721"
CANONICAL = PACKET / "data" / "canonical"
AGGRESSIVE = REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_results_20260723"

SEALED_SERIES_FRAGMENT = "targeted_pairwise"

# Phase-0 duration fractions, recovered exactly from each panel's tied counterpart
# by least squares and then asserted to machine precision.
SCALE_ALPHA = {
    "delphi_3e18": 0.7981376787495837,
    "300m": 0.80,
}
STARCODER_ALPHA = {
    "starcoder_cosine_50_50": 0.50,
    "starcoder_wsd80": 0.80,
}
AGGREGATE_MATCH_TOLERANCE = 1e-8
TIED_TV_TOLERANCE = 1e-9

TARGETS = ("uncheatable_bpb", "table9_macro_bpb")

CATALOG_PATH = PACKET / "data" / "catalog.json"

# The catalog's exposure multipliers are proportional to, but not equal to,
# simulated epochs. Calibrated against the aggressive panel's authoritative
# ``max_simulated_epoch`` and ``q95_simulated_epoch`` columns, the conversion is
# a single global constant, reproduced to 1e-11 relative error on all 290 rows.
EPOCH_SCALE = 4012.081936384326


@dataclass(frozen=True)
class ExposureSpec:
    """Per-bucket epoch conversions and canonical family assignment."""

    domains: tuple[str, ...]
    c0: np.ndarray
    c1: np.ndarray
    families: dict[str, list[str]]


def load_exposure_spec(dataset_id: str) -> ExposureSpec:
    """Return the catalog's per-bucket epoch conversions and family assignment.

    ``c0`` and ``c1`` convert a phase weight into simulated epochs. In these
    swarms ``c1 = c0 * (1 - alpha) / alpha``, so total physical exposure is a
    function of the aggregate alone and phase order never changes it.
    """
    catalog = json.loads(CATALOG_PATH.read_text())
    spec = catalog["datasets"][dataset_id]
    return ExposureSpec(
        domains=tuple(spec["domains"]),
        c0=np.asarray(spec["c0"], dtype=float) * EPOCH_SCALE,
        c1=np.asarray(spec["c1"], dtype=float) * EPOCH_SCALE,
        families=spec["families"],
    )


def family_index_for(domains: tuple[str, ...], families: dict[str, list[str]]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Map each bucket onto its canonical family id."""
    names = tuple(families)
    lookup = {bucket: names.index(name) for name, members in families.items() for bucket in members}
    missing = [b for b in domains if b not in lookup]
    assert not missing, f"buckets without a canonical family: {missing[:3]}"
    return np.asarray([lookup[b] for b in domains], dtype=int), names


def assert_sealed_absent(frame: pd.DataFrame, label: str) -> None:
    """Fail loudly if any row carries a sealed targeted-pairwise identifier."""
    text_columns = frame.select_dtypes(include=["object", "string"])
    if text_columns.empty:
        return
    hit = text_columns.astype("string").apply(
        lambda column: column.str.contains(SEALED_SERIES_FRAGMENT, case=False, na=False)
    )
    assert not bool(hit.to_numpy().any()), f"sealed targeted-pairwise rows present in {label}"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PairedPanel:
    """Aggregate/contrast coordinates with an observed tied counterpart."""

    scale: str
    alpha: float
    buckets: tuple[str, ...]
    row_id: np.ndarray
    aggregate: np.ndarray  # (n, k) aggregate exposure a
    contrast: np.ndarray  # (n, k) phase contrast d
    phase0: np.ndarray  # (n, k)
    phase1: np.ndarray  # (n, k)
    delta: dict[str, np.ndarray]  # target -> L(a, d) - L(a, 0)
    tied_bpb: dict[str, np.ndarray]  # target -> L(a, 0)
    two_phase_bpb: dict[str, np.ndarray]  # target -> L(a, d)
    candidate_kind: np.ndarray

    @property
    def phase_tv(self) -> np.ndarray:
        return 0.5 * np.abs(self.contrast).sum(axis=1)

    def __len__(self) -> int:
        return len(self.row_id)


def _weight_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], tuple[str, ...]]:
    phase0 = [c for c in frame.columns if c.startswith("phase_0_weight::")]
    phase1 = [c for c in frame.columns if c.startswith("phase_1_weight::")]
    buckets = tuple(c.split("::", 1)[1] for c in phase0)
    assert [c.split("::", 1)[1] for c in phase1] == list(buckets), "phase weight columns misaligned"
    return phase0, phase1, buckets


def load_paired_panel(scale: str, tied_prefix_regex: str, targets: tuple[str, ...] = TARGETS) -> PairedPanel:
    """Load a two-phase fit panel joined to its exact aggregate-matched tied panel."""
    alpha = SCALE_ALPHA[scale]
    two_phase = pd.read_csv(CANONICAL / f"{scale}_two_phase_fit.csv")
    tied = pd.read_csv(CANONICAL / f"{scale}_one_phase_fit.csv")
    assert_sealed_absent(two_phase, f"{scale} two-phase fit")
    assert_sealed_absent(tied, f"{scale} tied fit")

    p0_cols, p1_cols, buckets = _weight_columns(two_phase)
    tied = tied.copy()
    tied["base_id"] = tied["row_id"].str.replace(tied_prefix_regex, "", regex=True)
    assert set(tied["base_id"]) == set(two_phase["row_id"]), f"{scale}: tied/two-phase ids do not correspond"

    tied_indexed = tied.set_index("base_id").loc[two_phase["row_id"]]

    # The tied panel must be a genuine single-phase policy.
    tied_p0 = tied_indexed[p0_cols].to_numpy(float)
    tied_p1 = tied_indexed[p1_cols].to_numpy(float)
    assert np.abs(tied_p0 - tied_p1).max() < TIED_TV_TOLERANCE, f"{scale}: tied panel is not phase-tied"

    phase0 = two_phase[p0_cols].to_numpy(float)
    phase1 = two_phase[p1_cols].to_numpy(float)
    aggregate = alpha * phase0 + (1.0 - alpha) * phase1
    match_error = float(np.abs(aggregate - tied_p0).max())
    assert match_error < AGGREGATE_MATCH_TOLERANCE, f"{scale}: aggregate match error {match_error:.3e}"

    contrast = phase1 - phase0
    keep = 0.5 * np.abs(contrast).sum(axis=1) > TIED_TV_TOLERANCE
    for target in targets:
        keep &= two_phase[target].notna().to_numpy() & tied_indexed[target].notna().to_numpy()

    delta = {}
    tied_bpb = {}
    two_phase_bpb = {}
    for target in targets:
        lhs = two_phase[target].to_numpy(float)[keep]
        rhs = tied_indexed[target].to_numpy(float)[keep]
        delta[target] = lhs - rhs
        tied_bpb[target] = rhs
        two_phase_bpb[target] = lhs

    return PairedPanel(
        scale=scale,
        alpha=alpha,
        buckets=buckets,
        row_id=two_phase["row_id"].to_numpy()[keep],
        aggregate=aggregate[keep],
        contrast=contrast[keep],
        phase0=phase0[keep],
        phase1=phase1[keep],
        delta=delta,
        tied_bpb=tied_bpb,
        two_phase_bpb=two_phase_bpb,
        candidate_kind=two_phase["candidate_kind"].to_numpy()[keep],
    )


@dataclass(frozen=True)
class AntitheticPanel:
    """Exact antithetic triples: O and C observed separately at fixed aggregate."""

    buckets: tuple[str, ...]
    anchor_id: np.ndarray
    direction_id: np.ndarray
    target_phase_tv: np.ndarray
    realized_phase_tv: np.ndarray
    aggregate: np.ndarray  # (n, k) anchor aggregate a
    contrast: np.ndarray  # (n, k) plus-sign contrast d
    odd: dict[str, np.ndarray]  # target -> [L(+d) - L(-d)] / 2
    even: dict[str, np.ndarray]  # target -> [L(+d) + L(-d)] / 2 - L(0)
    plus_delta: dict[str, np.ndarray]
    minus_delta: dict[str, np.ndarray]


def load_antithetic_panel(buckets: tuple[str, ...]) -> AntitheticPanel:
    """Load the balanced antithetic panel, projected onto the canonical bucket order.

    The aggressive-panel export also carries ``phase_*_dolmino_share`` roll-up
    columns; selecting by the canonical bucket list excludes them.
    """
    pairs = pd.read_csv(AGGRESSIVE / "balanced_antithetic_pairs.csv")
    runs = pd.read_csv(AGGRESSIVE / "observed_results_with_control_deltas.csv")
    assert_sealed_absent(pairs, "antithetic pairs")
    assert_sealed_absent(runs, "aggressive runs")

    p0_cols = [f"phase_0_{bucket}" for bucket in buckets]
    p1_cols = [f"phase_1_{bucket}" for bucket in buckets]
    missing = [c for c in p0_cols + p1_cols if c not in runs.columns]
    assert not missing, f"aggressive panel missing bucket columns: {missing[:3]}"

    balanced = runs[runs["contrast_family"] == "balanced_partition"].copy()
    plus = balanced[balanced["sign"] == "plus"]
    key = ["anchor_id", "direction_id", "target_phase_tv"]
    plus_indexed = plus.set_index(key)

    index = pd.MultiIndex.from_frame(pairs[key])
    missing = [k for k in index if k not in plus_indexed.index]
    assert not missing, f"antithetic pairs without a plus-sign run: {missing[:3]}"
    aligned = plus_indexed.loc[index]

    alpha = SCALE_ALPHA["delphi_3e18"]
    phase0 = aligned[p0_cols].to_numpy(float)
    phase1 = aligned[p1_cols].to_numpy(float)
    aggregate = alpha * phase0 + (1.0 - alpha) * phase1

    odd = {}
    even = {}
    plus_delta = {}
    minus_delta = {}
    for target, stem in (("uncheatable_bpb", "uncheatable"), ("table9_macro_bpb", "table9")):
        odd[target] = pairs[f"{stem}_odd_effect"].to_numpy(float)
        even[target] = pairs[f"{stem}_curvature"].to_numpy(float)
        plus_delta[target] = pairs[f"{stem}_plus_delta"].to_numpy(float)
        minus_delta[target] = pairs[f"{stem}_minus_delta"].to_numpy(float)

    # The realized aggregate must be constant within an anchor: these are
    # fixed-aggregate interventions, so any drift would break the O/C estimands.
    for anchor in sorted(set(pairs["anchor_id"])):
        rows = pairs["anchor_id"].to_numpy() == anchor
        drift = float(np.abs(aggregate[rows] - aggregate[rows].mean(axis=0)).max())
        assert drift < 5e-3, f"anchor {anchor} aggregate drift {drift:.3e}"

    return AntitheticPanel(
        buckets=buckets,
        anchor_id=pairs["anchor_id"].to_numpy(),
        direction_id=pairs["direction_id"].to_numpy(),
        target_phase_tv=pairs["target_phase_tv"].to_numpy(float),
        realized_phase_tv=aligned["phase_tv"].to_numpy(float),
        aggregate=aggregate,
        contrast=phase1 - phase0,
        odd=odd,
        even=even,
        plus_delta=plus_delta,
        minus_delta=minus_delta,
    )


@dataclass(frozen=True)
class StarcoderSurface:
    """Dense low-dimensional two-phase surface with an interior tied reference."""

    name: str
    alpha: float
    buckets: tuple[str, ...]
    aggregate: np.ndarray
    contrast: np.ndarray
    bpb: np.ndarray


def load_starcoder(name: str) -> StarcoderSurface:
    alpha = STARCODER_ALPHA[name]
    frame = pd.read_csv(CANONICAL / f"{name}.csv")
    assert_sealed_absent(frame, name)
    p0_cols, p1_cols, buckets = _weight_columns(frame)
    phase0 = frame[p0_cols].to_numpy(float)
    phase1 = frame[p1_cols].to_numpy(float)
    return StarcoderSurface(
        name=name,
        alpha=alpha,
        buckets=buckets,
        aggregate=alpha * phase0 + (1.0 - alpha) * phase1,
        contrast=phase1 - phase0,
        bpb=frame["starcoder_bpb"].to_numpy(float),
    )


@dataclass(frozen=True)
class Spine:
    """Every frozen panel used for conditional phase-order development."""

    delphi_3e18: PairedPanel
    m300: PairedPanel
    antithetic: AntitheticPanel
    starcoder_cosine: StarcoderSurface
    starcoder_wsd: StarcoderSurface


def build_spine() -> Spine:
    """Load every panel and return the frozen spine with provenance."""
    delphi = load_paired_panel("delphi_3e18", r"^singleavg_fit_\d+_")
    m300 = load_paired_panel("300m", r"^singleavg_")
    assert delphi.buckets == m300.buckets, "Delphi and 300M bucket orders differ"
    antithetic = load_antithetic_panel(delphi.buckets)
    cosine = load_starcoder("starcoder_cosine_50_50")
    wsd = load_starcoder("starcoder_wsd80")

    return Spine(
        delphi_3e18=delphi,
        m300=m300,
        antithetic=antithetic,
        starcoder_cosine=cosine,
        starcoder_wsd=wsd,
    )


def provenance() -> dict[str, str]:
    files = [
        CANONICAL / "delphi_3e18_two_phase_fit.csv",
        CANONICAL / "delphi_3e18_one_phase_fit.csv",
        CANONICAL / "300m_two_phase_fit.csv",
        CANONICAL / "300m_one_phase_fit.csv",
        CANONICAL / "starcoder_cosine_50_50.csv",
        CANONICAL / "starcoder_wsd80.csv",
        AGGRESSIVE / "balanced_antithetic_pairs.csv",
        AGGRESSIVE / "observed_results_with_control_deltas.csv",
    ]
    return {str(path.relative_to(REFERENCE_OUTPUTS)): sha256_of(path) for path in files}


def main() -> None:
    spine = build_spine()
    delphi = spine.delphi_3e18
    m300 = spine.m300
    antithetic = spine.antithetic
    spec = load_exposure_spec("delphi_3e18_two_phase_fit")
    family_ids, family_names = family_index_for(delphi.buckets, spec.families)
    assert spec.domains == delphi.buckets, "catalog domain order differs from the fit panel"

    summary = {
        "alpha_by_scale": {"delphi_3e18": delphi.alpha, "300m": m300.alpha},
        "bucket_count": len(delphi.buckets),
        "delphi_paired_rows": len(delphi),
        "m300_paired_rows": len(m300),
        "antithetic_rows": len(antithetic.anchor_id),
        "antithetic_anchors": sorted(set(antithetic.anchor_id.tolist())),
        "antithetic_directions": len(set(antithetic.direction_id.tolist())),
        "antithetic_target_tv": sorted(set(antithetic.target_phase_tv.tolist())),
        "delphi_phase_tv_median": float(np.median(delphi.phase_tv)),
        "delphi_phase_tv_max": float(delphi.phase_tv.max()),
        "starcoder_cosine_rows": len(spine.starcoder_cosine.bpb),
        "starcoder_wsd_rows": len(spine.starcoder_wsd.bpb),
        "families": {name: int((family_ids == i).sum()) for i, name in enumerate(family_names)},
        "provenance_sha256": provenance(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    for name, panel in (("delphi_3e18", delphi), ("300m", m300)):
        for target in TARGETS:
            values = panel.delta[target]
            print(
                f"{name:11s} {target:18s} n={len(values):3d} "
                f"mean={values.mean():+.6f} sd={values.std(ddof=1):.6f} "
                f"min={values.min():+.6f} max={values.max():+.6f} "
                f"frac_better={float((values < 0).mean()):.3f}"
            )


if __name__ == "__main__":
    main()
