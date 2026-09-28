# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "tabulate",
# ]
# ///

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the Compact Retained State raw-optimum path validation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs"
DEFAULT_SOURCE_DIR = REFERENCE_OUTPUT_DIR / "delphi_grp_compact_raw_optimum_paths_20260721"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUT_DIR / "delphi_compact_optimum_path_validation_panel_20260721"
DEFAULT_FIT_SWARM = (
    REFERENCE_OUTPUT_DIR / "delphi_augmented_swarm_3e18_20260714" / "delphi_augmented_swarm_3e18_wide.csv"
)
DEFAULT_HELDOUTS = REFERENCE_OUTPUT_DIR / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"

ALPHA0 = 0.8
ALPHA1 = 0.2
EXACT_COORDINATE_TOLERANCE = 1e-10
MODEL_ID = "compact_retained_state"
TARGET_ORDER = {"uncheatable": 0, "table9": 1}


@dataclass(frozen=True)
class SelectionSummary:
    selected_paths: int
    unique_candidates: int
    two_phase_only_candidates: int
    tied_spine_endpoint_candidates: int
    uncheatable_candidates: int
    table9_candidates: int
    exact_fit_overlaps: int
    exact_heldout_overlaps: int
    source_panel_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-swarm", type=Path, default=DEFAULT_FIT_SWARM)
    parser.add_argument("--heldouts", type=Path, default=DEFAULT_HELDOUTS)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def policy_sha256(domains: list[str], weights: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update("\0".join(domains).encode())
    hasher.update(np.asarray(weights, dtype="<f8").tobytes())
    return hasher.hexdigest()


def weighted_policy_tv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim == 2:
        left = left[None, :, :]
    phase0 = 0.5 * np.abs(left[:, 0, :] - right[0]).sum(axis=1)
    phase1 = 0.5 * np.abs(left[:, 1, :] - right[1]).sum(axis=1)
    return ALPHA0 * phase0 + ALPHA1 * phase1


def source_policy(source_dir: Path, candidate_id: str, domains: list[str] | None) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(source_dir / "mixtures" / f"{candidate_id}.csv")
    required = {"domain", "phase_0_weight", "phase_1_weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{candidate_id} is missing mixture columns: {sorted(missing)}")
    candidate_domains = frame["domain"].astype(str).tolist()
    if domains is not None and candidate_domains != domains:
        raise ValueError(f"Domain order changed for {candidate_id}")
    weights = frame[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T
    if weights.shape != (2, len(candidate_domains)):
        raise ValueError(f"Unexpected policy shape for {candidate_id}: {weights.shape}")
    if np.any(weights < 0.0) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"Invalid phase weights for {candidate_id}")
    return candidate_domains, weights


def selected_path_groups(paths: pd.DataFrame) -> list[pd.DataFrame]:
    compact = paths.loc[paths["model"].eq(MODEL_ID)].copy()
    two_phase = compact.loc[compact["design"].eq("two_phase_only")]
    groups = [group.copy() for _, group in two_phase.groupby("candidate_id", sort=False)]

    tied = compact.loc[compact["design"].eq("tied_spine_plus_two_phase")]
    endpoints = tied.sort_values(["target", "total_unique_training_rows"]).groupby("target", as_index=False).tail(1)
    groups.extend([endpoints.loc[[index]].copy() for index in endpoints.index])
    return groups


def short_candidate_id(group: pd.DataFrame) -> str:
    target = str(group.iloc[0]["target"])
    target_tag = {"uncheatable": "unch", "table9": "t9"}[target]
    design = str(group.iloc[0]["design"])
    rows = sorted(group["total_unique_training_rows"].astype(int).unique().tolist())
    row_tag = "_".join(str(value) for value in rows)
    design_tag = {"two_phase_only": "2ponly", "tied_spine_plus_two_phase": "tied2p"}[design]
    return f"crspath_{target_tag}_{design_tag}_r{row_tag}"


def load_fit_weights(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    return np.stack(
        [
            frame[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float),
            frame[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float),
        ],
        axis=1,
    )


def load_heldout_weights(path: Path, domains: list[str]) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    weights: list[np.ndarray] = []
    identities: list[str] = []
    for row in frame.itertuples(index=False):
        phase0 = json.loads(str(row.phase_0_weights_json))
        phase1 = json.loads(str(row.phase_1_weights_json))
        weights.append(
            np.asarray(
                [
                    [float(phase0[domain]) for domain in domains],
                    [float(phase1[domain]) for domain in domains],
                ]
            )
        )
        identities.append(str(row.heldout_id))
    return np.stack(weights), identities


def build_panel(
    source_dir: Path,
    fit_swarm: Path,
    heldouts: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    paths = pd.read_csv(source_dir / "path_manifest.csv")
    groups = selected_path_groups(paths)
    if len(groups) != 15:
        raise ValueError(f"Expected 15 unique policy groups, found {len(groups)}")

    domains: list[str] | None = None
    policies: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    output_mixtures = output_dir / "mixtures"
    output_mixtures.mkdir(parents=True, exist_ok=True)

    for group in groups:
        source_id = str(group.iloc[0]["candidate_id"])
        candidate_domains, weights = source_policy(source_dir, source_id, domains)
        if domains is None:
            domains = candidate_domains
        policies.append(weights)
        representative = group.sort_values("total_unique_training_rows").iloc[-1]
        candidate_id = short_candidate_id(group)
        phase_tv = 0.5 * np.abs(weights[0] - weights[1]).sum()
        if phase_tv <= EXACT_COORDINATE_TOLERANCE:
            raise ValueError(f"Expected a two-phase policy for {candidate_id}")

        source_rows = sorted(group["total_unique_training_rows"].astype(int).unique().tolist())
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "target": str(representative["target"]),
            "policy_class": "two_phase",
            "candidate_kind": f"compact_raw_optimum_{representative['design']}",
            "fit_source": "delphi_3e18",
            "aggregate_kl_coefficient": "",
            "phase_information_budget": "",
            "model": MODEL_ID,
            "design": str(representative["design"]),
            "source_candidate_id": source_id,
            "source_fit_row_counts": ",".join(str(value) for value in source_rows),
            "source_grouped_fit_row_counts": str(representative["grouped_fit_row_counts"]),
            "proposal_predicted_bpb_latest": float(representative["predicted_bpb"]),
            "proposal_predicted_bpb_min": float(group["predicted_bpb"].min()),
            "proposal_predicted_bpb_max": float(group["predicted_bpb"].max()),
            "max_bucket_weight": float(representative["max_bucket_weight"]),
            "max_simulated_epochs": float(representative["max_simulated_epochs"]),
            "phase_total_variation": float(representative["phase_total_variation"]),
            "phase_information_kl_diagnostic": float(representative["phase_information_kl"]),
            "standardized_fit_support_distance": float(representative["standardized_fit_support_distance"]),
            "policy_sha256": policy_sha256(candidate_domains, weights),
        }
        for phase_index in range(2):
            for domain_index, domain in enumerate(candidate_domains):
                row[f"phase_{phase_index}_{domain}"] = float(weights[phase_index, domain_index])
        rows.append(row)
        shutil.copyfile(source_dir / "mixtures" / f"{source_id}.csv", output_mixtures / f"{candidate_id}.csv")

    assert domains is not None
    panel = pd.DataFrame(rows)
    panel["target_order"] = panel["target"].map(TARGET_ORDER)
    panel = panel.sort_values(["target_order", "design", "source_fit_row_counts", "candidate_id"]).drop(
        columns="target_order"
    )
    if panel["candidate_id"].duplicated().any() or panel["policy_sha256"].duplicated().any():
        raise ValueError("Selected panel contains duplicate identifiers or exact policies")

    fit_weights = load_fit_weights(fit_swarm, domains)
    heldout_weights, heldout_ids = load_heldout_weights(heldouts, domains)
    audit_rows: list[dict[str, object]] = []
    for row, weights in zip(rows, policies, strict=True):
        fit_max = np.max(np.abs(fit_weights - weights), axis=(1, 2))
        heldout_max = np.max(np.abs(heldout_weights - weights), axis=(1, 2))
        fit_tv = weighted_policy_tv(fit_weights, weights)
        heldout_tv = weighted_policy_tv(heldout_weights, weights)
        nearest_heldout = int(np.argmin(heldout_tv))
        audit_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "policy_sha256": row["policy_sha256"],
                "exact_fit_overlap_count": int(np.sum(fit_max <= EXACT_COORDINATE_TOLERANCE)),
                "exact_heldout_overlap_count": int(np.sum(heldout_max <= EXACT_COORDINATE_TOLERANCE)),
                "nearest_fit_weighted_tv": float(np.min(fit_tv)),
                "nearest_heldout_weighted_tv": float(heldout_tv[nearest_heldout]),
                "nearest_heldout_id": heldout_ids[nearest_heldout],
            }
        )
    audit = pd.DataFrame(audit_rows)
    if int(audit["exact_fit_overlap_count"].sum()) or int(audit["exact_heldout_overlap_count"].sum()):
        raise ValueError("At least one selected policy already exists in the fit or heldout archive")
    return panel, audit, domains


def write_report(panel: pd.DataFrame, audit: pd.DataFrame, source_sha256: str, output_dir: Path) -> None:
    counts = panel.groupby(["target", "design"]).size().rename("count").reset_index()
    lines = [
        "# Compact Retained State raw-optimum path validation panel",
        "",
        "This frozen Delphi 3e18 panel contains every coordinate-distinct Compact Retained State raw optimum "
        "from the two-phase-only learning curves, plus the maximum-evidence tied-spine endpoint for each target.",
        "",
        f"- Candidates: {len(panel)}.",
        f"- Source-panel SHA-256: `{source_sha256}`.",
        "- All candidates are two-phase policies and are scheduled for both Uncheatable and native Table-9 evaluation.",
        "- Exact overlap with the 280-row Delphi fit swarm: 0.",
        "- Exact overlap with the append-only Delphi heldout archive: 0.",
        f"- Nearest heldout weighted policy-TV range: {audit['nearest_heldout_weighted_tv'].min():.6f} to "
        f"{audit['nearest_heldout_weighted_tv'].max():.6f}.",
        "",
        "## Composition",
        "",
        counts.to_markdown(index=False),
        "",
        "The Table-9 520- and 560-row two-phase-only fits produce the same deduplicated coordinate and therefore "
        "share one validation checkpoint.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, audit, _ = build_panel(args.source_dir, args.fit_swarm, args.heldouts, args.output_dir)
    source_panel = args.output_dir / "launcher_source_panel.csv"
    panel.to_csv(source_panel, index=False, float_format="%.17g")
    audit.to_csv(args.output_dir / "coordinate_overlap_audit.csv", index=False, float_format="%.17g")
    source_sha256 = file_sha256(source_panel)
    summary = SelectionSummary(
        selected_paths=int(panel["source_fit_row_counts"].str.split(",").map(len).sum()),
        unique_candidates=len(panel),
        two_phase_only_candidates=int(panel["design"].eq("two_phase_only").sum()),
        tied_spine_endpoint_candidates=int(panel["design"].eq("tied_spine_plus_two_phase").sum()),
        uncheatable_candidates=int(panel["target"].eq("uncheatable").sum()),
        table9_candidates=int(panel["target"].eq("table9").sum()),
        exact_fit_overlaps=int(audit["exact_fit_overlap_count"].sum()),
        exact_heldout_overlaps=int(audit["exact_heldout_overlap_count"].sum()),
        source_panel_sha256=source_sha256,
    )
    (args.output_dir / "selection_summary.json").write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    write_report(panel, audit, source_sha256, args.output_dir)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
