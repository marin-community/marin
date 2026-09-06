# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "wandb>=0.21"]
# ///
"""Build a leakage-safe heldout registry for single-phase mixture surrogates.

The source archives use several historical definitions of ``heldout``. This
script ignores those labels for eligibility: a row must have tied phase
weights and its mixture must be coordinate-disjoint from the current canonical
fit panel. Repeated training seeds remain separate in ``heldout_runs.csv`` and
are collapsed only in ``heldout_coordinates.csv``.

Atomic metrics are exported only when their payload reconstructs the recorded
aggregate. W&B summaries that are stale or incomplete are recorded as gaps
rather than being mixed with final endpoint values.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "single_phase_heldout_benchmark_20260902"
CANONICAL = REFERENCE_OUTPUTS / "two_phase_surrogate_collaborator_packet_20260721" / "data" / "canonical"
TABLE9_METADATA = REFERENCE_OUTPUTS / "olmo_base_easy_one_phase_parity_panel_300m_20260628/component_metadata.json"

SIXTY_M_AUDIT = REFERENCE_OUTPUTS / "60m_39bucket_checkpoint_audit_20260724/heldout_observations.csv"
SIXTY_M_FIT = REFERENCE_OUTPUTS / "60m_39bucket_checkpoint_audit_20260724/fit_single_phase.csv"
SIXTY_M_TABLE9 = REFERENCE_OUTPUTS / "60m_table9_gap_completion_20260725/table9_eval_results.csv"
THREE_HUNDRED_M_PACKET = (
    REFERENCE_OUTPUTS / "two_phase_solver_gap_collaborator_packet_20260701/data/heldout_300m_checkpoint_metrics.csv"
)
DELPHI_ARCHIVE = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"

RECENT_DELPHI_SOURCES = {
    "shared_shape_dsp_epoch_cap": REFERENCE_OUTPUTS / "delphi_one_phase_dsp_epoch_cap_sweep_20260828",
    "aggregate_v_epoch_cap": REFERENCE_OUTPUTS / "delphi_one_phase_surrogate_challenger_validations_20260831",
    "full_canonical_dsp_epoch_cap": REFERENCE_OUTPUTS / "delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901",
    "weibull_softplus_unscaled_epoch_cap": (
        REFERENCE_OUTPUTS / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902"
    ),
}
EPOCH_DOSE_DIR = REFERENCE_OUTPUTS / "bucket_epoch_dose_response_20260729" / "full" / "delphi_3e18"
EPOCH_DOSE_RECOVERY_DIR = REFERENCE_OUTPUTS / "bucket_epoch_dose_response_20260729" / "recovery" / "delphi_3e18_20260902"
EPOCH_DOSE_RUN_MANIFEST = EPOCH_DOSE_DIR / "run_manifest.csv"
EPOCH_DOSE_PHASE_WEIGHTS = EPOCH_DOSE_DIR / "phase_weights.csv"
EPOCH_DOSE_RESULTS = EPOCH_DOSE_RECOVERY_DIR / "heldout_results.csv"
EPOCH_DOSE_UNCHEATABLE_COMPONENTS = EPOCH_DOSE_RECOVERY_DIR / "uncheatable_components.csv"
EPOCH_DOSE_TABLE9_COMPONENTS = EPOCH_DOSE_RECOVERY_DIR / "table9_components.csv"
EPOCH_DOSE_MATERIALIZATION_MANIFEST = EPOCH_DOSE_RECOVERY_DIR / "heldout_materialization_manifest.json"
APRIORI_SWARM_DIR = REFERENCE_OUTPUTS / "delphi_apriori_swarm_280_20260904"
APRIORI_SWARM_DESIGN = APRIORI_SWARM_DIR / "swarm_mixtures.csv"
APRIORI_SWARM_MATERIALIZATION_DIR = APRIORI_SWARM_DIR / "pilot_materialization"
APRIORI_SWARM_RESULTS = APRIORI_SWARM_MATERIALIZATION_DIR / "heldout_results.csv"
APRIORI_SWARM_UNCHEATABLE_COMPONENTS = APRIORI_SWARM_MATERIALIZATION_DIR / "uncheatable_components.csv"
APRIORI_SWARM_TABLE9_COMPONENTS = APRIORI_SWARM_MATERIALIZATION_DIR / "table9_components.csv"
APRIORI_SWARM_MATERIALIZATION_MANIFEST = APRIORI_SWARM_MATERIALIZATION_DIR / "heldout_materialization_manifest.json"

FIT_PANEL_PATHS = {
    "60m_39bucket": SIXTY_M_FIT,
    "300m_39bucket": CANONICAL / "300m_one_phase_fit.csv",
    "delphi_3e18_39bucket": CANONICAL / "delphi_3e18_one_phase_fit.csv",
}
PANEL_SCALE = {
    "60m_39bucket": "60M / 1.2B tokens",
    "300m_39bucket": "300M / 6B tokens",
    "delphi_3e18_39bucket": "3e18 FLOPs",
}

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
UNCHEATABLE_AGGREGATE = "eval/uncheatable_eval/bpb"
UNCHEATABLE_COMPONENTS = (
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
)
LEGACY_UNCHEATABLE_WEIGHTS = np.asarray(
    [0.1725475920, 0.1456785645, 0.1673678370, 0.1207508824, 0.1540951309, 0.1479443173, 0.0916157061]
)
DELPHI_UNCHEATABLE_WEIGHTS = np.asarray(
    [
        0.1725032419832083,
        0.1459150285318545,
        0.1673147061106415,
        0.1206224088062254,
        0.1543450059149921,
        0.1478598606213357,
        0.0914397685598754,
    ]
)
TABLE9_AGGREGATE_KEYS = (
    "olmo_base_easy/table9_51_component_macro_bpb",
    "olmo_base_easy/table9_macro_bpb",
    "olmo_base_eval/easy_bpb/_summary/primary_metric_mean",
)

PHASE_TOLERANCE = 1e-10
FIT_OVERLAP_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 3e-6
SCHEMA_VERSION = 2
POOL_FRACTION_PREFIX = "pool_fraction::"

AUDIT_COLUMNS = (
    "panel",
    "scale",
    "row_id",
    "source",
    "source_row_id",
    "source_experiment",
    "proposal_model",
    "proposal_target",
    "epoch_cap",
    "training_wandb_run_id",
    "training_wandb_url",
    "table9_eval_run_id",
    "table9_eval_url",
    "data_seed",
    "trainer_seed",
    "subset_seed",
    "phase_tv",
    "fit_panel_max_abs_distance",
    "coordinate_id",
    "uncheatable_bpb",
    "table9_macro_bpb",
    "eligible",
    "exclusion_reason",
)


@dataclasses.dataclass(frozen=True)
class ComponentRequest:
    row_id: str
    panel: str
    target: str
    project: str
    wandb_run_id: str
    expected_aggregate: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table9_components() -> tuple[str, ...]:
    components = tuple(json.loads(TABLE9_METADATA.read_text())["components"])
    if len(components) != 51 or len(set(components)) != 51:
        raise ValueError("Expected the fixed 51-component Table-9 inventory")
    return components


def table9_summary_keys() -> dict[str, str]:
    mapping = {
        str(component): str(summary_key)
        for component, summary_key in json.loads(TABLE9_METADATA.read_text())["native_component_keys"].items()
    }
    if set(mapping) != set(table9_components()):
        raise ValueError("Table-9 native component keys do not match the fixed component inventory")
    return mapping


def _canonical_table9_component(component: str) -> str:
    aliases: dict[str, str] = {}
    for canonical in table9_components():
        aliases[canonical] = canonical
        if canonical.startswith("olmo_base_eval/easy_bpb/") and canonical.endswith("/bpb"):
            aliases[canonical.removeprefix("olmo_base_eval/easy_bpb/").removesuffix("/bpb")] = canonical
    if component not in aliases:
        raise ValueError(f"Unknown Table-9 component: {component}")
    return aliases[component]


def domains() -> tuple[str, ...]:
    values = tuple(json.loads(TABLE9_METADATA.read_text())["domains"])
    if len(values) != 39 or len(set(values)) != 39:
        raise ValueError("Expected the fixed 39-bucket inventory")
    return values


def nearest_fit_distance(weights: np.ndarray, fit_weights: np.ndarray) -> np.ndarray:
    """Return each row's maximum-coordinate distance from its nearest fit row."""
    if weights.ndim != 2 or fit_weights.ndim != 2 or weights.shape[1] != fit_weights.shape[1]:
        raise ValueError("Heldout and fit weights must be compatible matrices")
    result = np.full(len(weights), np.inf)
    for start in range(0, len(fit_weights), 128):
        distance = np.max(np.abs(weights[:, None, :] - fit_weights[None, start : start + 128, :]), axis=2)
        result = np.minimum(result, distance.min(axis=1))
    return result


def coordinate_id(panel: str, weights: np.ndarray, pool_fractions: np.ndarray | None = None) -> str:
    """Return a stable coordinate identity over mixture shares and materialized support.

    Full-support rows retain the schema-v1 digest so frozen coordinate references do not
    change. Subsampled rows add all pool fractions to the digest.
    """
    normalized = np.asarray(weights, dtype="<f8")
    if normalized.shape != (39,) or not np.isfinite(normalized).all():
        raise ValueError(f"{panel}: invalid coordinate")
    rounded = np.round(normalized, 12)
    fractions = np.ones_like(normalized) if pool_fractions is None else np.asarray(pool_fractions, dtype="<f8")
    if fractions.shape != normalized.shape or not np.isfinite(fractions).all():
        raise ValueError(f"{panel}: invalid pool fractions")
    if np.any(fractions <= 0.0) or np.any(fractions > 1.0):
        raise ValueError(f"{panel}: pool fractions must lie in (0, 1]")
    rounded_fractions = np.round(fractions, 12)
    payload = panel.encode() + b"\0" + rounded.tobytes()
    if not np.array_equal(rounded_fractions, np.ones_like(rounded_fractions)):
        payload += b"\0pool_fractions\0" + rounded_fractions.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    return f"{panel}:{digest}"


def _fit_weights(panel: str, bucket_names: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(FIT_PANEL_PATHS[panel])
    plain = [f"phase_0_{bucket}" for bucket in bucket_names]
    namespaced = [f"phase_0_weight::{bucket}" for bucket in bucket_names]
    columns = plain if set(plain).issubset(frame) else namespaced
    weights = frame.loc[:, columns].to_numpy(float)
    if weights.shape[1] != 39 or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"{panel}: invalid canonical fit weights")
    return weights


def _finalize_audit(
    frame: pd.DataFrame,
    weights: np.ndarray,
    fit_weights: np.ndarray,
    pool_fractions: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    if len(frame) != len(weights):
        raise ValueError("Audit metadata and weights differ in length")
    fractions = np.ones_like(weights) if pool_fractions is None else np.asarray(pool_fractions, dtype=float)
    if fractions.shape != weights.shape or not np.isfinite(fractions).all():
        raise ValueError("Audit pool fractions differ from mixture weights")
    if np.any(fractions <= 0.0) or np.any(fractions > 1.0):
        raise ValueError("Audit pool fractions must lie in (0, 1]")
    if "subset_seed" not in frame:
        frame["subset_seed"] = np.nan
    fit_distance = nearest_fit_distance(weights, fit_weights)
    frame["fit_panel_max_abs_distance"] = fit_distance
    frame["coordinate_id"] = [
        coordinate_id(str(panel), row, row_fractions)
        for panel, row, row_fractions in zip(frame["panel"], weights, fractions, strict=True)
    ]
    frame["exclusion_reason"] = ""
    frame.loc[frame["phase_tv"] > PHASE_TOLERANCE, "exclusion_reason"] = "not_single_phase"
    full_support = np.all(np.isclose(fractions, 1.0, atol=FIT_OVERLAP_TOLERANCE, rtol=0.0), axis=1)
    overlap = (frame["exclusion_reason"] == "") & (fit_distance <= FIT_OVERLAP_TOLERANCE) & full_support
    frame.loc[overlap, "exclusion_reason"] = "fit_coordinate_overlap"
    missing = (frame["exclusion_reason"] == "") & frame["uncheatable_bpb"].isna() & frame["table9_macro_bpb"].isna()
    frame.loc[missing, "exclusion_reason"] = "missing_primary_targets"
    frame["eligible"] = frame["exclusion_reason"].eq("")
    for bucket_index, bucket in enumerate(domains()):
        frame[f"weight::{bucket}"] = weights[:, bucket_index]
        frame[f"{POOL_FRACTION_PREFIX}{bucket}"] = fractions[:, bucket_index]
    return frame


def _empty_metadata(rows: int) -> dict[str, list[Any]]:
    return {
        "proposal_model": [""] * rows,
        "proposal_target": [""] * rows,
        "epoch_cap": [np.nan] * rows,
        "training_wandb_url": [""] * rows,
        "table9_eval_run_id": [""] * rows,
        "table9_eval_url": [""] * rows,
        "data_seed": [np.nan] * rows,
        "trainer_seed": [np.nan] * rows,
        "subset_seed": [np.nan] * rows,
    }


def _audit_60m(bucket_names: tuple[str, ...]) -> pd.DataFrame:
    source = pd.read_csv(SIXTY_M_AUDIT)
    weights = source.loc[:, [f"phase_0_{bucket}" for bucket in bucket_names]].to_numpy(float)
    phase1 = source.loc[:, [f"phase_1_{bucket}" for bucket in bucket_names]].to_numpy(float)
    metadata = _empty_metadata(len(source))
    frame = pd.DataFrame(
        {
            "panel": "60m_39bucket",
            "scale": PANEL_SCALE["60m_39bucket"],
            "row_id": "60m::" + source["observation_id"].astype(str),
            "source": source["source_family"].astype(str),
            "source_row_id": source["run_name"].astype(str),
            "source_experiment": source["source_experiment"].astype(str),
            "training_wandb_run_id": source["wandb_run_id"].fillna("").astype(str),
            "phase_tv": 0.5 * np.abs(weights - phase1).sum(axis=1),
            "uncheatable_bpb": source["uncheatable_bpb"],
            "table9_macro_bpb": source["table9_macro_bpb"],
            **metadata,
        }
    )
    return _finalize_audit(frame, weights, _fit_weights("60m_39bucket", bucket_names))


def _audit_300m(bucket_names: tuple[str, ...]) -> pd.DataFrame:
    source = pd.read_csv(THREE_HUNDRED_M_PACKET, low_memory=False)
    if source["run_name"].duplicated().any():
        raise ValueError("300M checkpoint packet has duplicate run names")
    weights = source.loc[:, [f"phase_0_{bucket}" for bucket in bucket_names]].to_numpy(float)
    phase1 = source.loc[:, [f"phase_1_{bucket}" for bucket in bucket_names]].to_numpy(float)
    frame = pd.DataFrame(
        {
            "panel": "300m_39bucket",
            "scale": PANEL_SCALE["300m_39bucket"],
            "row_id": "300m::" + source["run_name"].astype(str),
            "source": source["packet_panel"].astype(str),
            "source_row_id": source["run_name"].astype(str),
            "source_experiment": source["source_experiment"].fillna("").astype(str),
            "proposal_model": source["packet_method"].fillna("").astype(str),
            "proposal_target": "",
            "epoch_cap": np.nan,
            "training_wandb_run_id": source["training_wandb_id"].fillna("").astype(str),
            "training_wandb_url": source["training_wandb_url"].fillna("").astype(str),
            "table9_eval_run_id": "",
            "table9_eval_url": "",
            "data_seed": np.nan,
            "trainer_seed": np.nan,
            "phase_tv": 0.5 * np.abs(weights - phase1).sum(axis=1),
            "uncheatable_bpb": source["eval_uncheatable_eval_bpb"],
            "table9_macro_bpb": source["table9_macro_bpb"],
        }
    )
    return _finalize_audit(frame, weights, _fit_weights("300m_39bucket", bucket_names))


def _json_weights(value: str, bucket_names: tuple[str, ...]) -> np.ndarray:
    payload = json.loads(value)
    if not isinstance(payload, dict) or set(payload) != set(bucket_names):
        raise ValueError("Delphi archive weight payload does not match the 39-bucket inventory")
    return np.asarray([payload[bucket] for bucket in bucket_names], dtype=float)


def _audit_delphi_archive(bucket_names: tuple[str, ...]) -> pd.DataFrame:
    source = pd.read_csv(DELPHI_ARCHIVE)
    weights = np.stack(source["phase_0_weights_json"].map(lambda value: _json_weights(value, bucket_names)))
    phase1 = np.stack(source["phase_1_weights_json"].map(lambda value: _json_weights(value, bucket_names)))
    frame = pd.DataFrame(
        {
            "panel": "delphi_3e18_39bucket",
            "scale": PANEL_SCALE["delphi_3e18_39bucket"],
            "row_id": "delphi_archive::" + source["wandb_run_id"].astype(str),
            "source": "archive::" + source["training_series"].astype(str),
            "source_row_id": source["wandb_run_name"].astype(str),
            "source_experiment": source["training_series"].astype(str),
            "proposal_model": source["candidate_kind"].fillna("").astype(str),
            "proposal_target": source["proposal_target"].fillna(source["objective"]).fillna("").astype(str),
            "epoch_cap": np.nan,
            "training_wandb_run_id": source["wandb_run_id"].astype(str),
            "training_wandb_url": source["wandb_url"].fillna("").astype(str),
            "table9_eval_run_id": source["table9_eval_run_id"].fillna("").astype(str),
            "table9_eval_url": source["table9_eval_url"].fillna("").astype(str),
            "data_seed": source["data_seed"],
            "trainer_seed": source["trainer_seed"],
            "phase_tv": 0.5 * np.abs(weights - phase1).sum(axis=1),
            "uncheatable_bpb": source["uncheatable_bpb"],
            "table9_macro_bpb": source["table9_macro_bpb"],
        }
    )
    return _finalize_audit(frame, weights, _fit_weights("delphi_3e18_39bucket", bucket_names))


def _wandb_id(url: object, eval_metrics_uri: object) -> str:
    if isinstance(url, str) and url:
        return url.rstrip("/").rsplit("/", 1)[-1]
    if isinstance(eval_metrics_uri, str) and eval_metrics_uri:
        return eval_metrics_uri.rstrip("/").split("/")[-3]
    return ""


def _audit_recent_delphi(bucket_names: tuple[str, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source_name, source_dir in RECENT_DELPHI_SOURCES.items():
        results = pd.read_csv(source_dir / "measured_results.csv")
        weight_rows = pd.read_csv(source_dir / "candidate_weights.csv")
        records: list[dict[str, object]] = []
        vectors: list[np.ndarray] = []
        for result in results.to_dict("records"):
            selected = weight_rows[weight_rows["candidate_id"].eq(result["candidate_id"])]
            if "target" in selected:
                selected = selected[selected["target"].eq(result["target"])]
            if len(selected) != 39 or selected["domain"].nunique() != 39:
                raise ValueError(f"{source_name}/{result['candidate_id']}: incomplete candidate weights")
            by_domain = dict(zip(selected["domain"], selected["weight"], strict=True))
            vector = np.asarray([by_domain[bucket] for bucket in bucket_names], dtype=float)
            table9_url = result.get("native_table9_wandb_url", result.get("table9_wandb_url", ""))
            training_url = result.get("training_wandb_url", "")
            run_id = _wandb_id(training_url, result.get("eval_metrics_uri", ""))
            records.append(
                {
                    "panel": "delphi_3e18_39bucket",
                    "scale": PANEL_SCALE["delphi_3e18_39bucket"],
                    "row_id": f"delphi_recent::{source_name}::{result['candidate_id']}",
                    "source": source_name,
                    "source_row_id": str(result["candidate_id"]),
                    "source_experiment": source_dir.name,
                    "proposal_model": source_name,
                    "proposal_target": str(result["target"]),
                    "epoch_cap": result["epoch_cap"],
                    "training_wandb_run_id": run_id,
                    "training_wandb_url": training_url,
                    "table9_eval_run_id": _wandb_id(table9_url, ""),
                    "table9_eval_url": table9_url,
                    "data_seed": np.nan,
                    "trainer_seed": np.nan,
                    "phase_tv": 0.0,
                    "uncheatable_bpb": result["uncheatable_bpb"],
                    "table9_macro_bpb": result["table9_macro_bpb"],
                }
            )
            vectors.append(vector)
        rows.append(
            _finalize_audit(
                pd.DataFrame(records),
                np.stack(vectors),
                _fit_weights("delphi_3e18_39bucket", bucket_names),
            )
        )
    return pd.concat(rows, ignore_index=True)


def _audit_epoch_dose_delphi(bucket_names: tuple[str, ...]) -> pd.DataFrame:
    manifest = pd.read_csv(EPOCH_DOSE_RUN_MANIFEST)
    results = pd.read_csv(EPOCH_DOSE_RESULTS)
    source = manifest.merge(results, on=["run_order", "run_name"], validate="one_to_one")
    if len(source) != len(manifest) or len(source) != 277:
        raise ValueError("Delphi epoch-dose materialization must contain all 277 designed runs")

    long_weights = pd.read_csv(EPOCH_DOSE_PHASE_WEIGHTS)
    weight_table = long_weights.pivot(index="run_name", columns=["phase", "domain"], values="weight")
    weights = np.asarray(
        [[weight_table.loc[run_name, ("phase_0", bucket)] for bucket in bucket_names] for run_name in source.run_name],
        dtype=float,
    )
    phase1 = np.asarray(
        [[weight_table.loc[run_name, ("phase_1", bucket)] for bucket in bucket_names] for run_name in source.run_name],
        dtype=float,
    )
    frame = pd.DataFrame(
        {
            "panel": "delphi_3e18_39bucket",
            "scale": PANEL_SCALE["delphi_3e18_39bucket"],
            "row_id": "delphi_epoch_dose::" + source["run_name"].astype(str),
            "source": "conditional_epoch_dose_response",
            "source_row_id": source["run_name"].astype(str),
            "source_experiment": "bucket_epoch_dose_response_20260729",
            "proposal_model": "designed_conditional_epoch_dose",
            "proposal_target": "uncheatable_and_table9",
            "epoch_cap": np.nan,
            "training_wandb_run_id": source["training_wandb_run_id"].astype(str),
            "training_wandb_url": source["training_wandb_url"].astype(str),
            "table9_eval_run_id": source["table9_eval_run_id"].fillna("").astype(str),
            "table9_eval_url": source["table9_eval_url"].fillna("").astype(str),
            "data_seed": source["data_seed"],
            "trainer_seed": source["trainer_seed"],
            "subset_seed": source["simulated_epoch_subset_seed"],
            "phase_tv": 0.5 * np.abs(weights - phase1).sum(axis=1),
            "uncheatable_bpb": source["uncheatable_bpb"],
            "table9_macro_bpb": source["table9_macro_bpb"],
        }
    )
    return _finalize_audit(frame, weights, _fit_weights("delphi_3e18_39bucket", bucket_names))


def _apriori_materialization_present() -> bool:
    paths = (
        APRIORI_SWARM_RESULTS,
        APRIORI_SWARM_UNCHEATABLE_COMPONENTS,
        APRIORI_SWARM_TABLE9_COMPONENTS,
        APRIORI_SWARM_MATERIALIZATION_MANIFEST,
    )
    present = [path.exists() for path in paths]
    if any(present) and not all(present):
        missing = [str(path) for path, exists in zip(paths, present, strict=True) if not exists]
        raise FileNotFoundError(f"A-priori swarm materialization is partial; missing {missing}")
    return all(present)


def _audit_apriori_delphi(bucket_names: tuple[str, ...]) -> pd.DataFrame:
    design = pd.read_csv(APRIORI_SWARM_DESIGN, low_memory=False).reset_index(names="run_order")
    results = pd.read_csv(APRIORI_SWARM_RESULTS, low_memory=False)
    source = design.merge(
        results, on=["run_order", "run_name"], suffixes=("_design", "_observed"), validate="one_to_one"
    )
    if len(source) != len(results) or not source["source"].eq("new").all():
        raise ValueError("A-priori materialization contains rows outside the frozen new-run design")

    for seed in ("data_seed", "trainer_seed", "subset_seed"):
        designed = pd.to_numeric(source[f"{seed}_design"], errors="raise").astype(int)
        observed = pd.to_numeric(source[f"{seed}_observed"], errors="raise").astype(int)
        if not designed.equals(observed):
            raise ValueError(f"A-priori materialized {seed} differs from the frozen design")

    weights = source.loc[:, [f"phase_0_{bucket}" for bucket in bucket_names]].to_numpy(float)
    phase1 = source.loc[:, [f"phase_1_{bucket}" for bucket in bucket_names]].to_numpy(float)
    designed_fractions = source.loc[:, [f"pool_fraction_{bucket}" for bucket in bucket_names]].to_numpy(float)
    observed_fractions = source.loc[:, [f"{POOL_FRACTION_PREFIX}{bucket}" for bucket in bucket_names]].to_numpy(float)
    if not np.allclose(designed_fractions, observed_fractions, atol=1e-12, rtol=0.0):
        raise ValueError("A-priori materialized pool fractions differ from the frozen design")

    frame = pd.DataFrame(
        {
            "panel": "delphi_3e18_39bucket",
            "scale": PANEL_SCALE["delphi_3e18_39bucket"],
            "row_id": "delphi_apriori::" + source["run_name"].astype(str),
            "source": "prospectively_frozen_apriori_swarm",
            "source_row_id": source["run_name"].astype(str),
            "source_experiment": "delphi_apriori_swarm_280_20260904",
            "proposal_model": "apriori_design",
            "proposal_target": "uncheatable_and_table9",
            "epoch_cap": np.nan,
            "training_wandb_run_id": source["training_wandb_run_id"].astype(str),
            "training_wandb_url": source["training_wandb_url"].astype(str),
            "table9_eval_run_id": source["table9_eval_run_id"].fillna("").astype(str),
            "table9_eval_url": source["table9_eval_url"].fillna("").astype(str),
            "data_seed": source["data_seed_observed"],
            "trainer_seed": source["trainer_seed_observed"],
            "subset_seed": source["subset_seed_observed"],
            "phase_tv": 0.5 * np.abs(weights - phase1).sum(axis=1),
            "uncheatable_bpb": source["uncheatable_bpb"],
            "table9_macro_bpb": source["table9_macro_bpb"],
        }
    )
    return _finalize_audit(
        frame,
        weights,
        _fit_weights("delphi_3e18_39bucket", bucket_names),
        observed_fractions,
    )


def audit_sources() -> pd.DataFrame:
    bucket_names = domains()
    sources = [
        _audit_60m(bucket_names),
        _audit_300m(bucket_names),
        _audit_delphi_archive(bucket_names),
        _audit_recent_delphi(bucket_names),
        _audit_epoch_dose_delphi(bucket_names),
    ]
    if _apriori_materialization_present():
        sources.append(_audit_apriori_delphi(bucket_names))
    audit = pd.concat(sources, ignore_index=True)
    if audit["row_id"].duplicated().any():
        raise ValueError("Audit row IDs are not unique")
    weight_columns = [f"weight::{bucket}" for bucket in bucket_names]
    pool_columns = [f"{POOL_FRACTION_PREFIX}{bucket}" for bucket in bucket_names]
    weights = audit.loc[:, weight_columns].to_numpy(float)
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Audited mixture weights are invalid")
    return audit.loc[:, [*AUDIT_COLUMNS, *weight_columns, *pool_columns]]


def _local_table9_components(eligible: pd.DataFrame) -> pd.DataFrame:
    components = table9_components()
    eligible_row_ids = set(eligible["row_id"])
    records: list[dict[str, object]] = []

    sixty = pd.read_csv(SIXTY_M_TABLE9).set_index("run_name")
    for row in eligible[eligible["panel"].eq("60m_39bucket")].itertuples(index=False):
        if row.source_row_id not in sixty.index:
            continue
        payload = sixty.loc[row.source_row_id]
        for position, component in enumerate(components):
            if component.startswith("olmo_base_eval/easy_bpb/"):
                column = "table9/" + component.removeprefix("olmo_base_eval/easy_bpb/")
            else:
                column = f"table9/{component}/bpb"
            records.append(
                _component_record(
                    row.row_id, row.panel, "table9", position, component, payload[column], "local_table9_csv"
                )
            )

    three_hundred = pd.read_csv(THREE_HUNDRED_M_PACKET, low_memory=False).set_index("run_name")
    for row in eligible[eligible["panel"].eq("300m_39bucket")].itertuples(index=False):
        payload = three_hundred.loc[row.source_row_id]
        for position, component in enumerate(components):
            records.append(
                _component_record(
                    row.row_id,
                    row.panel,
                    "table9",
                    position,
                    component,
                    payload[component],
                    "local_300m_checkpoint_packet",
                )
            )

    for source_name, source_dir in RECENT_DELPHI_SOURCES.items():
        payload = pd.read_csv(source_dir / "measured_table9_components.csv")
        for item in payload.to_dict("records"):
            row_id = f"delphi_recent::{source_name}::{item['candidate_id']}"
            if row_id not in eligible_row_ids:
                continue
            position = int(item["component_position"])
            component = _canonical_table9_component(str(item["component"]))
            if position >= len(components) or component != components[position]:
                raise ValueError(f"{row_id}: Table-9 component does not match position {position}")
            records.append(
                _component_record(
                    row_id,
                    "delphi_3e18_39bucket",
                    "table9",
                    position,
                    component,
                    item["bpb"],
                    "local_table9_csv",
                )
            )
    return pd.DataFrame(records)


def _local_uncheatable_components(eligible: pd.DataFrame) -> pd.DataFrame:
    packet = pd.read_csv(THREE_HUNDRED_M_PACKET, low_memory=False).set_index("run_name")
    if packet.index.has_duplicates:
        raise ValueError("300M checkpoint packet has duplicate run names")
    records: list[dict[str, object]] = []
    for row in eligible[eligible["panel"].eq("300m_39bucket")].itertuples(index=False):
        payload = packet.loc[row.source_row_id]
        component_values = {component: payload[component.replace("/", "_")] for component in UNCHEATABLE_COMPONENTS}
        if not all(np.isfinite(value) for value in component_values.values()):
            continue
        for position, (component, value) in enumerate(component_values.items()):
            records.append(
                _component_record(
                    row.row_id,
                    row.panel,
                    "uncheatable",
                    position,
                    component,
                    value,
                    "local_300m_checkpoint_packet",
                )
            )
    return pd.DataFrame(records)


def _local_epoch_dose_components(eligible: pd.DataFrame) -> pd.DataFrame:
    eligible_row_ids = set(eligible["row_id"])
    frames: list[pd.DataFrame] = []
    for target, path in (
        ("uncheatable", EPOCH_DOSE_UNCHEATABLE_COMPONENTS),
        ("table9", EPOCH_DOSE_TABLE9_COMPONENTS),
    ):
        source = pd.read_csv(path)
        source["row_id"] = "delphi_epoch_dose::" + source["run_name"].astype(str)
        source = source[source["row_id"].isin(eligible_row_ids)].copy()
        source["panel"] = "delphi_3e18_39bucket"
        source["target"] = target
        frames.append(source[["row_id", "panel", "target", "component_position", "component", "bpb", "provenance"]])
    return pd.concat(frames, ignore_index=True)


def _local_apriori_components(eligible: pd.DataFrame) -> pd.DataFrame:
    columns = ["row_id", "panel", "target", "component_position", "component", "bpb", "provenance"]
    if not _apriori_materialization_present():
        return pd.DataFrame(columns=columns)
    eligible_row_ids = set(eligible["row_id"])
    frames: list[pd.DataFrame] = []
    for target, path in (
        ("uncheatable", APRIORI_SWARM_UNCHEATABLE_COMPONENTS),
        ("table9", APRIORI_SWARM_TABLE9_COMPONENTS),
    ):
        source = pd.read_csv(path)
        source["row_id"] = "delphi_apriori::" + source["run_name"].astype(str)
        source = source[source["row_id"].isin(eligible_row_ids)].copy()
        source["panel"] = "delphi_3e18_39bucket"
        source["target"] = target
        if target == "table9" and not source.empty:
            source["component"] = source["component"].map(_canonical_table9_component)
        frames.append(source.loc[:, columns])
    return pd.concat(frames, ignore_index=True)


def _component_record(
    row_id: str,
    panel: str,
    target: str,
    position: int,
    component: str,
    value: object,
    provenance: str,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "panel": panel,
        "target": target,
        "component_position": position,
        "component": component,
        "bpb": float(value),
        "provenance": provenance,
    }


def component_requests(eligible: pd.DataFrame) -> list[ComponentRequest]:
    requests: list[ComponentRequest] = []
    for row in eligible.itertuples(index=False):
        if row.training_wandb_run_id and np.isfinite(row.uncheatable_bpb):
            requests.append(
                ComponentRequest(
                    row_id=row.row_id,
                    panel=row.panel,
                    target="uncheatable",
                    project=TRAIN_PROJECT,
                    wandb_run_id=row.training_wandb_run_id,
                    expected_aggregate=float(row.uncheatable_bpb),
                )
            )
        if row.source.startswith("archive::") and row.table9_eval_run_id and np.isfinite(row.table9_macro_bpb):
            requests.append(
                ComponentRequest(
                    row_id=row.row_id,
                    panel=row.panel,
                    target="table9",
                    project=EVAL_PROJECT,
                    wandb_run_id=row.table9_eval_run_id,
                    expected_aggregate=float(row.table9_macro_bpb),
                )
            )
    return requests


def _summary_aggregate(summary: dict[str, Any], request: ComponentRequest) -> float:
    if request.target == "uncheatable":
        return float(summary[UNCHEATABLE_AGGREGATE])
    for key in TABLE9_AGGREGATE_KEYS:
        if key in summary:
            return float(summary[key])
    summary_keys = table9_summary_keys()
    values = np.asarray([summary[summary_keys[component]] for component in table9_components()], dtype=float)
    return float(values.mean())


def _component_aggregate(panel: str, target: str, values: np.ndarray) -> float:
    if target == "table9":
        return float(values.mean())
    weights = DELPHI_UNCHEATABLE_WEIGHTS if panel == "delphi_3e18_39bucket" else LEGACY_UNCHEATABLE_WEIGHTS
    return float(values @ weights)


def _fetch_component_request(
    api: wandb.Api, request: ComponentRequest
) -> tuple[dict[str, object], list[dict[str, object]]]:
    identity = f"{request.project}/{request.wandb_run_id}"
    try:
        run = api.run(identity)
        summary = dict(run.summary)
        aggregate = _summary_aggregate(summary, request)
        error = abs(aggregate - request.expected_aggregate)
        components = UNCHEATABLE_COMPONENTS if request.target == "uncheatable" else table9_components()
        summary_keys = (
            {component: component for component in components}
            if request.target == "uncheatable"
            else table9_summary_keys()
        )
        missing = [component for component in components if summary_keys[component] not in summary]
        component_aggregate = np.nan
        component_error = np.nan
        if error > AGGREGATE_TOLERANCE:
            status = "aggregate_mismatch"
            records: list[dict[str, object]] = []
        elif missing:
            status = "missing_components"
            records = []
        else:
            values = np.asarray([summary[summary_keys[component]] for component in components], dtype=float)
            component_aggregate = _component_aggregate(request.panel, request.target, values)
            component_error = abs(component_aggregate - request.expected_aggregate)
            if component_error > AGGREGATE_TOLERANCE:
                status = "component_reconstruction_mismatch"
                records = []
            else:
                status = "complete"
                records = [
                    _component_record(
                        request.row_id,
                        request.panel,
                        request.target,
                        position,
                        component,
                        value,
                        "wandb_summary_validated",
                    )
                    for position, (component, value) in enumerate(zip(components, values, strict=True))
                ]
        result: dict[str, object] = {
            "row_id": request.row_id,
            "panel": request.panel,
            "target": request.target,
            "project": request.project,
            "wandb_run_id": request.wandb_run_id,
            "wandb_state": str(run.state),
            "expected_aggregate": request.expected_aggregate,
            "summary_aggregate": aggregate,
            "absolute_error": error,
            "component_aggregate": component_aggregate,
            "component_absolute_error": component_error,
            "component_count": len(records),
            "status": status,
            "detail": ";".join(missing),
        }
        return result, records
    except Exception as error:
        return (
            {
                "row_id": request.row_id,
                "panel": request.panel,
                "target": request.target,
                "project": request.project,
                "wandb_run_id": request.wandb_run_id,
                "wandb_state": "",
                "expected_aggregate": request.expected_aggregate,
                "summary_aggregate": np.nan,
                "absolute_error": np.nan,
                "component_aggregate": np.nan,
                "component_absolute_error": np.nan,
                "component_count": 0,
                "status": "fetch_error",
                "detail": f"{type(error).__name__}: {error}",
            },
            [],
        )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def fetch_wandb_components(
    requests: list[ComponentRequest], output_dir: Path, workers: int, offline: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_path = output_dir / "input/wandb_component_status.csv"
    component_path = output_dir / "input/wandb_components.csv"
    old_status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    old_components = pd.read_csv(component_path) if component_path.exists() else pd.DataFrame()
    complete_keys = set()
    if not old_status.empty and "component_absolute_error" in old_status:
        complete = old_status[
            old_status["status"].eq("complete") & old_status["component_absolute_error"].le(AGGREGATE_TOLERANCE)
        ]
        complete_keys = set(zip(complete["row_id"], complete["target"], strict=True))
    pending = [request for request in requests if (request.row_id, request.target) not in complete_keys]
    if offline and pending:
        print(f"offline: {len(pending)} component payloads remain unavailable", flush=True)
    elif pending:
        api = wandb.Api(timeout=120)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            fetched = list(executor.map(lambda request: _fetch_component_request(api, request), pending))
        status_rows = [status for status, _records in fetched]
        component_rows = [record for _status, records in fetched for record in records]
        new_status = pd.DataFrame(status_rows)
        new_components = pd.DataFrame(component_rows)
        pending_keys = {(request.row_id, request.target) for request in pending}
        if not old_status.empty:
            keep = ~pd.Series(list(zip(old_status["row_id"], old_status["target"], strict=True))).isin(pending_keys)
            old_status = old_status.loc[keep.to_numpy()].copy()
        if not old_components.empty:
            keep = ~pd.Series(list(zip(old_components["row_id"], old_components["target"], strict=True))).isin(
                pending_keys
            )
            old_components = old_components.loc[keep.to_numpy()].copy()
        old_status = pd.concat([old_status, new_status], ignore_index=True)
        old_components = pd.concat([old_components, new_components], ignore_index=True)
        _atomic_csv(old_status.sort_values(["panel", "row_id", "target"]), status_path)
        _atomic_csv(old_components.sort_values(["panel", "row_id", "target", "component_position"]), component_path)
    return old_status, old_components


def _validate_components(runs: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    expected_components = {"uncheatable": UNCHEATABLE_COMPONENTS, "table9": table9_components()}
    aggregates = runs.set_index("row_id")[["panel", "uncheatable_bpb", "table9_macro_bpb"]]
    checks: list[dict[str, object]] = []
    for (row_id, target), group in components.groupby(["row_id", "target"], sort=False):
        if target not in expected_components:
            raise ValueError(f"{row_id}: unknown component target {target}")
        if group["component"].duplicated().any():
            raise ValueError(f"{row_id}/{target}: duplicate atomic components")
        inventory = expected_components[target]
        expected_count = len(inventory)
        if len(group) != expected_count:
            raise ValueError(f"{row_id}/{target}: partial component payload included")
        ordered = group.sort_values("component_position")
        if ordered["component_position"].tolist() != list(range(expected_count)):
            raise ValueError(f"{row_id}/{target}: component positions do not match canonical inventory")
        if tuple(ordered["component"]) != inventory:
            raise ValueError(f"{row_id}/{target}: components do not match canonical component inventory")
        values = ordered["bpb"].to_numpy(float)
        panel = str(aggregates.loc[row_id, "panel"])
        if target == "table9":
            expected = float(aggregates.loc[row_id, "table9_macro_bpb"])
        else:
            expected = float(aggregates.loc[row_id, "uncheatable_bpb"])
        reconstructed = _component_aggregate(panel, target, values)
        error = abs(reconstructed - expected)
        if error > AGGREGATE_TOLERANCE:
            raise ValueError(f"{row_id}/{target}: components differ from aggregate by {error:.3g}")
        checks.append(
            {
                "row_id": row_id,
                "panel": panel,
                "target": target,
                "component_count": len(group),
                "expected_aggregate": expected,
                "reconstructed_aggregate": reconstructed,
                "absolute_error": error,
            }
        )
    return pd.DataFrame(checks)


def coordinate_table(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    weight_columns = [f"weight::{bucket}" for bucket in domains()]
    pool_columns = [f"{POOL_FRACTION_PREFIX}{bucket}" for bucket in domains()]
    for column in pool_columns:
        if column not in runs:
            runs[column] = 1.0
    for column in ("data_seed", "trainer_seed", "subset_seed"):
        if column not in runs:
            runs[column] = np.nan
    rows: list[dict[str, object]] = []
    for (_panel, _coordinate), group in runs.groupby(["panel", "coordinate_id"], sort=False):
        record = {
            column: group.iloc[0][column]
            for column in ["panel", "scale", "coordinate_id", *weight_columns, *pool_columns]
        }
        record.update(
            {
                "run_count": len(group),
                "source_count": group["source"].nunique(),
                "sources": ";".join(sorted(set(group["source"].astype(str)))),
                "row_ids": ";".join(group["row_id"].astype(str)),
            }
        )
        for seed_column in ("data_seed", "trainer_seed", "subset_seed"):
            values = sorted(set(pd.to_numeric(group[seed_column], errors="coerce").dropna().astype(int)))
            record[seed_column] = values[0] if len(values) == 1 else np.nan
            record[f"{seed_column}s"] = ";".join(str(value) for value in values)
        for target in ("uncheatable_bpb", "table9_macro_bpb"):
            values = group[target].dropna().to_numpy(float)
            prefix = target.removesuffix("_bpb")
            record[f"{prefix}_n"] = len(values)
            record[f"{prefix}_mean_bpb"] = float(values.mean()) if len(values) else np.nan
            record[f"{prefix}_sd_bpb"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["panel", "coordinate_id"]).reset_index(drop=True)


def coordinate_component_table(runs: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    joined = components.merge(runs[["row_id", "coordinate_id"]], on="row_id", validate="many_to_one")
    rows = (
        joined.groupby(["panel", "coordinate_id", "target", "component_position", "component"], as_index=False)
        .agg(bpb_mean=("bpb", "mean"), bpb_sd=("bpb", "std"), run_count=("bpb", "size"))
        .sort_values(["panel", "coordinate_id", "target", "component_position"])
    )
    return rows


def _source_inventory(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (panel, source), group in audit.groupby(["panel", "source"], sort=False):
        eligible = group[group["eligible"]]
        rows.append(
            {
                "panel": panel,
                "source": source,
                "audited_rows": len(group),
                "single_phase_rows": int((group["phase_tv"] <= PHASE_TOLERANCE).sum()),
                "fit_overlap_rows": int(group["exclusion_reason"].eq("fit_coordinate_overlap").sum()),
                "eligible_runs": len(eligible),
                "eligible_coordinates": eligible["coordinate_id"].nunique(),
                "uncheatable_aggregate_runs": int(eligible["uncheatable_bpb"].notna().sum()),
                "table9_aggregate_runs": int(eligible["table9_macro_bpb"].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["panel", "source"]).reset_index(drop=True)


def _coverage_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (panel, source), group in runs.groupby(["panel", "source"], sort=False):
        for target, aggregate_column, count_column, expected in (
            ("uncheatable", "uncheatable_bpb", "uncheatable_component_count", 7),
            ("table9", "table9_macro_bpb", "table9_component_count", 51),
        ):
            rows.append(
                {
                    "panel": panel,
                    "source": source,
                    "target": target,
                    "eligible_runs": len(group),
                    "aggregate_runs": int(group[aggregate_column].notna().sum()),
                    "complete_component_runs": int(group[count_column].eq(expected).sum()),
                    "missing_component_runs_with_aggregate": int(
                        (group[aggregate_column].notna() & group[count_column].ne(expected)).sum()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["panel", "source", "target"]).reset_index(drop=True)


def _write_report(
    output_dir: Path,
    audit: pd.DataFrame,
    runs: pd.DataFrame,
    coordinates: pd.DataFrame,
    components: pd.DataFrame,
    status: pd.DataFrame,
) -> None:
    delphi = audit[audit["panel"].eq("delphi_3e18_39bucket")]
    delphi_overlap_count = int(delphi["exclusion_reason"].eq("fit_coordinate_overlap").sum())
    recent_delphi_count = int(runs["source"].isin(RECENT_DELPHI_SOURCES).sum())
    epoch_dose_audit = audit[audit["source"].eq("conditional_epoch_dose_response")]
    epoch_dose_runs = runs[runs["source"].eq("conditional_epoch_dose_response")]
    lines = [
        "# Single-phase heldout benchmark inventory",
        "",
        "## Eligibility contract",
        "",
        "A row is eligible only when its two phase mixtures are equal within `1e-10` total-variation tolerance and",
        "its 39-bucket coordinate is farther than `1e-10` maximum-coordinate distance from every row in the current",
        "canonical fit panel. This recomputation supersedes historical `heldout` and `fit_panel_overlap` labels.",
        "Repeated seeds remain separate at run level and are averaged only in the coordinate export.",
        "",
        "## Usable evidence",
        "",
        (
            "| Panel | Eligible runs | Unique coordinates | Uncheatable aggregates | Table-9 aggregates | "
            "Full Uncheatable components | Full Table-9 components |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for panel, group in runs.groupby("panel", sort=False):
        lines.append(
            f"| {panel} | {len(group)} | {group.coordinate_id.nunique()} | {group.uncheatable_bpb.notna().sum()} | "
            f"{group.table9_macro_bpb.notna().sum()} | {group.uncheatable_component_count.eq(7).sum()} | "
            f"{group.table9_component_count.eq(51).sum()} |"
        )
    lines.extend(
        [
            "",
            f"Total: **{len(runs)} runs across {len(coordinates)} coordinates**. "
            f"The audit examined {len(audit)} source rows; "
            f"{(~audit.eligible).sum()} were excluded before modeling.",
            "",
            "## Important corrections and gaps",
            "",
            f"- The legacy Delphi archive contains {delphi_overlap_count} rows that "
            "match a current fit coordinate and are excluded. Its older overlap flag is not used.",
            (
                "- The Delphi set includes 187 archival one-phase validation runs plus "
                f"{recent_delphi_count} recent epoch-cap validation runs and "
                f"{len(epoch_dose_runs)} fit-panel-disjoint conditional epoch-dose runs."
            ),
            (
                f"- The conditional epoch-dose source contributes {len(epoch_dose_audit)} raw single-phase runs. "
                f"{epoch_dose_audit.exclusion_reason.eq('fit_coordinate_overlap').sum()} overlap the canonical fit "
                f"panel and remain development/noise evidence rather than fresh heldouts; all {len(epoch_dose_runs)} "
                f"remaining runs have complete Uncheatable payloads and "
                f"{epoch_dose_runs.table9_macro_bpb.notna().sum()} currently have complete Table-9 payloads."
            ),
            (
                "- The 300M packet contains all 280 current fit rows, which are excluded. Its 134 external coordinates "
                "comprise 78 proportional-controllability validations and 56 interventions; 117 have complete "
                "Uncheatable data and all 134 have complete Table-9 data."
            ),
            (
                "- Atomic metrics are present only when all components reproduce the recorded aggregate. Aggregate-only "
                "rows remain valid for aggregate-target scoring but not componentwise scoring."
            ),
            (
                "- Thirteen full-canonical epoch-cap rows currently lack Table-9 endpoints. They remain usable for "
                "Uncheatable scoring and should be augmented after those evaluations finish."
            ),
        ]
    )
    if not status.empty:
        failures = status[~status["status"].eq("complete")]
        lines.append(
            f"- W&B component recovery validated {status.status.eq('complete').sum()} payloads and rejected or "
            "could not "
            f"retrieve {len(failures)}; see `input/wandb_component_status.csv` for exact reasons."
        )
    lines.extend(
        [
            "",
            "## Modeling use",
            "",
            "Use `heldout_runs.csv` to estimate seed noise and score individual runs. Use `heldout_coordinates.csv` for",
            "mixture-blocked selection metrics so repeated seeds cannot cross train/test boundaries.",
            "Componentwise models must restrict to rows with the corresponding `*_components_complete` flag.",
            "Do not pool scales without an explicit scale-conditioned model.",
            "",
            "## Files",
            "",
            "- `heldout_runs.csv`: eligible run-level outcomes and 39 mixture weights.",
            "- `heldout_coordinates.csv`: coordinate means, SDs, replicate counts, and weights.",
            "- `heldout_components.csv`: validated run-level atomic BPBs.",
            "- `heldout_coordinate_components.csv`: coordinate-level atomic means and replicate SDs.",
            "- `mixture_weights.csv`: long-form run-level mixture weights.",
            "- `source_inventory.csv`, `target_coverage.csv`, and `excluded_rows.csv`: audit accounting.",
            "- `component_reconstruction.csv`: exact aggregate reconstruction checks.",
            "- `manifest.json`: source hashes, schema, tolerances, and frozen counts.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def prepare(output_dir: Path, workers: int, offline: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_sources()
    eligible = audit[audit["eligible"]].copy().reset_index(drop=True)
    local_components = pd.concat(
        [
            _local_table9_components(eligible),
            _local_uncheatable_components(eligible),
            _local_epoch_dose_components(eligible),
            _local_apriori_components(eligible),
        ],
        ignore_index=True,
    )
    local_component_keys = set(zip(local_components["row_id"], local_components["target"], strict=True))
    requests = [
        request
        for request in component_requests(eligible)
        if (request.row_id, request.target) not in local_component_keys
    ]
    status, wandb_components = fetch_wandb_components(requests, output_dir, workers, offline)
    components = pd.concat([local_components, wandb_components], ignore_index=True)
    if not components.empty:
        components = components.drop_duplicates(["row_id", "target", "component"], keep="first")
        components = components.sort_values(["panel", "row_id", "target", "component_position"]).reset_index(drop=True)
    reconstruction = _validate_components(eligible, components)

    counts = (
        components.groupby(["row_id", "target"]).size().unstack(fill_value=0) if not components.empty else pd.DataFrame()
    )
    for target, expected in (("uncheatable", 7), ("table9", 51)):
        column = f"{target}_component_count"
        eligible[column] = eligible["row_id"].map(counts[target] if target in counts else {}).fillna(0).astype(int)
        eligible[f"{target}_components_complete"] = eligible[column].eq(expected)

    coordinates = coordinate_table(eligible)
    coordinate_components = coordinate_component_table(eligible, components)
    inventory = _source_inventory(audit)
    coverage = _coverage_table(eligible)
    weight_columns = [f"weight::{bucket}" for bucket in domains()]
    long_weights = eligible.melt(
        id_vars=["panel", "scale", "row_id", "coordinate_id"],
        value_vars=weight_columns,
        var_name="bucket",
        value_name="weight",
    )
    long_weights["bucket"] = long_weights["bucket"].str.removeprefix("weight::")

    _atomic_csv(eligible, output_dir / "heldout_runs.csv")
    _atomic_csv(coordinates, output_dir / "heldout_coordinates.csv")
    _atomic_csv(components, output_dir / "heldout_components.csv")
    _atomic_csv(coordinate_components, output_dir / "heldout_coordinate_components.csv")
    _atomic_csv(long_weights, output_dir / "mixture_weights.csv")
    _atomic_csv(inventory, output_dir / "source_inventory.csv")
    _atomic_csv(coverage, output_dir / "target_coverage.csv")
    _atomic_csv(
        audit.loc[~audit["eligible"], [column for column in AUDIT_COLUMNS if column in audit]],
        output_dir / "excluded_rows.csv",
    )
    _atomic_csv(reconstruction, output_dir / "component_reconstruction.csv")

    source_paths = [
        TABLE9_METADATA,
        *FIT_PANEL_PATHS.values(),
        SIXTY_M_AUDIT,
        SIXTY_M_TABLE9,
        THREE_HUNDRED_M_PACKET,
        DELPHI_ARCHIVE,
        EPOCH_DOSE_RUN_MANIFEST,
        EPOCH_DOSE_PHASE_WEIGHTS,
        EPOCH_DOSE_RESULTS,
        EPOCH_DOSE_UNCHEATABLE_COMPONENTS,
        EPOCH_DOSE_TABLE9_COMPONENTS,
        EPOCH_DOSE_MATERIALIZATION_MANIFEST,
    ]
    for source_dir in RECENT_DELPHI_SOURCES.values():
        source_paths.extend(
            [
                source_dir / "measured_results.csv",
                source_dir / "candidate_weights.csv",
                source_dir / "measured_table9_components.csv",
            ]
        )
    if _apriori_materialization_present():
        source_paths.extend(
            [
                APRIORI_SWARM_DESIGN,
                APRIORI_SWARM_RESULTS,
                APRIORI_SWARM_UNCHEATABLE_COMPONENTS,
                APRIORI_SWARM_TABLE9_COMPONENTS,
                APRIORI_SWARM_MATERIALIZATION_MANIFEST,
            ]
        )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "eligibility": {
            "phase_tv_tolerance": PHASE_TOLERANCE,
            "fit_overlap_max_abs_tolerance": FIT_OVERLAP_TOLERANCE,
            "coordinate_identity": (
                "mixture weights plus per-bucket pool fractions; full-support IDs retain schema-v1 hashes"
            ),
            "fit_panels": {panel: str(path.relative_to(REPO_ROOT)) for panel, path in FIT_PANEL_PATHS.items()},
        },
        "aggregate_tolerance": AGGREGATE_TOLERANCE,
        "domains": list(domains()),
        "table9_components": list(table9_components()),
        "uncheatable_components": list(UNCHEATABLE_COMPONENTS),
        "counts": {
            "audited_rows": len(audit),
            "eligible_runs": len(eligible),
            "eligible_coordinates": len(coordinates),
            "components": len(components),
            "excluded_rows": int((~audit["eligible"]).sum()),
        },
        "panel_counts": {
            panel: {"runs": int(row.runs), "coordinates": int(row.coordinates)}
            for panel, row in (
                eligible.groupby("panel")
                .agg(runs=("row_id", "size"), coordinates=("coordinate_id", "nunique"))
                .iterrows()
            )
        },
        "source_hashes": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in source_paths},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_report(output_dir, audit, eligible, coordinates, components, status)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--offline", action="store_true", help="Use only already cached W&B component payloads")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    manifest = prepare(args.output_dir, args.workers, args.offline)
    print(json.dumps(manifest["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
