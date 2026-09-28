# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Build the portable 2026-07-21 two-phase surrogate collaborator packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
PACKET = SCRIPT_DIR / "reference_outputs/two_phase_surrogate_collaborator_packet_20260721"
ZIP_PATH = PACKET.with_suffix(".zip")
LEDGER = REPO_ROOT / ".experiments/ledger.sqlite"
PHASE_PREFIXES = ("phase_0_weight::", "phase_1_weight::")

OLD_PACKET = SCRIPT_DIR / "reference_outputs/two_phase_solver_gap_collaborator_packet_20260701"
DELPHI_FIT = SCRIPT_DIR / "reference_outputs/delphi_augmented_swarm_3e18_20260714"
DELPHI_ONE_PHASE = SCRIPT_DIR / "reference_outputs/delphi_one_phase_augmented_swarm_3e18_20260715"
DELPHI_HELDOUT = SCRIPT_DIR / "reference_outputs/delphi_3e18_append_only_heldouts_20260714"
PRODUCTION = SCRIPT_DIR / "reference_outputs/grug_moe_production_swarm_results_20260704"
PRODUCTION_MODEL = SCRIPT_DIR / (
    "reference_outputs/grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/model.json"
)
STARCODER_COSINE = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/paper_plots/data/two_phase_starcoder_combined_143_from_wandb.csv"
)
STARCODER_WSD = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_surface_refined_20260714"
METRIC_REGISTRY = SCRIPT_DIR / "metric_registry/metrics_wide.csv"

CORE_TARGETS = ("uncheatable_bpb", "table9_macro_bpb")
PRIVATE_GCS_PREFIX = re.compile(r"gs://marin-[^/]+/[^/]+/")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_ASSIGNMENT = re.compile(r"(?i)(token|secret|password|api[_-]?key)\s*=\s*[^\s,;]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theory",
        type=Path,
        help="Optional theory Markdown file to include as docs/THEORY.md.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path, *, patterns: tuple[str, ...] | None = None) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    if patterns is None:
        files = [path for path in source.rglob("*") if path.is_file()]
    else:
        files = []
        for pattern in patterns:
            files.extend(path for path in source.rglob(pattern) if path.is_file())
    for path in sorted(set(files)):
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        copy_file(path, destination / path.relative_to(source))


def sanitize_text(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    text = value.replace(str(REPO_ROOT), "<MARIN_REPO>").replace(str(Path.home()), "<USER_HOME>")
    text = PRIVATE_GCS_PREFIX.sub("gs://<redacted-private-prefix>/", text)
    text = EMAIL.sub("<REDACTED_EMAIL>", text)
    text = SECRET_ASSIGNMENT.sub(lambda match: match.group(1) + "=<REDACTED>", text)
    return text


def sanitize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=("object", "string")):
        result[column] = result[column].map(sanitize_text)
    return result


def generic_families(domains: list[str]) -> OrderedDict[str, list[str]]:
    families: OrderedDict[str, list[str]] = OrderedDict((("broad_text", []), ("tech_code", []), ("reasoning", [])))
    for domain in domains:
        is_broad = (
            domain.startswith("dolma3_cc/")
            or domain
            in {
                "dolma3_wikipedia",
                "dolmino_common_crawl_hq",
                "dolmino_olmocr_pdfs_hq",
                "dolmino_stem_heavy_crawl",
            }
            or domain.endswith("synth_qa")
        )
        is_tech = any(token in domain for token in ("stack_edu", "synth_code", "synth_math")) or domain in {
            "dolma3_arxiv",
            "dolma3_finemath_3plus",
        }
        is_reasoning = domain in {"dolmino_synth_instruction", "dolmino_synth_thinking"}
        if is_broad:
            families["broad_text"].append(domain)
        if is_tech:
            families["tech_code"].append(domain)
        if is_reasoning:
            families["reasoning"].append(domain)
    assigned = {domain for members in families.values() for domain in members}
    for domain in domains:
        if domain not in assigned:
            families["broad_text"].append(domain)
    return families


def production_families(domains: list[str]) -> OrderedDict[str, list[str]]:
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for domain in domains:
        match = re.fullmatch(r"c(?P<family>\d+)q\d+", domain)
        family = f"C{match.group('family')}" if match else domain
        grouped.setdefault(family, []).append(domain)
    return grouped


def canonical_frame(
    frame: pd.DataFrame,
    domains: list[str],
    phase0_columns: list[str],
    phase1_columns: list[str],
    *,
    row_ids: pd.Series,
    target_map: dict[str, str],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    phase0 = frame[phase0_columns].fillna(0.0).to_numpy(dtype=float, copy=True)
    phase1 = frame[phase1_columns].fillna(0.0).to_numpy(dtype=float, copy=True)
    phase0_mass = phase0.sum(axis=1, keepdims=True)
    phase1_mass = phase1.sum(axis=1, keepdims=True)
    if np.any(phase0_mass <= 0.0) or np.any(phase1_mass <= 0.0):
        raise ValueError("Canonical model table contains a phase with no resolved bucket mass")
    phase0 /= phase0_mass
    phase1 /= phase1_mass
    columns: dict[str, Any] = {"row_id": row_ids.astype(str).to_numpy()}
    for output_name, source_name in target_map.items():
        columns[output_name] = frame[source_name].to_numpy() if source_name in frame else np.nan
    for key, value in metadata.items():
        if isinstance(value, str) and value in frame:
            columns[key] = frame[value].to_numpy()
        else:
            columns[key] = value
    for index, domain in enumerate(domains):
        columns[f"{PHASE_PREFIXES[0]}{domain}"] = phase0[:, index]
        columns[f"{PHASE_PREFIXES[1]}{domain}"] = phase1[:, index]
    return pd.DataFrame(columns)


def write_canonical(frame: pd.DataFrame, name: str) -> Path:
    path = PACKET / "data/canonical" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_300m(catalog: dict[str, Any]) -> list[str]:
    source_data = OLD_PACKET / "data"
    copy_tree(source_data, PACKET / "data/raw/legacy_300m_packet_data")
    metadata = pd.read_csv(source_data / "grp_no_l2/two_phase_many_epoch_metadata.csv")
    domains = metadata["domain_name"].tolist()
    c0 = metadata["phase_0_epoch_multiplier"].tolist()
    c1 = metadata["phase_1_epoch_multiplier"].tolist()
    families = generic_families(domains)

    two = pd.read_csv(source_data / "fit_matrix_collapsed_proportional_300m.csv", low_memory=False)
    canonical_two = canonical_frame(
        two,
        domains,
        [f"phase_0_{domain}" for domain in domains],
        [f"phase_1_{domain}" for domain in domains],
        row_ids=two["run_name"],
        target_map={"uncheatable_bpb": "eval_uncheatable_eval_bpb", "table9_macro_bpb": "table9_macro_bpb"},
        metadata={
            "policy_class": "two_phase",
            "split": "fit",
            "training_series": "panel_source",
            "proposal_target": "",
            "candidate_kind": "packet_method",
            "group_id": "phase_correspondence_key",
        },
    )
    path = write_canonical(canonical_two, "300m_two_phase_fit")
    add_catalog(catalog, "300m_two_phase_fit", path, canonical_two, domains, c0, c1, families, "two_phase")

    all_rows = pd.read_csv(source_data / "all_300m_checkpoint_metrics.csv", low_memory=False)
    one = all_rows.loc[all_rows["training_phase_family"].eq("single_phase")].copy()
    if len(one) != 280:
        raise ValueError(f"Expected 280 one-phase 300M rows, found {len(one)}")
    phase0_columns = [f"phase_0_{domain}" for domain in domains]
    phase1_columns = [f"phase_1_{domain}" for domain in domains]
    canonical_one = canonical_frame(
        one,
        domains,
        phase0_columns,
        phase1_columns,
        row_ids=one["run_name"],
        target_map={"uncheatable_bpb": "eval_uncheatable_eval_bpb", "table9_macro_bpb": "table9_macro_bpb"},
        metadata={
            "policy_class": "one_phase",
            "split": "fit",
            "training_series": "panel_source",
            "proposal_target": "",
            "candidate_kind": "packet_method",
            "group_id": "phase_correspondence_key",
        },
    )
    path = write_canonical(canonical_one, "300m_one_phase_fit")
    add_catalog(catalog, "300m_one_phase_fit", path, canonical_one, domains, c0, c1, families, "one_phase")

    heldout = pd.read_csv(source_data / "heldout_300m_checkpoint_metrics.csv", low_memory=False)
    canonical_heldout = canonical_frame(
        heldout,
        domains,
        phase0_columns,
        phase1_columns,
        row_ids=heldout["run_name"],
        target_map={"uncheatable_bpb": "eval_uncheatable_eval_bpb", "table9_macro_bpb": "table9_macro_bpb"},
        metadata={
            "policy_class": "training_phase_family",
            "split": "heldout",
            "training_series": "panel_source",
            "proposal_target": "",
            "candidate_kind": "packet_method",
            "group_id": "phase_correspondence_key",
        },
    )
    path = write_canonical(canonical_heldout, "300m_heldouts")
    add_catalog(catalog, "300m_heldouts", path, canonical_heldout, domains, c0, c1, families, "mixed")
    return domains


def add_catalog(
    catalog: dict[str, Any],
    dataset_id: str,
    path: Path,
    frame: pd.DataFrame,
    domains: list[str],
    c0: list[float],
    c1: list[float],
    families: OrderedDict[str, list[str]],
    policy_class: str,
    *,
    group_column: str = "group_id",
) -> None:
    targets = [target for target in (*CORE_TARGETS, "starcoder_bpb") if target in frame and frame[target].notna().any()]
    target_coverage = {target: int(frame[target].notna().sum()) for target in targets}
    catalog["datasets"][dataset_id] = {
        "path": str(path.relative_to(PACKET)),
        "row_count": len(frame),
        "domains": domains,
        "c0": c0,
        "c1": c1,
        "families": families,
        "targets": targets,
        "target_coverage": target_coverage,
        "policy_class": policy_class,
        "group_column": group_column,
    }


def build_delphi(catalog: dict[str, Any], domains: list[str]) -> None:
    raw_root = PACKET / "data/raw"
    copy_tree(DELPHI_FIT, raw_root / "delphi_3e18_two_phase_fit")
    copy_tree(DELPHI_ONE_PHASE, raw_root / "delphi_3e18_one_phase_fit")
    for filename in (
        "heldout_current.csv",
        "heldout_registry.csv",
        "heldout_observations.csv",
        "heldout_provenance.csv",
        "audit_report.md",
        "summary.json",
    ):
        copy_file(DELPHI_HELDOUT / filename, raw_root / "delphi_3e18_heldouts" / filename)

    fit = pd.read_csv(DELPHI_FIT / "delphi_augmented_swarm_3e18_wide.csv", low_memory=False)
    token_metadata = pd.read_csv(OLD_PACKET / "data/grp_no_l2/two_phase_many_epoch_metadata.csv").set_index(
        "domain_name"
    )
    budget = float(fit["realized_train_tokens"].iloc[0])
    alpha0 = float(fit["phase_0_fraction"].iloc[0])
    token_counts = token_metadata.loc[domains, "token_count"].to_numpy(dtype=float)
    c0 = (alpha0 * budget / token_counts).tolist()
    c1 = ((1.0 - alpha0) * budget / token_counts).tolist()
    families = generic_families(domains)
    canonical_fit = canonical_frame(
        fit,
        domains,
        [f"phase_0_{domain}" for domain in domains],
        [f"phase_1_{domain}" for domain in domains],
        row_ids=fit["run_name"],
        target_map={"uncheatable_bpb": "uncheatable_bpb", "table9_macro_bpb": "table9_macro_bpb"},
        metadata={
            "policy_class": "two_phase",
            "split": "fit",
            "training_series": "source_experiment",
            "proposal_target": "",
            "candidate_kind": "panel_source",
            "group_id": "run_name",
        },
    )
    path = write_canonical(canonical_fit, "delphi_3e18_two_phase_fit")
    add_catalog(catalog, "delphi_3e18_two_phase_fit", path, canonical_fit, domains, c0, c1, families, "two_phase")

    heldout = pd.read_csv(DELPHI_HELDOUT / "heldout_current.csv", low_memory=False)
    completed = heldout.loc[
        heldout["training_state"].eq("finished") & heldout["checkpoint_declared_complete"].eq(1)
    ].reset_index(drop=True)

    def weight_matrix(column: str) -> np.ndarray:
        return np.asarray(
            [[float(json.loads(value)[domain]) for domain in domains] for value in completed[column]], dtype=float
        )

    phase0 = weight_matrix("phase_0_weights_json")
    phase1 = weight_matrix("phase_1_weights_json")
    completed_weights = completed.copy()
    for index, domain in enumerate(domains):
        completed_weights[f"source_phase_0::{domain}"] = phase0[:, index]
        completed_weights[f"source_phase_1::{domain}"] = phase1[:, index]
    canonical_heldout = canonical_frame(
        completed_weights,
        domains,
        [f"source_phase_0::{domain}" for domain in domains],
        [f"source_phase_1::{domain}" for domain in domains],
        row_ids=completed["heldout_id"],
        target_map={"uncheatable_bpb": "uncheatable_bpb", "table9_macro_bpb": "table9_macro_bpb"},
        metadata={
            "policy_class": "policy_class",
            "split": "heldout",
            "training_series": "training_series",
            "proposal_target": "proposal_target",
            "candidate_kind": "candidate_kind",
            "group_id": "mixture_sha256",
            "fit_panel_overlap": "fit_panel_overlap",
            "aggregate_kl_coefficient": "aggregate_kl_coefficient",
            "phase_information_budget": "phase_information_budget",
            "anchor_id": "anchor_id",
            "direction_id": "direction_id",
            "radius_fraction": "radius_fraction",
        },
    )
    path = write_canonical(canonical_heldout, "delphi_3e18_heldouts")
    add_catalog(catalog, "delphi_3e18_heldouts", path, canonical_heldout, domains, c0, c1, families, "mixed")

    manifest = pd.read_csv(DELPHI_ONE_PHASE / "training_manifest.csv").sort_values("run_order").reset_index(drop=True)
    long_weights = pd.read_csv(DELPHI_ONE_PHASE / "phase_weights.csv")
    weight_lookup = long_weights.set_index(["run_name", "phase", "domain"])["weight"]
    fit_lookup = fit.set_index("run_name")
    heldout_lookup = completed.set_index("wandb_run_base")
    one_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        run_name = str(row["run_name"])
        source_name = str(row["source_run_name"])
        disposition = str(row["disposition"])
        if disposition == "reused_exact_phase_tied_alias":
            source = fit_lookup.loc[source_name]
        elif disposition == "scheduled_new_training":
            source = heldout_lookup.loc[run_name]
        else:
            raise ValueError(f"Unknown one-phase disposition {disposition!r}")
        record = {
            "row_id": run_name,
            "uncheatable_bpb": float(source["uncheatable_bpb"]),
            "table9_macro_bpb": float(source["table9_macro_bpb"]),
            "policy_class": "one_phase",
            "split": "fit",
            "training_series": "delphi_3e18_one_phase_augmented_swarm",
            "proposal_target": "",
            "candidate_kind": disposition,
            "group_id": source_name,
        }
        for domain in domains:
            w0 = float(weight_lookup.loc[(run_name, "phase_0", domain)])
            w1 = float(weight_lookup.loc[(run_name, "phase_1", domain)])
            if not np.isclose(w0, w1, atol=1e-12):
                raise ValueError(f"One-phase row {run_name} is not tied")
            record[f"{PHASE_PREFIXES[0]}{domain}"] = w0
            record[f"{PHASE_PREFIXES[1]}{domain}"] = w1
        one_rows.append(record)
    canonical_one = pd.DataFrame(one_rows)
    if len(canonical_one) != 280:
        raise ValueError(f"Expected 280 one-phase Delphi rows, found {len(canonical_one)}")
    path = write_canonical(canonical_one, "delphi_3e18_one_phase_fit")
    add_catalog(catalog, "delphi_3e18_one_phase_fit", path, canonical_one, domains, c0, c1, families, "one_phase")


def build_production(catalog: dict[str, Any]) -> None:
    data_path = PRODUCTION / "production_swarm_840_wide.csv"
    copy_file(data_path, PACKET / "data/raw/production/production_swarm_840_wide.csv")
    copy_file(PRODUCTION_MODEL, PACKET / "data/raw/production/effective_exposure_model_metadata.json")
    frame = pd.read_csv(data_path, low_memory=False)
    model = json.loads(PRODUCTION_MODEL.read_text())
    domains = list(model["domain_names"])
    canonical = canonical_frame(
        frame,
        domains,
        [f"phase_0/{domain}" for domain in domains],
        [f"phase_1/{domain}" for domain in domains],
        row_ids=frame["candidate_name"],
        target_map={"uncheatable_bpb": "eval/uncheatable_eval/bpb"},
        metadata={
            "policy_class": "two_phase",
            "split": "fit",
            "training_series": "production_grug_moe_swarm",
            "proposal_target": "",
            "candidate_kind": "candidate_type",
            "group_id": "candidate_name",
        },
    )
    path = write_canonical(canonical, "production_two_phase_fit")
    add_catalog(
        catalog,
        "production_two_phase_fit",
        path,
        canonical,
        domains,
        list(model["c0"]),
        list(model["c1"]),
        production_families(domains),
        "two_phase",
    )


def build_starcoder(catalog: dict[str, Any]) -> None:
    copy_file(STARCODER_COSINE, PACKET / "data/raw/starcoder/cosine_50_50.csv")
    copy_tree(STARCODER_WSD, PACKET / "data/raw/starcoder/wsd_80_20", patterns=("*.csv", "*.json", "*.md"))
    target_column = "eval/paloma/dolma_100_programing_languages/bpb"
    frame = pd.read_csv(STARCODER_COSINE, low_memory=False)
    frame = frame.loc[frame["status"].eq("completed") & frame[target_column].notna()].reset_index(drop=True)
    domains = ["nemotron_full", "starcoder"]
    canonical = canonical_frame(
        frame,
        domains,
        [f"phase_0_{domain}" for domain in domains],
        [f"phase_1_{domain}" for domain in domains],
        row_ids=frame["run_id"],
        target_map={"starcoder_bpb": target_column},
        metadata={
            "policy_class": "two_phase",
            "split": "fit",
            "training_series": "starcoder_cosine_50_50",
            "proposal_target": "starcoder_bpb",
            "candidate_kind": "",
            "group_id": "run_id",
        },
    )

    def epoch_multiplier(phase: int, domain: str) -> float:
        weight = frame[f"phase_{phase}_{domain}"]
        epoch_domain = "nemotron" if domain == "nemotron_full" else domain
        epoch = frame[f"phase_{phase}_{epoch_domain}_epochs"]
        return float(np.median(epoch.loc[weight > 0] / weight.loc[weight > 0]))

    c0 = [epoch_multiplier(0, domain) for domain in domains]
    c1 = [epoch_multiplier(1, domain) for domain in domains]
    families = OrderedDict((("broad", ["nemotron_full"]), ("code", ["starcoder"])))
    path = write_canonical(canonical, "starcoder_cosine_50_50")
    add_catalog(catalog, "starcoder_cosine_50_50", path, canonical, domains, c0, c1, families, "two_phase")

    wsd = pd.read_csv(STARCODER_WSD / "wsd80_observed_metrics.csv")
    wsd_frame = wsd.copy()
    wsd_frame["phase_0_nemotron_full"] = 1.0 - wsd_frame["phase_0_starcoder"]
    wsd_frame["phase_1_nemotron_full"] = 1.0 - wsd_frame["phase_1_starcoder"]
    canonical_wsd = canonical_frame(
        wsd_frame,
        domains,
        [f"phase_0_{domain}" for domain in domains],
        [f"phase_1_{domain}" for domain in domains],
        row_ids=wsd_frame["selection_rank"].map(lambda value: f"wsd80_{int(value):03d}"),
        target_map={"starcoder_bpb": "wsd80_bpb"},
        metadata={
            "policy_class": "two_phase",
            "split": "fit",
            "training_series": "starcoder_wsd_80_20",
            "proposal_target": "starcoder_bpb",
            "candidate_kind": "panel",
            "group_id": "selection_rank",
        },
    )
    path = write_canonical(canonical_wsd, "starcoder_wsd80")
    add_catalog(
        catalog,
        "starcoder_wsd80",
        path,
        canonical_wsd,
        domains,
        [value * 0.8 / 0.5 for value in c0],
        [value * 0.2 / 0.5 for value in c1],
        families,
        "two_phase",
    )


def build_legacy_60m(catalog: dict[str, Any], domains: list[str]) -> None:
    frame = pd.read_csv(METRIC_REGISTRY, low_memory=False)
    frame = frame.loc[frame["scale"].eq("60m_1p2b")].reset_index(drop=True)
    raw_path = PACKET / "data/raw/legacy_60m/metrics_wide_60m.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)
    target = "eval/uncheatable_eval/bpb"
    usable = frame.loc[frame[target].notna()].copy()
    phase0_columns = [f"phase_0_{domain}" for domain in domains]
    phase1_columns = [f"phase_1_{domain}" for domain in domains]
    resolved = usable[phase0_columns].fillna(0.0).sum(axis=1).gt(0.0) & usable[phase1_columns].fillna(0.0).sum(
        axis=1
    ).gt(0.0)
    usable = usable.loc[resolved].reset_index(drop=True)
    metadata = pd.read_csv(OLD_PACKET / "data/grp_no_l2/two_phase_many_epoch_metadata.csv").set_index("domain_name")
    canonical = canonical_frame(
        usable,
        domains,
        phase0_columns,
        phase1_columns,
        row_ids=usable["run_name"],
        target_map={"uncheatable_bpb": target},
        metadata={
            "policy_class": "mixed",
            "split": "supplemental",
            "training_series": "source_experiment",
            "proposal_target": "",
            "candidate_kind": "cohort",
            "group_id": "run_name",
        },
    )
    path = write_canonical(canonical, "legacy_60m_uncheatable")
    add_catalog(
        catalog,
        "legacy_60m_uncheatable",
        path,
        canonical,
        domains,
        metadata.loc[domains, "phase_0_epoch_multiplier"].tolist(),
        metadata.loc[domains, "phase_1_epoch_multiplier"].tolist(),
        generic_families(domains),
        "mixed",
    )


def export_fieldbook() -> None:
    if not LEDGER.is_file():
        raise FileNotFoundError(LEDGER)
    connection = sqlite3.connect(LEDGER)
    selection = """
        tags LIKE '%data-mixing%' OR tags LIKE '%phase%' OR name LIKE '%DSP%'
        OR name LIKE '%GRP%' OR name LIKE '%swarm%'
    """
    experiments = pd.read_sql_query(f"SELECT * FROM v_experiments_v1 WHERE {selection} ORDER BY created_at", connection)
    ids = experiments["experiment_id"].tolist()
    placeholders = ",".join("?" for _ in ids)
    notes = pd.read_sql_query(
        f"SELECT * FROM v_notes_v1 WHERE entity_type='experiment' AND entity_id IN ({placeholders}) "
        "ORDER BY entity_id, created_at",
        connection,
        params=ids,
    )
    validations = pd.read_sql_query(
        f"SELECT * FROM v_validations_v1 WHERE entity_type='experiment' AND entity_id IN ({placeholders}) "
        "ORDER BY entity_id, created_at",
        connection,
        params=ids,
    )
    artifacts = pd.read_sql_query(
        f"SELECT * FROM v_artifacts_redacted_v1 WHERE experiment_id IN ({placeholders}) "
        "ORDER BY experiment_id, created_at",
        connection,
        params=ids,
    )
    connection.close()
    # The archive records itself in Fieldbook. Mask its digest in the exported
    # snapshot so rebuilding the archive is deterministic rather than a hash
    # self-reference with no fixed point.
    self_note = notes["title"].eq("Updated self-contained surrogate collaborator packet")
    notes.loc[self_note, "attrs_json"] = '{"packet.sha256":"<SELF_REFERENTIAL_ARCHIVE_DIGEST>"}'
    self_artifact = artifacts["attrs_json"].fillna("").str.contains('"packet.schema"', regex=False)
    artifacts.loc[self_artifact, "content_hash"] = "sha256:<SELF_REFERENTIAL_ARCHIVE_DIGEST>"
    artifacts.loc[self_artifact, "attrs_json"] = (
        '{"packet.schema":"v1","packet.scope":"all-current-swarms-and-heldouts"}'
    )
    output = PACKET / "fieldbook"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("experiments", experiments),
        ("notes", notes),
        ("validations", validations),
        ("artifacts", artifacts),
    ):
        sanitize_frame(frame).to_csv(output / f"{name}.csv", index=False)
    lines = [
        "# Redacted Fieldbook Index",
        "",
        (
            "This is a scientific export, not a copy of the operational ledger. Job commands, sessions, and raw local "
            "paths are excluded."
        ),
        "",
        f"- Experiments: {len(experiments)}",
        f"- Notes: {len(notes)}",
        f"- Validations: {len(validations)}",
        f"- Artifacts: {len(artifacts)}",
        "",
        "## Experiments",
        "",
    ]
    for row in experiments.itertuples(index=False):
        lines.append(f"- `{row.experiment_id}`: {sanitize_text(row.name)} ({row.status})")
    (output / "INDEX.md").write_text("\n".join(lines) + "\n")


def copy_evidence() -> None:
    evidence = PACKET / "evidence"
    selections = {
        "mechanistic_surrogate_discovery": [
            SCRIPT_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719/approach_registry.csv",
            SCRIPT_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719/data_use_ledger.csv",
            SCRIPT_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719/frozen_gate/acceptance_gate.json",
            SCRIPT_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719/frozen_gate/frozen_manifest.json",
        ],
        "mechanistic_surrogate_discovery/final_synthesis": [
            SCRIPT_DIR / f"reference_outputs/mechanistic_surrogate_discovery_20260719/final_synthesis/{name}"
            for name in (
                "final_report.md",
                "executive_summary.md",
                "data_dictionary.md",
                "acceptance_gate_evaluation.csv",
                "heldout_pareto_baseline.csv",
                "policy_class_metrics.csv",
                "one_phase_restriction_comparison.csv",
                "cross_scale_component_transfer.csv",
                "support_stratified_metrics.csv",
                "worst_optimism_rows.csv",
                "calibration_bins.csv",
                "adversarial_target_matched_metrics.csv",
                "adversarial_proposal_strata_metrics.csv",
                "future_confirmation_preregistration.md",
                "future_confirmation_preregistration.json",
            )
        ],
        "adversarial_stress": [
            SCRIPT_DIR / f"reference_outputs/delphi_3e18_adversarial_stress_panel_20260716/{name}"
            for name in ("report.md", "summary.json", "candidate_manifest.csv", "eligible_candidate_pool.csv")
        ],
        "adversarial_generalization": [
            SCRIPT_DIR / f"reference_outputs/delphi_3e18_adversarial_generalization_20260718/{name}"
            for name in (
                "report.md",
                "summary.json",
                "model_split_metrics.csv",
                "worst_adversarial_predictions.csv",
            )
        ],
        "sample_efficiency": [
            SCRIPT_DIR / f"reference_outputs/delphi_phase_policy_sample_efficiency_20260721/{name}"
            for name in (
                "report.md",
                "protocol.json",
                "learning_curve_runs.csv",
                "learning_curve_summary.csv",
                "endpoint_comparison.csv",
                "model_parameter_counts.csv",
                "selected_policy_report.md",
                "selected_policy_paths.csv",
                "selected_policy_endpoint_diagnostics.csv",
            )
        ],
        "hybrid_phase_ordering": [
            SCRIPT_DIR / f"reference_outputs/delphi_3e18_hybrid_phase_ordering_panel_20260720/{name}"
            for name in ("report.md", "summary.json", "candidate_manifest.csv", "fitted_models.json")
        ],
        "counterfactual_compact_transport": [
            SCRIPT_DIR / f"reference_outputs/counterfactual_compact_transport_3e18_20260721/{name}"
            for name in (
                "report.md",
                "approach_registry.csv",
                "acceptance_gate.csv",
                "data_use_ledger.csv",
                "cv_summary.csv",
                "heldout_metrics.csv",
                "heldout_predictions.csv",
                "raw_optima.csv",
            )
        ],
    }
    for group, paths in selections.items():
        for path in paths:
            copy_file(path, evidence / group / path.name)


def copy_exact_source() -> None:
    copy_file(SCRIPT_DIR / "standalone_code/dsp_exact.py", PACKET / "exact_source/standalone_code/dsp_exact.py")
    copy_file(
        SCRIPT_DIR / "standalone_code/grp_no_l2_exact.py", PACKET / "exact_source/standalone_code/grp_no_l2_exact.py"
    )
    sources = [
        REPO_ROOT / "experiments/domain_phase_mix/olmix_loglinear_fit.py",
        SCRIPT_DIR / "analyze_original_separate_heads_policy_ablation_300m.py",
        SCRIPT_DIR / "materialize_two_phase_canonical_bowl_candidates_300m.py",
        SCRIPT_DIR / "plot_lf_sepheads_kl_sweep_300m.py",
        SCRIPT_DIR / "benchmark_retained_weibull_replay_20260713.py",
        SCRIPT_DIR / "fit_production_grp_quality_variants.py",
        SCRIPT_DIR / "benchmark_hierarchical_coverage_grp_20260715.py",
        SCRIPT_DIR / "benchmark_grp_domain_saturation_phase_heads_20260714.py",
        SCRIPT_DIR / "benchmark_nested_coverage_dsp.py",
        SCRIPT_DIR / "benchmark_counterfactual_compact_transport_3e18_20260721.py",
        SCRIPT_DIR / "benchmark_production_grp_retained_hybrids_20260713.py",
        SCRIPT_DIR / "benchmark_grp_saturation_hierarchy_20260714.py",
        SCRIPT_DIR / "benchmark_partially_pooled_phase_bowls.py",
        SCRIPT_DIR / "surrogate_search/generic_family_followup.py",
        SCRIPT_DIR / "surrogate_search/generic_family_penalty_calibration.py",
        SCRIPT_DIR / "surrogate_search/generic_family_flexible_signal.py",
        SCRIPT_DIR / "surrogate_search/structured_epoch_family.py",
    ]
    records = []
    for source in sources:
        relative = source.relative_to(REPO_ROOT)
        destination = PACKET / "exact_source/repository_models" / relative
        copy_file(source, destination)
        records.append({"source": str(relative), "sha256": sha256(source)})
    pd.DataFrame(records).to_csv(PACKET / "exact_source/SOURCE_INDEX.csv", index=False)


def write_data_dictionary(catalog: dict[str, Any]) -> None:
    lines = [
        "# Data Dictionary",
        "",
        (
            "Canonical tables use one row per checkpoint observation and columns `phase_0_weight::<bucket>` and "
            "`phase_1_weight::<bucket>`. Weights are normalized within each phase. Exposure multipliers `c0` and `c1` "
            "are in `data/catalog.json`; multiplying a weight by its phase multiplier yields simulated epochs."
        ),
        "",
        "## Canonical datasets",
        "",
        "| ID | Rows | Buckets | Policy class | Complete targets |",
        "|---|---:|---:|---|---|",
    ]
    for dataset_id, spec in catalog["datasets"].items():
        lines.append(
            f"| `{dataset_id}` | {spec['row_count']} | {len(spec['domains'])} | {spec['policy_class']} | "
            f"{', '.join(spec['targets'])} |"
        )
    lines.extend(
        [
            "",
            "## Common columns",
            "",
            "- `row_id`: packet-unique observation identifier.",
            "- `policy_class`: `one_phase`, `two_phase`, or source-specific mixed label.",
            "- `split`: fit, heldout, or supplemental.",
            "- `training_series`: experimental series or proposal population.",
            "- `proposal_target`: objective used to generate a heldout candidate; blank for ordinary swarm rows.",
            "- `candidate_kind`: design stratum or candidate type.",
            "- `group_id`: grouping key for CV; paired or repeated observations should not be split across folds.",
            "- `uncheatable_bpb`, `table9_macro_bpb`, `starcoder_bpb`: lower-is-better targets.",
            "",
            "## 3e18 heldout archive",
            "",
            (
                "`delphi_3e18_heldouts` is the current append-only completed archive. It combines historical validation "
                "ladders, adversarial candidate panels, one-phase swarm checkpoints, random phase populations, HPR "
                "source comparisons, hybrid phase-ordering interventions, and repeat controls. It is not IID. Always "
                "stratify by `training_series`, `proposal_target`, `candidate_kind`, and `policy_class`. Exact aliases "
                "are identified by `fit_panel_overlap` and `group_id`."
            ),
            "",
            "## Raw data",
            "",
            (
                "Raw tables retain secondary evaluation metrics, Table-9 components, router/training summaries, "
                "append-only provenance, and source manifests. Use canonical tables for cross-panel model code and raw "
                "tables when auditing a metric or reconstructing a historical result."
            ),
            "",
            "## Schedule caveats",
            "",
            (
                "The StarCoder cosine and WSD surfaces differ in phase fractions and LR schedule. The legacy 60M "
                "registry and the production Grug-MoE swarm also differ in architecture, token budget, and bucket "
                "ontology. A shared functional form may transfer; parameter equality is not assumed."
            ),
        ]
    )
    (PACKET / "docs/DATA_DICTIONARY.md").write_text("\n".join(lines) + "\n")


def write_manifest(catalog: dict[str, Any]) -> None:
    excluded = {"MANIFEST.json", "CHECKSUMS.sha256"}
    files = []
    for path in sorted(item for item in PACKET.rglob("*") if item.is_file()):
        relative = str(path.relative_to(PACKET))
        if relative in excluded:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "packet": PACKET.name,
        "schema_version": 1,
        "build_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "dataset_count": len(catalog["datasets"]),
        "files": files,
        "privacy": (
            "Redacted scientific Fieldbook export; no raw ledger, job commands, credentials, or absolute home paths."
        ),
    }
    (PACKET / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = [f"{record['sha256']}  {record['path']}" for record in files]
    (PACKET / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")


def audit_privacy() -> None:
    failures: list[str] = []
    for path in PACKET.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json", ".txt", ".py"}:
            continue
        text = path.read_text(errors="ignore")
        if str(Path.home()) in text:
            failures.append(f"absolute home path in {path.relative_to(PACKET)}")
        if EMAIL.search(text):
            failures.append(f"email address in {path.relative_to(PACKET)}")
        if PRIVATE_GCS_PREFIX.search(text):
            failures.append(f"private GCS prefix in {path.relative_to(PACKET)}")
    if failures:
        raise RuntimeError("Privacy audit failed:\n" + "\n".join(failures[:50]))


def redact_copied_text_files() -> None:
    for path in PACKET.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json", ".txt", ".py"}:
            continue
        text = path.read_text(errors="ignore")
        redacted = sanitize_text(text)
        if redacted != text:
            path.write_text(redacted)


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in PACKET.rglob("*") if item.is_file()):
            archive.write(path, Path(PACKET.name) / path.relative_to(PACKET))


def main() -> None:
    args = parse_args()
    for relative in ("data/canonical", "data/raw", "evidence", "fieldbook", "exact_source"):
        path = PACKET / relative
        if path.exists():
            shutil.rmtree(path)
    catalog: dict[str, Any] = {"schema_version": 1, "datasets": OrderedDict()}
    domains = build_300m(catalog)
    build_delphi(catalog, domains)
    build_production(catalog)
    build_starcoder(catalog)
    build_legacy_60m(catalog, domains)
    (PACKET / "data/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    if args.theory is not None:
        copy_file(args.theory, PACKET / "docs/THEORY.md")
    export_fieldbook()
    copy_evidence()
    copy_exact_source()
    write_data_dictionary(catalog)
    redact_copied_text_files()
    audit_privacy()
    write_manifest(catalog)
    build_zip()
    print(f"Built {PACKET}")
    print(f"Archive {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
