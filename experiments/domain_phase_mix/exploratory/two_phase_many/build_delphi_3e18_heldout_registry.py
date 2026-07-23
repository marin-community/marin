# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec>=2024.10", "gcsfs>=2024.10", "wandb>=0.19"]
# ///
"""Build an append-only registry of historical Delphi 3e18 validation runs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import fsspec
import wandb

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FIT_PANEL = (
    Path(__file__).parent
    / "reference_outputs/olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628/fit_panel_table9_macro.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "reference_outputs/delphi_3e18_append_only_heldouts_20260714"
DEFAULT_LEDGER = REPO_ROOT / ".experiments/ledger.sqlite"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
EAST5_PREFIX = "gs://marin-us-east5/"
LEGACY_TRAIN_FILTER: dict[str, object] = {
    "$and": [
        {"display_name": {"$regex": "_3e18"}},
        {"tags": {"$in": ["FLOPs=3.0e+18"]}},
    ]
}
EXPLICIT_HELDOUT_TAGS = (
    "single-phase-ablation",
    "delphi-3e18-adversarial-stress",
    "delphi-3e18-frontier-phase-fiber",
    "delphi-3e18-hpr-optimum-validation",
    "delphi-3e18-frontier-random-phase-population",
    "delphi-3e18-compact-optimum-path-validation",
    "delphi-3e18-compact-sub280-optimum-validation",
)
OPTIONAL_EXPLICIT_HELDOUT_TAGS = ("delphi-3e18-hybrid-phase-ordering-validation",)
EVAL_FILTER: dict[str, object] = {"config.checkpoint_path": {"$regex": "_3e18[-_]"}}
FIT_SWARM_TAG = "delphi-3e18-augmented-swarm"
RUN_SUFFIX_RE = re.compile(r"-[0-9a-f]{6,8}$")
FIT_OVERLAP_TOLERANCE = 1e-10
FINAL_EVAL_METRICS_RELATIVE_PATH = "checkpoints/eval_metrics.jsonl"

REFERENCE_OUTPUTS = Path(__file__).parent / "reference_outputs"
HPR_300M_PANEL = REFERENCE_OUTPUTS / "hpr_300m_to_3e18_optimum_validation_panel_20260720/launcher_source_panel.csv"
HPR_3E18_PANEL = REFERENCE_OUTPUTS / "hpr_3e18_to_3e18_optimum_validation_panel_20260720/launcher_source_panel.csv"
RANDOM_PHASE_PANEL = (
    REFERENCE_OUTPUTS / "delphi_3e18_frontier_random_phase_population_20260720/launcher_source_panel.csv"
)
HYBRID_PHASE_PANEL = REFERENCE_OUTPUTS / "delphi_3e18_hybrid_phase_ordering_panel_20260720/launcher_source_panel.csv"
COMPACT_OPTIMUM_PATH_PANEL = (
    REFERENCE_OUTPUTS / "delphi_compact_optimum_path_validation_panel_20260721/launcher_source_panel.csv"
)
COMPACT_SUB280_OPTIMUM_PANEL = (
    REFERENCE_OUTPUTS / "delphi_compact_sub280_optimum_validation_panel_20260721/launcher_source_panel.csv"
)

TABLE9_KEYS = (
    "olmo_base_easy/table9_51_component_macro_bpb",
    "olmo_base_easy/table9_macro_bpb",
    "olmo_base_eval/easy_bpb/_summary/primary_metric_mean",
)

IDENTITY_FIELDS = (
    "heldout_id",
    "wandb_entity",
    "wandb_project",
    "wandb_run_id",
    "wandb_run_name",
    "wandb_run_base",
    "wandb_url",
    "created_at",
    "training_series",
    "objective",
    "tags_json",
    "data_seed",
    "trainer_seed",
    "policy_class",
    "configured_phase_count",
    "phase_boundary_step",
    "phase_0_fraction",
    "phase_0_weights_json",
    "phase_1_weights_json",
    "mixture_sha256",
    "fit_panel_overlap",
    "fit_panel_run_name",
    "fit_panel_max_abs_distance",
    "hf_save_path",
    "expected_hf_checkpoint",
)

OBSERVATION_FIELDS = (
    "heldout_id",
    "observed_at",
    "observation_fingerprint",
    "training_state",
    "global_step",
    "num_train_steps",
    "checkpoint_declared_complete",
    "parameter_count",
    "eval_bpb",
    "eval_macro_bpb",
    "uncheatable_bpb",
    "uncheatable_macro_bpb",
    "train_loss",
    "direct_table9_macro_bpb",
    "table9_macro_bpb",
    "table9_metric_source",
    "table9_eval_attempt_count",
    "table9_eval_failed_count",
    "table9_eval_run_id",
    "table9_eval_run_name",
    "table9_eval_state",
    "table9_eval_url",
    "table9_eval_created_at",
    "table9_eval_checkpoint_path",
    "fieldbook_match_count",
    "fieldbook_experiment_ids_json",
    "fieldbook_experiment_names_json",
    "fieldbook_run_ids_json",
    "fieldbook_run_attrs_json",
    "fieldbook_job_ids_json",
    "fieldbook_iris_parents_json",
)

EVAL_ATTEMPT_FIELDS = (
    "eval_wandb_run_id",
    "eval_wandb_run_name",
    "eval_wandb_url",
    "eval_state",
    "eval_created_at",
    "source_training_run_id",
    "checkpoint_path",
    "table9_macro_bpb",
    "eval_group",
)

MISSING_TABLE9_MANIFEST_FIELDS = (
    "eval_name",
    "checkpoint",
    "panel",
    "scale",
    "run_name",
    "source_experiment",
    "checkpoint_root",
    "expected_checkpoint_step",
    "method",
    "fit_panel_overlap",
    "wandb_url",
)

PROVENANCE_FIELDS = (
    "heldout_id",
    "panel_tag",
    "candidate_id",
    "proposal_target",
    "fit_source",
    "candidate_kind",
    "aggregate_kl_coefficient",
    "phase_information_budget",
    "anchor_id",
    "direction_id",
    "radius_fraction",
    "seed_block",
    "policy_sha256",
    "data_seed",
    "trainer_seed",
    "wandb_training_state",
    "training_metric_source",
    "gcs_eval_metrics_path",
    "gcs_final_step",
    "panel_manifest_path",
    "panel_manifest_sha256",
    "proposal_metadata_json",
)


@dataclass(frozen=True)
class PanelManifestSpec:
    tag: str
    training_series: str
    path: Path
    run_name_prefix: str
    use_manifest_run_order: bool = False


class FieldbookMatch(TypedDict):
    match_count: int
    experiment_ids: list[str]
    experiment_names: list[str]
    run_ids: list[str]
    run_attrs: dict[str, object]
    job_ids: list[str]
    iris_parents: list[str]


PANEL_MANIFEST_SPECS = (
    PanelManifestSpec(
        tag="delphi-3e18-hpr-optimum-validation",
        training_series="hpr_300m_to_3e18_optimum_validation_panel_20260720",
        path=HPR_300M_PANEL,
        run_name_prefix="hprv",
    ),
    PanelManifestSpec(
        tag="delphi-3e18-hpr-optimum-validation",
        training_series="hpr_3e18_to_3e18_optimum_validation_panel_20260720",
        path=HPR_3E18_PANEL,
        run_name_prefix="hprv",
    ),
    PanelManifestSpec(
        tag="delphi-3e18-frontier-random-phase-population",
        training_series="delphi_3e18_frontier_random_phase_population_20260720",
        path=RANDOM_PHASE_PANEL,
        run_name_prefix="rphase",
        use_manifest_run_order=True,
    ),
    PanelManifestSpec(
        tag="delphi-3e18-compact-optimum-path-validation",
        training_series="delphi_compact_optimum_path_validation_panel_20260721",
        path=COMPACT_OPTIMUM_PATH_PANEL,
        run_name_prefix="crsv",
    ),
    PanelManifestSpec(
        tag="delphi-3e18-compact-sub280-optimum-validation",
        training_series="delphi_compact_sub280_optimum_validation_panel_20260721",
        path=COMPACT_SUB280_OPTIMUM_PANEL,
        run_name_prefix="crslowv",
    ),
)
HYBRID_PANEL_MANIFEST_SPEC = PanelManifestSpec(
    tag="delphi-3e18-hybrid-phase-ordering-validation",
    training_series="delphi_3e18_hybrid_phase_ordering_validation_20260720",
    path=HYBRID_PHASE_PANEL,
    run_name_prefix="hprv",
)


def panel_eval_groups(include_hybrid_panel: bool) -> tuple[str, ...]:
    groups = [
        "olmo_base_eval_table9_hpr_300m_to_3e18_optimum_validation_panel_20260720",
        "olmo_base_eval_table9_hpr_3e18_to_3e18_optimum_validation_panel_20260720",
        "olmo_base_eval_table9_delphi_3e18_frontier_random_phase_population_20260720",
        "olmo_base_eval_table9_delphi_compact_optimum_path_validation_20260721",
        "olmo_base_eval_table9_delphi_compact_sub280_optimum_validation_20260721",
    ]
    if include_hybrid_panel:
        groups.append("olmo_base_eval_table9_delphi_3e18_hybrid_phase_ordering_validation_20260720")
    return tuple(groups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-panel", type=Path, default=DEFAULT_FIT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    parser.add_argument(
        "--include-hybrid-panel",
        action="store_true",
        help="Include the hybrid phase-ordering tag after its full train and native Table-9 graph succeeds.",
    )
    parser.add_argument(
        "--manifest-panels-only",
        action="store_true",
        help="Incrementally refresh only manifest-backed panels and preserve existing historical rows.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def scalar(value: object) -> float | int | str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return value
    return str(value)


def json_compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(json_compact(value).encode()).hexdigest()


def run_base(run_name: str) -> str:
    return RUN_SUFFIX_RE.sub("", run_name)


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def training_filters(include_hybrid_panel: bool, manifest_panels_only: bool = False) -> tuple[dict[str, object], ...]:
    if manifest_panels_only:
        tags = [spec.tag for spec in panel_manifest_specs(include_hybrid_panel)]
        return (
            {
                "$and": [
                    {"tags": {"$in": sorted(set(tags))}},
                    {"tags": {"$in": ["FLOPs=3.0e+18"]}},
                ]
            },
        )
    tags = [*EXPLICIT_HELDOUT_TAGS]
    if include_hybrid_panel:
        tags.extend(OPTIONAL_EXPLICIT_HELDOUT_TAGS)
    explicit_filter: dict[str, object] = {
        "$and": [
            {"tags": {"$in": tags}},
            {"tags": {"$in": ["FLOPs=3.0e+18"]}},
        ]
    }
    return LEGACY_TRAIN_FILTER, explicit_filter


def panel_manifest_specs(include_hybrid_panel: bool) -> tuple[PanelManifestSpec, ...]:
    if include_hybrid_panel:
        return *PANEL_MANIFEST_SPECS, HYBRID_PANEL_MANIFEST_SPEC
    return PANEL_MANIFEST_SPECS


def eval_filter(include_hybrid_panel: bool, manifest_panels_only: bool) -> dict[str, object]:
    if manifest_panels_only:
        return {"group": {"$in": list(panel_eval_groups(include_hybrid_panel))}}
    return EVAL_FILTER


def panel_run_base(spec: PanelManifestSpec, row: Mapping[str, str], row_index: int) -> str:
    run_order = int(row["run_order"]) if spec.use_manifest_run_order else row_index
    return f"{spec.run_name_prefix}_{run_order:03d}_{row['candidate_id']}"


def load_panel_provenance(
    domains: Sequence[str], include_hybrid_panel: bool
) -> dict[tuple[str, str], dict[str, object]]:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for spec in panel_manifest_specs(include_hybrid_panel):
        if not spec.path.exists():
            raise FileNotFoundError(f"Missing panel manifest: {spec.path}")
        manifest_sha256 = hashlib.sha256(spec.path.read_bytes()).hexdigest()
        with spec.path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        for row_index, row in enumerate(rows):
            run_name = panel_run_base(spec, row, row_index)
            key = (spec.training_series, run_name)
            if key in lookup:
                raise ValueError(f"Duplicate panel provenance key: {key}")
            phase_0 = [float(row[f"phase_0_{domain}"]) for domain in domains]
            phase_1 = [float(row[f"phase_1_{domain}"]) for domain in domains]
            proposal_metadata = {
                name: value
                for name, value in row.items()
                if not name.startswith("phase_0_") and not name.startswith("phase_1_") and name != "mixture_path"
            }
            anchor_id = row.get("anchor_id", "")
            proposal_target = row.get("target", "")
            if not proposal_target and anchor_id:
                proposal_target = "table9" if anchor_id.startswith("table9") else "uncheatable"
            lookup[key] = {
                "panel_tag": spec.tag,
                "candidate_id": row["candidate_id"],
                "proposal_target": proposal_target,
                "fit_source": row.get("fit_source", ""),
                "candidate_kind": row.get("candidate_kind", row.get("contrast_family", "")),
                "aggregate_kl_coefficient": row.get("aggregate_kl_coefficient", ""),
                "phase_information_budget": row.get("phase_information_budget", ""),
                "anchor_id": anchor_id,
                "direction_id": row.get("direction_id", ""),
                "radius_fraction": row.get("radius_fraction", ""),
                "seed_block": row.get("seed_block", ""),
                "policy_sha256": row.get("policy_sha256", row.get("coordinate_hash", "")),
                "data_seed": row.get("data_seed", ""),
                "trainer_seed": row.get("trainer_seed", ""),
                "panel_manifest_path": str(spec.path),
                "panel_manifest_sha256": manifest_sha256,
                "proposal_metadata_json": json_compact(proposal_metadata),
                "expected_phase_0": phase_0,
                "expected_phase_1": phase_1,
            }
    return lookup


def summary_value(summary: Mapping[str, object], keys: Sequence[str]) -> float | str:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return ""


def final_gcs_eval_metrics(hf_save_path: str, num_train_steps: int) -> tuple[dict[str, object], str] | None:
    if not hf_save_path.startswith(EAST5_PREFIX) or "/hf" not in hf_save_path or num_train_steps < 1:
        return None
    checkpoint_root = hf_save_path.rstrip("/").removesuffix("/hf")
    metrics_uri = f"{checkpoint_root}/{FINAL_EVAL_METRICS_RELATIVE_PATH}"
    fs, path = fsspec.core.url_to_fs(metrics_uri)
    if not fs.exists(path):
        return None
    final_line = ""
    with fs.open(path, "r") as source:
        for line in source:
            if line.strip():
                final_line = line
    if not final_line:
        return None
    payload = json.loads(final_line)
    step = int(payload.get("step", -1))
    if step < num_train_steps - 1:
        return None
    return payload, metrics_uri


def objective_for_run(name: str, tags: Sequence[str]) -> str:
    corpus = " ".join([name, *tags]).lower()
    if "table9" in corpus or re.search(r"(^|[_-])t9([_-]|$)", corpus):
        return "table9"
    if "uncheatable" in corpus or re.search(r"(^|[_-])unch([_-]|$)", corpus):
        return "uncheatable"
    if any(token in corpus for token in ("proportional", "unimax", "propnoise", "noise-panel")):
        return "baseline"
    return "other"


def training_series(hf_save_path: str) -> str:
    parts = [part for part in hf_save_path.split("/") if part]
    if len(parts) < 3:
        return ""
    return parts[-3]


def parse_train_weights(
    config: Mapping[str, object], domains: Sequence[str]
) -> tuple[list[float], list[float], int, int | str]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Training config is missing data mapping")
    raw_weights = data.get("train_weights")
    schedules: list[tuple[int, Mapping[str, object]]] = []
    if isinstance(raw_weights, Mapping):
        schedules.append((0, raw_weights))
    elif isinstance(raw_weights, list):
        for entry in raw_weights:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], Mapping):
                raise ValueError(f"Unexpected train_weights entry: {entry!r}")
            schedules.append((int(entry[0]), entry[1]))
    else:
        raise ValueError(f"Unexpected train_weights type: {type(raw_weights).__name__}")
    if not schedules:
        raise ValueError("Training config has no train-weight schedules")
    phase_0 = [float(schedules[0][1].get(domain, 0.0)) for domain in domains]
    phase_1 = [float(schedules[-1][1].get(domain, 0.0)) for domain in domains]
    boundary: int | str = schedules[1][0] if len(schedules) > 1 else ""
    return phase_0, phase_1, len(schedules), boundary


def load_fit_panel(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        domains = [column.removeprefix("phase_0_") for column in reader.fieldnames if column.startswith("phase_0_")]
        rows: list[dict[str, object]] = []
        for raw in reader:
            rows.append(
                {
                    "run_name": raw["run_name"],
                    "weights": [
                        *[float(raw[f"phase_0_{domain}"]) for domain in domains],
                        *[float(raw[f"phase_1_{domain}"]) for domain in domains],
                    ],
                }
            )
    if len(domains) != 39 or len(rows) != 280:
        raise ValueError(f"Expected a 39-domain, 280-row fit panel, found {len(domains)} domains and {len(rows)} rows")
    return domains, rows


def fit_panel_match(
    phase_0: Sequence[float], phase_1: Sequence[float], fit_rows: Sequence[Mapping[str, object]]
) -> tuple[str, str, float]:
    candidate = [*phase_0, *phase_1]
    best_name = ""
    best_distance = math.inf
    for fit_row in fit_rows:
        weights = fit_row["weights"]
        assert isinstance(weights, list)
        distance = max(abs(left - float(right)) for left, right in zip(candidate, weights, strict=True))
        if distance < best_distance:
            best_name = str(fit_row["run_name"])
            best_distance = distance
    overlap = "exact_coordinate" if best_distance <= FIT_OVERLAP_TOLERANCE else "coordinate_disjoint"
    return overlap, best_name, best_distance


def parse_checkpoint_training_run_id(checkpoint_path: str) -> str:
    match = re.search(r"/([^/]+)/hf(?:/|$)", checkpoint_path)
    return match.group(1) if match else ""


def load_fieldbook(ledger: Path) -> dict[str, FieldbookMatch]:
    if not ledger.exists():
        return {}
    connection = sqlite3.connect(ledger)
    connection.row_factory = sqlite3.Row
    run_rows = connection.execute(
        """
        SELECT r.id, r.name, r.external_id, r.attrs_json,
               e.id AS experiment_id, e.name AS experiment_name
        FROM runs r
        LEFT JOIN experiments e ON e.id = r.experiment_id
        WHERE r.deleted_at IS NULL
        """
    ).fetchall()
    job_rows = connection.execute(
        """
        SELECT jr.run_id, j.id AS job_id, j.external_id
        FROM job_runs jr
        JOIN jobs j ON j.id = jr.job_id
        WHERE jr.deleted_at IS NULL AND j.deleted_at IS NULL
        """
    ).fetchall()
    connection.close()

    jobs_by_run: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in job_rows:
        jobs_by_run[str(row["run_id"])].append(row)
    by_name: dict[str, list[sqlite3.Row]] = defaultdict(list)
    by_external_id: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in run_rows:
        by_name[str(row["name"])].append(row)
        if row["external_id"]:
            by_external_id[str(row["external_id"])].append(row)

    result: dict[str, FieldbookMatch] = {}
    all_keys = set(by_name) | set(by_external_id)
    for key in all_keys:
        matches = {str(row["id"]): row for row in [*by_name.get(key, []), *by_external_id.get(key, [])]}
        linked_jobs = [job for run_id in matches for job in jobs_by_run.get(run_id, [])]
        result[key] = {
            "match_count": len(matches),
            "experiment_ids": sorted({str(row["experiment_id"]) for row in matches.values() if row["experiment_id"]}),
            "experiment_names": sorted(
                {str(row["experiment_name"]) for row in matches.values() if row["experiment_name"]}
            ),
            "run_ids": sorted(matches),
            "run_attrs": {run_id: json.loads(str(row["attrs_json"])) for run_id, row in matches.items()},
            "job_ids": sorted({str(job["job_id"]) for job in linked_jobs if job["job_id"]}),
            "iris_parents": sorted(
                {str(job["external_id"]) for job in linked_jobs if str(job["external_id"] or "").startswith("/")}
            ),
        }
    return result


def batch_runs(api: wandb.Api, project: str, run_ids: Sequence[str], batch_size: int) -> Iterable[wandb.apis.public.Run]:
    for batch in chunks(run_ids, batch_size):
        runs = list(api.runs(project, filters={"name": {"$in": list(batch)}}, per_page=len(batch)))
        if {run.id for run in runs} != set(batch):
            missing = sorted(set(batch) - {run.id for run in runs})
            raise ValueError(f"W&B batch query did not return runs: {missing}")
        yield from runs
        runs.clear()
        gc.collect()


def collect_eval_attempts(
    api: wandb.Api, training_ids: set[str], batch_size: int, filters: Mapping[str, object]
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    metadata = api.runs(EVAL_PROJECT, filters=dict(filters), per_page=1000, lazy=True)
    eval_ids = sorted(run.id for run in metadata)
    rows: list[dict[str, object]] = []
    by_training: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in batch_runs(api, EVAL_PROJECT, eval_ids, batch_size):
        config = dict(run.config)
        summary = dict(run.summary)
        checkpoint_path = str(config.get("checkpoint_path") or "")
        source_id = parse_checkpoint_training_run_id(checkpoint_path)
        if source_id not in training_ids:
            continue
        row: dict[str, object] = {
            "eval_wandb_run_id": run.id,
            "eval_wandb_run_name": run.name,
            "eval_wandb_url": run.url,
            "eval_state": run.state,
            "eval_created_at": str(run.created_at),
            "source_training_run_id": source_id,
            "checkpoint_path": checkpoint_path,
            "table9_macro_bpb": summary_value(summary, TABLE9_KEYS),
            "eval_group": run.group or "",
        }
        rows.append(row)
        by_training[source_id].append(row)
    rows.sort(
        key=lambda row: (str(row["source_training_run_id"]), str(row["eval_created_at"]), str(row["eval_wandb_run_id"]))
    )
    return rows, by_training


def discover_training_ids(api: wandb.Api, include_hybrid_panel: bool, manifest_panels_only: bool) -> list[str]:
    run_ids: set[str] = set()
    explicit_tags = set(EXPLICIT_HELDOUT_TAGS)
    if include_hybrid_panel:
        explicit_tags.update(OPTIONAL_EXPLICIT_HELDOUT_TAGS)
    for filters in training_filters(include_hybrid_panel, manifest_panels_only):
        metadata = api.runs(TRAIN_PROJECT, filters=filters, per_page=1000, lazy=True)
        for run in metadata:
            tags = set(run.tags or [])
            explicitly_heldout = bool(tags.intersection(explicit_tags))
            if FIT_SWARM_TAG in tags and not explicitly_heldout:
                continue
            run_ids.add(run.id)
    return sorted(run_ids)


def best_table9_attempt(attempts: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    complete = [row for row in attempts if row["eval_state"] == "finished" and row["table9_macro_bpb"] != ""]
    if not complete:
        return None
    return max(complete, key=lambda row: (str(row["eval_created_at"]), str(row["eval_wandb_run_id"])))


def fieldbook_for_run(fieldbook: Mapping[str, FieldbookMatch], run_id: str, base: str) -> FieldbookMatch:
    matches = [fieldbook[key] for key in (run_id, base) if key in fieldbook]
    if not matches:
        return {
            "match_count": 0,
            "experiment_ids": [],
            "experiment_names": [],
            "run_ids": [],
            "run_attrs": {},
            "job_ids": [],
            "iris_parents": [],
        }
    matched_run_ids = {value for match in matches for value in match["run_ids"]}
    return {
        "match_count": len(matched_run_ids),
        "experiment_ids": sorted({value for match in matches for value in match["experiment_ids"]}),
        "experiment_names": sorted({value for match in matches for value in match["experiment_names"]}),
        "run_ids": sorted(matched_run_ids),
        "run_attrs": {key: value for match in matches for key, value in match["run_attrs"].items()},
        "job_ids": sorted({value for match in matches for value in match["job_ids"]}),
        "iris_parents": sorted({value for match in matches for value in match["iris_parents"]}),
    }


def collect_training_rows(
    api: wandb.Api,
    run_ids: Sequence[str],
    domains: Sequence[str],
    fit_rows: Sequence[Mapping[str, object]],
    eval_attempts: Mapping[str, Sequence[Mapping[str, object]]],
    fieldbook: Mapping[str, Mapping[str, object]],
    panel_provenance: Mapping[tuple[str, str], Mapping[str, object]],
    manifest_backed_tags: set[str],
    batch_size: int,
    observed_at: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    identities: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    for run in batch_runs(api, TRAIN_PROJECT, run_ids, batch_size):
        config = dict(run.config)
        summary = dict(run.summary)
        tags = sorted(str(tag) for tag in run.tags)
        base = run_base(run.name)
        phase_0, phase_1, phase_count, boundary = parse_train_weights(config, domains)
        overlap, matched_name, distance = fit_panel_match(phase_0, phase_1, fit_rows)
        trainer = config.get("trainer") if isinstance(config.get("trainer"), Mapping) else {}
        num_train_steps = int(trainer.get("num_train_steps") or 0)
        trainer_seed = config.get("trainer_seed", trainer.get("seed", ""))
        phase_0_fraction: float | str = ""
        if isinstance(boundary, int) and num_train_steps:
            phase_0_fraction = boundary / num_train_steps
        hf_save_path = str(config.get("hf_save_path") or "")
        series = training_series(hf_save_path)
        expected_checkpoint = (
            f"{hf_save_path.rstrip('/')}/step-{num_train_steps - 1}" if hf_save_path and num_train_steps else ""
        )
        policy_class = (
            "single_phase_tied"
            if max(abs(a - b) for a, b in zip(phase_0, phase_1, strict=True)) <= 1e-12
            else "two_phase"
        )
        mixture_payload = {"domains": list(domains), "phase_0": phase_0, "phase_1": phase_1}
        identity: dict[str, object] = {
            "heldout_id": f"wandb:{TRAIN_PROJECT}/{run.id}",
            "wandb_entity": TRAIN_PROJECT.split("/")[0],
            "wandb_project": TRAIN_PROJECT.split("/")[1],
            "wandb_run_id": run.id,
            "wandb_run_name": run.name,
            "wandb_run_base": base,
            "wandb_url": run.url,
            "created_at": str(run.created_at),
            "training_series": series,
            "objective": objective_for_run(base, tags),
            "tags_json": json_compact(tags),
            "data_seed": scalar(config.get("data_seed")),
            "trainer_seed": scalar(trainer_seed),
            "policy_class": policy_class,
            "configured_phase_count": phase_count,
            "phase_boundary_step": boundary,
            "phase_0_fraction": phase_0_fraction,
            "phase_0_weights_json": json_compact(dict(zip(domains, phase_0, strict=True))),
            "phase_1_weights_json": json_compact(dict(zip(domains, phase_1, strict=True))),
            "mixture_sha256": sha256_json(mixture_payload),
            "fit_panel_overlap": overlap,
            "fit_panel_run_name": matched_name,
            "fit_panel_max_abs_distance": distance,
            "hf_save_path": hf_save_path,
            "expected_hf_checkpoint": expected_checkpoint,
        }

        provenance = panel_provenance.get((series, base))
        run_manifest_tags = manifest_backed_tags.intersection(tags)
        if run_manifest_tags and provenance is None:
            raise ValueError(
                f"No frozen panel-manifest row matched {series}/{base} with tags {sorted(run_manifest_tags)}"
            )
        if provenance is not None:
            expected_phase_0 = provenance["expected_phase_0"]
            expected_phase_1 = provenance["expected_phase_1"]
            assert isinstance(expected_phase_0, list) and isinstance(expected_phase_1, list)
            if max(abs(left - float(right)) for left, right in zip(phase_0, expected_phase_0, strict=True)) > 1e-12:
                raise ValueError(f"Phase-0 weights differ from the frozen panel manifest for {base}")
            if max(abs(left - float(right)) for left, right in zip(phase_1, expected_phase_1, strict=True)) > 1e-12:
                raise ValueError(f"Phase-1 weights differ from the frozen panel manifest for {base}")

        attempts = list(eval_attempts.get(run.id, []))
        selected_eval = best_table9_attempt(attempts)
        direct_table9 = summary_value(summary, TABLE9_KEYS)
        table9_value = direct_table9
        table9_source = "training_summary" if direct_table9 != "" else ""
        if selected_eval is not None:
            table9_value = selected_eval["table9_macro_bpb"]
            table9_source = "native_eval_checkpoint_join"
        needs_gcs_recovery = run.state != "finished" or summary_value(summary, ("eval/uncheatable_eval/bpb",)) == ""
        recovered = final_gcs_eval_metrics(hf_save_path, num_train_steps) if needs_gcs_recovery else None
        recovered_metrics = recovered[0] if recovered is not None else {}
        recovered_metrics_path = recovered[1] if recovered is not None else ""
        logical_training_state = "finished" if recovered is not None else run.state
        metric_summary = recovered_metrics if recovered is not None else summary
        global_step = int(metric_summary.get("step") or summary.get("global_step") or summary.get("_step") or 0)
        fieldbook_match = fieldbook_for_run(fieldbook, run.id, base)
        observation: dict[str, object] = {
            "heldout_id": identity["heldout_id"],
            "observed_at": observed_at,
            "training_state": logical_training_state,
            "global_step": global_step,
            "num_train_steps": num_train_steps,
            "checkpoint_declared_complete": int(
                logical_training_state == "finished" and num_train_steps > 0 and global_step >= num_train_steps - 1
            ),
            "parameter_count": scalar(summary.get("parameter_count")),
            "eval_bpb": summary_value(metric_summary, ("eval/bpb",)),
            "eval_macro_bpb": summary_value(metric_summary, ("eval/macro_bpb",)),
            "uncheatable_bpb": summary_value(metric_summary, ("eval/uncheatable_eval/bpb",)),
            "uncheatable_macro_bpb": summary_value(metric_summary, ("eval/uncheatable_eval/macro_bpb",)),
            "train_loss": summary_value(summary, ("train/loss", "loss")),
            "direct_table9_macro_bpb": direct_table9,
            "table9_macro_bpb": table9_value,
            "table9_metric_source": table9_source,
            "table9_eval_attempt_count": len(attempts),
            "table9_eval_failed_count": sum(row["eval_state"] in {"crashed", "failed"} for row in attempts),
            "table9_eval_run_id": selected_eval["eval_wandb_run_id"] if selected_eval else "",
            "table9_eval_run_name": selected_eval["eval_wandb_run_name"] if selected_eval else "",
            "table9_eval_state": selected_eval["eval_state"] if selected_eval else "",
            "table9_eval_url": selected_eval["eval_wandb_url"] if selected_eval else "",
            "table9_eval_created_at": selected_eval["eval_created_at"] if selected_eval else "",
            "table9_eval_checkpoint_path": selected_eval["checkpoint_path"] if selected_eval else "",
            "fieldbook_match_count": fieldbook_match["match_count"],
            "fieldbook_experiment_ids_json": json_compact(fieldbook_match["experiment_ids"]),
            "fieldbook_experiment_names_json": json_compact(fieldbook_match["experiment_names"]),
            "fieldbook_run_ids_json": json_compact(fieldbook_match["run_ids"]),
            "fieldbook_run_attrs_json": json_compact(fieldbook_match["run_attrs"]),
            "fieldbook_job_ids_json": json_compact(fieldbook_match["job_ids"]),
            "fieldbook_iris_parents_json": json_compact(fieldbook_match["iris_parents"]),
        }
        fingerprint_payload = {
            key: observation[key] for key in OBSERVATION_FIELDS if key not in {"observed_at", "observation_fingerprint"}
        }
        observation["observation_fingerprint"] = sha256_json(fingerprint_payload)
        identities.append(identity)
        observations.append(observation)
        if provenance is not None:
            provenance_rows.append(
                {
                    **{field: provenance.get(field, "") for field in PROVENANCE_FIELDS},
                    "heldout_id": identity["heldout_id"],
                    "data_seed": identity["data_seed"],
                    "trainer_seed": identity["trainer_seed"],
                    "wandb_training_state": run.state,
                    "training_metric_source": (
                        "gcs_final_eval_metrics" if recovered is not None else "wandb_training_summary"
                    ),
                    "gcs_eval_metrics_path": recovered_metrics_path,
                    "gcs_final_step": recovered_metrics.get("step", ""),
                }
            )
    identities.sort(key=lambda row: (str(row["created_at"]), str(row["wandb_run_id"])))
    observations.sort(key=lambda row: str(row["heldout_id"]))
    provenance_rows.sort(key=lambda row: str(row["heldout_id"]))
    return identities, observations, provenance_rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def append_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def merge_eval_attempts(path: Path, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {row["eval_wandb_run_id"]: dict(row) for row in read_csv(path)}
    for row in rows:
        by_id[str(row["eval_wandb_run_id"])] = dict(row)
    merged = sorted(
        by_id.values(),
        key=lambda row: (
            str(row["source_training_run_id"]),
            str(row["eval_created_at"]),
            str(row["eval_wandb_run_id"]),
        ),
    )
    write_csv(path, EVAL_ATTEMPT_FIELDS, merged)
    return merged


def append_identities(path: Path, rows: Sequence[Mapping[str, object]]) -> tuple[int, list[dict[str, str]]]:
    existing = read_csv(path)
    by_id = {row["heldout_id"]: row for row in existing}
    new_rows: list[Mapping[str, object]] = []
    for row in rows:
        heldout_id = str(row["heldout_id"])
        old = by_id.get(heldout_id)
        if old is None:
            new_rows.append(row)
            continue
        changed = [field for field in IDENTITY_FIELDS if str(old[field]) != str(row[field])]
        if changed:
            raise ValueError(f"Immutable heldout identity changed for {heldout_id}: {changed}")
    append_csv(path, IDENTITY_FIELDS, new_rows)
    return len(new_rows), [
        *existing,
        *[{field: str(row.get(field, "")) for field in IDENTITY_FIELDS} for row in new_rows],
    ]


def append_observations(path: Path, rows: Sequence[Mapping[str, object]]) -> tuple[int, list[dict[str, str]]]:
    existing = read_csv(path)
    fingerprints = {(row["heldout_id"], row["observation_fingerprint"]) for row in existing}
    new_rows = [row for row in rows if (str(row["heldout_id"]), str(row["observation_fingerprint"])) not in fingerprints]
    append_csv(path, OBSERVATION_FIELDS, new_rows)
    return len(new_rows), [
        *existing,
        *[{field: str(row.get(field, "")) for field in OBSERVATION_FIELDS} for row in new_rows],
    ]


def append_provenance(path: Path, rows: Sequence[Mapping[str, object]]) -> tuple[int, list[dict[str, str]]]:
    existing = read_csv(path)
    by_id = {row["heldout_id"]: row for row in existing}
    new_rows: list[Mapping[str, object]] = []
    for row in rows:
        heldout_id = str(row["heldout_id"])
        old = by_id.get(heldout_id)
        if old is None:
            new_rows.append(row)
            continue
        changed = [field for field in PROVENANCE_FIELDS if str(old[field]) != str(row.get(field, ""))]
        if changed:
            raise ValueError(f"Immutable heldout provenance changed for {heldout_id}: {changed}")
    append_csv(path, PROVENANCE_FIELDS, new_rows)
    return len(new_rows), [
        *existing,
        *[{field: str(row.get(field, "")) for field in PROVENANCE_FIELDS} for row in new_rows],
    ]


def latest_observations(rows: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    latest: dict[str, Mapping[str, str]] = {}
    for row in rows:
        previous = latest.get(row["heldout_id"])
        if previous is None or row["observed_at"] >= previous["observed_at"]:
            latest[row["heldout_id"]] = row
    return latest


def audit_summary(
    current: Sequence[Mapping[str, object]], eval_attempts: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    state_counts = Counter(str(row["training_state"]) for row in current)
    objective_counts = Counter(str(row["objective"]) for row in current)
    policy_counts = Counter(str(row["policy_class"]) for row in current)
    overlap_counts = Counter(str(row["fit_panel_overlap"]) for row in current)
    series_counts = Counter(str(row["training_series"]) for row in current)
    complete = [row for row in current if str(row["checkpoint_declared_complete"]) == "1"]
    disjoint_complete = [row for row in complete if row["fit_panel_overlap"] == "coordinate_disjoint"]
    unique_mixtures = {str(row["mixture_sha256"]) for row in current}
    complete_unique_mixtures = {str(row["mixture_sha256"]) for row in complete}
    disjoint_complete_unique_mixtures = {str(row["mixture_sha256"]) for row in disjoint_complete}
    return {
        "training_attempt_count": len(current),
        "training_state_counts": dict(sorted(state_counts.items())),
        "checkpoint_complete_count": sum(str(row["checkpoint_declared_complete"]) == "1" for row in current),
        "uncheatable_metric_count": sum(row["uncheatable_bpb"] != "" for row in current),
        "table9_metric_count": sum(row["table9_macro_bpb"] != "" for row in current),
        "usable_complete_count": len(complete),
        "usable_complete_with_table9_count": sum(row["table9_macro_bpb"] != "" for row in complete),
        "usable_disjoint_complete_count": len(disjoint_complete),
        "usable_disjoint_complete_with_table9_count": sum(row["table9_macro_bpb"] != "" for row in disjoint_complete),
        "unique_policy_coordinate_count": len(unique_mixtures),
        "usable_complete_unique_policy_coordinate_count": len(complete_unique_mixtures),
        "usable_disjoint_complete_unique_policy_coordinate_count": len(disjoint_complete_unique_mixtures),
        "repeat_observation_count": len(current) - len(unique_mixtures),
        "missing_table9_eval_ready_count": sum(row["table9_macro_bpb"] == "" for row in complete),
        "fieldbook_match_count": sum(int(str(row["fieldbook_match_count"] or 0)) > 0 for row in current),
        "objective_counts": dict(sorted(objective_counts.items())),
        "policy_class_counts": dict(sorted(policy_counts.items())),
        "fit_panel_overlap_counts": dict(sorted(overlap_counts.items())),
        "training_series_counts": dict(sorted(series_counts.items())),
        "native_table9_eval_attempt_count": len(eval_attempts),
        "incomplete_training_runs": [row["wandb_run_name"] for row in current if row["training_state"] != "finished"],
        "missing_uncheatable_runs": [row["wandb_run_name"] for row in current if row["uncheatable_bpb"] == ""],
        "missing_table9_runs": [row["wandb_run_name"] for row in current if row["table9_macro_bpb"] == ""],
    }


def missing_table9_manifest(current: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in current:
        if str(row["checkpoint_declared_complete"]) != "1" or row["table9_macro_bpb"] != "":
            continue
        checkpoint = str(row["expected_hf_checkpoint"])
        if not checkpoint.startswith(EAST5_PREFIX):
            raise ValueError(f"Expected east5 checkpoint, got {checkpoint}")
        checkpoint_root, separator, _ = checkpoint.partition("/hf/")
        if not separator:
            raise ValueError(f"Expected checkpoint path containing /hf/: {checkpoint}")
        eval_name = re.sub(r"[^A-Za-z0-9_]+", "_", f"t9_gap_{row['wandb_run_name']}").strip("_")
        rows.append(
            {
                "eval_name": eval_name,
                "checkpoint": checkpoint.removeprefix(EAST5_PREFIX),
                "panel": "delphi_3e18_append_only_heldouts",
                "scale": "3e18",
                "run_name": row["wandb_run_name"],
                "source_experiment": row["training_series"],
                "checkpoint_root": checkpoint_root,
                "expected_checkpoint_step": row["global_step"],
                "method": row["objective"],
                "fit_panel_overlap": row["fit_panel_overlap"],
                "wandb_url": row["wandb_url"],
            }
        )
    return rows


def write_report(path: Path, summary: Mapping[str, object], fit_panel: Path, observed_at: str) -> None:
    states = summary["training_state_counts"]
    objectives = summary["objective_counts"]
    policies = summary["policy_class_counts"]
    overlaps = summary["fit_panel_overlap_counts"]
    lines = [
        "# Delphi 3e18 append-only heldout audit",
        "",
        f"Snapshot time: `{observed_at}`",
        f"Fit-panel reference: `{fit_panel}`",
        "",
        "## Coverage",
        "",
        f"- Training attempts: **{summary['training_attempt_count']}** ({states})",
        f"- Declared-complete checkpoints: **{summary['checkpoint_complete_count']}**",
        f"- Uncheatable BPB coverage: **{summary['uncheatable_metric_count']}**",
        f"- native/direct Table-9 macro coverage: **{summary['table9_metric_count']}**",
        f"- Strict coordinate-disjoint usable heldouts with Table-9: "
        f"**{summary['usable_disjoint_complete_with_table9_count']} / "
        f"{summary['usable_disjoint_complete_count']}**",
        f"- Unique policy coordinates: **{summary['unique_policy_coordinate_count']}** "
        f"({summary['repeat_observation_count']} repeated observations)",
        f"- Completed checkpoints ready for missing Table-9 eval: **{summary['missing_table9_eval_ready_count']}**",
        f"- Native Table-9 eval attempts retained: **{summary['native_table9_eval_attempt_count']}**",
        f"- Fieldbook-linked training attempts: **{summary['fieldbook_match_count']}**",
        f"- Objectives: `{objectives}`",
        f"- Policy classes: `{policies}`",
        f"- Fit-panel coordinate overlap: `{overlaps}`",
        "",
        "## Semantics",
        "",
        "`heldout_registry.csv` is immutable and append-only by W&B run identity. "
        "`heldout_observations.csv` appends a row only when mutable job or metric state changes. "
        "`heldout_provenance.csv` is immutable panel-manifest provenance for designed panels. "
        "`heldout_current.csv` is the reproducible latest-state join used for analysis.",
        "",
        "A run marked `exact_coordinate` reproduces a mixture coordinate from the 280-row 300M fit panel, "
        "but remains a scale/seed heldout observation. It is not coordinate-disjoint and must be reported separately "
        "when measuring mixture-level generalization.",
        "",
        "The registry includes crashed and incomplete attempts. "
        "Missing metrics are explicit rather than silently filtered.",
        "",
        "Legacy validation discovery requires both the exact `FLOPs=3.0e+18` tag and `_3e18` in the training "
        "run name. Explicit heldout panels are additionally admitted by audited panel tags. This excludes "
        "unrelated isoflop studies, inherited-tag TPP=10/20 runs with larger token budgets, writeback-only runs, "
        f"and runs tagged only `{FIT_SWARM_TAG}` from the 280-row fit swarm.",
        "",
        "## Gaps",
        "",
        f"- Incomplete training runs: `{summary['incomplete_training_runs']}`",
        f"- Missing Uncheatable BPB: **{len(summary['missing_uncheatable_runs'])}**",
        f"- Missing Table-9 macro BPB: **{len(summary['missing_table9_runs'])}**",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    domains, fit_rows = load_fit_panel(args.fit_panel)
    fieldbook = load_fieldbook(args.ledger)
    observed_at = utc_now()
    api = wandb.Api(timeout=args.wandb_timeout)

    panel_provenance = load_panel_provenance(domains, args.include_hybrid_panel)
    manifest_specs = panel_manifest_specs(args.include_hybrid_panel)
    manifest_backed_tags = {spec.tag for spec in manifest_specs}
    training_ids = discover_training_ids(api, args.include_hybrid_panel, args.manifest_panels_only)
    if not training_ids:
        raise ValueError("No Delphi 3e18 training runs matched the audited W&B filter")
    applied_eval_filter = eval_filter(args.include_hybrid_panel, args.manifest_panels_only)
    eval_rows, eval_by_training = collect_eval_attempts(api, set(training_ids), args.batch_size, applied_eval_filter)
    identities, observations, provenance_rows = collect_training_rows(
        api,
        training_ids,
        domains,
        fit_rows,
        eval_by_training,
        fieldbook,
        panel_provenance,
        manifest_backed_tags,
        args.batch_size,
        observed_at,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.output_dir / "heldout_registry.csv"
    observations_path = args.output_dir / "heldout_observations.csv"
    provenance_path = args.output_dir / "heldout_provenance.csv"
    new_identities, all_identities = append_identities(registry_path, identities)
    new_observations, all_observations = append_observations(observations_path, observations)
    new_provenance, all_provenance = append_provenance(provenance_path, provenance_rows)
    latest = latest_observations(all_observations)
    provenance_by_id = {row["heldout_id"]: row for row in all_provenance}
    empty_provenance = {field: "" for field in PROVENANCE_FIELDS if field != "heldout_id"}
    current = [
        {
            **identity,
            **latest[identity["heldout_id"]],
            **empty_provenance,
            **{
                field: value
                for field, value in provenance_by_id.get(identity["heldout_id"], {}).items()
                if field != "heldout_id"
            },
        }
        for identity in all_identities
    ]
    current.sort(key=lambda row: (row["created_at"], row["wandb_run_id"]))
    current_fields = (
        *IDENTITY_FIELDS,
        *[field for field in OBSERVATION_FIELDS if field != "heldout_id"],
        *[field for field in PROVENANCE_FIELDS if field != "heldout_id"],
    )
    write_csv(args.output_dir / "heldout_current.csv", current_fields, current)
    all_eval_rows = merge_eval_attempts(args.output_dir / "table9_eval_attempts.csv", eval_rows)
    write_csv(
        args.output_dir / "missing_table9_eval_manifest.csv",
        MISSING_TABLE9_MANIFEST_FIELDS,
        missing_table9_manifest(current),
    )

    summary = audit_summary(current, all_eval_rows)
    summary.update(
        {
            "observed_at": observed_at,
            "new_identity_rows": new_identities,
            "new_observation_rows": new_observations,
            "new_provenance_rows": new_provenance,
            "fit_panel_path": str(args.fit_panel),
            "fit_panel_sha256": hashlib.sha256(args.fit_panel.read_bytes()).hexdigest(),
            "training_wandb_filters": training_filters(args.include_hybrid_panel, args.manifest_panels_only),
            "eval_wandb_filter": applied_eval_filter,
            "included_optional_hybrid_panel": args.include_hybrid_panel,
            "manifest_panels_only": args.manifest_panels_only,
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(args.output_dir / "audit_report.md", summary, args.fit_panel, observed_at)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
