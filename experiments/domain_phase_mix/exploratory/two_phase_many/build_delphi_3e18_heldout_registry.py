# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["wandb>=0.19"]
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
from datetime import UTC, datetime
from pathlib import Path

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
TRAIN_FILTER = {
    "$and": [
        {"display_name": {"$regex": "_3e18"}},
        {"tags": {"$in": ["FLOPs=3.0e+18"]}},
    ]
}
EVAL_FILTER = {"config.checkpoint_path": {"$regex": "_3e18-"}}
FIT_SWARM_TAG = "delphi-3e18-augmented-swarm"
RUN_SUFFIX_RE = re.compile(r"-[0-9a-f]{6,8}$")
FIT_OVERLAP_TOLERANCE = 1e-10

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-panel", type=Path, default=DEFAULT_FIT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wandb-timeout", type=int, default=180)
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


def summary_value(summary: Mapping[str, object], keys: Sequence[str]) -> float | str:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return ""


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


def load_fieldbook(ledger: Path) -> dict[str, dict[str, object]]:
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

    result: dict[str, dict[str, object]] = {}
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
    api: wandb.Api, training_ids: set[str], batch_size: int
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    metadata = api.runs(EVAL_PROJECT, filters=EVAL_FILTER, per_page=1000, lazy=True)
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


def best_table9_attempt(attempts: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    complete = [row for row in attempts if row["eval_state"] == "finished" and row["table9_macro_bpb"] != ""]
    if not complete:
        return None
    return max(complete, key=lambda row: (str(row["eval_created_at"]), str(row["eval_wandb_run_id"])))


def fieldbook_for_run(fieldbook: Mapping[str, Mapping[str, object]], run_id: str, base: str) -> Mapping[str, object]:
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
    batch_size: int,
    observed_at: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    identities: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
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
            "training_series": training_series(hf_save_path),
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

        attempts = list(eval_attempts.get(run.id, []))
        selected_eval = best_table9_attempt(attempts)
        direct_table9 = summary_value(summary, TABLE9_KEYS)
        table9_value = direct_table9
        table9_source = "training_summary" if direct_table9 != "" else ""
        if selected_eval is not None:
            table9_value = selected_eval["table9_macro_bpb"]
            table9_source = "native_eval_checkpoint_join"
        global_step = int(summary.get("global_step") or summary.get("_step") or 0)
        fieldbook_match = fieldbook_for_run(fieldbook, run.id, base)
        observation: dict[str, object] = {
            "heldout_id": identity["heldout_id"],
            "observed_at": observed_at,
            "training_state": run.state,
            "global_step": global_step,
            "num_train_steps": num_train_steps,
            "checkpoint_declared_complete": int(
                run.state == "finished" and num_train_steps > 0 and global_step >= num_train_steps - 1
            ),
            "parameter_count": scalar(summary.get("parameter_count")),
            "eval_bpb": summary_value(summary, ("eval/bpb",)),
            "eval_macro_bpb": summary_value(summary, ("eval/macro_bpb",)),
            "uncheatable_bpb": summary_value(summary, ("eval/uncheatable_eval/bpb",)),
            "uncheatable_macro_bpb": summary_value(summary, ("eval/uncheatable_eval/macro_bpb",)),
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
    identities.sort(key=lambda row: (str(row["created_at"]), str(row["wandb_run_id"])))
    observations.sort(key=lambda row: str(row["heldout_id"]))
    return identities, observations


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
        "`heldout_current.csv` is the reproducible latest-state join used for analysis.",
        "",
        "A run marked `exact_coordinate` reproduces a mixture coordinate from the 280-row 300M fit panel, "
        "but remains a scale/seed heldout observation. It is not coordinate-disjoint and must be reported separately "
        "when measuring mixture-level generalization.",
        "",
        "The registry includes crashed and incomplete attempts. "
        "Missing metrics are explicit rather than silently filtered.",
        "",
        "Selection requires both the exact `FLOPs=3.0e+18` tag and `_3e18` in the training run name. "
        "This excludes unrelated isoflop studies, inherited-tag TPP=10/20 runs with larger token budgets, "
        f"writeback-only runs, and runs tagged `{FIT_SWARM_TAG}` from the 280-row fit swarm.",
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

    training_metadata = api.runs(TRAIN_PROJECT, filters=TRAIN_FILTER, per_page=1000, lazy=True)
    training_ids = sorted(run.id for run in training_metadata if FIT_SWARM_TAG not in (run.tags or []))
    if not training_ids:
        raise ValueError("No Delphi 3e18 training runs matched the audited W&B filter")
    eval_rows, eval_by_training = collect_eval_attempts(api, set(training_ids), args.batch_size)
    identities, observations = collect_training_rows(
        api,
        training_ids,
        domains,
        fit_rows,
        eval_by_training,
        fieldbook,
        args.batch_size,
        observed_at,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.output_dir / "heldout_registry.csv"
    observations_path = args.output_dir / "heldout_observations.csv"
    new_identities, all_identities = append_identities(registry_path, identities)
    new_observations, all_observations = append_observations(observations_path, observations)
    latest = latest_observations(all_observations)
    current = [{**identity, **latest[identity["heldout_id"]]} for identity in all_identities]
    current.sort(key=lambda row: (row["created_at"], row["wandb_run_id"]))
    current_fields = (*IDENTITY_FIELDS, *[field for field in OBSERVATION_FIELDS if field != "heldout_id"])
    write_csv(args.output_dir / "heldout_current.csv", current_fields, current)
    write_csv(args.output_dir / "table9_eval_attempts.csv", EVAL_ATTEMPT_FIELDS, eval_rows)
    write_csv(
        args.output_dir / "missing_table9_eval_manifest.csv",
        MISSING_TABLE9_MANIFEST_FIELDS,
        missing_table9_manifest(current),
    )

    summary = audit_summary(current, eval_rows)
    summary.update(
        {
            "observed_at": observed_at,
            "new_identity_rows": new_identities,
            "new_observation_rows": new_observations,
            "fit_panel_path": str(args.fit_panel),
            "fit_panel_sha256": hashlib.sha256(args.fit_panel.read_bytes()).hexdigest(),
            "training_wandb_filter": TRAIN_FILTER,
            "eval_wandb_filter": EVAL_FILTER,
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(args.output_dir / "audit_report.md", summary, args.fit_panel, observed_at)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
