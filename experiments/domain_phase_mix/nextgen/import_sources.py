# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import sources for legacy domain/phase mixture trajectories."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import fsspec
import pandas as pd

from experiments.domain_phase_mix.analysis import match_runs_to_configs, query_wandb_runs
from experiments.domain_phase_mix.nextgen.contracts import ImportSource, RunRecord
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

logger = logging.getLogger(__name__)

_SWARM_RUN_RE = re.compile(r"/run_(\d+)$")
_BASE_RUN_RE = re.compile(r"/base_(\d+)$")
_PHASE_COL_RE = re.compile(r"^phase_(\d+)_(.+)$")

TWO_PHASE_STARCODER_EXPERIMENT = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_4"
THREE_PHASE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/3_partitions_3_phases_6"
THREE_PHASE_STARCODER_EXPERIMENT = "pinlin_calvin_xu/data_mixture/three_phase_starcoder_1"


@dataclass(frozen=True)
class LegacyDomainPhaseImportSource:
    """Import source for existing domain_phase_mix experiments.

    Supports legacy experiments that saved `weight_configs.json` and logged to
    W&B with tags based on experiment name.
    """

    source_experiment: str
    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    wandb_tags: tuple[str, ...] = ()
    weight_configs_path: str | None = None
    metric_prefixes: tuple[str, ...] = ("eval/", "lm_eval/")

    def _load_weight_configs(self) -> list[dict]:
        if self.weight_configs_path is None:
            return []

        path = self.weight_configs_path
        if "*" in path:
            fs, base = fsspec.core.url_to_fs(path)
            matches = fs.glob(base)
            if not matches:
                raise FileNotFoundError(f"No matching weight_configs files found for pattern: {path}")
            if isinstance(fs.protocol, str) and fs.protocol:
                path = f"{fs.protocol}://{matches[0]}"
            elif isinstance(fs.protocol, (tuple, list)) and fs.protocol and fs.protocol[0]:
                path = f"{fs.protocol[0]}://{matches[0]}"
            else:
                path = matches[0]
            logger.info("Resolved weight_configs pattern %s -> %s", self.weight_configs_path, path)

        with fsspec.open(path) as f:
            data = json.load(f)

        configs = data.get("configs", [])
        if not isinstance(configs, list):
            raise ValueError(f"Malformed weight configs at {self.weight_configs_path}")
        return configs

    def _query_runs(self) -> list[dict]:
        tags = list(self.wandb_tags) if self.wandb_tags else [self.source_experiment]
        return query_wandb_runs(
            entity=self.wandb_entity,
            project=self.wandb_project,
            tags=tags,
            metrics=[],
            metric_prefixes=self.metric_prefixes,
        )

    def collect_runs(self) -> list[RunRecord]:
        runs = self._query_runs()
        configs = self._load_weight_configs()

        if configs:
            matched = match_runs_to_configs(runs=runs, configs=configs, experiment_name=self.source_experiment)
            records: list[RunRecord] = []
            for row in matched:
                run_record = RunRecord(
                    wandb_run_id=row.get("wandb_run_id"),
                    source_experiment=self.source_experiment,
                    local_run_id=int(row["run_id"]) if row.get("run_id") is not None else None,
                    run_name=row.get("wandb_run_name"),
                    phase_weights=normalize_phase_weights({k: v for k, v in row.items() if isinstance(v, dict)}),
                    status=row.get("status", "unknown"),
                    metrics={
                        key: float(value)
                        for key, value in row.items()
                        if isinstance(value, int | float)
                        and any(key.startswith(prefix) for prefix in self.metric_prefixes)
                    },
                )
                records.append(run_record)
            return records

        # Fallback if weight configs are unavailable: infer local run ids from run names.
        records = []
        for run in runs:
            run_name = run.get("wandb_run_name")
            local_run_id = _infer_local_run_id_from_name(run_name)
            records.append(
                RunRecord(
                    wandb_run_id=run.get("wandb_run_id"),
                    source_experiment=self.source_experiment,
                    local_run_id=local_run_id,
                    run_name=run_name,
                    phase_weights={},
                    status=run.get("status", "unknown"),
                    metrics={
                        key: float(value)
                        for key, value in run.items()
                        if isinstance(value, int | float)
                        and any(key.startswith(prefix) for prefix in self.metric_prefixes)
                    },
                )
            )
        return records

    def collect_trajectories(self, objective_metric: str) -> pd.DataFrame:
        import wandb

        tags = list(self.wandb_tags) if self.wandb_tags else [self.source_experiment]
        api = wandb.Api()
        wb_runs = api.runs(
            f"{self.wandb_entity}/{self.wandb_project}",
            filters={"tags": {"$in": tags}},
        )

        rows: list[dict] = []
        for wb_run in wb_runs:
            run_id = wb_run.id
            run_name = wb_run.name
            local_run_id = _infer_local_run_id_from_name(run_name)

            keys = ["_step", objective_metric, "throughput/total_tokens"]
            try:
                history_iter = wb_run.scan_history(keys=keys)
            except Exception:
                logger.exception("Failed to scan history for run %s", run_id)
                continue

            for entry in history_iter:
                value = entry.get(objective_metric)
                step = entry.get("_step")
                if value is None or step is None:
                    continue
                rows.append(
                    {
                        "wandb_run_id": run_id,
                        "source_experiment": self.source_experiment,
                        "local_run_id": local_run_id,
                        "run_name": run_name,
                        "step": int(step),
                        "total_tokens": (
                            float(entry["throughput/total_tokens"])
                            if entry.get("throughput/total_tokens") is not None
                            else None
                        ),
                        "metric_key": objective_metric,
                        "metric_value": float(value),
                    }
                )

        return pd.DataFrame(rows)


@dataclass(frozen=True)
class CsvDomainPhaseImportSource:
    """Import source that loads run rows from a local or remote CSV."""

    source_experiment: str
    csv_path: str
    status_filter: str | None = "completed"
    metric_prefixes: tuple[str, ...] = ("eval/", "lm_eval/")

    def _load_csv(self) -> pd.DataFrame:
        with fsspec.open(self.csv_path, "r") as f:
            return pd.read_csv(f)

    def collect_runs(self) -> list[RunRecord]:
        df = self._load_csv()
        if self.status_filter is not None and "status" in df.columns:
            df = df[df["status"] == self.status_filter].copy()

        records: list[RunRecord] = []
        for _, row in df.iterrows():
            phase_weights: dict[str, dict[str, float]] = {}
            metrics: dict[str, float] = {}

            for column in df.columns:
                match = _PHASE_COL_RE.match(column)
                if match:
                    value = row.get(column)
                    if pd.isna(value):
                        continue
                    phase_idx = int(match.group(1))
                    domain = match.group(2)
                    phase_weights.setdefault(f"phase_{phase_idx}", {})
                    phase_weights[f"phase_{phase_idx}"][domain] = float(value)
                    continue

                if any(column.startswith(prefix) for prefix in self.metric_prefixes):
                    value = row.get(column)
                    if pd.isna(value):
                        continue
                    metrics[column] = float(value)

            local_run_id = _coerce_int(row.get("run_id"))
            wandb_run_id = _coerce_str(row.get("wandb_run_id"))
            run_name = _coerce_str(row.get("run_name")) or wandb_run_id
            status = _coerce_str(row.get("status")) or "unknown"

            records.append(
                RunRecord(
                    wandb_run_id=wandb_run_id,
                    source_experiment=self.source_experiment,
                    local_run_id=local_run_id,
                    run_name=run_name,
                    phase_weights=normalize_phase_weights(phase_weights),
                    status=status,
                    metrics=metrics,
                )
            )

        return records

    def collect_trajectories(self, objective_metric: str) -> pd.DataFrame:
        del objective_metric
        return pd.DataFrame(
            columns=[
                "wandb_run_id",
                "source_experiment",
                "local_run_id",
                "run_name",
                "step",
                "total_tokens",
                "metric_key",
                "metric_value",
            ]
        )


def _extract_summary_metrics(run_row: dict[str, object], metric_prefixes: tuple[str, ...]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in run_row.items()
        if isinstance(value, int | float) and any(key.startswith(prefix) for prefix in metric_prefixes)
    }


def _trajectory_frame_for_run(
    *,
    wandb_run,
    objective_metric: str,
    source_experiment: str,
    local_run_id: int | None,
    run_name: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["_step", objective_metric, "throughput/total_tokens"]
    for entry in wandb_run.scan_history(keys=keys):
        metric_value = entry.get(objective_metric)
        step = entry.get("_step")
        if metric_value is None or step is None:
            continue
        rows.append(
            {
                "wandb_run_id": wandb_run.id,
                "source_experiment": source_experiment,
                "local_run_id": local_run_id,
                "run_name": run_name,
                "step": int(step),
                "total_tokens": (
                    float(entry["throughput/total_tokens"]) if entry.get("throughput/total_tokens") is not None else None
                ),
                "metric_key": objective_metric,
                "metric_value": float(metric_value),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "wandb_run_id",
            "source_experiment",
            "local_run_id",
            "run_name",
            "step",
            "total_tokens",
            "metric_key",
            "metric_value",
        ],
    )


@dataclass(frozen=True)
class NamedWandbRunImportSource:
    """Import source for one explicitly named W&B run with known phase weights."""

    source_experiment: str
    local_run_id: int
    run_name: str
    phase_weights: dict[str, dict[str, float]]
    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    wandb_tags: tuple[str, ...] = ()
    wandb_name_substring: str | None = None
    metric_prefixes: tuple[str, ...] = ("eval/", "lm_eval/")

    def _candidate_rows(self) -> list[dict[str, object]]:
        tags = list(self.wandb_tags) if self.wandb_tags else [self.source_experiment]
        return query_wandb_runs(
            entity=self.wandb_entity,
            project=self.wandb_project,
            tags=tags,
            metrics=[],
            metric_prefixes=self.metric_prefixes,
        )

    def _resolve_run_row(self) -> dict[str, object]:
        name_substring = self.wandb_name_substring or f"/{self.run_name}"
        matches = [row for row in self._candidate_rows() if name_substring in str(row.get("wandb_run_name") or "")]
        if not matches:
            raise ValueError(
                f"No W&B run found for source_experiment={self.source_experiment!r} "
                f"matching substring {name_substring!r}"
            )

        finished = [row for row in matches if row.get("status") == "finished"]
        return finished[0] if finished else matches[0]

    def collect_runs(self) -> list[RunRecord]:
        run_row = self._resolve_run_row()
        return [
            RunRecord(
                wandb_run_id=_coerce_str(run_row.get("wandb_run_id")),
                source_experiment=self.source_experiment,
                local_run_id=self.local_run_id,
                run_name=self.run_name,
                phase_weights=normalize_phase_weights(self.phase_weights),
                status="completed" if run_row.get("status") == "finished" else str(run_row.get("status", "unknown")),
                metrics=_extract_summary_metrics(run_row, self.metric_prefixes),
            )
        ]

    def collect_trajectories(self, objective_metric: str) -> pd.DataFrame:
        import wandb

        run_row = self._resolve_run_row()
        wandb_run_id = _coerce_str(run_row.get("wandb_run_id"))
        if wandb_run_id is None:
            raise ValueError(f"Resolved W&B row for {self.run_name!r} is missing wandb_run_id")

        api = wandb.Api()
        wandb_run = api.run(f"{self.wandb_entity}/{self.wandb_project}/{wandb_run_id}")
        return _trajectory_frame_for_run(
            wandb_run=wandb_run,
            objective_metric=objective_metric,
            source_experiment=self.source_experiment,
            local_run_id=self.local_run_id,
            run_name=self.run_name,
        )


def _coerce_str(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def _coerce_int(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _infer_local_run_id_from_name(run_name: str | None) -> int | None:
    if not run_name:
        return None

    swarm = _SWARM_RUN_RE.search(run_name)
    if swarm:
        return int(swarm.group(1))

    base = _BASE_RUN_RE.search(run_name)
    if base:
        raw = int(base.group(1))
        return raw if raw >= 90000 else 90000 + raw

    return None


def default_legacy_sources() -> tuple[LegacyDomainPhaseImportSource, ...]:
    """Convenience defaults for immediate legacy coverage."""
    prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")

    return (
        LegacyDomainPhaseImportSource(
            source_experiment=TWO_PHASE_STARCODER_EXPERIMENT,
            weight_configs_path=(f"{prefix}/{TWO_PHASE_STARCODER_EXPERIMENT}/" "weight_configs-*/weight_configs.json"),
        ),
        LegacyDomainPhaseImportSource(
            source_experiment=THREE_PHASE_EXPERIMENT,
            weight_configs_path=(f"{prefix}/{THREE_PHASE_EXPERIMENT}/" "weight_configs-*/weight_configs.json"),
        ),
        LegacyDomainPhaseImportSource(
            source_experiment=THREE_PHASE_STARCODER_EXPERIMENT,
            weight_configs_path=(f"{prefix}/{THREE_PHASE_STARCODER_EXPERIMENT}/" "weight_configs-*/weight_configs.json"),
        ),
    )


def source_from_dict(payload: dict) -> ImportSource:
    """Deserialize an import source dictionary."""
    source_type = payload.get("type", "legacy_domain_phase")
    if source_type == "legacy_domain_phase":
        return LegacyDomainPhaseImportSource(
            source_experiment=payload["source_experiment"],
            wandb_entity=payload.get("wandb_entity", "marin-community"),
            wandb_project=payload.get("wandb_project", "marin"),
            wandb_tags=tuple(payload.get("wandb_tags", ())),
            weight_configs_path=payload.get("weight_configs_path"),
            metric_prefixes=tuple(payload.get("metric_prefixes", ("eval/", "lm_eval/"))),
        )
    if source_type == "csv_domain_phase":
        csv_path = payload["csv_path"]
        if not Path(csv_path).exists() and not str(csv_path).startswith(("gs://", "s3://")):
            logger.warning("CSV import source path does not currently exist: %s", csv_path)
        return CsvDomainPhaseImportSource(
            source_experiment=payload["source_experiment"],
            csv_path=csv_path,
            status_filter=payload.get("status_filter", "completed"),
            metric_prefixes=tuple(payload.get("metric_prefixes", ("eval/", "lm_eval/"))),
        )
    if source_type == "named_wandb_run":
        return NamedWandbRunImportSource(
            source_experiment=payload["source_experiment"],
            local_run_id=int(payload["local_run_id"]),
            run_name=payload["run_name"],
            phase_weights=normalize_phase_weights(payload["phase_weights"]),
            wandb_entity=payload.get("wandb_entity", "marin-community"),
            wandb_project=payload.get("wandb_project", "marin"),
            wandb_tags=tuple(payload.get("wandb_tags", ())),
            wandb_name_substring=payload.get("wandb_name_substring"),
            metric_prefixes=tuple(payload.get("metric_prefixes", ("eval/", "lm_eval/"))),
        )

    raise ValueError(f"Unsupported import source type: {source_type}")
