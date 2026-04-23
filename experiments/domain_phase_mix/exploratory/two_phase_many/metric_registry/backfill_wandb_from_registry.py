# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "wandb"]
# ///
"""Backfill W&B summaries from canonical metric registry facts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import re

import pandas as pd
import wandb

from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    METRICS_LONG_CSV,
    RUNS_CSV,
    SCRIPT_DIR,
    canonicalize_metric_key,
)

DEFAULT_ENTITY = "marin-community"
DEFAULT_PROJECT = "marin"
DEFAULT_MANIFEST_CSV = SCRIPT_DIR / "wandb_backfill_manifest.csv"
VALUE_TOLERANCE = 1e-10
SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_key(metric_key: str) -> str:
    return SAFE_KEY_RE.sub("_", metric_key).strip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--metric-prefix", action="append", default=[])
    parser.add_argument("--scale", default="60m_1p2b")
    parser.add_argument("--cohort", default="signal")
    parser.add_argument(
        "--run-set",
        choices=("fit_swarm_60m_default", "qsplit240_core", "all"),
        default="fit_swarm_60m_default",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--overwrite-conflicts", action="store_true")
    parser.add_argument("--write-source-metadata", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Mutate W&B. Omit for a dry-run manifest.")
    return parser.parse_args()


def _selected_metric_keys(metrics: list[str], metric_prefixes: list[str]) -> tuple[set[str], tuple[str, ...]]:
    canonical_metrics = {str(canonicalize_metric_key(metric)["canonical_metric_key"]) for metric in metrics}
    return canonical_metrics, tuple(metric_prefixes)


def _load_backfill_candidates(
    *,
    metrics: list[str],
    metric_prefixes: list[str],
    scale: str,
    cohort: str,
    run_set: str,
) -> pd.DataFrame:
    if not metrics and not metric_prefixes:
        raise ValueError("Specify at least one --metric or --metric-prefix")

    metric_keys, prefixes = _selected_metric_keys(metrics, metric_prefixes)
    facts = pd.read_csv(METRICS_LONG_CSV)
    selected = facts["canonical_metric_key"].isin(metric_keys)
    for prefix in prefixes:
        selected |= facts["canonical_metric_key"].str.startswith(prefix)
    facts = facts.loc[selected].copy()

    runs = pd.read_csv(RUNS_CSV, low_memory=False)
    run_mask = runs["scale"].eq(scale) & runs["cohort"].eq(cohort)
    if run_set == "fit_swarm_60m_default":
        run_mask &= runs["is_fit_swarm_60m_default"].fillna(False)
    elif run_set == "qsplit240_core":
        run_mask &= runs["is_qsplit240_core"].fillna(False)
    runs = runs.loc[run_mask].copy()

    merged = facts.merge(
        runs[
            [
                "registry_run_key",
                "run_name",
                "wandb_run_id",
                "scale",
                "cohort",
                "source_names",
            ]
        ],
        on="registry_run_key",
        how="inner",
        suffixes=("", "_run"),
    )
    return merged.dropna(subset=["wandb_run_id"]).reset_index(drop=True)


def _manifest_row(row: pd.Series, *, action: str, existing_value: float | None) -> dict[str, object]:
    return {
        "action": action,
        "run_name": row["run_name"],
        "wandb_run_id": row["wandb_run_id"],
        "registry_run_key": row["registry_run_key"],
        "canonical_metric_key": row["canonical_metric_key"],
        "value": float(row["value"]),
        "existing_value": existing_value,
        "source_name": row["source_name"],
        "source_uri": row["source_uri"],
    }


def backfill_wandb(
    *,
    entity: str,
    project: str,
    metrics: list[str],
    metric_prefixes: list[str],
    scale: str,
    cohort: str,
    run_set: str,
    manifest: Path,
    apply_updates: bool,
    overwrite_conflicts: bool,
    write_source_metadata: bool,
) -> pd.DataFrame:
    """Create a backfill manifest and optionally update W&B summaries."""
    candidates = _load_backfill_candidates(
        metrics=metrics,
        metric_prefixes=metric_prefixes,
        scale=scale,
        cohort=cohort,
        run_set=run_set,
    )
    api = wandb.Api()
    rows: list[dict[str, object]] = []
    backfilled_at = datetime.now(UTC).isoformat()

    for wandb_run_id, group in candidates.groupby("wandb_run_id", sort=True):
        run = api.run(f"{entity}/{project}/{wandb_run_id}")
        changed = False
        for _, row in group.sort_values("canonical_metric_key").iterrows():
            metric_key = str(row["canonical_metric_key"])
            value = float(row["value"])
            existing = run.summary.get(metric_key)
            existing_value = float(existing) if isinstance(existing, int | float) else None
            if existing_value is None:
                action = "write_missing"
            elif abs(existing_value - value) <= VALUE_TOLERANCE:
                action = "skip_existing_equal"
            elif overwrite_conflicts:
                action = "overwrite_conflict"
            else:
                action = "skip_existing_conflict"

            rows.append(_manifest_row(row, action=action, existing_value=existing_value))
            if not apply_updates or action not in {"write_missing", "overwrite_conflict"}:
                continue
            run.summary[metric_key] = value
            if write_source_metadata:
                safe_metric = _safe_key(metric_key)
                run.summary[f"metric_registry/source_name/{safe_metric}"] = str(row["source_name"])
                run.summary[f"metric_registry/source_uri/{safe_metric}"] = str(row["source_uri"])
            run.summary["metric_registry/backfilled_at_utc"] = backfilled_at
            changed = True
        if changed:
            run.summary.update()

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_frame = pd.DataFrame.from_records(rows)
    manifest_frame.to_csv(manifest, index=False)
    print(f"Wrote {len(manifest_frame)} manifest rows to {manifest}")
    if not apply_updates:
        print("Dry run only. Re-run with --apply to mutate W&B.")
    return manifest_frame


def main() -> None:
    args = _parse_args()
    backfill_wandb(
        entity=args.entity,
        project=args.project,
        metrics=args.metric,
        metric_prefixes=args.metric_prefix,
        scale=args.scale,
        cohort=args.cohort,
        run_set=args.run_set,
        manifest=args.manifest,
        apply_updates=args.apply,
        overwrite_conflicts=args.overwrite_conflicts,
        write_source_metadata=args.write_source_metadata,
    )


if __name__ == "__main__":
    main()
