# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Corpus-wide census sweep for the false Marin persona (ruling 2026-11-23).

Sweeps every TaskTrove config's archived tasks and every prepared Atlas population's
model-visible task text for the identity defect: a persona or creator claim that names a
non-Marin developer for Marin. The TaskTrove sweep covers the upstream archive surface,
including configs that were never converted; the prepared sweep covers the model-visible
conversation text of the fingerprinted Atlas sources. Both write resumable per-part JSON
reports beside the aggregate, so an interrupted job resumes without rescanning finished parts.
"""

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.identity_following_repair import (
    TASKTROVE_REPO,
    TASKTROVE_REVISION,
    census_hits,
    census_tasktrove_config,
)

logger = logging.getLogger(__name__)

REGISTRY_REPO = "open-athena/marin-rl-data-atlas-registry"
REGISTRY_REVISION = "0c03567df8e11df5dde672513da42995ccc2c428"
LOCATOR_FILES = (
    "experiments/post_training/task_curation/source_population_locators_archived326.json",
    "experiments/post_training/task_curation/source_population_locators.json",
    "experiments/post_training/task_curation/source_population_locators_forge_20261120_a_v1.json",
    "experiments/post_training/task_curation/source_population_locators_scan_20261120_b_v1.json",
)
DOWNLOAD_WORKERS = 8


def tasktrove_configs() -> list[str]:
    """Every TaskTrove config folder with a tasks.parquet at the pinned revision."""
    info = HfApi().repo_info(TASKTROVE_REPO, repo_type="dataset", revision=TASKTROVE_REVISION, files_metadata=False)
    return sorted(
        sibling.rfilename.removesuffix("/tasks.parquet")
        for sibling in info.siblings
        if sibling.rfilename.endswith("/tasks.parquet")
    )


def sweep_tasktrove(output_root: StoragePath, *, limit_configs: int | None, limit_rows: int | None) -> dict[str, Any]:
    """Census every TaskTrove config, resuming from parts already written."""
    configs = tasktrove_configs()
    if limit_configs is not None:
        configs = configs[:limit_configs]
    logger.info("TaskTrove sweep over %d configs", len(configs))
    reports: dict[str, Any] = {}
    missing: dict[str, str] = {}

    def download(config: str) -> str | None:
        try:
            return hf_hub_download(
                TASKTROVE_REPO, f"{config}/tasks.parquet", repo_type="dataset", revision=TASKTROVE_REVISION
            )
        except Exception as error:
            logger.warning("Download failed for %s: %s", config, error)
            return None

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        downloads = dict(zip(configs, pool.map(download, configs), strict=True))
    for config in configs:
        part = output_root / "tasktrove" / f"{config}.json"
        if part.exists():
            reports[config] = json.loads(part.read_bytes())
            continue
        local = downloads.get(config)
        if local is None:
            missing[config] = "download_failed"
            continue
        try:
            report = census_tasktrove_config(local, config, limit=limit_rows)
        except Exception as error:
            logger.warning("Census failed for %s: %s", config, error)
            missing[config] = f"census_failed: {error}"
            continue
        report["affected"] = report["affected_tasks"] > 0
        StoragePath(str(part)).write_bytes(json.dumps(report, indent=1, sort_keys=True).encode())
        reports[config] = report
        if report["affected"]:
            logger.info("AFFECTED %s: %d/%d tasks", config, report["affected_tasks"], report["rows"])
    totals = {
        "configs": len(configs),
        "scanned": len(reports),
        "affected_configs": sum(1 for report in reports.values() if report["affected"]),
        "rows": sum(report["rows"] for report in reports.values()),
        "affected_tasks": sum(report["affected_tasks"] for report in reports.values()),
        "missing": missing,
    }
    return {"totals": totals, "configs": reports}


def prepared_locators() -> dict[str, dict[str, Any]]:
    """Union the prepared-population locators across locator revisions, newest label per source."""
    merged: dict[str, dict[str, Any]] = {}
    for name in LOCATOR_FILES:
        payload = hf_hub_download(REGISTRY_REPO, name, repo_type="dataset", revision=REGISTRY_REVISION)
        entries = json.loads(Path(payload).read_bytes()).get("prepared", {})
        for source_id, entry in entries.items():
            merged.setdefault(source_id, entry)
    return merged


def sweep_prepared(output_root: StoragePath, *, limit_sources: int | None, limit_rows: int | None) -> dict[str, Any]:
    """Census the model-visible text of every prepared population, resuming from written parts."""
    locators = prepared_locators()
    source_ids = sorted(locators)
    if limit_sources is not None:
        source_ids = source_ids[:limit_sources]
    logger.info("Prepared sweep over %d sources", len(source_ids))
    reports: dict[str, Any] = {}
    missing: dict[str, str] = {}
    for index, source_id in enumerate(source_ids):
        part = output_root / "prepared" / f"{_safe_name(source_id)}.json"
        if part.exists():
            reports[source_id] = json.loads(part.read_bytes())
            continue
        entry = locators[source_id]
        uri = entry.get("uri", "")
        if not uri:
            missing[source_id] = "no_uri"
            continue
        try:
            report = _census_prepared_parquet(uri, limit=limit_rows)
        except Exception as error:
            logger.warning("Prepared census failed for %s: %s", source_id, error)
            missing[source_id] = f"census_failed: {error}"
            continue
        report["uri"] = uri
        report["declared_task_count"] = entry.get("task_count")
        report["affected"] = report["affected_tasks"] > 0
        StoragePath(str(part)).write_bytes(json.dumps(report, indent=1, sort_keys=True).encode())
        reports[source_id] = report
        if index % 25 == 0:
            logger.info("Prepared progress %d/%d", index, len(source_ids))
        if report["affected"]:
            logger.info("AFFECTED %s: %d/%d tasks", source_id, report["affected_tasks"], report["rows"])
    totals = {
        "sources": len(source_ids),
        "scanned": len(reports),
        "affected_sources": sum(1 for report in reports.values() if report["affected"]),
        "rows": sum(report["rows"] for report in reports.values()),
        "affected_tasks": sum(report["affected_tasks"] for report in reports.values()),
        "missing": missing,
    }
    return {"totals": totals, "sources": reports}


def _census_prepared_parquet(uri: str, *, limit: int | None) -> dict[str, Any]:
    """Census the text column of one prepared population parquet."""
    rows = 0
    affected = 0
    pattern_totals: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    affected_ids: list[str] = []
    with StoragePath(uri).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(batch_size=256, columns=["task_id", "text"]):
            for row in batch.to_pylist():
                rows += 1
                if limit is not None and rows > limit:
                    break
                hits, row_samples = census_hits(row["text"] or "")
                if not hits:
                    continue
                affected += 1
                pattern_totals.update(hits)
                if len(affected_ids) < 20:
                    affected_ids.append(row["task_id"])
                if len(samples) < 20:
                    samples.extend({"task_id": row["task_id"], **sample} for sample in row_samples)
            if limit is not None and rows > limit:
                break
    return {
        "rows": rows,
        "affected_tasks": affected,
        "pattern_totals": dict(pattern_totals),
        "affected_ids_prefix": affected_ids,
        "samples": samples,
    }


def _safe_name(source_id: str) -> str:
    return source_id.replace("/", "_").replace(":", "_")


def main() -> None:
    """Run the sweep CLI against TaskTrove and the prepared populations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=("tasktrove", "prepared", "both"), default="both")
    parser.add_argument("--output", required=True, help="Aggregate report URI; parts are written beside it")
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_coreweave_s3()
    output_root = StoragePath(args.output.rsplit("/", 1)[0] + "/")
    report: dict[str, Any] = {"sweep": args.sweep}
    if args.sweep in ("tasktrove", "both"):
        report["tasktrove"] = sweep_tasktrove(output_root, limit_configs=args.limit_configs, limit_rows=args.limit_rows)
    if args.sweep in ("prepared", "both"):
        report["prepared"] = sweep_prepared(output_root, limit_sources=args.limit_sources, limit_rows=args.limit_rows)
    aggregate = StoragePath(args.output)
    aggregate.write_bytes(json.dumps(report, indent=1, sort_keys=True).encode())
    logger.info("Aggregate report written to %s", args.output)


if __name__ == "__main__":
    main()
