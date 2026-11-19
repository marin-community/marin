# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit current Atlas sources; never mutate admission or create approval decisions."""

import hashlib
import json
import logging
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from finestore.rl.licenses import LicenseClass, LicenseReceipt
from huggingface_hub.errors import HfHubHTTPError
from requests import RequestException

from experiments.post_training.task_curation.license_resolution import LicenseTarget, resolve_license
from experiments.post_training.task_curation.sources import all_sources

logger = logging.getLogger(__name__)


@click.command()
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--workers", type=click.IntRange(1, 4), default=2)
def main(output: Path, workers: int) -> None:
    """Write resumable per-source receipts and an audit table for coordinator review."""
    output.mkdir(parents=True, exist_ok=True)
    sources = all_sources()
    targets = {}
    inventory = {}
    for source_id, source in sources.items():
        dataset = source.dataset
        inventory[source_id] = {
            "source_id": source_id,
            "previously_admitted": not source.admission_blockers,
            "dataset_id": dataset.name if dataset else None,
            "revision": dataset.revision if dataset else None,
            "url": dataset.url if dataset else source.info.upstream_url,
        }
        if dataset is not None and dataset.revision:
            targets[source_id] = LicenseTarget(source_id, dataset.name, dataset.revision, dataset.url)
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    resolved = {}
    errors = {}
    # Resolve each shared repository pin once, then retain source-specific receipt identity.
    groups = {}
    for source_id, target in targets.items():
        groups.setdefault((target.dataset_id, target.revision, target.url), []).append(source_id)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for key, source_ids in groups.items():
            digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()
            path = output / f"receipt-{digest}.json"
            if path.exists():
                receipt = LicenseReceipt.model_validate_json(path.read_text())
                for source_id in source_ids:
                    resolved[source_id] = receipt.model_copy(update={"source_id": source_id})
                continue
            futures[executor.submit(resolve_license, targets[source_ids[0]])] = (source_ids, path)
        for future in as_completed(futures):
            source_ids, path = futures[future]
            try:
                receipt = future.result()
            except (ValueError, RequestException, HfHubHTTPError, subprocess.CalledProcessError) as error:
                # An explicit unresolved audit row is a failure, never inferred permission.
                for source_id in source_ids:
                    errors[source_id] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
                logger.error("Unresolved license for %s: %s", source_ids, type(error).__name__)
                continue
            path.write_text(receipt.model_dump_json(indent=2) + "\n")
            for source_id in source_ids:
                resolved[source_id] = receipt.model_copy(update={"source_id": source_id})
                if receipt.classification is not LicenseClass.PERMISSIVE:
                    logger.warning(
                        "LICENSE AUDIT %s: %s; already admitted=%s; evidence=%s",
                        source_id,
                        receipt.classification.value,
                        inventory[source_id]["previously_admitted"],
                        [item.url for item in receipt.evidence],
                    )
    counts = Counter(receipt.classification.value for receipt in resolved.values())
    counts["unresolved"] = len(sources) - len(resolved)
    report = {
        "counts": dict(counts),
        "sources": [
            {
                **row,
                "classification": resolved[source_id].classification.value if source_id in resolved else "unresolved",
                "receipt": resolved[source_id].model_dump(mode="json") if source_id in resolved else None,
                "error": errors.get(source_id, "no pinned license target") if source_id not in resolved else None,
            }
            for source_id, row in inventory.items()
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Atlas license backfill",
        "",
        "Audit only: existing admissions are unchanged; unresolved rows are blocked for new ingest.",
        "",
        "| Source | Previously admitted | Class | License | Evidence |",
        "|---|---|---|---|---|",
    ]
    for row in report["sources"]:
        receipt = resolved.get(row["source_id"])
        ids = ", ".join(receipt.license_ids) if receipt else row["error"]
        links = (
            " ".join(f"[evidence {i}]({e.url})" for i, e in enumerate(receipt.evidence, 1))
            if receipt
            else row["url"] or ""
        )
        lines.append(
            f"| {row['source_id']} | {row['previously_admitted']} | {row['classification']} | {ids} | {links} |"
        )
    (output / "table.md").write_text("\n".join(lines) + "\n")
    logger.warning("License backfill counts: %s", dict(counts))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
