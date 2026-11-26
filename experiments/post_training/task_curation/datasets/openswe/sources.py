# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned OpenSWE keep-only population the dedicated profile converts.

Ruling 48 (user 2026-11-17) lifted the OpenSWE license ban for one pin: the
permissive-license keep partition, built as a NEW private release from the
base-commit license verification (coordinator rulings 2026-11-26 5:16/5:25 AM).
The keep-only release is the admission boundary: every row carries its license
partition evidence verbatim, and rows outside it never convert.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

UPSTREAM_DATASET = "GAIR/OpenSWE"
UPSTREAM_REVISION = "a8db93af5335df2c8baac0cd1ff367e4d475d3d7"
KEEP_RELEASE_REPO = "open-athena/openswe-oss-keep-only-20261126a"
KEEP_RELEASE_REVISION = "20a87df9d9ba"
KEEP_RELEASE_ROWS = "openswe_oss-keep-only.jsonl"
KEEP_RELEASE_MANIFEST = "manifest.json"
PINNED_POPULATION = 21137


@dataclass(frozen=True)
class OpenSweRow:
    """One keep-only row: the upstream record plus its license partition evidence."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    eval_script: str
    dockerfile: str
    image_name: str
    license_partition: dict


def keep_release_rows(release_dir: str | None = None) -> list[OpenSweRow]:
    """Load the keep-only population from the pinned release.

    ``release_dir`` reads a local checkout of the release (the Hub download stays
    the default). The manifest's row count must match the pinned population and
    the loaded rows must match it too: the license verification binds this
    exact population, so a changed release fails instead of converting.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    if release_dir is not None:
        root = Path(release_dir)
        rows_path = root / KEEP_RELEASE_ROWS
        manifest = json.loads((root / KEEP_RELEASE_MANIFEST).read_text())
    else:
        rows_path = Path(
            hf_hub_download(KEEP_RELEASE_REPO, KEEP_RELEASE_ROWS, repo_type="dataset", revision=KEEP_RELEASE_REVISION)
        )
        manifest = json.loads(
            Path(
                hf_hub_download(
                    KEEP_RELEASE_REPO, KEEP_RELEASE_MANIFEST, repo_type="dataset", revision=KEEP_RELEASE_REVISION
                )
            ).read_text()
        )
    recorded = manifest.get("rows")
    if recorded != PINNED_POPULATION:
        raise ValueError(
            f"The keep-only release manifest records {recorded} rows; the pinned population is {PINNED_POPULATION}"
        )
    rows: list[OpenSweRow] = []
    seen: set[str] = set()
    with rows_path.open() as stream:
        for line in stream:
            payload = json.loads(line)
            instance_id = payload["instance_id"]
            if instance_id in seen:
                raise ValueError(f"Duplicate keep-only instance_id: {instance_id}")
            seen.add(instance_id)
            rows.append(
                OpenSweRow(
                    instance_id=instance_id,
                    repo=payload["repo"],
                    base_commit=payload["base_commit"],
                    problem_statement=payload["problem_statement"],
                    patch=payload["patch"],
                    test_patch=payload["test_patch"],
                    eval_script=payload["eval_script"],
                    dockerfile=payload["Dockerfile"],
                    image_name=payload["image_name"],
                    license_partition=payload["license_partition"],
                )
            )
    if len(rows) != PINNED_POPULATION:
        raise ValueError(f"The keep-only release carries {len(rows)} rows; the pinned population is {PINNED_POPULATION}")
    LOGGER.info("keep-only population loaded: %d rows", len(rows))
    return rows
