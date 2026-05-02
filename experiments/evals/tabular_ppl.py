# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone tabular raw-text slices for issue #5059.

This split keeps the byte-preserving CSV/TSV staging helper live with one real
public CSV artifact while broader structured-table sources stay out of the
landing path. The first slice uses FiveThirtyEight's Marvel Wikia character
table, which is a public GitHub-hosted CSV with ordinary header, missing-value,
string, and numeric surfaces.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

import requests
from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep
from marin.execution.step_spec import StepSpec
from marin.transform.structured_text.tabular import TabularStagingConfig, stage_tabular_source
from marin.utils import fsspec_mkdirs
from rigging.filesystem import open_url, url_to_fs
from zephyr.writers import atomic_rename

ISSUE_5059 = 5059
EPIC_5005 = 5005

FIVETHIRTYEIGHT_COMIC_CHARACTERS_DATASET_URL = "https://github.com/fivethirtyeight/data/tree/master/comic-characters"
FIVETHIRTYEIGHT_MARVEL_WIKIA_CSV_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/marvel-wikia-data.csv"
)
FIVETHIRTYEIGHT_MARVEL_SLICE_KEY = "structured_text/fivethirtyeight_marvel_wikia_csv"
FIVETHIRTYEIGHT_MARVEL_SOURCE_LABEL = "fivethirtyeight:marvel_wikia_csv"
FIVETHIRTYEIGHT_MARVEL_OUTPUT_FILENAME = "data.jsonl.gz"
FIVETHIRTYEIGHT_MARVEL_SOURCE_FILENAME = "marvel-wikia-data.csv"
FIVETHIRTYEIGHT_DOWNLOAD_TIMEOUT = 300
FIVETHIRTYEIGHT_MARVEL_MAX_BYTES_PER_SOURCE = 512 * 1024
FIVETHIRTYEIGHT_MARVEL_MAX_BYTES_PER_DOCUMENT = 32 * 1024


@dataclass(frozen=True)
class TabularPplSlice:
    slice_key: str
    source_label: str
    source_urls: tuple[str, ...]
    source_license: str
    tags: tuple[str, ...]

    def manifest(self) -> IngestionSourceManifest:
        return IngestionSourceManifest(
            dataset_key="tabular/fivethirtyeight_marvel_wikia_csv",
            slice_key=self.slice_key,
            source_label=self.source_label,
            source_urls=self.source_urls,
            source_license=self.source_license,
            source_format="raw_csv_file",
            surface_form="byte_preserved_csv_chunks",
            epic_issue=EPIC_5005,
            issue_numbers=(ISSUE_5059,),
            policy=IngestionPolicy(
                usage_policy=UsagePolicy.EVAL_ONLY,
                use_policy="Eval-only structured-text perplexity slice.",
                requires_sanitization=False,
                identity_treatment=IdentityTreatment.PRESERVE,
                secret_redaction=SecretRedaction.NONE,
                contamination_risk="low: public CSV from FiveThirtyEight's data repository",
                provenance_notes="Marvel character table from FiveThirtyEight's comic-characters dataset.",
            ),
            staging=StagingMetadata(
                transform_name="stage_tabular_source",
                preserve_header=True,
                metadata={"output_filename": FIVETHIRTYEIGHT_MARVEL_OUTPUT_FILENAME},
            ),
            sample_caps=SampleCapConfig(
                max_bytes_per_source=FIVETHIRTYEIGHT_MARVEL_MAX_BYTES_PER_SOURCE,
                max_bytes_per_document=FIVETHIRTYEIGHT_MARVEL_MAX_BYTES_PER_DOCUMENT,
            ),
        )


FIVETHIRTYEIGHT_MARVEL_SLICE = TabularPplSlice(
    slice_key=FIVETHIRTYEIGHT_MARVEL_SLICE_KEY,
    source_label=FIVETHIRTYEIGHT_MARVEL_SOURCE_LABEL,
    source_urls=(FIVETHIRTYEIGHT_COMIC_CHARACTERS_DATASET_URL, FIVETHIRTYEIGHT_MARVEL_WIKIA_CSV_URL),
    source_license="CC BY 4.0",
    tags=(
        "structured_text",
        "tabular",
        f"epic:{EPIC_5005}",
        f"issue:{ISSUE_5059}",
        FIVETHIRTYEIGHT_MARVEL_SLICE_KEY,
    ),
)


def _path_exists(path: str) -> bool:
    fs, resolved_path = url_to_fs(path)
    return fs.exists(resolved_path)


def _download_source_file(*, source_url: str, output_file: str) -> None:
    output_dir = posixpath.dirname(output_file)
    fsspec_mkdirs(output_dir, exist_ok=True)

    with requests.get(source_url, stream=True, timeout=FIVETHIRTYEIGHT_DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with atomic_rename(output_file) as temp_path:
            with open_url(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)


def materialize_fivethirtyeight_marvel_wikia(output_path: str) -> dict[str, int | str]:
    """Download and stage the FiveThirtyEight Marvel Wikia CSV into raw-text eval documents."""
    raw_dir = posixpath.join(output_path, "raw")
    raw_file = posixpath.join(raw_dir, FIVETHIRTYEIGHT_MARVEL_SOURCE_FILENAME)
    if not _path_exists(raw_file):
        _download_source_file(source_url=FIVETHIRTYEIGHT_MARVEL_WIKIA_CSV_URL, output_file=raw_file)

    manifest = FIVETHIRTYEIGHT_MARVEL_SLICE.manifest()
    return stage_tabular_source(
        TabularStagingConfig(
            input_path=raw_dir,
            output_path=output_path,
            source_label=FIVETHIRTYEIGHT_MARVEL_SOURCE_LABEL,
            file_extensions=(".csv",),
            max_bytes_per_source=FIVETHIRTYEIGHT_MARVEL_MAX_BYTES_PER_SOURCE,
            max_bytes_per_document=FIVETHIRTYEIGHT_MARVEL_MAX_BYTES_PER_DOCUMENT,
            preserve_header=True,
            output_filename=FIVETHIRTYEIGHT_MARVEL_OUTPUT_FILENAME,
            source_manifest=manifest,
            content_fingerprint=manifest.fingerprint(),
        )
    )


FIVETHIRTYEIGHT_MARVEL_RAW = StepSpec(
    name="raw/structured_text/fivethirtyeight_marvel_wikia_csv",
    fn=materialize_fivethirtyeight_marvel_wikia,
    hash_attrs={
        "version": "v1",
        "content_fingerprint": FIVETHIRTYEIGHT_MARVEL_SLICE.manifest().fingerprint(),
    },
)


def tabular_raw_validation_sets(
    *,
    tabular_raw: ExecutorStep | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Return the standalone tabular raw-text eval slice(s)."""
    if tabular_raw is None:
        tabular_raw = FIVETHIRTYEIGHT_MARVEL_RAW.as_executor_step()

    return {
        FIVETHIRTYEIGHT_MARVEL_SLICE_KEY: raw_text_dataset(
            tabular_raw.cd(FIVETHIRTYEIGHT_MARVEL_OUTPUT_FILENAME),
            tags=FIVETHIRTYEIGHT_MARVEL_SLICE.tags,
        )
    }
