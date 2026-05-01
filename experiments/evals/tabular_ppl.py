# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone tabular raw-text slices for issue #5059.

This split keeps the byte-preserving CSV/TSV staging helper live with one small
public dataset while broader structured-table sources stay out of the landing
path. The first slice uses the official UCI Wine Quality red-wine CSV because
it is public, small, and preserves the ordinary delimiter/numeric/header
surfaces we want to probe.
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

UCI_WINE_QUALITY_DATASET_URL = "https://archive.ics.uci.edu/ml/datasets/wine_quality"
UCI_WINE_QUALITY_RED_CSV_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)
UCI_WINE_QUALITY_SLICE_KEY = "structured_text/uci_wine_quality_red_csv"
UCI_WINE_QUALITY_SOURCE_LABEL = "uci_wine_quality:red_csv"
UCI_WINE_QUALITY_OUTPUT_FILENAME = "data.jsonl.gz"
UCI_WINE_QUALITY_SOURCE_FILENAME = "winequality-red.csv"
UCI_WINE_QUALITY_DOWNLOAD_TIMEOUT = 300
UCI_WINE_QUALITY_MAX_BYTES_PER_SOURCE = 128 * 1024
UCI_WINE_QUALITY_MAX_BYTES_PER_DOCUMENT = 32 * 1024


@dataclass(frozen=True)
class TabularPplSlice:
    slice_key: str
    source_label: str
    source_urls: tuple[str, ...]
    source_license: str
    tags: tuple[str, ...]

    def manifest(self) -> IngestionSourceManifest:
        return IngestionSourceManifest(
            dataset_key="tabular/uci_wine_quality_red_csv",
            slice_key=self.slice_key,
            source_label=self.source_label,
            source_urls=self.source_urls,
            source_license=self.source_license,
            source_format="raw_csv_file",
            surface_form="byte_preserved_csv_chunks",
            policy=IngestionPolicy(
                usage_policy=UsagePolicy.EVAL_ONLY,
                use_policy="Eval-only structured-text perplexity slice.",
                requires_sanitization=False,
                identity_treatment=IdentityTreatment.PRESERVE,
                secret_redaction=SecretRedaction.NONE,
                contamination_risk="moderate: standard public ML benchmark table",
                provenance_notes="Official UCI Wine Quality red-wine CSV.",
            ),
            staging=StagingMetadata(
                transform_name="stage_tabular_source",
                preserve_header=True,
                metadata={"output_filename": UCI_WINE_QUALITY_OUTPUT_FILENAME},
            ),
            sample_caps=SampleCapConfig(
                max_bytes_per_source=UCI_WINE_QUALITY_MAX_BYTES_PER_SOURCE,
                max_bytes_per_document=UCI_WINE_QUALITY_MAX_BYTES_PER_DOCUMENT,
            ),
        )


UCI_WINE_QUALITY_RED_SLICE = TabularPplSlice(
    slice_key=UCI_WINE_QUALITY_SLICE_KEY,
    source_label=UCI_WINE_QUALITY_SOURCE_LABEL,
    source_urls=(UCI_WINE_QUALITY_DATASET_URL, UCI_WINE_QUALITY_RED_CSV_URL),
    source_license="CC BY 4.0",
    tags=("structured_text", "tabular", f"epic:{EPIC_5005}", f"issue:{ISSUE_5059}", UCI_WINE_QUALITY_SLICE_KEY),
)


def _path_exists(path: str) -> bool:
    fs, resolved_path = url_to_fs(path)
    return fs.exists(resolved_path)


def _download_source_file(*, source_url: str, output_file: str) -> None:
    output_dir = posixpath.dirname(output_file)
    fsspec_mkdirs(output_dir, exist_ok=True)

    with requests.get(source_url, stream=True, timeout=UCI_WINE_QUALITY_DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with atomic_rename(output_file) as temp_path:
            with open_url(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)


def materialize_uci_wine_quality_red(output_path: str) -> dict[str, int | str]:
    """Download and stage the UCI red-wine CSV into raw-text eval documents."""
    raw_dir = posixpath.join(output_path, "raw")
    raw_file = posixpath.join(raw_dir, UCI_WINE_QUALITY_SOURCE_FILENAME)
    if not _path_exists(raw_file):
        _download_source_file(source_url=UCI_WINE_QUALITY_RED_CSV_URL, output_file=raw_file)

    manifest = UCI_WINE_QUALITY_RED_SLICE.manifest()
    return stage_tabular_source(
        TabularStagingConfig(
            input_path=raw_dir,
            output_path=output_path,
            source_label=UCI_WINE_QUALITY_SOURCE_LABEL,
            file_extensions=(".csv",),
            max_bytes_per_source=UCI_WINE_QUALITY_MAX_BYTES_PER_SOURCE,
            max_bytes_per_document=UCI_WINE_QUALITY_MAX_BYTES_PER_DOCUMENT,
            preserve_header=True,
            output_filename=UCI_WINE_QUALITY_OUTPUT_FILENAME,
            source_manifest=manifest,
            content_fingerprint=manifest.fingerprint(),
        )
    )


UCI_WINE_QUALITY_RED_RAW = StepSpec(
    name="raw/structured_text/uci_wine_quality_red_csv",
    fn=materialize_uci_wine_quality_red,
    hash_attrs={
        "version": "v1",
        "source_urls": UCI_WINE_QUALITY_RED_SLICE.source_urls,
        "source_license": UCI_WINE_QUALITY_RED_SLICE.source_license,
        "max_bytes_per_source": UCI_WINE_QUALITY_MAX_BYTES_PER_SOURCE,
        "max_bytes_per_document": UCI_WINE_QUALITY_MAX_BYTES_PER_DOCUMENT,
    },
)


def tabular_raw_validation_sets(
    *,
    tabular_raw: ExecutorStep | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Return the standalone tabular raw-text eval slice(s)."""
    if tabular_raw is None:
        tabular_raw = UCI_WINE_QUALITY_RED_RAW.as_executor_step()

    return {
        UCI_WINE_QUALITY_SLICE_KEY: raw_text_dataset(
            tabular_raw.cd(UCI_WINE_QUALITY_OUTPUT_FILENAME),
            tags=UCI_WINE_QUALITY_RED_SLICE.tags,
        )
    }
