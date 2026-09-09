# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare both pinned evaluation cohorts in central2 without moving checkpoints."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
from huggingface_hub import hf_hub_download
from marin.datakit.download.uncheatable_eval import (
    DownloadTask,
    UncheatableEvalDataset,
    UncheatableEvalDownloadConfig,
    _download_and_convert_single,
)

from experiments.evaluation.prepare_uncheatable_losses import DATASET_ID, SOURCE_FILE, SOURCE_REVISION, prepare_manifest

PREFIX = "gs://marin-us-central2/evaluation/document-losses/snowball-history-8975/data"
LEGACY_SUMMARY = "https://storage.googleapis.com/marin-public/dlwh/uncheatable-snowball-losses/2026-09-08/summary.json"


def main():
    logging.basicConfig(level=logging.INFO)
    with fsspec.open(LEGACY_SUMMARY, "rt") as stream:
        legacy = json.load(stream)["manifest"]
    legacy_path = f"{PREFIX}/legacy-2025"
    cfg = UncheatableEvalDownloadConfig(
        output_path=legacy_path, repo_owner="ziqing-huang", branch=legacy["source_revision"], skip_existing=False
    )
    tasks = []
    for row in legacy["subsets"]:
        filename = Path(row["input_path"]).name.removesuffix(".jsonl.gz") + ".json"
        url = f"https://raw.githubusercontent.com/ziqing-huang/uncheatable_eval/{legacy['source_revision']}/data/{filename}"
        dataset = UncheatableEvalDataset(
            row["name"],
            row["start_date"],
            row["end_date"],
            filename,
            url,
            sha=row["source_blob_sha"],
            size=row["source_bytes"],
        )
        output = f"{legacy_path}/{dataset.output_filename()}"
        tasks.append(DownloadTask(url, output, dataset, cfg))
    with ThreadPoolExecutor(max_workers=4) as pool:
        converted = list(pool.map(_download_and_convert_single, tasks))
    for row, converted_row in zip(legacy["subsets"], converted, strict=True):
        if row["expected_documents"] != converted_row["records"]:
            raise ValueError(f"Unexpected legacy document count: {row['name']}")
        row["input_path"] = converted_row["output_file"]
    july_path = f"{PREFIX}/2026-07"
    source_path = hf_hub_download(DATASET_ID, SOURCE_FILE, repo_type="dataset", revision=SOURCE_REVISION)
    july = prepare_manifest(source_path, july_path)
    for name, manifest in [("legacy-2025", legacy), ("2026-07", july)]:
        output = f"{PREFIX}/{name}/manifest.json"
        contents = json.dumps(manifest, indent=2) + "\n"
        with fsspec.open(output, "wt") as stream:
            stream.write(contents)
        Path(os.environ["IRIS_OUTPUT_DIR"], f"{name}-manifest.json").write_text(contents)
        logging.info("REGIONAL_DATA_READY %s %s", output, json.dumps(manifest))


if __name__ == "__main__":
    main()
