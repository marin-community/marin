# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a pinned, complete Uncheatable Eval snapshot for document scoring."""

import json
import logging
import os
from pathlib import Path

import fsspec
from marin.datakit.download.uncheatable_eval import UncheatableEvalDownloadConfig, download_latest_uncheatable_eval

SOURCE_REVISION = "185c463882a8ae0f203e51ae5852d8cf4fe299cf"
OUTPUT_PREFIX = "s3://marin-us-east-02a/evaluation/document-losses/uncheatable/185c463882a8/data"
EXPECTED_SUBSETS = 14
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    result = download_latest_uncheatable_eval(
        UncheatableEvalDownloadConfig(
            output_path=OUTPUT_PREFIX,
            repo_owner="ziqing-huang",
            repo_name="uncheatable_eval",
            branch=SOURCE_REVISION,
            skip_existing=False,
        )
    )
    if not result["success"]:
        raise RuntimeError(f"Uncheatable preparation failed: {result}")
    records = result["processed"]
    if len(records) != EXPECTED_SUBSETS:
        raise ValueError(f"Expected {EXPECTED_SUBSETS} benchmark subsets, got {len(records)}")
    manifest = {
        "source_repository": "https://github.com/ziqing-huang/uncheatable_eval",
        "source_revision": SOURCE_REVISION,
        "subsets": [
            {
                "name": record["benchmark"],
                "input_path": record["output_file"],
                "expected_documents": record["records"],
                "source_blob_sha": record["sha"],
                "source_bytes": record["size"],
                "start_date": record["start_date"],
                "end_date": record["end_date"],
            }
            for record in sorted(records, key=lambda record: record["benchmark"])
        ],
    }
    manifest_path = f"{OUTPUT_PREFIX}/manifest.json"
    contents = json.dumps(manifest, indent=2) + "\n"
    with fsspec.open(manifest_path, "wt") as stream:
        stream.write(contents)
    Path(os.environ["IRIS_OUTPUT_DIR"], "manifest.json").write_text(contents)
    logger.info("UNCHEATABLE_MANIFEST_READY %s %s", manifest_path, json.dumps(manifest))


if __name__ == "__main__":
    main()
