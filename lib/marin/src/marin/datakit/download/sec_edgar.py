# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TeraflopAI/SEC-EDGAR dataset download + normalize helpers.

8M filings (~43B tokens) from the SEC EDGAR database, organized into
per-filing-type subdirectories: 10-K, 10-Q, 8-K, 20-F, S-1, S-8, 144,
and Form 3/4/5. Each subdirectory holds raw parquet shards. Text lives
in the ``content`` column; ``text`` also exists upstream but is reserved
for the parsed-plaintext field — we keep ``content`` to match the
upstream documentation.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "TeraflopAI/SEC-EDGAR"
HF_REVISION = "43de32c"

FILING_TYPES = ["10-K", "10-Q", "8-K", "20-F", "S-1", "S-8", "144", "3", "4", "5"]


def sec_edgar_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, normalize)`` chain for SEC-EDGAR."""
    download = download_hf_step(
        "raw/sec-edgar",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{filing}/*.parquet" for filing in FILING_TYPES],
    )
    normalize = normalize_step(
        name="normalized/sec-edgar",
        download=download,
        text_field="content",
        file_extensions=(".parquet",),
    )
    return (download, normalize)
