# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``dolma4pdfs``: OCR'd PDF text from allenai/dolma3.5_pool.

The subset holds three upstream components; two are ingested:

* ``olmo-crawled-pdfs_ngram_filtered`` -- PDFs from OLMo's crawl.
* ``s2orcforolmo_nogpl_ngram_filtered_license_partioned`` -- S2ORC papers, license-partitioned.

``finepdfs_wo_partitioned_qual_ngram_filtered`` is excluded: it is the same corpus
as the ``finepdfs`` source (``finepdfs.py``).

The two components have different schemas -- s2orc carries ``fos``, ``fos_max`` and
``license``, and both nest ``attributes``/``metadata`` dicts whose value types vary
per record. Parquet cannot widen a column mid-write, so they normalize to the bare
``id``/``text``/``source_id`` schema. At ~39k files the download is bound by per-file
task dispatch, not bandwidth, so it runs at raised parallelism.
"""

from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "allenai/dolma3.5_pool"
HF_REVISION = "d2bf6ae"

MARIN_NAME = "dolma4pdfs"
COMPONENTS = (
    "olmo-crawled-pdfs_ngram_filtered",
    "s2orcforolmo_nogpl_ngram_filtered_license_partioned",
)
DOWNLOAD_PARALLELISM = 32


def dolma4pdfs_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return the ``(download, normalize)`` chain for ``dolma4pdfs``."""
    return {
        MARIN_NAME: hf_normalize_steps(
            marin_name=MARIN_NAME,
            hf_dataset_id=HF_DATASET_ID,
            revision=HF_REVISION,
            staged_path=f"raw/dolma3_5_pool-{HF_REVISION}/{MARIN_NAME}",
            hf_urls_glob=tuple(f"{MARIN_NAME}/{c}/**/*.jsonl.zst" for c in COMPONENTS),
            file_extensions=(".jsonl.zst",),
            zephyr_max_parallelism=DOWNLOAD_PARALLELISM,
            bare=True,
        )
    }
