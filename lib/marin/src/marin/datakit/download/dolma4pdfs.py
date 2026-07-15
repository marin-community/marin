# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``dolma4pdfs``: OCR'd PDF text from allenai/dolma3.5_pool.

The subset holds three upstream components; two are ingested:

* ``olmo-crawled-pdfs_ngram_filtered`` -- PDFs from OLMo's crawl.
* ``s2orcforolmo_nogpl_ngram_filtered_license_partioned`` -- S2ORC papers, license-partitioned.

``finepdfs_wo_partitioned_qual_ngram_filtered`` is excluded: it is the same corpus
as the ``finepdfs`` source (``finepdfs.py``).
"""

from rigging.filesystem import prefix_join

from marin.datakit.download.dolma3_5 import (
    DATA_FILE_EXTENSION,
    DATA_FILE_GLOB,
    HF_DATASET_ID,
    HF_REVISION,
    STAGED_ROOT,
)
from marin.datakit.download.hf_simple_util import NormalizationSchema, hf_normalize_steps
from marin.execution.step_spec import StepSpec

MARIN_NAME = "dolma4pdfs"
COMPONENTS = (
    "olmo-crawled-pdfs_ngram_filtered",
    "s2orcforolmo_nogpl_ngram_filtered_license_partioned",
)
DOWNLOAD_PARALLELISM = 32


def dolma4pdfs_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    return {
        MARIN_NAME: hf_normalize_steps(
            marin_name=MARIN_NAME,
            hf_dataset_id=HF_DATASET_ID,
            revision=HF_REVISION,
            staged_path=prefix_join(STAGED_ROOT, MARIN_NAME),
            hf_urls_glob=tuple(
                prefix_join(prefix_join(MARIN_NAME, component), DATA_FILE_GLOB) for component in COMPONENTS
            ),
            file_extensions=(DATA_FILE_EXTENSION,),
            zephyr_max_parallelism=DOWNLOAD_PARALLELISM,
            normalization_schema=NormalizationSchema.BARE,
        )
    }
