# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EssentialAI/eai-taxonomy-code-w-dclm download + normalize helpers.

Common Crawl web text kept for intermediate-to-advanced code/CS reasoning (via
the EAI taxonomy classifier) and instruction density (via the DCLM
classifier), plus adjacent math content. 273,847,260 documents; 591.9B tokens
under the Marin tokenizer (the dataset card's ~564B is a different tokenizer's
count). Single flat HF repo (``data/train-*.parquet``) whose ``text``/``id``
fields already match the default schema, so this is a thin wrapper over
:func:`hf_normalize_steps`.
"""

from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "EssentialAI/eai-taxonomy-code-w-dclm"
HF_REVISION = "2e1eb21"


def eai_taxonomy_code_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, normalize)`` chain for eai-taxonomy-code-w-dclm."""
    return hf_normalize_steps(
        marin_name="eai-taxonomy-code-w-dclm",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=("data/train-*.parquet",),
    )
