# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EssentialAI eai-taxonomy-code-w-dclm dataset as a lazy ``Dataset`` handle.

Common Crawl web text kept for intermediate-to-advanced code/CS reasoning (via
the EAI taxonomy classifier) and instruction density (via the DCLM classifier),
plus adjacent math content. ~564B tokens across 273.8M documents; the ``text``
field is the raw document, so no field remapping is needed.
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

EAI_TAXONOMY_CODE_HF_ID = "EssentialAI/eai-taxonomy-code-w-dclm"
EAI_TAXONOMY_CODE_REVISION = "2e1eb21"
_URLS_GLOB = ["data/train-*.parquet"]


def eai_taxonomy_code_dataset(*, tokenizer: str = llama3_tokenizer) -> ArtifactStep[TokenizedCache]:
    """EssentialAI eai-taxonomy-code-w-dclm as a tokenized ``Dataset`` handle (~564B tokens)."""
    raw = hf_download(
        "raw/eai-taxonomy-code-w-dclm",
        hf_id=EAI_TAXONOMY_CODE_HF_ID,
        revision=EAI_TAXONOMY_CODE_REVISION,
        urls_glob=_URLS_GLOB,
        version="2026.07.11",
    )
    return tokenized(
        "eai-taxonomy-code-w-dclm-llama3",
        tokenizer=tokenizer,
        raw=raw,
        glob=_URLS_GLOB[0],
        version="2026.07.11",
    )


if __name__ == "__main__":
    dataset_main({"eai_taxonomy_code": eai_taxonomy_code_dataset()})
