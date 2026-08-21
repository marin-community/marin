# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
The Paloma eval sets, downloaded and tokenized

https://huggingface.co/datasets/allenai/paloma
"""

import os.path

from marin.evaluation.perplexity_gap import raw_text_dataset
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"


# The datasets in the Paloma eval set and their paths within the HF dataset
# https://huggingface.co/datasets/allenai/paloma
PALOMA_DATASETS_TO_DIR = {
    "4chan": "4chan_meta_sep",
    "c4_100_domains": "c4_100_domains",
    "c4_en": "c4_en",
    "dolma-v1_5": "dolma-v1_5",
    "dolma_100_programing_languages": "dolma_100_programing_languages",
    "dolma_100_subreddits": "dolma_100_subreddits",
    "falcon-refinedweb": "falcon-refinedweb",
    "gab": "gab",
    "m2d2_s2orc_unsplit": "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit": "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep": "manosphere_meta_sep",
    "mc4": "mc4",
    "ptb": "ptb",
    "redpajama": "redpajama",
    "twitterAAE_HELM_fixed": "twitterAAE_HELM_fixed",
    "wikitext_103": "wikitext_103",
}

paloma = hf_download(
    "raw/paloma",
    hf_id="allenai/paloma",
    revision="65cd6fc",
    version="2026.07.03",
    pin="raw/paloma-fc6827/65cd6fc",
)


def paloma_tokenized(
    *, base_path="tokenized/", tokenizer: str = llama3_tokenizer, paloma_raw: ArtifactStep = paloma
) -> dict[str, ArtifactStep[TokenizedCache]]:
    """
    Returns a dictionary of steps to tokenize the Paloma eval sets. Keys are the subset names (with `paloma/` prefix)
    """
    paloma_steps: dict[str, ArtifactStep[TokenizedCache]] = {}
    for dataset, path_part in PALOMA_DATASETS_TO_DIR.items():
        name = os.path.join("paloma", dataset)
        paloma_steps[name] = tokenized(
            f"{base_path}{name}",
            tokenizer=tokenizer,
            version="2026.07.03",
            raw=paloma_raw,
            glob=f"{path_part}/val/val*.jsonl.gz",
            validation=True,
        )

    return paloma_steps


def paloma_raw_validation_sets(*, paloma_raw: ArtifactStep = paloma):
    return {
        os.path.join("paloma", dataset): raw_text_dataset(
            os.path.join(paloma_raw.path(), f"{path_part}/val/val*.jsonl.gz")
        )
        for dataset, path_part in PALOMA_DATASETS_TO_DIR.items()
    }


if __name__ == "__main__":
    dataset_main(paloma_tokenized())
