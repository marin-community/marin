# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
The Paloma eval sets, downloaded and tokenized

https://huggingface.co/datasets/allenai/paloma
"""

import os.path

from marin.download.huggingface.download_hf import DownloadConfig as HfDownloadConfig, download_hf
from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner

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

paloma = StepSpec(
    name="raw/paloma",
    hash_attrs={"hf_dataset_id": "allenai/paloma", "revision": "65cd6fc"},
    override_output_path="raw/paloma-fc6827",
    fn=lambda output_path: download_hf(
        HfDownloadConfig(
            hf_dataset_id="allenai/paloma",
            revision="65cd6fc",
            gcs_output_path=output_path,
            wait_for_completion=True,
            append_sha_to_path=True,
        )
    ),
)

# The paloma data lives under the revision subdirectory
paloma_data_path = os.path.join(paloma.output_path, "65cd6fc")


def paloma_tokenized(*, base_path="tokenized/", tokenizer: str = llama3_tokenizer) -> dict[str, StepSpec]:
    """
    Returns a dictionary of steps to tokenize the Paloma eval sets. Keys are the subset names (with `paloma/` prefix)
    """
    # avoid cyclic dependency
    from experiments.defaults import default_tokenize

    paloma_steps: dict[str, StepSpec] = {}
    for dataset, path_part in PALOMA_DATASETS_TO_DIR.items():
        paloma_steps[os.path.join("paloma", dataset)] = default_tokenize(
            name=os.path.join("paloma", dataset),
            dataset=os.path.join(paloma_data_path, path_part, "val", "val*.jsonl.gz"),
            tokenizer=tokenizer,
            is_validation=True,
        )

    return paloma_steps


if __name__ == "__main__":
    StepRunner().run([paloma, *paloma_tokenized().values()])
