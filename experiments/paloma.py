# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
The Paloma eval sets, downloaded and tokenized

https://huggingface.co/datasets/allenai/paloma
"""

import os.path

from marin.datakit.download.huggingface import DownloadConfig as HfDownloadConfig, download_hf

# cyclic dependency
# from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, mirrored, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import TokenizerStep

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

paloma = (
    ExecutorStep(
        name="raw/paloma",
        fn=download_hf,
        config=HfDownloadConfig(
            hf_dataset_id=versioned("allenai/paloma"),
            revision=versioned("65cd6fc"),
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            append_sha_to_path=True,
        ),
    )
    .with_output_path("raw/paloma-fc6827")
    .cd("65cd6fc")
)


def paloma_tokenized(
    *, base_path="tokenized/", tokenizer: str = llama3_tokenizer, paloma_raw: ExecutorStep = paloma
) -> dict[str, TokenizerStep]:
    """
    Returns a dictionary of steps to tokenize the Paloma eval sets. Keys are the subset names (with `paloma/` prefix)

    When ``paloma_raw`` is the default global ``paloma`` step, the source paths are wrapped with
    ``mirrored()`` so that DPO/LM validation pipelines can run from any GCS region without
    re-downloading the HF raw files (the raw download was already materialized in us-central1).
    If a caller passes a custom ``paloma_raw``, the old ``.cd()`` dependency path is used so the
    executor still enforces the raw-download step.
    """
    # avoid cyclic dependency
    from experiments.defaults import default_tokenize

    use_mirror = paloma_raw is paloma

    paloma_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset, path_part in PALOMA_DATASETS_TO_DIR.items():
        if use_mirror:
            # Known-good materialized path: raw/paloma-fc6827/65cd6fc/<subset>/val/val*.jsonl.gz
            source = mirrored(f"raw/paloma-fc6827/65cd6fc/{path_part}/val/val*.jsonl.gz", budget_gb=1)
        else:
            source = paloma_raw.cd(f"{path_part}/val/val*.jsonl.gz")
        paloma_steps[os.path.join("paloma", dataset)] = default_tokenize(
            name=os.path.join("paloma", dataset),
            dataset=source,
            tokenizer=tokenizer,
            is_validation=True,
        )

    return paloma_steps


if __name__ == "__main__":
    executor_main(steps=[paloma, *paloma_tokenized().values()])
