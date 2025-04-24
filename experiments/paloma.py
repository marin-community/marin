"""
The Paloma eval sets, downloaded and tokenized

https://huggingface.co/datasets/allenai/paloma
"""

import os.path

# cyclic dependency
# from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import TokenizerStep
from operations.download import HfDownloadConfig, download_hf_gated_manual

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
        fn=download_hf_gated_manual,
        config=HfDownloadConfig(
            hf_dataset_id=versioned("allenai/paloma"),
            revision=versioned("65cd6fc"),
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    )
    .with_output_path("raw/paloma-fc6827")
    .cd("65cd6fc")
)


def paloma_tokenized(*, base_path="tokenized/", tokenizer: str = llama3_tokenizer) -> dict[str, TokenizerStep]:
    """
    Returns a dictionary of steps to tokenize the Paloma eval sets. Keys are the subset names (with `paloma/` prefix)
    """
    # avoid cyclic dependency
    from experiments.defaults import default_tokenize

    paloma_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset, path_part in PALOMA_DATASETS_TO_DIR.items():
        paloma_steps[os.path.join("paloma", dataset)] = default_tokenize(
            name=os.path.join("paloma", dataset),
            dataset=paloma.cd(f"{path_part}/val/val*.jsonl.gz"),
            tokenizer=tokenizer,
            is_validation=True,
        )

    return paloma_steps


if __name__ == "__main__":
    executor_main(steps=[paloma, *paloma_tokenized().values()])
