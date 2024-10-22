"""
The Paloma eval sets, downloaded and tokenized
"""

import os.path

# cyclic dependency
# from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.tokenize import TokenizerStep
from operations.download import HfDownloadConfig, download_hf_gated_manual

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"


# The datasets in the Paloma eval set and their paths within the HF dataset
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

download_paloma = ExecutorStep(
    name="raw/paloma",
    fn=download_hf_gated_manual,
    config=HfDownloadConfig(
        hf_dataset_id=versioned("allenai/paloma"),
        revision=versioned("65cd6fc"),
        output_path=this_output_path(),
        wait_for_completion=True,
    ),
).cd("65cd6fc")


def tokenize_paloma_steps(*, base_path="tokenized/", tokenizer=llama3_tokenizer) -> dict[str, TokenizerStep]:
    paloma_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset, path_part in PALOMA_DATASETS_TO_DIR.items():
        paloma_steps[os.path.join("paloma", dataset)] = ExecutorStep(
            name=os.path.join(base_path, "paloma", dataset),
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=versioned([]),
                validation_paths=[download_paloma.cd(f"{path_part}/val/val*.jsonl.gz")],
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
        )

    return paloma_steps


if __name__ == "__main__":
    executor_main(steps=[download_paloma, *tokenize_paloma_steps().values()])
