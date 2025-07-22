"""
Tokenizes the Fineweb2-HQ dataset splits.

This module defines a function that returns tokenization steps for each dataset split available in the
Fineweb2 dataset.
"""

import os.path

from levanter.store.cache import CacheOptions

from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download import DownloadConfig
from marin.download.huggingface.download_hf import download_hf
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

FINEWEB2_DATASETS = {
    "arb_Arab": ["arb_Arab/*.parquet"],
    "rus_Cyrl": ["rus_Cyrl/*.parquet"],
    "cmn_Hani": ["cmn_Hani/*.parquet"],
    "deu_Latn": ["deu_Latn/*.parquet"],
    "spa_Latn": ["spa_Latn/*.parquet"],
    "jpn_Jpan": ["jpn_Jpan/*.parquet"],
    "fra_Latn": ["fra_Latn/*.parquet"],
    "ita_Latn": ["ita_Latn/*.parquet"],
    "por_Latn": ["por_Latn/*.parquet"],
    "pol_Latn": ["pol_Latn/*.parquet"],
    "nld_Latn": ["nld_Latn/*.parquet"],
    "ind_Latn": ["ind_Latn/*.parquet"],
    "tur_Latn": ["tur_Latn/*.parquet"],
    "ces_Latn": ["ces_Latn/*.parquet"],
    "fas_Arab": ["fas_Arab/*.parquet"],
    "hun_Latn": ["hun_Latn/*.parquet"],
    "swe_Latn": ["swe_Latn/*.parquet"],
    "ell_Grek": ["ell_Grek/*.parquet"],
    "dan_Latn": ["dan_Latn/*.parquet"],
    "vie_Latn": ["vie_Latn/*.parquet"],
}

FINEWEB2_HQ_MIXTURE_WEIGHTS = {  # From https://huggingface.co/datasets/epfml/FineWeb2-HQ
    "arb_Arab": 94 / 1024,
    "rus_Cyrl": 1.2,  # TiB
    "cmn_Hani": 784 / 1024,  # in GiB
    "deu_Latn": 618 / 1024,
    "spa_Latn": 515 / 1024,
    "jpn_Jpan": 393 / 1024,
    "fra_Latn": 483 / 1024,
    "ita_Latn": 269 / 1024,
    "por_Latn": 222 / 1024,
    "pol_Latn": 168 / 1024,
    "nld_Latn": 160 / 1024,
    "ind_Latn": 125 / 1024,
    "tur_Latn": 100 / 1024,
    "ces_Latn": 104 / 1024,
    "fas_Arab": 69 / 1024,
    "hun_Latn": 79 / 1024,
    "swe_Latn": 61 / 1024,
    "ell_Grek": 84 / 1024,
    "dan_Latn": 61 / 1024,
    "vie_Latn": 59 / 1024,
}

LLAMA3_TOKENS_PER_WORD = { # https://colab.research.google.com/drive/1SAd9lg2xnD69pzRwbUgLfKpcOx1sE9rk#scrollTo=DpkLnQLW75jx
    "eng_Latn": 1.0,
    "arb_Arab": 1.86,     
    "ces_Latn": 1.88,    
    "fra_Latn": 1.6018518518518519, 
    "ell_Grek": 2.29,     
    "cmn_Hani": 1.0,      
    "por_Latn": 1.53,    
    "rus_Cyrl": 1.72,     
    "tur_Latn": 1.95,    
    "pol_Latn": 2.05,    
    "nld_Latn": 1.41,    
    "jpn_Jpan": 1.46,     
    "fas_Arab": 1.69,    
    "deu_Latn": 1.5727272727272728,  
    "spa_Latn": 1.43,     
    "ita_Latn": 1.81,    
    "ind_Latn": 1.85,     
    "dan_Latn": 1.47,     
    "swe_Latn": 1.59,     
    "hun_Latn": 1.84,     
    "vie_Latn": 1.92     
}

fineweb2_raw = ExecutorStep(
    name="raw/fineweb2_hq",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="epfml/FineWeb2-HQ",
        gcs_output_path=this_output_path(),
        revision="c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
        wait_for_completion=True,
    ),
).with_output_path("raw/fineweb2-hq")


def _get_fineweb2_split_paths(split):
    patterns = FINEWEB2_DATASETS[split]
    fineweb2_split_paths = [output_path_of(fineweb2_raw, pattern) for pattern in patterns]
    return fineweb2_split_paths


def tokenize_fineweb2hq_steps(*, base_path="tokenized/", tokenizer=llama3_tokenizer) -> dict[str, TokenizerStep]:
    steps = []
    for split in FINEWEB2_DATASETS.keys():
        fineweb2_split_output_path = os.path.join(base_path, "fineweb2_hq", split)
        fineweb2_split_paths = _get_fineweb2_split_paths(split)
        step = ExecutorStep(
            name=fineweb2_split_output_path,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=fineweb2_split_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
                cache_options=CacheOptions(num_shard_groups=256),
            ),
            pip_dependency_groups=["sentencepiece"],
        )
        steps.append(step)
    return steps


if __name__ == "__main__":
    executor_main(steps=tokenize_fineweb2hq_steps(), description="Tokenize Fineweb2-HQ dataset")
