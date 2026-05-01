# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
FineWeb2 multilingual data and held-out eval bundles.

The eval bundle tokenizes FineWeb2's per-language ``test`` split directly from Hugging Face parquet files. This avoids
downloading the full train split while still making held-out documents available as Levanter validation caches.
"""

import os.path
from collections.abc import Sequence
from typing import Literal

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import TokenizerStep

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

FINEWEB2_DATASET_ID = "HuggingFaceFW/fineweb-2"
FINEWEB2_PARQUET_REVISION = "345aeeb34ec379862323beb9b5530d9e7f94522d"
FineWeb2Split = Literal["train", "test"]
LevanterCacheSplit = Literal["train", "validation"]
FINEWEB2_EVAL_SPLIT: FineWeb2Split = "test"

# Top 50 configs by total row count from the Hugging Face Dataset Viewer /size endpoint for
# HuggingFaceFW/fineweb-2 source revision af9c13333eb981300149d5ca60a8e9d659b276b9.
FINEWEB2_TOP_50_BY_ROWS = (
    "rus_Cyrl",
    "cmn_Hani",
    "deu_Latn",
    "jpn_Jpan",
    "fra_Latn",
    "ita_Latn",
    "por_Latn",
    "pol_Latn",
    "nld_Latn",
    "ind_Latn",
    "ces_Latn",
    "arb_Arab",
    "vie_Latn",
    "kor_Hang",
    "swe_Latn",
    "fas_Arab",
    "ron_Latn",
    "ukr_Cyrl",
    "hun_Latn",
    "ell_Grek",
    "dan_Latn",
    "nob_Latn",
    "fin_Latn",
    "tha_Thai",
    "slk_Latn",
    "bul_Cyrl",
    "hin_Deva",
    "bos_Latn",
    "cat_Latn",
    "ben_Beng",
    "heb_Hebr",
    "lit_Latn",
    "slv_Latn",
    "ekk_Latn",
    "zsm_Latn",
    "als_Latn",
    "lvs_Latn",
    "azj_Latn",
    "hrv_Latn",
    "tam_Taml",
    "npi_Deva",
    "urd_Arab",
    "mkd_Cyrl",
    "srp_Cyrl",
    "mar_Deva",
    "kat_Geor",
    "kaz_Cyrl",
    "mal_Mlym",
    "isl_Latn",
    "glg_Latn",
)

# Native-script South Asian/Indic configs available in FineWeb2, including every config written in an Indic script.
# Romanized variants are deliberately omitted so the supplement tracks the primary written form of each language.
FINEWEB2_INDIC_LANGUAGE_CONFIGS = (
    "anp_Deva",
    "asm_Beng",
    "awa_Deva",
    "ben_Beng",
    "bho_Deva",
    "bpy_Beng",
    "brx_Deva",
    "div_Thaa",
    "doi_Deva",
    "gom_Deva",
    "grt_Beng",
    "guj_Gujr",
    "hin_Deva",
    "hne_Deva",
    "kan_Knda",
    "kas_Arab",
    "kas_Deva",
    "kle_Deva",
    "lif_Deva",
    "mag_Deva",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Beng",
    "mni_Mtei",
    "mup_Deva",
    "new_Deva",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "rav_Deva",
    "san_Deva",
    "sat_Olck",
    "sck_Deva",
    "sin_Sinh",
    "skr_Arab",
    "snd_Arab",
    "snd_Deva",
    "suz_Deva",
    "taj_Deva",
    "tam_Taml",
    "tcy_Knda",
    "tel_Telu",
    "thl_Deva",
    "urd_Arab",
    "xsr_Deva",
)

FINEWEB2_MULTILINGUAL_EVAL_CONFIGS = tuple(dict.fromkeys((*FINEWEB2_TOP_50_BY_ROWS, *FINEWEB2_INDIC_LANGUAGE_CONFIGS)))

_FINEWEB2_TOP_50_BY_ROWS_SET = frozenset(FINEWEB2_TOP_50_BY_ROWS)
_FINEWEB2_INDIC_LANGUAGE_CONFIGS_SET = frozenset(FINEWEB2_INDIC_LANGUAGE_CONFIGS)


def fineweb2_multilingual_parquet_pattern(config: str, split: FineWeb2Split) -> str:
    """Return the pinned Hugging Face parquet pattern for a FineWeb2 language config split."""
    return f"hf://datasets/{FINEWEB2_DATASET_ID}@{FINEWEB2_PARQUET_REVISION}/{config}/{split}/*.parquet"


def fineweb2_multilingual_tags(config: str) -> list[str]:
    """Return Levanter eval tags for aggregate multilingual, script, language, and subset metrics."""
    assert "_" in config, f"Expected FineWeb2 config in lang_Script form, got {config!r}"
    language, script = config.rsplit("_", maxsplit=1)
    tags = [
        "fineweb2_multilingual",
        f"fineweb2_multilingual/script/{script}",
        f"fineweb2_multilingual/language/{language}",
    ]
    if config in _FINEWEB2_TOP_50_BY_ROWS_SET:
        tags.append("fineweb2_multilingual/top_50_by_rows")
    if config in _FINEWEB2_INDIC_LANGUAGE_CONFIGS_SET:
        tags.append("fineweb2_multilingual/indic")
    return tags


def fineweb2_multilingual_tokenized(
    *,
    split: FineWeb2Split,
    configs: Sequence[str] = FINEWEB2_MULTILINGUAL_EVAL_CONFIGS,
    cache_split: LevanterCacheSplit = "train",
    name_prefix: str | None = None,
    tokenizer: str = llama3_tokenizer,
) -> dict[str, TokenizerStep]:
    """Return tokenization steps for selected FineWeb2 multilingual configs and split."""
    steps: dict[str, TokenizerStep] = {}
    if name_prefix is None:
        name_prefix = os.path.join("fineweb2_multilingual", split)
    for config in configs:
        name = os.path.join(name_prefix, config)
        steps[name] = default_tokenize(
            name=name,
            dataset=fineweb2_multilingual_parquet_pattern(config, split),
            tokenizer=tokenizer,
            is_validation=cache_split == "validation",
            tags=fineweb2_multilingual_tags(config),
        )
    return steps


def fineweb2_multilingual_eval_bundle(*, tokenizer: str = llama3_tokenizer) -> dict[str, TokenizerStep]:
    """Return the opt-in tokenization bundle for selected FineWeb2 multilingual held-out eval configs."""
    return fineweb2_multilingual_tokenized(
        split=FINEWEB2_EVAL_SPLIT,
        cache_split="validation",
        name_prefix="fineweb2_multilingual_eval",
        tokenizer=tokenizer,
    )


def fineweb2_multilingual_raw_validation_sets(
    *,
    configs: Sequence[str] = FINEWEB2_MULTILINGUAL_EVAL_CONFIGS,
    name_prefix: str = "fineweb2_multilingual",
) -> dict[str, RawTextEvaluationDataset]:
    """Return raw FineWeb2 multilingual held-out eval sets for perplexity-gap reports."""
    return {
        os.path.join(name_prefix, config): raw_text_dataset(
            fineweb2_multilingual_parquet_pattern(config, FINEWEB2_EVAL_SPLIT),
            tags=tuple(fineweb2_multilingual_tags(config)),
        )
        for config in configs
    }


if __name__ == "__main__":
    executor_main(
        steps=list(fineweb2_multilingual_eval_bundle().values()),
        description="Tokenize FineWeb2 multilingual held-out eval sets",
    )
