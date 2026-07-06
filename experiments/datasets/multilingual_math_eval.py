# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Held-out multilingual (de/ru/zh) and math validation subsets for the tokenizer soak.

:mod:`experiments.datasets.uncheatable` covers the English + code domains of the
tokenizer bake-off. The 24h tokenizer soak (:mod:`experiments.tokenization.train_model`)
also trains on German, Russian, and Chinese Wikipedia plus FineMath
(:data:`~experiments.tokenization.train_model.SOAK_SOURCES`), so its BPB eval needs
held-out data in those domains too, disjoint from what the soak trains on:

- ``ml-de`` / ``ml-ru`` / ``ml-zh``: one parquet shard each from
  `wikimedia/wikisource <https://huggingface.co/datasets/wikimedia/wikisource>`_. Wikisource is
  a separate Wikimedia sister project — public-domain and CC-licensed primary-source texts
  (books, legal codes, translations, poetry) hosted at a different domain
  (``xx.wikisource.org``) from Wikipedia's encyclopedia articles (``xx.wikipedia.org``). It
  shares no content with the ``wikimedia/wikipedia`` dump ``SOAK_SOURCES`` trains on, for any
  snapshot date, because the two projects hold disjoint text by editorial scope, not by
  timing.
- ``math``: the `EleutherAI/hendrycks_math <https://huggingface.co/datasets/EleutherAI/hendrycks_math>`_
  test split, reusing the raw download already pinned at ``raw/hendrycks/mathhf`` (see
  ``extra_raw_downloads`` in :mod:`experiments.datasets.eval`) instead of fetching it again.
  FineMath's decontamination pass removes any 13-gram overlap between ``finemath-3plus`` (the
  corpus ``SOAK_SOURCES`` trains on for ``math``) and the GSM8k/MATH/MMLU/ARC test sets (see the
  FineMath dataset card's "Decontamination" section and its contamination report at
  ``HuggingFaceTB/finemath_contamination_report``), so the MATH test split shares no text with
  the training corpus by the training corpus's own construction.
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

# wikimedia/wikisource is published as a single snapshot (2023-12-01); pinned to its current
# (and only) revision.
_WIKISOURCE_REVISION = "f31a033f5f3d2107b3e864e578710df104a00baa"

# One parquet shard per language (~150-300MB each, ~700MB total) is ample for a held-out BPB
# eval: Wikisource is never part of soak training, so any shard is a valid validation slice.
_WIKISOURCE_SHARDS = {
    "ml-de": "20231201.de/train-00000-of-00002.parquet",
    "ml-ru": "20231201.ru/train-00000-of-00019.parquet",
    "ml-zh": "20231201.zh/train-00000-of-00011.parquet",
}

_WIKISOURCE_RAW = hf_download(
    "raw/hf/wikisource-deruzh-val",
    hf_id="wikimedia/wikisource",
    revision=_WIKISOURCE_REVISION,
    urls_glob=list(_WIKISOURCE_SHARDS.values()),
    version="2026.07.05",
)

# EleutherAI/hendrycks_math (MATH), downloading only the per-config test parquets. Declared as a
# self-contained hf_download so the eval DAG materializes it, rather than assuming another
# experiment already pinned raw/hendrycks/mathhf in this prefix.
_HENDRYCKS_MATH_RAW = hf_download(
    "raw/hf/hendrycks-math-val",
    hf_id="EleutherAI/hendrycks_math",
    revision="21a5633",
    urls_glob=["**/test-*.parquet"],
    version="2026.07.05",
)
_HENDRYCKS_MATH_TEST_GLOB = "*/test-00000-of-00001.parquet"


def multilingual_math_validation(arm_name: str, tokenizer: str) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Held-out ml-de/ml-ru/ml-zh (Wikisource) + math (MATH test) validation, tokenized with ``tokenizer``.

    Named per arm for the same reason as
    :func:`experiments.tokenization.proxy_ladder.bakeoff_validation`: the artifact store
    adopts a cache by name@version, so a shared name would silently reuse another tokenizer's
    tokens.
    """
    datasets = {
        subset: tokenized(
            f"bakeoff-val/{subset}-{arm_name}",
            tokenizer=tokenizer,
            version="2026.07.05",
            raw=_WIKISOURCE_RAW,
            glob=glob,
            validation=True,
        )
        for subset, glob in _WIKISOURCE_SHARDS.items()
    }
    datasets["math"] = tokenized(
        f"bakeoff-val/math-{arm_name}",
        tokenizer=tokenizer,
        version="2026.07.05",
        raw=_HENDRYCKS_MATH_RAW,
        glob=_HENDRYCKS_MATH_TEST_GLOB,
        text_key="solution",
        validation=True,
    )
    return datasets


if __name__ == "__main__":
    dataset_main(multilingual_math_validation("llama3", llama3_tokenizer))
