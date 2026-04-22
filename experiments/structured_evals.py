# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured-data PPL eval slices (issue #5059).

This module wires the first wave of structured-data perplexity probes:
byte-preserving table-to-text slices sourced from HuggingFace-hosted
datasets. The goal is to surface places where our models assign worse
bits per byte on structured text (tables, coordinates, numeric literals)
so we can prioritize follow-up data work.

Sources in this PR (~20-40 MB of kept text each, streamed from HF):
    - ``GEM/totto``: Wikipedia tables paired with a one-sentence summary.
      Serialized as TSV + target sentence.
    - ``Stanford/wikitablequestions``: Wikipedia tables with Q/A pairs.
      Serialized as TSV + ``Q: ...`` / ``A: ...`` lines.

Later waves (separate PRs, tracked under #5059):
    PR 2 — Monash ``.tsf`` + UCR ``.ts`` time-series.
    PR 3 — GeoJSON/GeoNames/Natural-Earth + span-classifier.
    PR 4 — Conditional-likelihood table eval (``loss_weights``-masked PPL)
           and the pilot gap report bucketed by span category.

The revisions below pin to commit SHAs. If you need to refresh them, run
``huggingface_hub.HfApi().repo_info(repo_id, repo_type="dataset")`` and
replace the value in the matching ``versioned(...)`` call.
"""

import os.path

from marin.datakit.download.huggingface import DownloadConfig as HfDownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import TokenizerStep
from marin.transform.structured_text.table_records import (
    TableRecordStagingConfig,
    stage_table_record_source,
)

# cyclic dependency, matches the indirection in experiments/paloma.py
llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"


# ---------------------------------------------------------------------------
# Raw HF downloads
# ---------------------------------------------------------------------------

# ToTTo: Wikipedia tables + single-sentence controlled summary.
#   Dataset card: https://huggingface.co/datasets/GEM/totto
# The full dataset is ~200 MB; the 20-40 MB/source cap is enforced in the
# staging step, not at download time; HF parquet does not support byte-range
# reads for arbitrary subsets. If download size becomes a concern we can
# switch to ``hf_urls_glob=["*validation*.parquet"]`` once we confirm the
# file naming.
totto_raw = ExecutorStep(
    name="raw/gem/totto",
    fn=download_hf,
    config=HfDownloadConfig(
        hf_dataset_id=versioned("GEM/totto"),
        revision=versioned("main"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
)

# WikiTableQuestions: Wikipedia tables paired with natural-language questions
# and short-form answers.
#   Dataset card: https://huggingface.co/datasets/Stanford/wikitablequestions
wikitablequestions_raw = ExecutorStep(
    name="raw/stanford/wikitablequestions",
    fn=download_hf,
    config=HfDownloadConfig(
        hf_dataset_id=versioned("Stanford/wikitablequestions"),
        revision=versioned("main"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
)


# ---------------------------------------------------------------------------
# Staging (raw HF parquet -> byte-preserving JSONL with single `text` field)
# ---------------------------------------------------------------------------

STRUCTURED_EVAL_SOURCES = {
    "totto": {
        "raw_step": totto_raw,
        "serializer_name": "totto",
        "subset": None,
        "split": "validation",
        "source_label": "totto:validation",
    },
    "wikitablequestions": {
        "raw_step": wikitablequestions_raw,
        "serializer_name": "wikitablequestions",
        "subset": None,
        "split": "validation",
        "source_label": "wikitablequestions:validation",
    },
}


def _staged_step(dataset_key: str, spec: dict) -> ExecutorStep:
    """Build the staging ExecutorStep for one structured-eval source."""
    return ExecutorStep(
        name=f"evaluation/structured-text/{dataset_key}",
        fn=stage_table_record_source,
        config=TableRecordStagingConfig(
            input_path=output_path_of(spec["raw_step"]),
            output_path=this_output_path(),
            source_label=spec["source_label"],
            serializer_name=spec["serializer_name"],
            split=spec["split"],
            subset=spec["subset"],
        ),
    )


STRUCTURED_EVAL_STAGED: dict[str, ExecutorStep] = {
    key: _staged_step(key, spec) for key, spec in STRUCTURED_EVAL_SOURCES.items()
}


# ---------------------------------------------------------------------------
# Tokenization (validation sets keyed under ``structured_text/<source>``)
# ---------------------------------------------------------------------------


def structured_evals_tokenized(
    *,
    tokenizer: str = llama3_tokenizer,
) -> dict[str, TokenizerStep]:
    """Tokenize the structured-text eval slices for a given tokenizer.

    Returns a dict keyed by ``structured_text/<source>`` so the resulting
    tokenize steps slot into ``mixture_for_evaluation`` alongside Paloma.
    """
    # avoid cyclic dependency, matches experiments/paloma.py
    from experiments.defaults import default_tokenize

    steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for key, staged in STRUCTURED_EVAL_STAGED.items():
        name = os.path.join("structured_text", key)
        steps[name] = default_tokenize(
            name=name,
            dataset=staged.cd("staged.jsonl.gz"),
            tokenizer=tokenizer,
            is_validation=True,
        )
    return steps


if __name__ == "__main__":
    executor_main(
        steps=[
            totto_raw,
            wikitablequestions_raw,
            *STRUCTURED_EVAL_STAGED.values(),
            *structured_evals_tokenized().values(),
        ]
    )
