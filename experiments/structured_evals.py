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

from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.datakit.download.huggingface import DownloadConfig as HfDownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import TokenizerStep
from marin.transform.structured_text.table_records import (
    DEFAULT_MAX_BYTES_PER_SOURCE,
    TableRecordStagingConfig,
    stage_table_record_source,
)

# cyclic dependency, matches the indirection in experiments/paloma.py
llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"
LONG_TAIL_PPL_EPIC_ISSUE = 5005
STRUCTURED_TEXT_ISSUE = 5059


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
        revision=versioned("5e745cedfd0050cc18aa143e5325d03061941d7d"),
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
        revision=versioned("fac45b3184e0ce9b79eecac454acf17e0a51f94e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
)


# ---------------------------------------------------------------------------
# Staging (raw HF parquet -> byte-preserving JSONL with single `text` field)
# ---------------------------------------------------------------------------


def _eval_only_policy(provenance_notes: str) -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only structured-text PPL probe. Do not mix into training without explicit follow-up review.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: direct contamination if the held-out probe slice is copied into training data",
        provenance_notes=provenance_notes,
    )


STRUCTURED_EVAL_MANIFESTS: dict[str, IngestionSourceManifest] = {
    "totto": IngestionSourceManifest(
        dataset_key="GEM/totto",
        slice_key="structured_text/totto/validation",
        source_label="totto:validation",
        source_urls=("https://huggingface.co/datasets/GEM/totto",),
        source_license="CC BY-SA 3.0",
        source_format="huggingface_parquet_table_records",
        surface_form="wikipedia_table_tsv_plus_summary_sentence",
        policy=_eval_only_policy(
            "Public Wikipedia-derived table-to-text dataset mirrored on Hugging Face; "
            "staged from a pinned dataset revision."
        ),
        staging=StagingMetadata(
            transform_name="stage_table_record_source",
            serializer_name="totto",
            split="validation",
            metadata={"raw_source_type": "huggingface_parquet"},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(STRUCTURED_TEXT_ISSUE,),
        sample_caps=SampleCapConfig(max_bytes_per_source=DEFAULT_MAX_BYTES_PER_SOURCE),
        source_metadata={
            "hf_dataset_id": "GEM/totto",
            "hf_revision": "5e745cedfd0050cc18aa143e5325d03061941d7d",
            "hf_urls_glob": "**/*.parquet,*.md",
        },
    ),
    "wikitablequestions": IngestionSourceManifest(
        dataset_key="Stanford/wikitablequestions",
        slice_key="structured_text/wikitablequestions/validation",
        source_label="wikitablequestions:validation",
        source_urls=("https://huggingface.co/datasets/Stanford/wikitablequestions",),
        source_license="CC BY 4.0",
        source_format="huggingface_parquet_table_records",
        surface_form="wikipedia_table_tsv_plus_question_answer_lines",
        policy=_eval_only_policy(
            "Public Wikipedia-table QA dataset mirrored on Hugging Face; staged from a pinned dataset revision."
        ),
        staging=StagingMetadata(
            transform_name="stage_table_record_source",
            serializer_name="wikitablequestions",
            split="validation",
            metadata={"raw_source_type": "huggingface_parquet"},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(STRUCTURED_TEXT_ISSUE,),
        sample_caps=SampleCapConfig(max_bytes_per_source=DEFAULT_MAX_BYTES_PER_SOURCE),
        source_metadata={
            "hf_dataset_id": "Stanford/wikitablequestions",
            "hf_revision": "fac45b3184e0ce9b79eecac454acf17e0a51f94e",
            "hf_urls_glob": "**/*.parquet,*.md",
        },
    ),
}


STRUCTURED_EVAL_SOURCES = {
    "totto": {"raw_step": totto_raw, "manifest": STRUCTURED_EVAL_MANIFESTS["totto"]},
    "wikitablequestions": {
        "raw_step": wikitablequestions_raw,
        "manifest": STRUCTURED_EVAL_MANIFESTS["wikitablequestions"],
    },
}


def _staged_step(dataset_key: str, spec: dict[str, ExecutorStep | IngestionSourceManifest]) -> ExecutorStep:
    """Build the staging ExecutorStep for one structured-eval source."""
    manifest = spec["manifest"]
    raw_step = spec["raw_step"]
    assert isinstance(manifest, IngestionSourceManifest)
    assert isinstance(raw_step, ExecutorStep)
    return ExecutorStep(
        name=f"evaluation/structured-text/{dataset_key}",
        fn=stage_table_record_source,
        config=TableRecordStagingConfig(
            input_path=output_path_of(raw_step),
            output_path=this_output_path(),
            source_label=manifest.source_label,
            serializer_name=manifest.staging.serializer_name or "",
            split=manifest.staging.split or "validation",
            subset=manifest.staging.subset,
            max_bytes_per_source=manifest.sample_caps.max_bytes_per_source or DEFAULT_MAX_BYTES_PER_SOURCE,
            source_manifest=manifest,
            content_fingerprint=manifest.fingerprint(),
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
