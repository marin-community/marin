# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured-data PPL eval slices (issue #5059).

This module wires the higher-value first pass of structured-data perplexity
probes: byte-preserving table-to-text slices from Hugging Face-hosted table
datasets and small WebDataCommons sample archives. The goal is to surface
places where our models assign worse bits per byte on schema text, delimiters,
numeric literals, file paths, and table metadata so we can prioritize
follow-up data work.

Sources in this PR (~20-40 MB of kept text each):
    - ``GEM/totto``: Wikipedia tables paired with a one-sentence summary.
      Serialized as TSV + target sentence.
    - ``Stanford/wikitablequestions``: Wikipedia tables with Q/A pairs.
      Serialized as TSV + ``Q: ...`` / ``A: ...`` lines.
    - ``target-benchmark/gittables-corpus``: GitHub-derived relational tables.
      Serialized as TSV plus compact table metadata like CSV URL and license.
    - WebDataCommons WebTables ``sample10`` and ``sample1K``: CSV tables from
      the public sample archives, preserving original CSV text and page
      metadata provenance where available.

Later waves (separate PRs, tracked under #5059):
    PR 2 — OpenStreetMap/GeoJSON slices.
    PR 3 — Monash ``.tsf`` + UCR ``.ts`` time-series.
    PR 4 — Conditional-likelihood table eval (``loss_weights``-masked PPL)
           and pilot gap reports bucketed by span category.
"""

from __future__ import annotations

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
from marin.datakit.download.huggingface import download_hf_step
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import TokenizerStep
from marin.transform.structured_text.table_records import (
    DEFAULT_MAX_BYTES_PER_SOURCE,
    TableRecordStagingConfig,
    stage_table_record_source,
)
from marin.transform.structured_text.web_data_commons import (
    WebDataCommonsStagingConfig,
    stage_web_data_commons_source,
)

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"
LONG_TAIL_PPL_EPIC_ISSUE = 5005
STRUCTURED_TEXT_ISSUE = 5059

TOTTO_DATASET_ID = "GEM/totto"
TOTTO_REVISION = "5e745cedfd0050cc18aa143e5325d03061941d7d"
WIKITABLEQUESTIONS_DATASET_ID = "Stanford/wikitablequestions"
WIKITABLEQUESTIONS_REVISION = "fac45b3184e0ce9b79eecac454acf17e0a51f94e"
GITTABLES_DATASET_ID = "target-benchmark/gittables-corpus"
GITTABLES_REVISION = "401ebb35a14b2d6aa1135bce5d81d55e1f3cbf51"
WEBDATACOMMONS_WEBTABLES_SAMPLE10_URL = "https://webdatacommons.org/webtables/data/sample10.zip"
WEBDATACOMMONS_WEBTABLES_SAMPLE1K_URL = "https://webdatacommons.org/webtables/data/sample1K.zip"


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


totto_raw = download_hf_step(
    "raw/gem/totto",
    hf_dataset_id=TOTTO_DATASET_ID,
    revision=TOTTO_REVISION,
    hf_urls_glob=["**/*.parquet", "*.md"],
)

wikitablequestions_raw = download_hf_step(
    "raw/stanford/wikitablequestions",
    hf_dataset_id=WIKITABLEQUESTIONS_DATASET_ID,
    revision=WIKITABLEQUESTIONS_REVISION,
    hf_urls_glob=["**/*.parquet", "*.md"],
)

gittables_raw = download_hf_step(
    "raw/target-benchmark/gittables-corpus",
    hf_dataset_id=GITTABLES_DATASET_ID,
    revision=GITTABLES_REVISION,
    hf_urls_glob=["**/*.parquet", "*.md"],
)


STRUCTURED_EVAL_MANIFESTS: dict[str, IngestionSourceManifest] = {
    "totto": IngestionSourceManifest(
        dataset_key=TOTTO_DATASET_ID,
        slice_key="structured_text/totto/validation",
        source_label="totto:validation",
        source_urls=(f"https://huggingface.co/datasets/{TOTTO_DATASET_ID}",),
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
            "hf_dataset_id": TOTTO_DATASET_ID,
            "hf_revision": TOTTO_REVISION,
            "hf_urls_glob": "**/*.parquet,*.md",
        },
    ),
    "wikitablequestions": IngestionSourceManifest(
        dataset_key=WIKITABLEQUESTIONS_DATASET_ID,
        slice_key="structured_text/wikitablequestions/validation",
        source_label="wikitablequestions:validation",
        source_urls=(f"https://huggingface.co/datasets/{WIKITABLEQUESTIONS_DATASET_ID}",),
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
            "hf_dataset_id": WIKITABLEQUESTIONS_DATASET_ID,
            "hf_revision": WIKITABLEQUESTIONS_REVISION,
            "hf_urls_glob": "**/*.parquet,*.md",
        },
    ),
    "gittables": IngestionSourceManifest(
        dataset_key=GITTABLES_DATASET_ID,
        slice_key="structured_text/gittables/train",
        source_label="gittables:train",
        source_urls=("https://gittables.github.io/", f"https://huggingface.co/datasets/{GITTABLES_DATASET_ID}"),
        source_license="Mixed per-table licenses recorded in the dataset metadata; eval-only until reviewed.",
        source_format="huggingface_parquet_relational_tables",
        surface_form="tsv_table_plus_csv_url_and_license",
        policy=_eval_only_policy(
            "GitTables Hugging Face mirror of GitHub-derived relational tables with per-table license metadata."
        ),
        staging=StagingMetadata(
            transform_name="stage_table_record_source",
            serializer_name="gittables",
            split="train",
            metadata={"raw_source_type": "huggingface_parquet"},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(STRUCTURED_TEXT_ISSUE,),
        sample_caps=SampleCapConfig(max_bytes_per_source=DEFAULT_MAX_BYTES_PER_SOURCE),
        source_metadata={
            "hf_dataset_id": GITTABLES_DATASET_ID,
            "hf_revision": GITTABLES_REVISION,
            "hf_urls_glob": "**/*.parquet",
        },
    ),
    "web_data_commons_sample10": IngestionSourceManifest(
        dataset_key="webdatacommons/webtables",
        slice_key="structured_text/web_data_commons/sample10",
        source_label="webdatacommons:webtables:sample10",
        source_urls=(WEBDATACOMMONS_WEBTABLES_SAMPLE10_URL, "https://webdatacommons.org/webtables/englishTables.html"),
        source_license="WebDataCommons public WebTables sample; redistributed for research use.",
        source_format="zip_of_raw_csv_tables_with_json_page_metadata",
        surface_form="byte_preserved_csv_plus_page_metadata",
        policy=_eval_only_policy("Small public sample archive from WebDataCommons WebTables."),
        staging=StagingMetadata(
            transform_name="stage_web_data_commons_source",
            serializer_name="csv",
            preserve_header=True,
            metadata={"sample_name": "sample10"},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(STRUCTURED_TEXT_ISSUE,),
        sample_caps=SampleCapConfig(
            max_bytes_per_source=DEFAULT_MAX_BYTES_PER_SOURCE,
            max_bytes_per_document=32 * 1024,
            max_files=1,
        ),
        source_metadata={"release": "webtables-2012-english"},
    ),
    "web_data_commons_sample1k": IngestionSourceManifest(
        dataset_key="webdatacommons/webtables",
        slice_key="structured_text/web_data_commons/sample1k",
        source_label="webdatacommons:webtables:sample1k",
        source_urls=(WEBDATACOMMONS_WEBTABLES_SAMPLE1K_URL, "https://webdatacommons.org/webtables/englishTables.html"),
        source_license="WebDataCommons public WebTables sample; redistributed for research use.",
        source_format="zip_of_tarred_gzipped_csv_tables",
        surface_form="byte_preserved_csv_plus_page_metadata",
        policy=_eval_only_policy("Small public sample archive from WebDataCommons WebTables."),
        staging=StagingMetadata(
            transform_name="stage_web_data_commons_source",
            serializer_name="csv",
            preserve_header=True,
            metadata={"sample_name": "sample1K"},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(STRUCTURED_TEXT_ISSUE,),
        sample_caps=SampleCapConfig(
            max_bytes_per_source=DEFAULT_MAX_BYTES_PER_SOURCE,
            max_bytes_per_document=32 * 1024,
            max_files=1,
        ),
        source_metadata={"release": "webtables-2012-english"},
    ),
}


STRUCTURED_EVAL_SOURCES: dict[str, dict[str, StepSpec | IngestionSourceManifest]] = {
    "totto": {"raw_step": totto_raw, "manifest": STRUCTURED_EVAL_MANIFESTS["totto"]},
    "wikitablequestions": {
        "raw_step": wikitablequestions_raw,
        "manifest": STRUCTURED_EVAL_MANIFESTS["wikitablequestions"],
    },
    "gittables": {"raw_step": gittables_raw, "manifest": STRUCTURED_EVAL_MANIFESTS["gittables"]},
}


def _staged_hf_step(dataset_key: str, spec: dict[str, StepSpec | IngestionSourceManifest]) -> StepSpec:
    """Build the staging StepSpec for one HF-backed structured-eval source."""
    manifest = spec["manifest"]
    raw_step = spec["raw_step"]
    assert isinstance(manifest, IngestionSourceManifest)
    assert isinstance(raw_step, StepSpec)
    return StepSpec(
        name=f"evaluation/structured-text/{dataset_key}",
        deps=[raw_step],
        fn=lambda output_path: stage_table_record_source(
            TableRecordStagingConfig(
                input_path=raw_step.output_path,
                output_path=output_path,
                source_label=manifest.source_label,
                serializer_name=manifest.staging.serializer_name or "",
                split=manifest.staging.split or "validation",
                subset=manifest.staging.subset,
                max_bytes_per_source=manifest.sample_caps.max_bytes_per_source or DEFAULT_MAX_BYTES_PER_SOURCE,
                source_manifest=manifest,
                content_fingerprint=manifest.fingerprint(),
            )
        ),
        hash_attrs={
            "dataset_key": dataset_key,
            "manifest_fingerprint": manifest.fingerprint(),
            "serializer_name": manifest.staging.serializer_name,
            "split": manifest.staging.split,
            "subset": manifest.staging.subset,
            "max_bytes_per_source": manifest.sample_caps.max_bytes_per_source or DEFAULT_MAX_BYTES_PER_SOURCE,
        },
    )


def _staged_wdc_step(dataset_key: str, manifest: IngestionSourceManifest) -> StepSpec:
    metadata = manifest.staging.metadata
    sample_name = metadata["sample_name"]
    assert isinstance(sample_name, str)
    return StepSpec(
        name=f"evaluation/structured-text/{dataset_key}",
        fn=lambda output_path: stage_web_data_commons_source(
            WebDataCommonsStagingConfig(
                sample_url=manifest.source_urls[0],
                output_path=output_path,
                source_label=manifest.source_label,
                sample_name=sample_name,
                max_bytes_per_source=manifest.sample_caps.max_bytes_per_source or DEFAULT_MAX_BYTES_PER_SOURCE,
                max_bytes_per_document=manifest.sample_caps.max_bytes_per_document or 32 * 1024,
                preserve_header=(
                    manifest.staging.preserve_header if manifest.staging.preserve_header is not None else True
                ),
                extra_metadata={"release": "webtables-2012-english"},
                source_manifest=manifest,
                content_fingerprint=manifest.fingerprint(),
            )
        ),
        hash_attrs={
            "dataset_key": dataset_key,
            "manifest_fingerprint": manifest.fingerprint(),
            "sample_url": manifest.source_urls[0],
            "sample_name": sample_name,
            "max_bytes_per_source": manifest.sample_caps.max_bytes_per_source or DEFAULT_MAX_BYTES_PER_SOURCE,
            "max_bytes_per_document": manifest.sample_caps.max_bytes_per_document or 32 * 1024,
            "preserve_header": (
                manifest.staging.preserve_header if manifest.staging.preserve_header is not None else True
            ),
        },
    )


STRUCTURED_EVAL_STAGED: dict[str, StepSpec] = {
    key: _staged_hf_step(key, spec) for key, spec in STRUCTURED_EVAL_SOURCES.items()
}
STRUCTURED_EVAL_STAGED["web_data_commons_sample10"] = _staged_wdc_step(
    "web_data_commons_sample10", STRUCTURED_EVAL_MANIFESTS["web_data_commons_sample10"]
)
STRUCTURED_EVAL_STAGED["web_data_commons_sample1k"] = _staged_wdc_step(
    "web_data_commons_sample1k", STRUCTURED_EVAL_MANIFESTS["web_data_commons_sample1k"]
)


def _dataset_tags(dataset_key: str) -> tuple[str, ...]:
    return ("structured_text", f"epic:{LONG_TAIL_PPL_EPIC_ISSUE}", f"issue:{STRUCTURED_TEXT_ISSUE}", dataset_key)


def structured_evals_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Return staged raw-text eval datasets for perplexity-gap reports."""
    return {
        os.path.join("structured_text", key): raw_text_dataset(
            staged.as_executor_step().cd("staged.jsonl.gz"),
            tags=_dataset_tags(key),
        )
        for key, staged in STRUCTURED_EVAL_STAGED.items()
    }


def structured_evals_tokenized(
    *,
    tokenizer: str = llama3_tokenizer,
) -> dict[str, TokenizerStep]:
    """Tokenize the structured-text eval slices for a given tokenizer."""
    from experiments.defaults import default_tokenize

    steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for key, staged in STRUCTURED_EVAL_STAGED.items():
        name = os.path.join("structured_text", key)
        steps[name] = default_tokenize(
            name=name,
            dataset=staged.as_executor_step().cd("staged.jsonl.gz"),
            tokenizer=tokenizer,
            is_validation=True,
            tags=list(_dataset_tags(key)),
        )
    return steps


if __name__ == "__main__":
    executor_main(
        steps=[
            totto_raw.as_executor_step(),
            wikitablequestions_raw.as_executor_step(),
            gittables_raw.as_executor_step(),
            *[step.as_executor_step() for step in STRUCTURED_EVAL_STAGED.values()],
            *structured_evals_tokenized().values(),
        ]
    )
