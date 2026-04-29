# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal raw PPL bridge for selected LM-eval task datasets.

This first pass keeps scope intentionally narrow: stage the exact prompt/answer
surface for a few high-value public datasets so they can be scored in the raw
perplexity-gap pipeline without pulling in the full LM-eval harness.
"""

from __future__ import annotations


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
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.transform.evaluation.raw_lm_eval import (
    GSM8K_COT_DEFAULT_NUM_FEWSHOT,
    MMLU_DEFAULT_FEWSHOT_SPLIT,
    MMLU_DEFAULT_NUM_FEWSHOT,
    LmEvalRawRenderer,
    LmEvalRawStagingConfig,
    stage_lm_eval_source,
)

LONG_TAIL_PPL_EPIC_ISSUE = 5005
LM_EVAL_BRIDGE_ISSUE = 5053

MMLU_DATASET_ID = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def _eval_only_policy(provenance_notes: str) -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only LM-eval raw-text bridge. Do not mix into training without explicit follow-up review.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: direct contamination if benchmark text is copied into training data",
        provenance_notes=provenance_notes,
    )


mmlu_raw = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download_hf,
    config=HfDownloadConfig(
        hf_dataset_id=versioned(MMLU_DATASET_ID),
        revision=versioned(MMLU_REVISION),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
)

gsm8k_raw = ExecutorStep(
    name="raw/openai/gsm8k",
    fn=download_hf,
    config=HfDownloadConfig(
        hf_dataset_id=versioned(GSM8K_DATASET_ID),
        revision=versioned(GSM8K_REVISION),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
)


LM_EVAL_SOURCE_MANIFESTS: dict[str, IngestionSourceManifest] = {
    "lm_eval/mmlu_auxiliary_train": IngestionSourceManifest(
        dataset_key=MMLU_DATASET_ID,
        slice_key="lm_eval/mmlu_auxiliary_train",
        source_label="mmlu:all:auxiliary_train",
        source_urls=(f"https://huggingface.co/datasets/{MMLU_DATASET_ID}",),
        source_license="MIT",
        source_format="huggingface_parquet_multiple_choice",
        surface_form="multiple_choice_prompt_plus_correct_answer",
        policy=_eval_only_policy("MMLU public dataset mirrored on Hugging Face; staged from a pinned revision."),
        staging=StagingMetadata(
            transform_name="stage_lm_eval_source",
            serializer_name=LmEvalRawRenderer.MMLU.value,
            split="auxiliary_train",
            subset="all",
            metadata={"num_fewshot": MMLU_DEFAULT_NUM_FEWSHOT, "fewshot_split": MMLU_DEFAULT_FEWSHOT_SPLIT},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(LM_EVAL_BRIDGE_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=99_842),
        source_metadata={"hf_revision": MMLU_REVISION},
    ),
    "lm_eval/mmlu_dev": IngestionSourceManifest(
        dataset_key=MMLU_DATASET_ID,
        slice_key="lm_eval/mmlu_dev",
        source_label="mmlu:all:dev",
        source_urls=(f"https://huggingface.co/datasets/{MMLU_DATASET_ID}",),
        source_license="MIT",
        source_format="huggingface_parquet_multiple_choice",
        surface_form="multiple_choice_prompt_plus_correct_answer",
        policy=_eval_only_policy("MMLU public dataset mirrored on Hugging Face; staged from a pinned revision."),
        staging=StagingMetadata(
            transform_name="stage_lm_eval_source",
            serializer_name=LmEvalRawRenderer.MMLU.value,
            split="dev",
            subset="all",
            metadata={"num_fewshot": MMLU_DEFAULT_NUM_FEWSHOT, "fewshot_split": MMLU_DEFAULT_FEWSHOT_SPLIT},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(LM_EVAL_BRIDGE_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=285),
        source_metadata={"hf_revision": MMLU_REVISION},
    ),
    "lm_eval/gsm8k_train": IngestionSourceManifest(
        dataset_key=GSM8K_DATASET_ID,
        slice_key="lm_eval/gsm8k_train",
        source_label="gsm8k:main:train",
        source_urls=(f"https://huggingface.co/datasets/{GSM8K_DATASET_ID}",),
        source_license="MIT",
        source_format="huggingface_parquet_math_qa",
        surface_form="question_plus_worked_answer",
        policy=_eval_only_policy("GSM8K public dataset mirrored on Hugging Face; staged from a pinned revision."),
        staging=StagingMetadata(
            transform_name="stage_lm_eval_source",
            serializer_name=LmEvalRawRenderer.GSM8K.value,
            split="train",
            subset="main",
            metadata={"num_fewshot": GSM8K_COT_DEFAULT_NUM_FEWSHOT},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(LM_EVAL_BRIDGE_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=7_473),
        source_metadata={"hf_revision": GSM8K_REVISION},
    ),
}


def _stage_step(dataset_key: str, raw_step: ExecutorStep, manifest: IngestionSourceManifest) -> ExecutorStep:
    return ExecutorStep(
        name=f"evaluation/{dataset_key}",
        fn=stage_lm_eval_source,
        config=LmEvalRawStagingConfig(
            input_path=output_path_of(raw_step),
            output_path=this_output_path(),
            source_label=manifest.source_label,
            renderer_name=LmEvalRawRenderer(manifest.staging.serializer_name or ""),
            split=manifest.staging.split or "train",
            subset=manifest.staging.subset,
            max_examples=manifest.sample_caps.max_examples,
            num_fewshot=int((manifest.staging.metadata or {}).get("num_fewshot", 0)),
            fewshot_split=(manifest.staging.metadata or {}).get("fewshot_split"),
            source_manifest=manifest,
            content_fingerprint=manifest.fingerprint(),
        ),
    )


LM_EVAL_STAGED: dict[str, ExecutorStep] = {
    "lm_eval/mmlu_auxiliary_train": _stage_step(
        "lm_eval/mmlu_auxiliary_train",
        mmlu_raw,
        LM_EVAL_SOURCE_MANIFESTS["lm_eval/mmlu_auxiliary_train"],
    ),
    "lm_eval/mmlu_dev": _stage_step(
        "lm_eval/mmlu_dev",
        mmlu_raw,
        LM_EVAL_SOURCE_MANIFESTS["lm_eval/mmlu_dev"],
    ),
    "lm_eval/gsm8k_train": _stage_step(
        "lm_eval/gsm8k_train",
        gsm8k_raw,
        LM_EVAL_SOURCE_MANIFESTS["lm_eval/gsm8k_train"],
    ),
}


def lm_eval_bridge_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Return the staged raw-text bridge datasets for perplexity-gap reports."""
    return {
        key: raw_text_dataset(
            step.cd("staged.jsonl.gz"),
            tags=("lm_eval_bridge", f"epic:{LONG_TAIL_PPL_EPIC_ISSUE}", f"issue:{LM_EVAL_BRIDGE_ISSUE}", key),
        )
        for key, step in LM_EVAL_STAGED.items()
    }


if __name__ == "__main__":
    executor_main(steps=[mmlu_raw, gsm8k_raw, *LM_EVAL_STAGED.values()])
