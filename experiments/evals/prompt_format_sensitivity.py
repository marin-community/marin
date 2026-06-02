# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt-format sensitivity supervised PPL slices for issue #6067."""

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
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.execution.executor import executor_main
from marin.execution.step_spec import StepSpec
from marin.transform.evaluation.prompt_format_sensitivity import (
    DEFAULT_PROMPT_FORMAT_OUTPUT_FILENAME,
    PROMPT_FORMAT_NUM_FEWSHOT,
    PROMPT_FORMAT_RENDERER_VERSION,
    PROMPT_FORMAT_TASKS,
    PROMPT_FORMAT_TEMPLATES,
    PromptFormatSensitivityStagingConfig,
    stage_prompt_format_sensitivity_source,
)

PROMPT_FORMAT_SENSITIVITY_ISSUE = 6067
PROMPT_FORMAT_SENSITIVITY_PREFIX = "prompt_format_sensitivity"
PROMPT_FORMAT_SENSITIVITY_DATASET_ID = "marin/prompt_format_sensitivity_static"


def _eval_only_policy() -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only prompt-format sensitivity probes. Do not mix into training.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: examples are intentionally held-out eval probes",
        provenance_notes=(
            "Deterministic static examples created to isolate prompt surface-form sensitivity. "
            "Every slice uses the same 5-shot support/query semantics for a task and changes only rendering."
        ),
    )


def _manifest_for_slice(task_key: str, template_key: str, heldout_count: int) -> IngestionSourceManifest:
    slice_key = f"{PROMPT_FORMAT_SENSITIVITY_PREFIX}/{task_key}/{template_key}"
    return IngestionSourceManifest(
        dataset_key=PROMPT_FORMAT_SENSITIVITY_DATASET_ID,
        slice_key=slice_key,
        source_label=slice_key,
        source_urls=(f"https://github.com/marin-community/marin/issues/{PROMPT_FORMAT_SENSITIVITY_ISSUE}",),
        source_license="Apache-2.0",
        source_format="deterministic_static_prompt_format_examples",
        surface_form=f"{PROMPT_FORMAT_NUM_FEWSHOT}shot_{template_key}_target_only",
        policy=_eval_only_policy(),
        staging=StagingMetadata(
            transform_name="stage_prompt_format_sensitivity_source",
            serializer_name=template_key,
            split="validation",
            subset=task_key,
            metadata={
                "task_key": task_key,
                "template_key": template_key,
                "renderer_version": PROMPT_FORMAT_RENDERER_VERSION,
                "num_fewshot": PROMPT_FORMAT_NUM_FEWSHOT,
            },
        ),
        issue_numbers=(PROMPT_FORMAT_SENSITIVITY_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=heldout_count),
        source_metadata={"construction": "static deterministic examples; no external download"},
    )


PROMPT_FORMAT_SENSITIVITY_SOURCE_MANIFESTS: dict[str, IngestionSourceManifest] = {
    f"{PROMPT_FORMAT_SENSITIVITY_PREFIX}/{task.key}/{template.key}": _manifest_for_slice(
        task.key,
        template.key,
        len(task.heldout_examples),
    )
    for task in PROMPT_FORMAT_TASKS
    for template in PROMPT_FORMAT_TEMPLATES
}


def _stage_step(task_key: str, template_key: str) -> StepSpec:
    dataset_key = f"{PROMPT_FORMAT_SENSITIVITY_PREFIX}/{task_key}/{template_key}"
    manifest = PROMPT_FORMAT_SENSITIVITY_SOURCE_MANIFESTS[dataset_key]
    return StepSpec(
        name=f"evaluation/{dataset_key}",
        deps=[],
        fn=lambda output_path: stage_prompt_format_sensitivity_source(
            PromptFormatSensitivityStagingConfig(
                output_path=output_path,
                task_key=task_key,
                template_key=template_key,
                source_manifest=manifest,
                content_fingerprint=manifest.fingerprint(),
            )
        ),
        hash_attrs={
            "dataset_key": dataset_key,
            "manifest_fingerprint": manifest.fingerprint(),
            "task_key": task_key,
            "template_key": template_key,
            "renderer_version": PROMPT_FORMAT_RENDERER_VERSION,
            "output_filename": DEFAULT_PROMPT_FORMAT_OUTPUT_FILENAME,
        },
    )


PROMPT_FORMAT_SENSITIVITY_STAGED: dict[str, StepSpec] = {
    dataset_key: _stage_step(
        str(manifest.staging.metadata["task_key"]),
        str(manifest.staging.metadata["template_key"]),
    )
    for dataset_key, manifest in PROMPT_FORMAT_SENSITIVITY_SOURCE_MANIFESTS.items()
}


def prompt_format_sensitivity_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Return prompt-format sensitivity slices for supervised target-only PPL scoring."""

    datasets: dict[str, RawTextEvaluationDataset] = {}
    for dataset_key, step in PROMPT_FORMAT_SENSITIVITY_STAGED.items():
        manifest = PROMPT_FORMAT_SENSITIVITY_SOURCE_MANIFESTS[dataset_key]
        metadata = manifest.staging.metadata or {}
        task_key = str(metadata["task_key"])
        template_key = str(metadata["template_key"])
        datasets[dataset_key] = supervised_text_dataset(
            step.as_executor_step().cd(DEFAULT_PROMPT_FORMAT_OUTPUT_FILENAME),
            tags=(
                PROMPT_FORMAT_SENSITIVITY_PREFIX,
                f"issue:{PROMPT_FORMAT_SENSITIVITY_ISSUE}",
                f"task:{task_key}",
                f"template:{template_key}",
                f"num_fewshot:{PROMPT_FORMAT_NUM_FEWSHOT}",
                "format:supervised_target_only",
            ),
        )
    return datasets


if __name__ == "__main__":
    executor_main(
        steps=[step.as_executor_step() for step in PROMPT_FORMAT_SENSITIVITY_STAGED.values()],
        description="Stage prompt-format sensitivity supervised PPL slices for issue #6067.",
    )
