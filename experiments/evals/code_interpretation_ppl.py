# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Code-interpretation supervised PPL slices for issue #6070."""

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
from marin.transform.evaluation.code_interpretation import (
    CODE_INTERPRETATION_NUM_FEWSHOT,
    CODE_INTERPRETATION_RENDERER_VERSION,
    CODE_INTERPRETATION_TASKS,
    CODE_INTERPRETATION_TEMPLATES,
    DEFAULT_CODE_INTERPRETATION_OUTPUT_FILENAME,
    CodeInterpretationStagingConfig,
    stage_code_interpretation_source,
)

CODE_INTERPRETATION_ISSUE = 6070
CODE_INTERPRETATION_PREFIX = "code_interpretation"
CODE_INTERPRETATION_DATASET_ID = "marin/code_interpretation_static"


def _eval_only_policy() -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only code-interpretation probes. Do not mix into training.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: examples are intentionally held-out eval probes",
        provenance_notes=(
            "Deterministic static examples created to isolate Python-like expression and helper-definition "
            "interpretation. Slices use 5-shot support and score only the held-out output continuation."
        ),
    )


def _manifest_for_slice(
    task_key: str, task_family: str, template_key: str, heldout_count: int
) -> IngestionSourceManifest:
    slice_key = f"{CODE_INTERPRETATION_PREFIX}/{task_key}/{template_key}"
    return IngestionSourceManifest(
        dataset_key=CODE_INTERPRETATION_DATASET_ID,
        slice_key=slice_key,
        source_label=slice_key,
        source_urls=(f"https://github.com/marin-community/marin/issues/{CODE_INTERPRETATION_ISSUE}",),
        source_license="Apache-2.0",
        source_format="deterministic_static_code_interpretation_examples",
        surface_form=f"{CODE_INTERPRETATION_NUM_FEWSHOT}shot_{template_key}_target_only",
        policy=_eval_only_policy(),
        staging=StagingMetadata(
            transform_name="stage_code_interpretation_source",
            serializer_name=template_key,
            split="validation",
            subset=task_key,
            metadata={
                "task_key": task_key,
                "task_family": task_family,
                "template_key": template_key,
                "renderer_version": CODE_INTERPRETATION_RENDERER_VERSION,
                "num_fewshot": CODE_INTERPRETATION_NUM_FEWSHOT,
            },
        ),
        issue_numbers=(CODE_INTERPRETATION_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=heldout_count),
        source_metadata={"construction": "static deterministic examples; no external download"},
    )


CODE_INTERPRETATION_SOURCE_MANIFESTS: dict[str, IngestionSourceManifest] = {
    f"{CODE_INTERPRETATION_PREFIX}/{task.key}/{template.key}": _manifest_for_slice(
        task.key,
        task.family,
        template.key,
        len(task.heldout_examples),
    )
    for task in CODE_INTERPRETATION_TASKS
    for template in CODE_INTERPRETATION_TEMPLATES
}


def _stage_step(task_key: str, template_key: str) -> StepSpec:
    dataset_key = f"{CODE_INTERPRETATION_PREFIX}/{task_key}/{template_key}"
    manifest = CODE_INTERPRETATION_SOURCE_MANIFESTS[dataset_key]
    return StepSpec(
        name=f"evaluation/{dataset_key}",
        deps=[],
        fn=lambda output_path: stage_code_interpretation_source(
            CodeInterpretationStagingConfig(
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
            "renderer_version": CODE_INTERPRETATION_RENDERER_VERSION,
            "output_filename": DEFAULT_CODE_INTERPRETATION_OUTPUT_FILENAME,
        },
    )


CODE_INTERPRETATION_STAGED: dict[str, StepSpec] = {
    dataset_key: _stage_step(
        str(manifest.staging.metadata["task_key"]),
        str(manifest.staging.metadata["template_key"]),
    )
    for dataset_key, manifest in CODE_INTERPRETATION_SOURCE_MANIFESTS.items()
}


def code_interpretation_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Return code-interpretation slices for supervised target-only PPL scoring."""

    datasets: dict[str, RawTextEvaluationDataset] = {}
    for dataset_key, step in CODE_INTERPRETATION_STAGED.items():
        manifest = CODE_INTERPRETATION_SOURCE_MANIFESTS[dataset_key]
        metadata = manifest.staging.metadata or {}
        task_key = str(metadata["task_key"])
        task_family = str(metadata["task_family"])
        template_key = str(metadata["template_key"])
        datasets[dataset_key] = supervised_text_dataset(
            step.as_executor_step().cd(DEFAULT_CODE_INTERPRETATION_OUTPUT_FILENAME),
            tags=(
                CODE_INTERPRETATION_PREFIX,
                f"issue:{CODE_INTERPRETATION_ISSUE}",
                f"task:{task_key}",
                f"task_family:{task_family}",
                f"template:{template_key}",
                f"num_fewshot:{CODE_INTERPRETATION_NUM_FEWSHOT}",
                "format:supervised_target_only",
            ),
        )
    return datasets


if __name__ == "__main__":
    executor_main(
        steps=[step.as_executor_step() for step in CODE_INTERPRETATION_STAGED.values()],
        description="Stage code-interpretation supervised PPL slices for issue #6070.",
    )
