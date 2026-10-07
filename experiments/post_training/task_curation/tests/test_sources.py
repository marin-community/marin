# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public source artifact identity and acquired input provenance."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from taskcompendium.datasets.kto_components import COMPONENTS, TRAIN_FILE
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import FilterPolicy, RawRow
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourcePipelineResult, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.sources import staged_raw_file_rows
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewTransport
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation import pipeline
from experiments.post_training.task_curation.pipeline import (
    RecordedReviewInput,
    SourceRuntime,
    SourceRuntimeConfig,
)
from experiments.post_training.task_curation.sources import rl_data_pipelines


@pytest.fixture
def source_config():
    return SourcePipelineConfig(
        mode=SourceProcessingMode.SAMPLE,
        normalized_shards=2,
        quality_policy=SourceQualityPolicy(),
        verification_policy=SourceVerificationPolicy(10, 0, 2, 0.9),
        review=ReviewConfig("fixture-model", "fixture-revision", transport=ReviewTransport.PROVIDER_BATCH),
        execution=AuditExecution(),
        filter_policy=FilterPolicy(),
    )


@pytest.fixture
def source_runtime():
    # Constructing factories is inert; these nonexistent images cannot execute a machine.
    images = {
        name: SourceRuntime("iris-gvisor", "example.org/grader@sha256:" + "a" * 64)
        for name in (
            definition.source_key for definition in rl_data_pipelines().values() if definition.source_key is not None
        )
    }
    runtime = SourceRuntimeConfig(images, "http://unused-controller.invalid")
    with runtime.campaign.activate(ZephyrContext()):
        yield runtime


@pytest.mark.parametrize("change", ["model", "policy", "image", "mode", "input", "execution_image", "transport"])
def test_source_behavior_changes_invalidate_cached_identity(source_config, source_runtime, change):
    original = rl_data_pipelines()["Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2"]
    original_step = original.bind(source_config, source_runtime)
    if change == "model":
        source_config = replace(source_config, review=replace(source_config.review, model_revision="new-revision"))
    elif change == "transport":
        source_config = replace(
            source_config, review=replace(source_config.review, transport=ReviewTransport.DIRECT_CHAT)
        )
    elif change == "policy":
        source_config = replace(source_config, quality_policy=replace(source_config.quality_policy, seed=42))
    elif change == "mode":
        source_config = replace(source_config, mode=SourceProcessingMode.FULL)
    elif change == "execution_image":
        source_config = replace(
            source_config,
            execution=replace(
                source_config.execution,
                worker_resources=ResourceConfig(cpu=1, ram="1g", image="example.org/worker@sha256:" + "b" * 64),
            ),
        )
    elif change == "input":
        source_runtime = replace(
            source_runtime,
            source_inputs={
                "Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2": ArtifactStep.adopt(
                    "staged-fixture", "2026.10.06", source="/tmp/staged-fixture"
                )
            },
        )
    else:
        source_runtime = replace(
            source_runtime,
            images={
                **source_runtime.images,
                "Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2": replace(
                    source_runtime.images["Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2"],
                    image="example.org/grader@sha256:" + "b" * 64,
                ),
            },
        )
    changed = original.bind(source_config, source_runtime)
    assert changed.fingerprint() != original_step.fingerprint()
    assert changed.name != original_step.name


@pytest.mark.parametrize("source", ["Task Trove:laion__nemotron-gym-multichallenge-advanced-v4"])
def test_private_credential_references_do_not_change_source_identity(source_config, source_runtime, source):
    declaration = next(item for item in rl_data_pipelines().values() if item.source_key == source)
    first_runtime = replace(source_runtime, verifier_secret_env={"TOGETHER_API_KEY": ("env:PRIVATE_FIRST",)})
    second_runtime = replace(source_runtime, verifier_secret_env={"TOGETHER_API_KEY": ("env:PRIVATE_SECOND",)})
    first = declaration.bind(source_config, first_runtime)
    second = declaration.bind(source_config, second_runtime)
    assert first.fingerprint() == second.fingerprint()
    assert first.name == second.name


def test_component_artifacts_share_acquisitions_and_resolve_separate_parent_roots(
    source_config, source_runtime, tmp_path, monkeypatch
):
    primary = tmp_path / "primary"
    auxiliary = tmp_path / "auxiliary"
    (primary / "data").mkdir(parents=True)
    (auxiliary / "data").mkdir(parents=True)
    parent = []
    kto = []
    for index, component in enumerate(COMPONENTS):
        history = [{"role": "system", "content": "Keep context"}, {"role": "user", "content": f"Request {index}"}]
        chosen = [*history, {"role": "assistant", "content": "Preferred"}]
        rejected = [*history, {"role": "assistant", "content": "Other"}]
        parent.append({"dataset": component, "chosen": chosen, "rejected": rejected})
        kto.extend(
            [
                {"prompt": history, "completion": chosen[-1:], "label": True},
                {"prompt": history, "completion": rejected[-1:], "label": False},
            ]
        )
    pq.write_table(pa.Table.from_pylist(parent), auxiliary / TRAIN_FILE)
    pq.write_table(pa.Table.from_pylist(kto[::-1]), primary / TRAIN_FILE)

    def normalize_selected(
        recipe,
        context,
        source_input,
        output_path,
        files,
        config,
        suite,
        *,
        previous_verification_report,
        previous_sample_path,
        canonical_source,
    ):
        # Substitute provider-bound processing while exercising actual artifact
        # path resolution, source reader/selector and native task normalization.
        selected = []
        for record in staged_raw_file_rows(source_input, TRAIN_FILE, files):
            task = recipe.policy.normalize(
                RawRow(
                    record["locator"],
                    Source(
                        dataset=recipe.source.dataset,
                        revision=recipe.source.revision,
                        row=record["locator"],
                        importer_revision=recipe.version,
                    ),
                    record["data"],
                )
            )
            assert isinstance(task, TaskSpec)
            selected.append(task.model_dump(mode="json"))
        destination = Path(output_path)
        destination.mkdir()
        (destination / "normalized.json").write_text(json.dumps(selected))
        return SourcePipelineResult(
            source_input,
            output_path,
            output_path,
            output_path,
            str(destination / "report.json"),
            output_path,
            "completed",
        )

    monkeypatch.setattr(pipeline, "run_source_pipeline", normalize_selected)
    steps = [
        rl_data_pipelines()["MarinSkyRL:kto_mix/" + component].bind(source_config, source_runtime)
        for component in COMPONENTS
    ]
    assert len({step.deps[0].fingerprint() for step in steps}) == 1
    assert len({step.deps[1].fingerprint() for step in steps}) == 1
    assert steps[0].deps[0].fingerprint() != steps[0].deps[1].fingerprint()
    for index, step in enumerate(steps):
        ctx = StepContext(
            output_path=str(tmp_path / f"result-{index}"),
            prefix=str(tmp_path),
            region=None,
            is_fingerprint=False,
            _dep_ref=lambda dep: str(primary if dep.fingerprint() == steps[0].deps[0].fingerprint() else auxiliary),
            _runtime_args={},
            _deps=step.deps,
        )
        result = step.run(step.build_config(ctx))
        tasks = [
            TaskSpec.model_validate(task) for task in json.loads((Path(result.path) / "normalized.json").read_text())
        ]
        assert len(tasks) == 2
        assert {task.source.row for task in tasks} == {f"{TRAIN_FILE}:{5-index*2}", f"{TRAIN_FILE}:{4-index*2}"}
        assert all(task.context.events[0].content == "Keep context" for task in tasks)
        assert all(
            any(resource.path == "acquisition/kto_component_provenance.json" for resource in task.resources.verifier)
            for task in tasks
        )


def test_source_variants_share_primary_blend_acquisition(source_config, source_runtime):
    registry = rl_data_pipelines()
    selection = "MarinSkyRL:nemotron_ultra_mopd/"
    pure = registry[selection + "hs3_en"].bind(source_config, source_runtime).deps[0]
    math = registry[selection + "ultra_sft_step3200_math_cot"].bind(source_config, source_runtime).deps[0]
    repository = registry[selection + "swe_pivot_len40k/SWE-Gym/SWE-Gym"].bind(source_config, source_runtime).deps[0]
    assert pure.path("/tmp/source-cache") == math.path("/tmp/source-cache") == repository.path("/tmp/source-cache")
    assert pure.fingerprint() == math.fingerprint() == repository.fingerprint()


def test_recorded_review_is_resolved_as_dependency_and_tampering_fails_before_execution(
    tmp_path, source_config, source_runtime
):
    bundle = tmp_path / "manual.json"
    bundle.write_text('{"schema_version":"recorded-review-v1"}')
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    runtime = replace(source_runtime, recorded_reviews={"MarinSkyRL:aime24": RecordedReviewInput(str(bundle), digest)})
    step = rl_data_pipelines()["MarinSkyRL:aime24"].bind(source_config, runtime)
    adopted = [dependency for dependency in step.deps if dependency.adopt_source == str(bundle)]
    assert len(adopted) == 1
    binding = step.build_config(StepContext.for_run(str(tmp_path / "result"), str(tmp_path), deps=step.deps))
    assert Path(binding.recorded_review_path).read_bytes() == bundle.read_bytes()
    bundle.write_text('{"schema_version":"changed"}')
    with pytest.raises(ValueError, match="SHA"):
        step.run(binding)
    assert not (tmp_path / "result").exists()
