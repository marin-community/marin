# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import threading
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.iris.machine import IrisMachineFactory
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.models import TaskSpec, TextMessage, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.jsonl_normalization import normalized_jsonl_path
from taskcompendium.pipeline.models import Controls, FilterPolicy, ReviewRubric
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode
from verifyit.spec import ExactSpec
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation import pipeline as pipeline_module
from experiments.post_training.task_curation.campaign import CampaignRuntime, PipelineResult, error_chain
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions, RecipeSettings
from experiments.post_training.task_curation.datasets.nemotron_ultra.components import prepare_blend
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import (
    built_environment,
    environment_artifact,
)
from experiments.post_training.task_curation.pipeline import (
    CampaignMachines,
    CurationRecipe,
    DownloadRequest,
    HfSource,
    UrlSource,
    download_source,
    download_step,
    environment_requirements,
    process_rows,
    source_recipe,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.tests.image_builds import (
    REPOSITORY,
    install_fake_build_tools,
    tracked_lock,
)

AGENT_IMAGE = "ghcr.io/marin-community/iris-task@sha256:" + "d" * 64


def math500() -> CurationRecipe:
    return cast(CurationRecipe, next(source.config for source in skyrl_math.sources() if source.name == "math500"))


@pytest.fixture
def config() -> SourcePipelineConfig:
    return SourcePipelineConfig(
        mode=SourceProcessingMode.SAMPLE,
        quality_policy=SourceQualityPolicy(),
        verification_policy=SourceVerificationPolicy(10, 0, 2, 0.9),
        review=ReviewConfig("fixture-model", "fixture-revision", mode=ReviewMode.BATCH),
        execution=AuditExecution(),
        filter_policy=FilterPolicy(),
        normalized_shards=2,
        machines=CampaignMachines(),
    )


def source_step(recipe: CurationRecipe, options: PipelineOptions):
    source = RlDataSource(
        info=SourceInfo(id=f"fixture:{recipe.name}", title=recipe.name, origin="fixture"),
        config=recipe,
        pipeline=process_rows,
    )
    return process_rows(source, options)


def step_name(pipeline, config) -> str:
    return source_step(
        pipeline, PipelineOptions(config.mode, CampaignRuntime(), recipe_settings=RecipeSettings(config=config))
    ).name


@pytest.mark.parametrize(
    "change",
    [
        partial(replace, version="2"),
        partial(replace, rubric="Reject problems whose reference answer is wrong."),
        partial(replace, controls=None),
    ],
)
def test_declaration_changes_rename_the_artifact(config, change):
    assert step_name(change(math500()), config) != step_name(math500(), config)


def test_review_settings_enter_identity_only_with_a_rubric(config):
    revised = replace(config, review=replace(config.review, model_revision="other-revision"))
    assert step_name(math500(), revised) != step_name(math500(), config)
    unreviewed = replace(math500(), rubric=None)
    assert step_name(unreviewed, revised) == step_name(unreviewed, config)


def test_image_factory_enters_identity_only_with_controls(config):
    iris = replace(config, machines=CampaignMachines(IrisMachineFactory(controller_url="http://controller.invalid")))
    assert step_name(math500(), iris) != step_name(math500(), config)
    unchecked = replace(math500(), controls=None)
    assert step_name(unchecked, iris) == step_name(unchecked, config)


@pytest.fixture
def grader_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    install_fake_build_tools(tmp_path, monkeypatch)
    return tracked_lock(tmp_path)


def test_a_changed_grader_environment_renames_the_artifact(grader_lock, config):
    grader = Environment(lock=grader_lock)
    pipeline = replace(math500(), grader=grader)
    run(environment_artifact(grader, REPOSITORY))
    original = source_step(
        pipeline, PipelineOptions(config.mode, CampaignRuntime(), recipe_settings=RecipeSettings(config=config))
    )
    assert environment_artifact(grader).name in [dep.name for dep in original.deps]
    grader_lock.write_text("numpy==2.3.4\n")
    run(environment_artifact(grader, REPOSITORY))
    assert step_name(pipeline, config) != original.name


def test_apt_packages_beyond_the_worker_image_run_in_a_sandbox_of_the_built_image(grader_lock):
    environment = Environment(lock=grader_lock, apt=("build-essential", "jq"))
    (built,) = run(environment_artifact(environment, REPOSITORY))
    requirements = environment_requirements(environment, built_environment(environment))
    assert built.image is not None and built.image.startswith(f"{REPOSITORY}@sha256:")
    assert requirements.docker_image == built.image
    assert requirements.packages_lock is None


@pytest.mark.parametrize(
    "declare",
    [
        pytest.param(lambda lock: Environment(lock=lock, apt=("build-essential", "git")), id="worker-image-apt"),
        pytest.param(lambda lock: Environment(lock=lock), id="lock-only"),
        pytest.param(lambda lock: Environment(pypi=("numpy==2.3.5",)), id="pypi-only"),
    ],
)
def test_environments_the_worker_image_covers_run_in_the_worker_from_their_lock(grader_lock, declare):
    environment = declare(grader_lock)
    (built,) = run(environment_artifact(environment, REPOSITORY))
    requirements = environment_requirements(environment, built_environment(environment))
    assert requirements.docker_image is None
    assert requirements.packages_lock == built.lock_url
    assert hashlib.sha256(StoragePath(requirements.packages_lock).read_bytes()).hexdigest() == built.lock_sha256


@pytest.mark.parametrize(
    "declare",
    [
        pytest.param(lambda lock: Environment(pypi=("numpy==2.3.5",), lock=lock), id="pypi-and-lock"),
        pytest.param(lambda lock: Environment(pypi=("numpy>=2",)), id="unpinned-pypi"),
        pytest.param(lambda lock: Environment(image="ghcr.io/marin-community/iris-task:latest"), id="unpinned-image"),
        pytest.param(lambda lock: Environment(image=AGENT_IMAGE, lock=lock), id="image-with-packages"),
        pytest.param(lambda lock: Environment(lock=lock, data=("punkt_tab",)), id="data-without-a-downloader"),
        pytest.param(lambda lock: Environment(lock=lock, apt=("jq; rm -rf /",)), id="apt-not-a-package-name"),
    ],
)
def test_environment_declarations_reject_contradictory_or_unpinned_needs(grader_lock, declare):
    with pytest.raises(ValueError):
        declare(grader_lock)


def test_rubric_paragraphs_become_review_criteria():
    pipeline = replace(math500(), rubric="\nFirst criterion\nspans two lines.\n\nSecond criterion.\n")
    assert source_recipe(pipeline, {}, None).rubric == ReviewRubric(
        id="math500", version="1", criteria=("First criterion spans two lines.", "Second criterion.")
    )


def test_declarations_with_the_same_pinned_files_share_one_download():
    pipeline = math500()
    selected = replace(pipeline.source, select=lambda row, context: True)
    assert download_step(selected, CampaignRuntime()).name == download_step(pipeline.source, CampaignRuntime()).name


def convert_fixture(row, _context):
    return exact_answer_task(row, prompt=row.data["prompt"], answers=(row.data["answer"],), ignore_case=False)


@pytest.mark.parametrize("prepared", [False, True])
def test_recipe_result_omits_skipped_review_and_verification_stages(tmp_path, config, monkeypatch, prepared):
    primary = tmp_path / "source"
    primary.mkdir()
    (primary / "rows.jsonl").write_text('{"dataset": "fixture", "prompt": "Two plus two?", "answer": "4"}\n')
    pipeline = replace(
        math500(),
        source=HfSource(
            "fixture/questions",
            "a" * 40,
            ("rows.jsonl",),
            SourceFormat.JSONL,
            prepare=prepare_blend if prepared else None,
            prepared_component="fixture" if prepared else None,
        ),
        convert=convert_fixture,
        rubric=None,
        controls=None,
        grader=None,
    )
    runtime = CampaignRuntime()
    options = PipelineOptions(
        SourceProcessingMode.SAMPLE,
        runtime,
        recipe_settings=RecipeSettings(config=replace(config, review=None, execution=AuditExecution(), machines=None)),
    )
    step = source_step(pipeline, options)
    # Substitute only the HF transfer boundary; materialize the real declared dependency.
    original_plan = pipeline_module.plan_download

    def local_plan(request):
        return original_plan(replace(request, source_url_override=str(primary)))

    monkeypatch.setattr(pipeline_module, "plan_download", local_plan)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            artifact = run(step, max_concurrent=1)[0]
        result = cast(PipelineResult, artifact.result)
        assert result.stages == ("download", "normalize", "final")
        rows = [
            row
            for shard in (StoragePath(result.outputs["final"]) / "*.parquet").glob()
            for row in load_parquet(str(shard))
        ]
        assert [row["source_row"] for row in rows] == ["rows.jsonl:0"]
        if prepared:
            (staged,) = (StoragePath(str(tmp_path / "artifacts")) / "task-curation/download/**/rows.jsonl").glob()
            records = list(load_parquet(str(normalized_jsonl_path(staged))))
            assert records == [
                {"component": "fixture", "source_index": 0, "raw_json": (primary / "rows.jsonl").read_bytes()}
            ]
        review = json.loads((StoragePath(result.evidence["review"]) / "report.json").read_text())
        verification = json.loads((StoragePath(result.evidence["verify"]) / "report.json").read_text())
        assert review["status"] == "unreviewed"
        assert verification["status"] == "skipped"
    finally:
        client.shutdown()


def decode_auxiliary_answer(row, context):
    return {**row, "answer": (context.inputs["answers"] / "answer.txt").read_text()}


@pytest.mark.parametrize("mode", [SourceProcessingMode.SAMPLE, SourceProcessingMode.FULL])
@pytest.mark.parametrize("primary_override", ["root", "file"])
@pytest.mark.parametrize("prepared", [False, True])
def test_reviewed_local_inputs_keep_controls_and_invalidate_changed_bytes(
    tmp_path, config, monkeypatch, mode, primary_override, prepared
):
    primary = tmp_path / "input"
    primary.mkdir()
    file = primary / ("rows.jsonl" if primary_override == "root" else "sample.jsonl")
    file.write_text(
        '{"dataset":"fixture","prompt":"First question"}\n' '{"dataset":"fixture","prompt":"Second question"}\n'
    )
    if prepared:
        prepare_blend(str(file))
    auxiliary = tmp_path / "answers"
    auxiliary.mkdir()
    answer = auxiliary / "answer.txt"
    answer.write_text("two")
    recipe = replace(
        math500(),
        name="local-inputs",
        source=HfSource(
            "fixture/questions",
            "a" * 40,
            ("rows.jsonl",),
            SourceFormat.JSONL,
            decode=decode_auxiliary_answer,
            prepare=prepare_blend if prepared else None,
            prepared_component="fixture" if prepared else None,
        ),
        inputs={"answers": HfSource("fixture/answers", "a" * 40, ("answer.txt",), SourceFormat.JSONL)},
        convert=convert_fixture,
        rubric=None,
        controls=Controls(golden=reference_reply),
        grader=None,
    )
    overrides = InputOverrides(
        root=str(primary) if primary_override == "root" else None,
        files={"rows.jsonl": file} if primary_override == "file" else {},
        auxiliary={"answers": str(auxiliary)},
    )
    runtime = CampaignRuntime()
    options = PipelineOptions(
        mode,
        runtime,
        inputs=overrides,
        recipe_settings=RecipeSettings(
            config=replace(
                config,
                review=None,
                machines=None,
                quality_policy=SourceQualityPolicy(sample_size=1),
                verification_policy=SourceVerificationPolicy(10, 0, 1, 1),
            )
        ),
    )
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    original = source_step(recipe, options)
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            first = run(original, max_concurrent=1)[0]
            file.write_text(
                '{"dataset":"fixture","prompt":"Updated first question"}\n'
                '{"dataset":"fixture","prompt":"Updated second question"}\n'
            )
            changed_primary = source_step(recipe, options)
            assert changed_primary.name != original.name
            answer.write_text("updated answer")
            changed_auxiliary = source_step(recipe, options)
            assert changed_auxiliary.name != changed_primary.name
            second = run(changed_auxiliary, max_concurrent=1)[0]
        for artifact, expected_answer in [(first, "two"), (second, "updated answer")]:
            result = cast(PipelineResult, artifact.result)
            records = [
                row
                for shard in (StoragePath(result.outputs["final"]) / "*.parquet").glob()
                for row in load_parquet(str(shard))
            ]
            assert len(records) == (1 if mode == SourceProcessingMode.SAMPLE else 2)
            for record in records:
                task = TaskSpec.model_validate_json(record["task_json"])
                assert task.source.row == record["source_locator"]
                assert task.source.row in {"rows.jsonl:0", "rows.jsonl:1"}
                assert task.source.dataset == "fixture/questions"
                assert verifyit_spec(cast(VerifyitGrader, task.grader)) == ExactSpec(
                    (expected_answer,), ignore_case=False
                )
                if artifact is second:
                    assert cast(TextMessage, task.context.events[0]).content.startswith("Updated ")
            review = json.loads((StoragePath(result.evidence["review"]) / "report.json").read_text())
            verification = json.loads((StoragePath(result.evidence["verify"]) / "report.json").read_text())
            assert review["status"] == "unreviewed"
            assert verification["status"] == "passed"
            assert verification["counts"]["checked"] == len(records)
            assert all(
                check["check"] == "golden" and check["status"] == "pass"
                for row in verification["results"]
                for trial in row["trials"]
                for check in trial["checks"]
            )
        assert not (tmp_path / "artifacts" / "task-curation" / "download").exists()
        manifest = json.loads(StoragePath(second.result.evidence["manifest"]).read_text())
        expected_files = {
            "rows.jsonl": {"path": str(file.resolve()), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        }
        assert manifest["source_file_overrides"] == expected_files
    finally:
        client.shutdown()


def convert_with_changed_input(row, context, *, input_path: str):
    Path(input_path).write_text('{"prompt":"Changed during conversion"}\n')
    return convert_fixture(row, context)


@pytest.mark.parametrize("changed_input", ["root", "file", "auxiliary", "during_conversion"])
def test_reviewed_input_mutation_fails_instead_of_publishing_under_planned_identity(
    tmp_path, config, monkeypatch, changed_input
):
    primary = tmp_path / "primary"
    primary.mkdir()
    file = primary / "rows.jsonl"
    file.write_text('{"prompt":"Original question"}\n')
    auxiliary = tmp_path / "answers"
    auxiliary.mkdir()
    answer = auxiliary / "answer.txt"
    answer.write_text("two")
    recipe = replace(
        math500(),
        source=HfSource("fixture/questions", "a" * 40, (file.name,), SourceFormat.JSONL, decode=decode_auxiliary_answer),
        inputs={"answers": HfSource("fixture/answers", "a" * 40, ("answer.txt",), SourceFormat.JSONL)},
        convert=(
            partial(convert_with_changed_input, input_path=str(file))
            if changed_input == "during_conversion"
            else convert_fixture
        ),
        rubric=None,
        controls=None,
        grader=None,
    )
    runtime = CampaignRuntime()
    options = PipelineOptions(
        SourceProcessingMode.FULL,
        runtime,
        inputs=InputOverrides(
            root=str(primary) if changed_input != "file" else None,
            files={file.name: file} if changed_input == "file" else {},
            auxiliary={"answers": str(auxiliary)},
        ),
        recipe_settings=RecipeSettings(config=replace(config, review=None, machines=None)),
    )
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    planned = source_step(recipe, options)
    if changed_input in ("root", "file"):
        file.write_text('{"prompt":"Changed before execution"}\n')
    elif changed_input == "auxiliary":
        answer.write_text("changed answer")
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            with pytest.raises(Exception) as failure:
                run(planned, max_concurrent=1)
        assert "Input overrides changed after planning" in error_chain(failure.value)
        if changed_input != "during_conversion":
            assert not (StoragePath(planned.path()) / "final").exists()
    finally:
        client.shutdown()


@pytest.fixture
def served(tmp_path):
    root = tmp_path / "served"
    root.mkdir()
    (root / "rows.jsonl").write_bytes(b'{"prompt": "1 + 1", "answer": "2"}\n')
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/rows.jsonl", (root / "rows.jsonl").read_bytes()
    server.shutdown()
    server.server_close()


def test_url_download_stages_the_file_only_when_its_digest_matches(tmp_path, served):
    url, data = served
    good = UrlSource(url, hashlib.sha256(data).hexdigest(), "rows.jsonl", SourceFormat.JSONL)
    download_source(DownloadRequest(good, str(tmp_path / "good")), campaign=CampaignRuntime())
    assert (tmp_path / "good" / "rows.jsonl").read_bytes() == data

    bad = replace(good, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        download_source(DownloadRequest(bad, str(tmp_path / "bad")), campaign=CampaignRuntime())
    assert not (tmp_path / "bad" / "rows.jsonl").exists()
