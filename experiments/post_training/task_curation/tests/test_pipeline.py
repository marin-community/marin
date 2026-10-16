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
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    verifyit_spec,
)
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import Controls, FilterPolicy, ReviewRubric
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.sources import prepared_component_path
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode
from verifyit.spec import EmptyOutputPolicy, ExactSpec, JudgeSpec
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation import pipeline as pipeline_module
from experiments.post_training.task_curation.campaign import CampaignRuntime, PipelineResult, error_chain
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions, RecipeSettings
from experiments.post_training.task_curation.datasets.nemotron_ultra.components import prepare_blend
from experiments.post_training.task_curation.datasets.nemotron_ultra.graders import semantic_empty_negative
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.environment import Archive, Environment
from experiments.post_training.task_curation.images.build import (
    built_environment,
    environment_artifact,
)
from experiments.post_training.task_curation.pipeline import (
    CampaignMachines,
    CurationRecipe,
    DownloadRequest,
    HarborHubSource,
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
    COMPILED_LOCK,
    DATA_PATH,
    REPOSITORY,
    REQUIREMENTS,
    install_fake_build_tools,
)
from experiments.post_training.task_curation.tests.review_server import JudgeReply

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
def grader_build(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    install_fake_build_tools(tmp_path, monkeypatch)


def test_a_changed_grader_environment_renames_the_artifact(grader_build, config):
    grader = Environment(requirements=REQUIREMENTS)
    pipeline = replace(math500(), grader=grader)
    run(environment_artifact(grader, REPOSITORY))
    original = source_step(
        pipeline, PipelineOptions(config.mode, CampaignRuntime(), recipe_settings=RecipeSettings(config=config))
    )
    assert environment_artifact(grader).name in [dep.name for dep in original.deps]
    changed = replace(grader, requirements=("numpy==2.3.4",))
    run(environment_artifact(changed, REPOSITORY))
    assert step_name(replace(pipeline, grader=changed), config) != original.name


def test_apt_packages_beyond_the_worker_image_run_in_a_sandbox_of_the_built_image(grader_build):
    environment = Environment(requirements=REQUIREMENTS, apt=("build-essential", "jq"))
    (built,) = run(environment_artifact(environment, REPOSITORY))
    requirements = environment_requirements(environment, built_environment(environment))
    assert built.image is not None and built.image.startswith(f"{REPOSITORY}@sha256:")
    assert requirements.docker_image == built.image
    assert requirements.packages_lock is None


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param(Environment(requirements=REQUIREMENTS, apt=("build-essential", "git")), id="worker-image-apt"),
        pytest.param(Environment(requirements=REQUIREMENTS), id="requirements-only"),
    ],
)
def test_environments_the_worker_image_covers_run_in_the_worker_from_their_lock(grader_build, environment):
    (built,) = run(environment_artifact(environment, REPOSITORY))
    requirements = environment_requirements(environment, built_environment(environment))
    assert requirements.docker_image is None
    assert requirements.packages_lock == built.lock_url
    assert StoragePath(requirements.packages_lock).read_text() == COMPILED_LOCK


PINNED_ARCHIVE = Archive("https://archives.invalid/data.zip", "0" * 64, "english")


@pytest.mark.parametrize(
    "declare",
    [
        pytest.param(lambda: Environment(requirements=("numpy>=2",)), id="unpinned-requirement"),
        pytest.param(
            lambda: Environment(requirements=("verifiable-instructions @ https://github.com/x/y/archive/main.tar.gz",)),
            id="url-requirement-without-a-commit",
        ),
        pytest.param(lambda: Environment(image="ghcr.io/marin-community/iris-task:latest"), id="unpinned-image"),
        pytest.param(lambda: Environment(image=AGENT_IMAGE, requirements=REQUIREMENTS), id="image-with-packages"),
        pytest.param(lambda: Environment(files={"share/data/english": PINNED_ARCHIVE}), id="relative-file-path"),
        pytest.param(
            lambda: Environment(files={"/usr/local/share/data/english": PINNED_ARCHIVE}),
            id="file-under-system-directory",
        ),
        pytest.param(
            lambda: Environment(files={DATA_PATH: PINNED_ARCHIVE, f"{DATA_PATH}/extra": PINNED_ARCHIVE}),
            id="file-inside-another-file",
        ),
        pytest.param(lambda: Environment(files={DATA_PATH: replace(PINNED_ARCHIVE, sha256="abc")}), id="bad-sha256"),
        pytest.param(
            lambda: Environment(files={DATA_PATH: replace(PINNED_ARCHIVE, member="../etc")}), id="escaping-member"
        ),
        pytest.param(lambda: Environment(apt=("jq; rm -rf /",)), id="apt-not-a-package-name"),
    ],
)
def test_environment_declarations_reject_contradictory_or_unpinned_needs(declare):
    with pytest.raises(ValueError):
        declare()


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
            records = list(load_parquet(str(prepared_component_path(staged, "fixture"))))
            assert [(row["component"], row["source_index"], json.loads(row["raw_json"])) for row in records] == [
                ("fixture", 0, json.loads((primary / "rows.jsonl").read_text()))
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
        with ZephyrContext(
            client=LocalClient(), max_workers=2, chunk_storage_prefix=str(tmp_path / "prepare")
        ) as context:
            prepare_blend(str(file), context)
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
            trials = json.loads(StoragePath(verification["trial_report"]).read_text())
            assert all(
                check["check"] == "golden" and check["status"] == "pass"
                for row in trials["results"]
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


def convert_judge_fixture(row, context):
    task = convert_fixture(row, context)
    assert isinstance(task, TaskSpec)
    package = verifyit_package(
        JudgeSpec(
            references=(row.data["answer"],),
            rubric="labels",
            model="fixture-judge",
            prompt_template="Reference: {reference}\nCandidate: {candidate}",
            label_scores={"PASS": 1.0, "FAIL": 0.0},
            empty_output=EmptyOutputPolicy.GRADE,
        ),
        environment=EnvironmentRequirements(docker_image=AGENT_IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS),
    )
    return task.model_copy(update={"grader": package.grader})


def test_recipe_without_review_uses_its_explicit_connection_for_judge_controls(
    tmp_path, config, monkeypatch, review_server
):
    primary = tmp_path / "source"
    primary.mkdir()
    (primary / "rows.jsonl").write_text('{"prompt":"Two plus two?","answer":"4"}\n')
    recipe = replace(
        math500(),
        source=HfSource("fixture/questions", "a" * 40, ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert_judge_fixture,
        rubric=None,
        controls=Controls(negative=semantic_empty_negative),
        grader=None,
    )
    runtime = CampaignRuntime()
    options = PipelineOptions(
        SourceProcessingMode.FULL,
        runtime,
        inputs=InputOverrides(root=str(primary)),
        recipe_settings=RecipeSettings(
            config=replace(
                config,
                review=None,
                execution=AuditExecution(),
                machines=None,
                verification_policy=SourceVerificationPolicy(10, 0, 1, 1),
            ),
            base_url=f"http://127.0.0.1:{review_server.server_port}/v1",
            api_key="fixture-key",
        ),
    )
    review_server.judge_replies = [JudgeReply("FAIL")]
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            artifact = run(source_step(recipe, options), max_concurrent=1)[0]
        result = cast(PipelineResult, artifact.result)
        verification = json.loads((StoragePath(result.evidence["verify"]) / "report.json").read_text())
        assert verification["status"] == "passed"
        assert len(review_server.requests) == 1
        assert review_server.requests[0]["model"] == "fixture-judge"
    finally:
        client.shutdown()


def test_harbor_hub_download_reuses_cache_and_separates_release_pins(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    original_run = pipeline_module.subprocess.run
    acquired = []

    def exporter(command, **kwargs):
        if str(pipeline_module.HARBOR_EXPORTER) not in command:
            return original_run(command, **kwargs)
        release = command[command.index("--release") + 1]
        output = StoragePath(command[command.index("--output") + 1])
        acquired.append(release)
        (output / "tasks/task.jsonl").write_text(json.dumps({"release": release}) + "\n", auto_mkdir=True)
        (output / "release.json").write_text(json.dumps({"release": release}))
        return None

    # Replace the isolated SDK process, not the artifact executor or its caching.
    monkeypatch.setattr(pipeline_module.subprocess, "run", exporter)
    source = HarborHubSource("fixture/tasks", "sha256:" + "a" * 64)
    runtime = CampaignRuntime()
    client = LocalClient()
    try:
        with set_current_client(client):
            first = run(download_step(source, runtime), max_concurrent=1)[0]
            repeated = run(download_step(source, runtime), max_concurrent=1)[0]
            changed = replace(source, revision="sha256:" + "c" * 64)
            other = run(download_step(changed, runtime), max_concurrent=1)[0]
        assert first.path == repeated.path != other.path
        assert acquired == [f"{source.name}@{source.revision}", f"{changed.name}@{changed.revision}"]
        for artifact, expected in ((first, acquired[0]), (other, acquired[1])):
            row = json.loads((StoragePath(artifact.path) / "tasks/task.jsonl").read_text())
            assert row["release"] == expected
    finally:
        client.shutdown()


@pytest.mark.parametrize("mode", [SourceProcessingMode.QUICK, SourceProcessingMode.SAMPLE, SourceProcessingMode.FULL])
def test_harbor_hub_input_override_bypasses_acquisition_and_tracks_content(tmp_path, config, mode):
    source = HarborHubSource("fixture/tasks", "sha256:" + "a" * 64)
    recipe = replace(math500(), source=source, convert=convert_fixture, rubric=None, controls=None, grader=None)
    options = PipelineOptions(mode, CampaignRuntime(), recipe_settings=RecipeSettings(config=replace(config, mode=mode)))
    automatic = source_step(recipe, options)
    assert len(automatic.deps) == 1
    root = tmp_path / "input"
    (root / "tasks").mkdir(parents=True)
    payload = root / "tasks/task.jsonl"
    payload.write_text('{"prompt": "one", "answer": "1"}\n')
    local_options = replace(options, inputs=InputOverrides(root=str(root)))
    first = source_step(recipe, local_options)
    assert first.deps == ()
    payload.write_text('{"prompt": "two", "answer": "2"}\n')
    second = source_step(recipe, local_options)
    if mode != SourceProcessingMode.QUICK:
        assert first.name != second.name
