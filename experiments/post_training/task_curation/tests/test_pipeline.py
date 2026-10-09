# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import re
import sys
import threading
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import Backend
from taskcompendium.convert.environment import IMAGE_BACKENDS
from taskcompendium.pipeline.controls import GradingMachines
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import FilterPolicy, ReviewRubric
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import VerificationBackend, campaign_machines
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import (
    MissingEnvironmentArtifact,
    built_environment,
    environment_artifact,
)
from experiments.post_training.task_curation.invocation import PipelineRun
from experiments.post_training.task_curation.pipeline import (
    DownloadRequest,
<<<<<<< HEAD
||||||| parent of a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
    HfSource,
=======
    HfSource,
    RlDataPipeline,
>>>>>>> a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
    UrlSource,
    download_source,
    download_step,
    environment_requirements,
    source_recipe,
    source_step,
)
from experiments.post_training.task_curation.tests.image_builds import (
    REPOSITORY,
    install_fake_build_tools,
    tracked_lock,
)

AGENT_IMAGE = "ghcr.io/marin-community/iris-task@sha256:" + "d" * 64

CONVERTER_MODULE = """
from taskcompendium.convert.answers import exact_answer_task


def convert(row, _context):
    return exact_answer_task(row, prompt=row.data["prompt"], answers=(row.data["answer"],), ignore_case=False)
"""


def math500():
    return next(
        source.pipeline
        for source in skyrl_math.sources()
        if source.name == "math500" and isinstance(source.pipeline, RlDataPipeline)
    )


def machines(backend: VerificationBackend) -> GradingMachines:
    return campaign_machines(backend, "fixture-worker", "http://controller.invalid")


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
        machines=machines(VerificationBackend.GVISOR),
    )


def step_name(pipeline, config) -> str:
    return source_step(pipeline, config, CampaignRuntime(), source=pipeline).name


def test_step_is_named_for_the_declaration_and_stable(config):
    first, second = step_name(math500(), config), step_name(math500(), config)
    assert first == second
    assert first.startswith("data/rl/math500-")


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


def test_verification_backend_enters_identity_only_with_controls(config):
    iris = replace(config, machines=machines(VerificationBackend.IRIS))
    assert step_name(math500(), iris) != step_name(math500(), config)
    unchecked = replace(math500(), controls=None)
    assert step_name(unchecked, iris) == step_name(unchecked, config)


@pytest.fixture
def fixture_converter(tmp_path, monkeypatch):
    """A converter module in its own directory, so tests can change the files beside it."""
    directory = tmp_path / "fixture_family"
    directory.mkdir()
    (directory / "fixture_source.py").write_text(CONVERTER_MODULE)
    spec = importlib.util.spec_from_file_location("fixture_source", directory / "fixture_source.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Identity finds the converter's files through sys.modules; each test registers its own module.
    monkeypatch.setitem(sys.modules, "fixture_source", module)
    spec.loader.exec_module(module)
    return directory, module.convert


def test_python_files_beside_the_converter_rename_the_artifact(fixture_converter, config):
    directory, convert = fixture_converter
    script = directory / "fixture_grade.py"
    script.write_text("print(1)\n")
    pipeline = replace(math500(), name="fixture", convert=convert)
    original = step_name(pipeline, config)
    script.write_text("print(0)\n")
    assert step_name(pipeline, config) != original


def test_shipped_scorer_bytes_rename_the_artifact(fixture_converter, config):
    directory, convert = fixture_converter
    scorer = directory / "scorers" / "upstream" / "score.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("REWARD = 1\n")
    pipeline = replace(math500(), name="fixture", convert=convert, ships=(directory / "scorers",))
    original = step_name(pipeline, config)
    scorer.write_text("REWARD = 0\n")
    assert step_name(pipeline, config) != original
    unshipped = replace(pipeline, ships=())
    scorer.write_text("REWARD = 1\n")
    assert step_name(unshipped, config) != original


def test_declarations_ship_only_existing_directories(tmp_path):
    with pytest.raises(ValueError, match="do not exist"):
        replace(math500(), ships=(tmp_path / "missing",))


@pytest.fixture
def grader_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    install_fake_build_tools(tmp_path, monkeypatch)
    return tracked_lock(tmp_path)


def test_a_changed_grader_environment_renames_the_artifact(grader_lock, config):
    grader = Environment(lock=grader_lock)
    pipeline = replace(math500(), grader=grader)
    run(environment_artifact(grader, REPOSITORY))
    original = source_step(pipeline, config, CampaignRuntime(), source=pipeline)
    assert environment_artifact(grader).name in [dep.name for dep in original.deps]
    grader_lock.write_text("numpy==2.3.4\n")
    run(environment_artifact(grader, REPOSITORY))
    assert step_name(pipeline, config) != original.name


def test_a_grader_environment_without_a_built_artifact_names_the_build_command(grader_lock, config):
    pipeline = replace(math500(), grader=Environment(lock=grader_lock))
    with pytest.raises(
        MissingEnvironmentArtifact,
        match=re.escape("run: uv run python -m experiments.post_training.task_curation.images --identity "),
    ):
        source_step(pipeline, config, CampaignRuntime(), source=pipeline)


def test_an_environment_image_runs_as_declared_in_a_sandbox():
    requirements = environment_requirements(Environment(image=AGENT_IMAGE))
    assert (requirements.docker_image, requirements.compatible_backends) == (AGENT_IMAGE, IMAGE_BACKENDS)
    assert requirements.packages_lock is None


def test_apt_packages_beyond_the_worker_image_run_in_a_sandbox_of_the_built_image(grader_lock):
    environment = Environment(lock=grader_lock, apt=("build-essential", "jq"))
    (built,) = run(environment_artifact(environment, REPOSITORY))
    requirements = environment_requirements(environment, built_environment(environment))
    assert built.image is not None and built.image.startswith(f"{REPOSITORY}@sha256:")
    assert (requirements.docker_image, requirements.compatible_backends) == (built.image, IMAGE_BACKENDS)
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
    assert requirements.compatible_backends == (Backend.LOCAL,)
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


def test_an_agent_environment_must_name_its_image(grader_lock):
    with pytest.raises(ValueError, match="agent environment's image"):
        replace(math500(), environment=Environment(lock=grader_lock))


def test_rubric_paragraphs_become_review_criteria():
    pipeline = replace(math500(), rubric="\nFirst criterion\nspans two lines.\n\nSecond criterion.\n")
    assert source_recipe(pipeline, {}, None).rubric == ReviewRubric(
        id="math500", version="1", criteria=("First criterion spans two lines.", "Second criterion.")
    )


def test_declarations_with_the_same_pinned_files_share_one_download():
    pipeline = math500()
    selected = replace(pipeline.source, select=lambda row, context: True)
    assert download_step(selected, CampaignRuntime()).name == download_step(pipeline.source, CampaignRuntime()).name


<<<<<<< HEAD
||||||| parent of a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
@pytest.mark.parametrize("mode, expected_rows", [(SourceProcessingMode.SAMPLE, 2), (SourceProcessingMode.FULL, 5)])
def test_reviewed_invocation_uses_selected_mode_for_panel_or_full_conversion(
    tmp_path, fixture_converter, config, mode, expected_rows
):
    _, convert = fixture_converter
    source = tmp_path / "source"
    source.mkdir()
    (source / "rows.jsonl").write_text(
        "".join(json.dumps({"prompt": f"Question {index}?", "answer": str(index)}) + "\n" for index in range(5))
    )
    pipeline = replace(
        math500(),
        source=HfSource("fixture/questions", "a" * 40, ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert,
        rubric=None,
        controls=None,
        grader=None,
    )
    config = replace(config, mode=mode, quality_policy=SourceQualityPolicy(sample_size=2))
    client = LocalClient()
    with (
        set_current_client(client),
        ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
    ):
        result = run_curation(
            pipeline,
            mode=mode,
            context=context,
            source_input=str(source),
            output_path=str(tmp_path / "output"),
            inputs={},
            config=config,
        )
    rows = [row for shard in (tmp_path / "output/normalize").glob("*.parquet") for row in load_parquet(str(shard))]
    assert len(rows) == expected_rows
    assert {row["source_row"] for row in rows} <= {f"rows.jsonl:{index}" for index in range(5)}
    manifest = json.loads(StoragePath(result.manifest_path).read_text())
    assert manifest["mode"] == mode


=======
@pytest.mark.parametrize("mode, expected_rows", [(SourceProcessingMode.SAMPLE, 2), (SourceProcessingMode.FULL, 5)])
def test_reviewed_invocation_uses_invocation_mode_for_panel_or_full_conversion(
    tmp_path, fixture_converter, config, mode, expected_rows
):
    _, convert = fixture_converter
    source = tmp_path / "source"
    source.mkdir()
    (source / "rows.jsonl").write_text(
        "".join(json.dumps({"prompt": f"Question {index}?", "answer": str(index)}) + "\n" for index in range(5))
    )
    pipeline = replace(
        math500(),
        source=HfSource("fixture/questions", "a" * 40, ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert,
        rubric=None,
        controls=None,
        grader=None,
    )
    config = replace(config, mode=mode, quality_policy=SourceQualityPolicy(sample_size=2))
    client = LocalClient()
    with (
        set_current_client(client),
        ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
    ):
        result = replace(pipeline, config=config)(
            pipeline, PipelineRun(mode, context, str(tmp_path / "output"), source_input=str(source))
        )
    rows = [row for shard in (tmp_path / "output/normalize").glob("*.parquet") for row in load_parquet(str(shard))]
    assert len(rows) == expected_rows
    assert {row["source_row"] for row in rows} <= {f"rows.jsonl:{index}" for index in range(5)}
    manifest = json.loads(StoragePath(result.evidence["manifest"]).read_text())
    assert manifest["mode"] == mode


>>>>>>> a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
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
