# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import threading
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import FilterPolicy, ReviewRubric
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewTransport

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import CampaignMachines, VerificationBackend
from experiments.post_training.task_curation.pipeline import (
    DownloadRequest,
    UrlSource,
    download_source,
    download_step,
    source_recipe,
    source_step,
)

CONVERTER_MODULE = """
from taskcompendium.convert.answers import exact_answer_task


def convert(row):
    return exact_answer_task(row, prompt=row.data["prompt"], answers=(row.data["answer"],), ignore_case=False)
"""


def math500():
    return next(pipeline for pipeline in skyrl_math.pipelines() if pipeline.name == "math500")


def machines(backend: VerificationBackend) -> CampaignMachines:
    return CampaignMachines(backend, "fixture-worker", "http://controller.invalid", {})


@pytest.fixture
def config() -> SourcePipelineConfig:
    return SourcePipelineConfig(
        mode=SourceProcessingMode.SAMPLE,
        quality_policy=SourceQualityPolicy(),
        verification_policy=SourceVerificationPolicy(10, 0, 2, 0.9),
        review=ReviewConfig("fixture-model", "fixture-revision", transport=ReviewTransport.PROVIDER_BATCH),
        execution=AuditExecution(),
        filter_policy=FilterPolicy(),
        normalized_shards=2,
        machines=machines(VerificationBackend.GVISOR),
    )


def step_name(pipeline, config) -> str:
    return source_step(pipeline, config, CampaignRuntime()).name


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


def test_packaged_grader_script_changes_rename_the_artifact(tmp_path, monkeypatch, config):
    (tmp_path / "fixture_source.py").write_text(CONVERTER_MODULE)
    script = tmp_path / "fixture_grade.py"
    script.write_text("print(1)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("fixture_source")
    pipeline = replace(math500(), name="fixture", convert=module.convert)
    original = step_name(pipeline, config)
    script.write_text("print(0)\n")
    assert step_name(pipeline, config) != original


def test_rubric_paragraphs_become_review_criteria():
    pipeline = replace(math500(), rubric="\nFirst criterion\nspans two lines.\n\nSecond criterion.\n")
    assert source_recipe(pipeline, {}).rubric == ReviewRubric(
        id="math500", version="1", criteria=("First criterion spans two lines.", "Second criterion.")
    )


def test_declarations_with_the_same_pinned_files_share_one_download():
    pipeline = math500()
    selected = replace(pipeline.source, select=lambda row, inputs: True)
    assert download_step(selected, CampaignRuntime()).name == download_step(pipeline.source, CampaignRuntime()).name


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
