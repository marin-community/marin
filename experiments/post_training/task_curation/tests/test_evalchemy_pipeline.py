# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact graph regressions for source preparation identity and release commitment."""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from taskcompendium.convert.answers import answer_task
from taskcompendium.evaluation import EvaluationProtocol, Metric, MetricKind, Sampling, TrialPolicy
from taskcompendium.pipeline.evaluation import MANIFEST_FILENAME, chat_item, evaluation_cases
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, RawRow
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from verifyit.spec import ExactSpec
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def decode_prefix(data: dict, _context: ConversionContext, *, prefix: str) -> dict:
    return {**data, "prompt": prefix + data["prompt"]}


def convert_fixture(row: RawRow, _context: ConversionContext):
    task = answer_task(row, prompt=row.data["prompt"], spec=ExactSpec(expected=("red",)))
    return task.model_copy(update={"tags": ("eval",)})


def convert_mutating_fixture(row: RawRow, context: ConversionContext):
    task = convert_fixture(row, context)
    path = Path(row.data["mutate_path"])
    path.write_bytes(path.read_bytes() + b"\n")
    return task


def fixture_source() -> RlDataSource[EvalDataSet]:
    return RlDataSource(
        info=SourceInfo(
            id="eval-identity", title="Identity fixture", origin="test", family="test", tags=("eval",), count=1
        ),
        config=EvalDataSet(
            CurationRecipe(
                name="eval-identity",
                source=HfSource("test/identity", "0" * 40, ("rows.jsonl",), SourceFormat.JSONL),
                convert=convert_fixture,
                version="1",
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark="identity",
                version="1",
                operation="chat_generation",
                sampling=Sampling(temperature=0, top_p=1, top_k=None, max_tokens=10, stop=(), chat_template_kwargs={}),
                trials=TrialPolicy(count=1, seed=None),
                metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
                primary_metric="accuracy",
                reducer="mean_v1",
                expected_population=1,
            ),
            chat_item,
        ),
        pipeline=curate,
    )


@pytest.fixture
def local_graph(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    runtime = CampaignRuntime()
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime, inputs=InputOverrides(root=str(root)))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            yield root, options
    finally:
        client.shutdown()


def test_evaluation_artifact_cache_distinguishes_source_preparation(local_graph):
    root, options = local_graph
    (root / "rows.jsonl").write_text(json.dumps({"prompt": "Name a color"}) + "\n")
    source = fixture_source()
    dataset = source.config
    assert dataset is not None
    outputs = []
    for prefix in ("First: ", "Second: "):
        prepared = replace(dataset.recipe.source, decode=partial(decode_prefix, prefix=prefix))
        configured = replace(source, config=replace(dataset, recipe=replace(dataset.recipe, source=prepared)))
        artifact = run(curate(configured, options), max_concurrent=1)[0]
        outputs.append(next(evaluation_cases(artifact.path)).task.context.events[0].content)
    assert outputs == ["First: Name a color", "Second: Name a color"]


def test_evaluation_input_change_during_conversion_withdraws_the_release_marker(local_graph):
    root, options = local_graph
    path = root / "rows.jsonl"
    path.write_text(json.dumps({"prompt": "Name a color", "mutate_path": str(path)}) + "\n")
    source = fixture_source()
    dataset = source.config
    assert dataset is not None
    source = replace(source, config=replace(dataset, recipe=replace(dataset.recipe, convert=convert_mutating_fixture)))
    step = curate(source, options)
    with pytest.raises(RuntimeError):
        run(step, max_concurrent=1)
    assert not (Path(step.path()) / MANIFEST_FILENAME).exists()
