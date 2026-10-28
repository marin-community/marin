# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare real evaluation conversion graphs with a frozen, independent Evalchemy oracle."""

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from taskcompendium.evaluation import ChatGeneration, ChoiceLikelihood, likelihood_metrics
from taskcompendium.grading import grade_answer
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage
from taskcompendium.pipeline.evaluation import evaluation_cases, evaluation_protocol
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.submission import conversation_messages
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.evalchemy import sources
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import curate

ORACLE = Path(__file__).parent / "fixtures/evalchemy_oracle.json"


@pytest.fixture(params=["MATH500", "AIME24", "piqa"])
def published_case(request, tmp_path, monkeypatch):
    reference = next(case for case in json.loads(ORACLE.read_text())["cases"] if case["name"] == request.param)
    source = next(source for source in sources() if source.config.protocol.benchmark == request.param)
    dataset = source.config
    root = tmp_path / "source"
    root.mkdir()
    file = root / dataset.files[0]
    file.parent.mkdir(parents=True, exist_ok=True)
    if dataset.recipe.source.format is SourceFormat.PARQUET:
        pq.write_table(pa.Table.from_pylist(reference["rows"]), file)
    else:
        file.write_text("".join(json.dumps(row) + "\n" for row in reference["rows"]))
    protocol = dataset.protocol.model_copy(
        update={
            "expected_population": len(reference["rows"]),
            "trials": dataset.protocol.trials.model_copy(update={"seed": 7 if request.param == "AIME24" else None}),
        }
    )
    source = replace(source, config=replace(dataset, protocol=protocol))
    runtime = CampaignRuntime()
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime, inputs=InputOverrides(root=str(root)))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            artifact = run(curate(source, options), max_concurrent=1)[0]
        yield reference, artifact.path, protocol
    finally:
        client.shutdown()


def test_evalchemy_corpus_preserves_source_requests_and_grades(published_case):
    reference, path, expected_protocol = published_case
    protocol, manifest = evaluation_protocol(path)
    cases = list(evaluation_cases(path))
    assert manifest.intended_use == "eval"
    assert protocol == expected_protocol
    assert [case.item.source_ordinal for case in cases] == list(range(len(reference["rows"])))
    original_specs = [spec for spec in reference["specs"] if spec["trial"] in (None, 0)]
    for case, original in zip(cases, original_specs, strict=True):
        assert "eval" in case.task.tags
        assert "source:evalchemy" in case.task.tags
        if isinstance(case.item.operation, ChatGeneration):
            assert conversation_messages(case.task.context.events) == original["requests"][0]["prompt"]
        else:
            operation = case.item.operation
            assert isinstance(operation, ChoiceLikelihood)
            assert [request.context for request in operation.requests] == [
                request["prompt"] for request in original["requests"]
            ]
            assert [request.continuation for request in operation.requests] == [
                request["continuation"] for request in original["requests"]
            ]
    for original in reference["grade_cases"]:
        if "response" in original:
            case = cases[0]
            attempt = GradingAttempt(
                ConversationTrace(
                    events=(
                        *case.task.context.events,
                        TextMessage(role="assistant", content=original["response"]),
                    )
                )
            )
            grade = grade_answer(case.task, attempt)
            assert grade.reward == original["grade"]["metrics"]["accuracy"]
        else:
            case = next(case for case in cases if original["target"] in case.item.operation.gold)
            scores = tuple(score[0] for score in original["likelihoods"])
            assert likelihood_metrics(case.item.operation, scores, protocol) == original["grade"]["metrics"]
    if reference["name"] == "AIME24":
        assert [protocol.trials.seed_for(trial) for trial in range(protocol.trials.count)] == list(range(7, 17))
