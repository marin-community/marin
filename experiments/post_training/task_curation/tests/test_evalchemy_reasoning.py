# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay GPQA/MMLU source prompts and grades captured from independent vendor execution."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from taskcompendium.grading import grade_answer
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage
from taskcompendium.pipeline.evaluation import evaluation_cases, evaluation_protocol
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.submission import conversation_messages
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import curate
from experiments.post_training.task_curation.datasets.evalchemy.reasoning import (
    MMLU_VALIDATION_FILE,
    mmlu_prompt,
    sources,
)

ORACLE = Path(__file__).parent / "fixtures/evalchemy_choices_oracle.json"


@pytest.fixture(params=["GPQADiamond", "MMLUPro", "MMLUPro-url"])
def reasoning_corpus(request, tmp_path, monkeypatch):
    benchmark = request.param.removesuffix("-url")
    reference = next(case for case in json.loads(ORACLE.read_text())["cases"] if case["name"] == benchmark)
    source = next(source for source in sources() if source.config.protocol.benchmark == benchmark)
    dataset = source.config
    root = tmp_path / "source"
    file = root / dataset.files[0]
    file.parent.mkdir(parents=True)
    if benchmark == "MMLUPro":
        pq.write_table(pa.Table.from_pylist(reference["rows"]), file)
        support = tmp_path / "fewshot"
        support_file = support / MMLU_VALIDATION_FILE
        support_file.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(reference["fewshot_rows"]), support_file)
        auxiliary = {"fewshot": support.as_uri() + "/" if request.param.endswith("-url") else str(support)}
    else:
        with file.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(reference["rows"][0]))
            writer.writeheader()
            writer.writerows(reference["rows"])
        auxiliary = {}
    protocol = dataset.protocol.model_copy(
        update={
            "expected_population": len(reference["rows"]),
            "trials": dataset.protocol.trials.model_copy(update={"seed": 7 if benchmark == "GPQADiamond" else None}),
        }
    )
    source = replace(source, config=replace(dataset, protocol=protocol))
    runtime = CampaignRuntime()
    options = PipelineOptions(
        SourceProcessingMode.QUICK,
        runtime,
        inputs=InputOverrides(root=str(root), auxiliary=auxiliary),
    )
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            artifact = run(curate(source, options), max_concurrent=1)[0]
        yield reference, artifact.path
    finally:
        client.shutdown()


def test_reasoning_release_preserves_vendor_requests_and_grades(reasoning_corpus):
    reference, path = reasoning_corpus
    protocol, _ = evaluation_protocol(path)
    cases = list(evaluation_cases(path))
    originals = [spec for spec in reference["specs"] if spec["trial"] in (None, 0)]
    assert len(cases) == len(reference["rows"])
    for case, original in zip(cases, originals, strict=True):
        assert conversation_messages(case.task.context.events) == original["requests"][0]["prompt"]
        assert case.item.source_ordinal == original["ordinal"]
        if reference["name"] == "MMLUPro":
            assert case.item.groups == original["tags"]
    if reference["name"] == "GPQADiamond":
        assert [protocol.trials.seed_for(trial) for trial in range(protocol.trials.count)] == [
            spec["requests"][0]["generation"]["seed"] for spec in reference["specs"][:3]
        ]
    for recorded in reference["grade_cases"]:
        case = cases[0]
        attempt = GradingAttempt(
            ConversationTrace(
                events=(
                    *case.task.context.events,
                    TextMessage(role="assistant", content=recorded["response"]),
                )
            )
        )
        result = grade_answer(case.task, attempt)
        assert result.reward == recorded["grade"]["metrics"]["accuracy"]


def test_mmlu_all_fewshot_variants_match_original_source():
    reference = next(case for case in json.loads(ORACLE.read_text())["cases"] if case["name"] == "MMLUPro")
    actual = [
        mmlu_prompt(tuple(reference["fewshot_rows"]), reference["rows"][0], num_fewshot=shots) for shots in range(6)
    ]
    assert actual == reference["variants"]
