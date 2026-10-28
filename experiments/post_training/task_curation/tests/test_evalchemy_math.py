# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-semantic fixtures captured from the independent legacy evaluator."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from taskcompendium.pipeline.evaluation import evaluation_cases
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.submission import conversation_messages
from verifyit.adapters.evalchemy_amc import grade_amc
from verifyit.adapters.evalchemy_olympiad import first_boxed_answer, grade_olympiad, judge_prompt, reference_answers
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.evalchemy.math import sources
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import curate

ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_math_oracle.json").read_text())


def test_olympiad_judge_request_matches_the_frozen_source_prompt():
    case = ORACLE["judge_case"]
    candidate = first_boxed_answer(case["response"])
    assert judge_prompt(case["question"], tuple(case["answers"]), candidate) == case["plan"]["requests"][0]["prompt"]


@pytest.mark.parametrize("case", ORACLE["olympiad_grades"])
def test_olympiad_first_box_any_reference_matches_legacy(case):
    verdict = grade_olympiad(reference_answers(case["answers"]), case["response"])
    assert verdict.reward == case["grade"]["metrics"]["accuracy"]
    assert verdict.detail["extracted"] == case["grade"]["filtered_resps"][0]


@pytest.mark.parametrize("case", ORACLE["amc_grades"])
def test_amc_normalized_literal_grading_matches_legacy(case):
    verdict = grade_amc(case["expected"], case["response"])
    assert verdict.reward == case["grade"]["metrics"]["accuracy"]
    assert verdict.detail["extracted"] == case["grade"]["filtered_resps"][0]


@pytest.mark.parametrize("reference", ORACLE["cases"], ids=lambda reference: reference["name"])
def test_math_publication_preserves_original_population_and_exact_prompt(reference, tmp_path, monkeypatch):
    source = next(source for source in sources() if source.config.protocol.benchmark == reference["name"])
    dataset = source.config
    assert dataset is not None
    root = tmp_path / "source"
    root.mkdir()
    (root / dataset.files[0]).write_text("".join(json.dumps(row) + "\n" for row in reference["rows"]))
    source = replace(
        source,
        config=replace(
            dataset,
            protocol=dataset.protocol.model_copy(update={"expected_population": len(reference["rows"])}),
        ),
    )
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
        cases = list(evaluation_cases(artifact.path))
    finally:
        client.shutdown()
    assert len(cases) == len(reference["rows"])
    for case, original in zip(cases, reference["specs"], strict=True):
        assert case.item.source_ordinal == original["ordinal"]
        assert conversation_messages(case.task.context.events) == original["requests"][0]["prompt"]
