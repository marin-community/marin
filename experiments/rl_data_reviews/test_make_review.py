# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib
import io
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("prompt_name", ["JUDGE_PROMPT", "COALESCE_PROMPT"])
@pytest.mark.parametrize(
    "empty_findings",
    [[], [{"dimension": "verifier_coverage", "text": "...", "severity": "info", "kind": "observation"}]],
)
def test_empty_provider_opinion_stays_incomplete_and_resume_preserves_raw_calls(
    tmp_path, monkeypatch, prompt_name, empty_findings
):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    runner = importlib.import_module("make_review")
    opinion = {"summary": "...", "verdict": "keep", "metrics": [], "findings": empty_findings, "tags": []}
    if prompt_name == "COALESCE_PROMPT":
        opinion["subject_id"] = "sample"
    completed = {
        **opinion,
        "summary": "The native verifier ran successfully; this sample has no observed grading defect.",
        "findings": [],
    }
    outputs = [{"syntheses": [value]} if prompt_name == "COALESCE_PROMPT" else value for value in [opinion, completed]]
    responses = [
        {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}]} for output in outputs
    ]
    pending = iter(responses)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda _request, **_kwargs: io.BytesIO(json.dumps(next(pending)).encode())
    )
    model = {"name": "reviewer", "base_url": "https://provider.invalid/v1", "parameters": {}, "timeout": 10}
    prompt = getattr(runner, prompt_name)

    with pytest.raises(ValueError):
        runner.judgment(model, prompt, {}, tmp_path, 100_000, None)

    assert not (tmp_path / "parsed.json").exists()
    assert json.loads((tmp_path / "call-000/response.json").read_text()) == responses[0]
    assert runner.judgment(model, prompt, {}, tmp_path, 100_000, None) == outputs[1]
    assert json.loads((tmp_path / "parsed.json").read_text()) == outputs[1]
    assert json.loads((tmp_path / "call-000/response.json").read_text()) == responses[0]
    assert json.loads((tmp_path / "call-001/response.json").read_text()) == responses[1]

    # Old cached placeholders must not bypass validation when resuming a prior run.
    (tmp_path / "parsed.json").write_text(json.dumps(outputs[0]))
    with pytest.raises(ValueError):
        runner.judgment(model, prompt, {}, tmp_path, 100_000, None)


def test_synthesis_records_all_applicable_reviews_without_model_copied_ids(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    runner = importlib.import_module("make_review")
    subjects = []
    reviews = []
    opinions = []
    source_populations = {}
    task_sources = {}
    task_coverages = {}
    for source in ["source-a", "source-b"]:
        task = source + "/task"
        source_populations[source] = 10
        task_sources[task] = source
        task_coverages[task] = {"scope": "single_task", "sample_count": 1, "samples": [{"task_id": task}]}
        for subject, level in [(source, "source"), (task, "task")]:
            subjects.append({"id": subject, "level": level})
            opinions.append(
                {
                    "subject_id": subject,
                    "summary": "The independent reviews disagree about verifier coverage.",
                    "verdict": "conditional",
                    "metrics": [],
                    "findings": [],
                    "tags": [],
                }
            )
        for method in ["runtime", "judge-1", "judge-2", "judge-3"]:
            reviews.append(
                {
                    "id": task + "/" + method,
                    "subject_id": task,
                    "tests_executed": True,
                    "summary": "Saved native result or independent opinion.",
                    "verdict": "keep" if method == "judge-1" else "conditional",
                    "metrics": [],
                    "findings": [],
                    "evidence": [],
                }
            )
    original_reviews = copy.deepcopy(reviews)
    collection = {"subjects": subjects, "reviews": reviews, "tag_assignments": [], "execution_provenance": {}}
    stage = tmp_path / "coalescer"
    stage.mkdir()
    (stage / "parsed.json").write_text(json.dumps({"syntheses": opinions}))
    panel = runner.PanelReviews(collection, source_populations, task_sources, task_coverages)

    runner.coalesce_reviews(panel, {"name": "reviewer"}, tmp_path, 100_000, {}, None)

    assert collection["reviews"][:8] == original_reviews
    for synthesis in collection["reviews"][8:]:
        source = synthesis["subject_id"].split("/")[0]
        assert synthesis["derived_from_review_ids"] == [
            source + "/task/runtime",
            source + "/task/judge-1",
            source + "/task/judge-2",
            source + "/task/judge-3",
        ]
        assert synthesis["verdict"] == "conditional"
