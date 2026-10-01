# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib
import json
from pathlib import Path


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
