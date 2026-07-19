# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fsspec.core import url_to_fs
from marin.evaluation.samples import EvalSample, SampleKind, write_sample_parquet

from infra.evaldash.src.samples import fetch_samples, list_sample_tasks


def test_sample_reader_returns_typed_filtered_page(tmp_path) -> None:
    fs, root = url_to_fs(str(tmp_path))
    samples = [
        EvalSample(
            task="arc",
            doc_id="correct",
            kind=SampleKind.GENERATION,
            prompt_text="Question one",
            output="A",
            target_text="A",
            metrics={"acc_norm,none": 1.0},
            correct=True,
        ),
        EvalSample(
            task="arc",
            doc_id="incorrect",
            kind=SampleKind.GENERATION,
            prompt_text="Question two",
            output="B",
            target_text="A",
            metrics={"acc_norm,none": 0.0},
            correct=False,
        ),
    ]
    write_sample_parquet(fs, f"{root}/samples_arc_20260719.parquet", samples)

    tasks = list_sample_tasks(str(tmp_path))
    page = fetch_samples(str(tmp_path), "arc", offset=0, limit=1, correct="incorrect")

    assert tasks.model_dump(mode="json") == {
        "available": True,
        "error": None,
        "tasks": [{"task": "arc", "files": 1}],
    }
    assert page.primary_metric == "acc_norm,none"
    assert page.counts.model_dump() == {"all": 2, "correct": 1, "incorrect": 1}
    assert page.total == 1
    assert page.offset == 0
    assert page.limit == 1
    assert [row.doc_id for row in page.rows] == ["incorrect"]
