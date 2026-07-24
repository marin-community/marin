# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fsspec.core import url_to_fs
from marin.evaluation.samples import EvalSample, SampleKind, sample_from_lm_eval, write_sample_parquet

from infra.evaldash.src.samples import fetch_artifact, fetch_samples, list_sample_tasks


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


def test_grading_is_derived_and_round_trips(tmp_path) -> None:
    fs, root = url_to_fs(str(tmp_path))
    sample = sample_from_lm_eval(
        "gsm8k",
        {
            "doc_id": 3,
            "arguments": [["2+2?", " 4"]],
            "resps": [[" 4"]],
            "target": "4",
            "exact_match,flexible-extract": 1.0,
        },
    )
    write_sample_parquet(fs, f"{root}/samples_gsm8k_20260723.parquet", [sample])

    page = fetch_samples(str(tmp_path), "gsm8k", offset=0, limit=1, correct="all")

    (row,) = page.rows
    assert row.grading is not None
    assert row.grading.method == "lm-eval:exact_match"
    assert row.grading.metric == "exact_match,flexible-extract"
    assert row.grading.filter == "flexible-extract"
    assert row.grading.passed is True


def test_artifact_fetch_returns_run_local_object(tmp_path) -> None:
    results = tmp_path / "results"
    trajectory = results / "trajectories" / "aime_68.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text('{"steps": []}')

    artifact = fetch_artifact(str(results), str(trajectory))

    assert artifact.available is True
    assert artifact.reason is None
    assert artifact.media_type == "application/json"
    assert artifact.truncated is False
    assert artifact.text == '{"steps": []}'


def test_artifact_fetch_rejects_out_of_tree_uri(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret": true}')

    artifact = fetch_artifact(str(results), str(secret))

    assert artifact.available is False
    assert artifact.text is None
    assert "outside the run results directory" in (artifact.reason or "")


def test_artifact_fetch_rejects_parent_traversal(tmp_path) -> None:
    results = tmp_path / "results"
    (results / "trajectories").mkdir(parents=True)

    artifact = fetch_artifact(str(results), f"{results}/trajectories/../../secret.json")

    assert artifact.available is False
    assert "outside the run results directory" in (artifact.reason or "")


def test_artifact_fetch_missing_object_degrades(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()

    artifact = fetch_artifact(str(results), f"{results}/trajectories/absent.json")

    assert artifact.available is False
    assert artifact.text is None
    assert artifact.reason is not None


def test_artifact_fetch_enforces_size_cap(tmp_path) -> None:
    results = tmp_path / "results"
    big = results / "trajectories" / "big.json"
    big.parent.mkdir(parents=True)
    big.write_text("x" * 4096)

    artifact = fetch_artifact(str(results), str(big), max_bytes=1024)

    assert artifact.available is False
    assert artifact.truncated is True
    assert artifact.text is None


def test_artifact_fetch_without_results_path_degrades() -> None:
    artifact = fetch_artifact(None, "gs://bucket/runs/x/results/trajectories/aime_68.json")

    assert artifact.available is False
    assert artifact.reason == "run has no results_path"
