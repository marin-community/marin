"""Regression checks for exact swaps, source semantics and held-out exclusions."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.inventory.build_taskforge_pair import build_pair


def task(identifier: str, dataset: str) -> str:
    return TaskSpec(
        id=identifier,
        source=Source(dataset=dataset, revision="test-revision", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=f"Solve {identifier}."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("python", "/tests/grade.py"),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image="python@sha256:" + "a" * 64,
            ),
        ),
    ).model_dump_json()


def write_tasks(path: Path, values: list[str]) -> None:
    pq.write_table(pa.table({"task_spec": values}), path)


def pair_inputs(tmp_path):
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.jsonl"
    validation = tmp_path / "validation.parquet"
    exclusions = tmp_path / "full_validation.parquet"
    admitted_ids = tmp_path / "admitted.txt"
    write_tasks(baseline, [task(f"control-{i}", "openscience") for i in range(2200)])
    candidates = [task(f"candidate-{i}", "taskforge") for i in range(210)]
    candidate.write_text("\n".join(candidates) + "\n")
    admitted_ids.write_text("\n".join(f"candidate-{i}" for i in range(210)) + "\n")
    write_tasks(validation, [task("validation", "evaluation")])
    write_tasks(exclusions, [task("validation", "evaluation"), task("heldout", "evaluation")])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "admitted_ids": admitted_ids,
        "validation": validation,
        "exclusions": [exclusions],
        "stem_datasets": {"openscience"},
        "baseline_uri": "fixture://baseline",
        "candidate_uri": "fixture://candidate",
    }


def test_exact_200_row_swap_is_deterministic_and_preserves_images_and_panel(tmp_path):
    inputs = pair_inputs(tmp_path)
    output = tmp_path / "pair"
    manifest = build_pair(**inputs, output=output)
    repeated = build_pair(**inputs, output=tmp_path / "repeat")
    assert manifest == repeated
    control = pq.read_table(output / "control/train.parquet").column("task_spec").to_pylist()
    candidate = pq.read_table(output / "candidate/train.parquet").column("task_spec").to_pylist()
    assert len(control) == len(candidate) == 2000
    assert sum(a != b for a, b in zip(control, candidate, strict=True)) == 200
    source_candidate_bytes = set(inputs["candidate"].read_text().splitlines())
    assert set(candidate) - set(control) <= source_candidate_bytes
    for name in ("control", "candidate"):
        assert (output / name / "validation.parquet").read_bytes() == inputs["validation"].read_bytes()
        assert (output / name / "full_validation.parquet").read_bytes() == inputs["exclusions"][0].read_bytes()


def test_admitted_candidate_matching_heldout_is_refused(tmp_path):
    inputs = pair_inputs(tmp_path)
    candidate = inputs["candidate"]
    values = candidate.read_text().splitlines()
    original = json.loads(values[0])
    original["context"] = json.loads(task("heldout", "evaluation"))["context"]
    values[0] = json.dumps(original)
    candidate.write_text("\n".join(values) + "\n")
    with pytest.raises(ValueError, match="heldout exclusion"):
        build_pair(**inputs, output=tmp_path / "pair")
    assert not (tmp_path / "pair").exists()


def test_failed_publication_leaves_no_output(tmp_path, monkeypatch):
    inputs = pair_inputs(tmp_path)

    def fail_copy(*args):
        raise OSError("fixture copy failure")

    monkeypatch.setattr("experiments.post_training.inventory.build_taskforge_pair.shutil.copyfile", fail_copy)
    with pytest.raises(OSError, match="fixture copy failure"):
        build_pair(**inputs, output=tmp_path / "pair")
    assert not (tmp_path / "pair").exists()
    assert not list(tmp_path.glob("taskforge-pair-*"))
