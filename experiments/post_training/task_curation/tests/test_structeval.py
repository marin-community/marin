# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ScriptGrader, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

sys.path.insert(0, str(Path(__file__).parents[1] / "datasets"))

from experiments.post_training.task_curation.datasets import structeval
from experiments.post_training.task_curation.datasets.structeval_grade import extract_code, load_structure
from experiments.post_training.task_curation.datasets.structeval_paths import path_exists, tokenize_path

NATIVE = {
    "task_id": "000500",
    "task_name": "Text to JSON",
    "input_type": "Text",
    "output_type": "JSON",
    "query": "Please output JSON code:\n\nTask:\nSummarize a fictional novel as structured JSON.",
    "feature_requirements": "1. The field novel.title must be a string.",
    "query_example": "example",
    "VQA": [],
    "raw_output_metric": ["novel.title", "novel.author.name", "novel.publication.year"],
    "rendering": False,
}

REPLY = '<|BEGIN_CODE|>\n{"novel": {"title": "T", "author": {"name": "A"}, "publication": {"year": 2001}}}\n<|END_CODE|>'


def row(native):
    return RawRow(
        id="original-structeval-row",
        source=Source(dataset=structeval.REPO, revision=structeval.REVISION, row="fixture", importer_revision="test"),
        data=native,
    )


def converted(native):
    result = structeval.convert_problem(row(native), _context())
    assert isinstance(result, NormalizedTask)
    return result


def rejected(native):
    result = structeval.convert_problem(row(native), _context())
    assert isinstance(result, ImportRejection)
    return result


def _context():
    # The census compiles the recipe's pinned requirements to a hash lock and supplies its URL;
    # the test only needs a structurally valid requirements-only grading environment.
    return ConversionContext(
        {},
        EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="s3://test/structeval-requirements.lock"
        ),
    )


def test_the_non_renderable_row_converts_with_the_official_grader():
    task = converted(NATIVE).task
    assert isinstance(task.grader, ScriptGrader) and task.grader.judge is None
    config = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    payload = json.loads(resource_bytes(config))
    assert payload["output_type"] == "json" and payload["raw_output_metric"] == NATIVE["raw_output_metric"]
    assert payload["task_id"] == "000500"
    assert "VQA" not in json.dumps(payload)
    grade = next(resource for resource in task.resources.verifier if resource.path == "grade.py")
    script = resource_bytes(grade).decode()
    assert "BEGIN_CODE" in script and "path_exists" in script and "0.2 * parse_ok" in script
    assert not task.resources.oracle and not task.resources.worker


def test_renderable_rows_are_the_typed_grader_missing_exclusion():
    rejection = rejected({**NATIVE, "rendering": True, "VQA": [{"question": "q", "answer": "a"}]})
    assert rejection.reason.startswith("grader_missing_renderable_structeval")


def test_contract_defects_fail_closed():
    assert rejected({**NATIVE, "task_id": ""}).reason.startswith("invalid_structeval_metadata")
    assert rejected({key: value for key, value in NATIVE.items() if key != "query"}).reason.startswith(
        "incomplete_structeval_contract"
    )
    assert rejected({**NATIVE, "raw_output_metric": []}).reason.startswith("invalid_metric_paths")
    assert rejected({**NATIVE, "output_type": "Angular"}).reason.startswith("unsupported_structeval_output")
    assert rejected({**NATIVE, "raw_output_metric": ["novel.author[0.name"]}).reason.startswith("invalid_metric_path")


def test_the_completion_instruction_is_recorded():
    result = converted(NATIVE)
    (user,) = result.task.context.events
    assert user.role == "user" and user.content == NATIVE["query"] + structeval.COMPLETION_INSTRUCTION
    assert len(result.changes) == 1 and "completion-format instruction" in result.changes[0].reason


def test_the_official_reward_semantics():
    # A complete reply under the official envelope scores 1.0.
    structure, parse_ok = load_structure(extract_code(REPLY, "json"), "json")
    assert parse_ok == 1
    assert all(path_exists(structure, metric) for metric in NATIVE["raw_output_metric"])
    # A partial reply scores the path fraction; garbage scores 0 with parse_ok 0.
    partial, ok = load_structure(extract_code('<|BEGIN_CODE|>{"novel": {"title": "T"}}<|END_CODE|>', "json"), "json")
    assert ok == 1 and sum(path_exists(partial, m) for m in NATIVE["raw_output_metric"]) == 1
    assert load_structure("not json at all", "json")[1] == 0
    # Fenced replies win over markers, matching the official extractor's override.
    fenced = extract_code(
        '```json\n{"novel": {"title": "F"}}\n```\n<|BEGIN_CODE|>{"novel": {"title": "M"}}<|END_CODE|>', "json"
    )
    structure, _ = load_structure(fenced, "json")
    assert structure["novel"]["title"] == "F"


def test_the_official_path_grammar():
    assert tokenize_path("a.b[0].c") == ["a", "b", "[0]", "c"]
    assert tokenize_path("mission.payloads[1]..experiments") == ["mission", "payloads", "[1]", "experiments"]
    assert tokenize_path("`a.b`.c") == ["a.b", "c"]
    assert path_exists({"a": [{"b": 1}, {"b": 2}]}, "a.*.b")
    assert not path_exists({"a": {"*": 1}}, "a.*")
    assert path_exists({"@id": "x"}, "@id") and path_exists({"id": "x"}, "@id")
    csv_structure, ok = load_structure("name,year\nT,2001\n", "csv")
    assert ok == 1 and path_exists(csv_structure, "csv::name") and not path_exists(csv_structure, "csv::missing")


def test_the_source_module_binds_the_ruled_population():
    files = ("data/train-00000-of-00001.parquet",)
    source = structeval.source(files)
    assert source.info.id == structeval.SOURCE_ID and source.info.count == 2035
    assert source.config.source.revision == structeval.REVISION
    assert source.config.intended_use.value == "eval"
    assert source.config.grader.requirements == structeval.GRADER_REQUIREMENTS
    assert source.info.origin.value == "Tiger-lab"
    assert "grader-missing exclusion" in source.info.notes and "official" in source.info.notes
