# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the decontam eval-corpus text building (marin#6852 cluster B).

Passage-bearing reading-comprehension / QA eval docs must index question+answer
but not the public passage, so a corpus doc that merely quotes the passage is
not falsely flagged. Non-passage docs are unchanged.
"""

import builtins
import io
import json
import sys
import types
import zipfile
from dataclasses import replace

import pyarrow.parquet as pq
import pytest
from rigging.timing import ExponentialBackoff

from experiments.datakit.decontam import prepare_eval_corpus
from experiments.datakit.decontam.prepare_eval_corpus import (
    _PASSAGE_FIELDS,
    AA_EVALS,
    AAEvalConfig,
    AARecordType,
    _aa_records,
    _iter_aa_rows,
    _lmh_doc_text,
)

# A rendered prompt embeds the passage (as lm-eval-harness doc_to_text does).
_PASSAGE = "The rain had continued for a week and a flood created a big river by the farm."


def _prompt(doc: dict) -> str:
    body = " ".join(str(doc.get(k, "")) for k in ("article", "passage", "context", "premise", "story"))
    return f"Read: {body}\nQuestion: {doc.get('question', '')}"


def _target(doc: dict) -> str:
    return str(doc.get("answer", ""))


def test_lmh_passage_doc_drops_passage_keeps_qa():
    """RC doc: passage dropped (raw field + doc_to_text), question/answer/options kept."""
    doc = {
        "article": _PASSAGE,
        "question": "What did Nancy do when the flood came",
        "answer": "C",
        "options": ["ran away", "hid inside", "gathered her cows", "slept through it"],
    }
    text = _lmh_doc_text(doc, _prompt, _target)
    assert _PASSAGE not in text, "public passage must not be indexed"
    assert "What did Nancy do when the flood came" in text  # question kept
    assert "gathered her cows" in text  # options kept
    assert "C" in text  # answer kept


def test_lmh_non_passage_doc_unchanged():
    """No passage field → doc_to_text (the question) is kept as before."""
    doc = {"question": "What is the sum of two and two", "answer": "4"}
    text = _lmh_doc_text(doc, _prompt, _target)
    assert "What is the sum of two and two" in text
    assert "4" in text


@pytest.mark.parametrize("field", sorted(_PASSAGE_FIELDS))
def test_lmh_every_passage_field_suppresses_passage(field: str):
    """Each passage-like field name triggers suppression; the question survives."""
    doc = {
        field: "UNIQUE_PUBLIC_PASSAGE_MARKER spanning several ordinary words here",
        "question": "Q_MARKER here",
        "answer": "yes",
    }
    text = _lmh_doc_text(doc, lambda d: f"renders {d.get(field, '')} then {d['question']}", _target)
    assert "UNIQUE_PUBLIC_PASSAGE_MARKER" not in text, field
    assert "Q_MARKER" in text, field
    assert "yes" in text, field


def test_lmh_doc_to_text_exception_is_tolerated():
    """A doc_to_text that raises still yields the answer + raw fields (no crash)."""

    def boom(_doc):
        raise RuntimeError("no template")

    doc = {"question": "Q here", "answer": "42"}
    text = _lmh_doc_text(doc, boom, _target)
    assert "Q here" in text  # from raw fields
    assert "42" in text  # answer


def _aa_config(**changes) -> AAEvalConfig:
    values = {
        "name": "Example Eval",
        "subdir": "example",
        "source_revision": "0123456789abcdef",
        "expected_records": 1,
        "official_records": 1,
        "hf_id": "owner/example",
        "subset": None,
        "split": "test",
        "text_fields": ("question", "answer"),
    }
    values.update(changes)
    return AAEvalConfig(**values)


def test_aa_hf_loader_uses_pinned_revision(monkeypatch):
    calls = []

    def load_dataset(dataset_id, *, name, split, revision):
        calls.append((dataset_id, name, split, revision))
        return [{"question": "Question", "answer": "Answer"}]

    monkeypatch.setattr(prepare_eval_corpus, "load_dataset", load_dataset)

    assert list(_iter_aa_rows(_aa_config())) == [{"question": "Question", "answer": "Answer"}]
    assert calls == [("owner/example", None, "test", "0123456789abcdef")]


def test_hle_includes_multimodal_rows():
    hle = next(cfg for cfg in AA_EVALS if cfg.name == "Humanity's Last Exam")
    cfg = replace(hle, expected_records=1, official_records=1)
    [record] = _aa_records(
        [{"id": "image-task", "question": "Question with a figure", "answer": "Answer", "image": {"bytes": b"x"}}],
        cfg,
    )
    assert record["id"] == "hle-image-task"
    assert record["text"] == "Question with a figure\n\nAnswer"


def test_scicode_expands_each_subproblem_with_its_prompt_context():
    cfg = _aa_config(
        name="SciCode",
        subdir="scicode",
        expected_records=2,
        official_records=2,
        record_type=AARecordType.SCICODE_SUBPROBLEM,
        text_fields=(),
    )
    rows = [
        {
            "problem_id": "7",
            "problem_description_main": "Main problem",
            "problem_background_main": "Main background",
            "required_dependencies": "import numpy as np",
            "sub_steps": [
                {
                    "step_number": "7.1",
                    "step_description_prompt": "First prompt",
                    "step_background": "First background",
                    "function_header": "def first():",
                    "test_cases": ["assert first() == 1"],
                },
                {
                    "step_number": "7.2",
                    "step_description_prompt": "Second prompt",
                    "step_background": "Second background",
                    "function_header": "def second():",
                    "test_cases": ["assert second() == 2"],
                },
            ],
        }
    ]

    records = _aa_records(rows, cfg)

    assert [record["id"] for record in records] == ["scicode-7.1", "scicode-7.2"]
    assert "Main problem" in records[0]["text"]
    assert "First background" in records[0]["text"]
    assert "def first():" in records[0]["text"]
    assert "assert first() == 1" in records[0]["text"]
    assert "Second prompt" not in records[0]["text"]


def test_tau3_records_include_user_scenario_and_hidden_criteria():
    cfg = _aa_config(
        name="tau3-Banking",
        subdir="tau3_banking",
        expected_records=1,
        official_records=1,
        record_type=AARecordType.TAU3_TASK,
        text_fields=(),
    )
    rows = [
        {
            "id": "task_001",
            "user_scenario": {"instructions": "Ask for the highest cash-back card."},
            "evaluation_criteria": {"actions": [{"name": "apply_for_credit_card"}]},
        }
    ]

    [record] = _aa_records(rows, cfg)

    assert record["id"] == "tau3_banking-task_001"
    assert "highest cash-back card" in record["text"]
    assert "apply_for_credit_card" in record["text"]


def test_terminal_bench_archive_pairs_each_instruction_with_its_solution(monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("terminal-bench/tasks/task-b/instruction.md", "Instruction B")
        zf.writestr("terminal-bench/tasks/task-b/solution/solve.sh", "Solution B")
        zf.writestr("terminal-bench/tasks/task-a/instruction.md", "Instruction A")
        zf.writestr("terminal-bench/tasks/task-a/solution/solve.sh", "Solution A")
    monkeypatch.setattr(prepare_eval_corpus, "_download", lambda _url: archive.getvalue())
    cfg = _aa_config(
        name="Terminal-Bench v2.1",
        subdir="terminal_bench_2_1",
        source_url="https://example.test/archive.zip",
        expected_records=2,
        official_records=2,
        record_type=AARecordType.TERMINAL_BENCH_TASK,
        hf_id=None,
        text_fields=("instruction", "solution"),
    )

    records = _aa_records(list(_iter_aa_rows(cfg)), cfg)

    assert records == [
        {"id": "terminal_bench_2_1-task-a", "text": "Instruction A\n\nSolution A"},
        {"id": "terminal_bench_2_1-task-b", "text": "Instruction B\n\nSolution B"},
    ]


def test_aa_record_count_mismatch_stops_preparation():
    cfg = _aa_config(expected_records=2, official_records=2)

    with pytest.raises(ValueError, match="Example Eval: expected 2 records, extracted 1"):
        _aa_records([{"question": "Only question", "answer": "Only answer"}], cfg)


def test_aa_manifest_is_immutable_within_a_corpus_version(tmp_path, monkeypatch):
    manifest = prepare_eval_corpus._aa_manifest("complete")
    manifest_path = tmp_path / prepare_eval_corpus.AA_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    original_bytes = manifest_path.read_bytes()
    monkeypatch.setattr(prepare_eval_corpus, "_output_root", lambda: str(tmp_path))
    monkeypatch.setattr(
        prepare_eval_corpus,
        "_iter_aa_rows",
        lambda _cfg: pytest.fail("a sealed corpus version must not load AA source rows"),
    )

    result = prepare_eval_corpus._prepare_aa()

    assert result == manifest
    assert manifest_path.read_bytes() == original_bytes


def test_aa_manifest_rejects_config_change_within_a_corpus_version(tmp_path, monkeypatch):
    manifest = prepare_eval_corpus._aa_manifest("complete")
    manifest["benchmarks"][0]["source_revision"] = "changed-revision"
    manifest_path = tmp_path / prepare_eval_corpus.AA_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(prepare_eval_corpus, "_output_root", lambda: str(tmp_path))

    with pytest.raises(ValueError, match="EVAL_CORPUS_VERSION"):
        prepare_eval_corpus._prepare_aa()


def test_lmh_manifest_is_immutable_within_a_corpus_version(tmp_path, monkeypatch):
    manifest = {
        "schema_version": 1,
        "corpus_version": prepare_eval_corpus.EVAL_CORPUS_VERSION,
        "required": False,
        "status": "complete_with_failures",
        "artifact_root": prepare_eval_corpus.LMH_EVALS_RELATIVE,
        "extraction_version": prepare_eval_corpus._LMH_EXTRACTION_VERSION,
        "required_tasks": {
            task: [f"{task}_{i}" for i in range(leaves)]
            for task, leaves in prepare_eval_corpus.required_lmh_tasks().items()
        },
        "included_leaf_tasks": ["existing-task"],
        "artifacts": [{"task": "existing-task", "artifact": "existing-task/eval.parquet", "records": 1}],
        "failed": [],
    }
    manifest_path = tmp_path / prepare_eval_corpus.LMH_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    original_bytes = manifest_path.read_bytes()
    monkeypatch.setattr(prepare_eval_corpus, "_output_root", lambda: str(tmp_path))
    monkeypatch.setattr(
        prepare_eval_corpus,
        "patch_hf_dataset_loading",
        lambda: pytest.fail("a sealed corpus version must not load tasks again"),
    )

    result = prepare_eval_corpus._prepare_lmh()

    assert result == manifest
    assert manifest_path.read_bytes() == original_bytes


def test_lmh_manifest_rejects_leaf_count_change_within_a_corpus_version(tmp_path, monkeypatch):
    manifest = {
        "corpus_version": prepare_eval_corpus.EVAL_CORPUS_VERSION,
        "status": "complete",
        "artifact_root": prepare_eval_corpus.LMH_EVALS_RELATIVE,
        "extraction_version": prepare_eval_corpus._LMH_EXTRACTION_VERSION,
        "required_tasks": {"group": ["group_a", "group_b"]},
    }
    manifest_path = tmp_path / prepare_eval_corpus.LMH_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(prepare_eval_corpus, "_output_root", lambda: str(tmp_path))
    monkeypatch.setattr(prepare_eval_corpus, "_lmh_task_names", lambda: ["group"])
    monkeypatch.setattr(prepare_eval_corpus, "BEST_EFFORT_LMH_TASKS", {})
    monkeypatch.setattr(prepare_eval_corpus, "LMH_LEAF_TASK_COUNTS", {"group": 3})

    with pytest.raises(ValueError, match="EVAL_CORPUS_VERSION"):
        prepare_eval_corpus._prepare_lmh()


def _fake_task(docs: list[dict]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        test_docs=lambda: docs,
        validation_docs=lambda: [],
        training_docs=lambda: [],
        doc_to_text=lambda doc: doc["question"],
        doc_to_target=lambda doc: doc["answer"],
    )


def _use_fake_lm_eval(
    monkeypatch,
    tmp_path,
    get_task_dict,
    task_names: list[str],
    best_effort: dict[str, str],
    leaf_counts: dict[str, int],
) -> None:
    lm_eval = types.ModuleType("lm_eval")
    lm_eval_tasks = types.ModuleType("lm_eval.tasks")
    lm_eval_tasks.get_task_dict = get_task_dict
    lm_eval.tasks = lm_eval_tasks
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", lm_eval_tasks)
    monkeypatch.setattr(prepare_eval_corpus, "marin_prefix", lambda: str(tmp_path))
    monkeypatch.setattr(prepare_eval_corpus, "patch_hf_dataset_loading", lambda: None)
    monkeypatch.setattr(prepare_eval_corpus, "_lmh_task_names", lambda: task_names)
    monkeypatch.setattr(prepare_eval_corpus, "BEST_EFFORT_LMH_TASKS", best_effort)
    monkeypatch.setattr(prepare_eval_corpus, "LMH_LEAF_TASK_COUNTS", leaf_counts)
    monkeypatch.setattr(
        prepare_eval_corpus, "LOAD_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001, jitter=0.0)
    )


def test_lmh_artifacts_are_stored_below_the_versioned_corpus_root(tmp_path, monkeypatch):
    task = _fake_task([{"question": "What is two plus two?", "answer": "4"}])
    _use_fake_lm_eval(
        monkeypatch, tmp_path, lambda _names: {"example-task": task}, ["example-task"], best_effort={}, leaf_counts={}
    )

    manifest = prepare_eval_corpus._prepare_lmh()

    artifact_path = tmp_path / prepare_eval_corpus.EVALS_RELATIVE / "lmh/example-task/eval.parquet"
    assert artifact_path.exists()
    assert manifest["artifacts"] == [{"task": "example-task", "artifact": "example-task/eval.parquet", "records": 1}]
    assert manifest["required_tasks"] == {"example-task": ["example-task"]}


def _load_group(names: list[str]) -> dict:
    [name] = names
    if name != "group":
        return {name: _fake_task([{"question": "What is two plus two?", "answer": "4"}])}
    return {
        "group": {
            "group_a": _fake_task([{"question": "What is one plus one?", "answer": "2"}]),
            "group_b": _fake_task([{"question": "What is one plus two?", "answer": "3"}]),
        }
    }


def _load_group_with_empty_leaf(names: list[str]) -> dict:
    if names != ["group"]:
        return _load_group(names)
    return {"group": {**_load_group(names)["group"], "group_b": _fake_task([])}}


def _load_group_with_leaf_without_text(names: list[str]) -> dict:
    if names != ["group"]:
        return _load_group(names)
    return {"group": {**_load_group(names)["group"], "group_b": _fake_task([{"question": "", "answer": ""}])}}


def _load_group_without_one_leaf(names: list[str]) -> dict:
    if names != ["group"]:
        return _load_group(names)
    return {"group": {"group_a": _load_group(names)["group"]["group_a"]}}


def _reject_group(names: list[str]) -> dict:
    if names == ["group"]:
        raise ConnectionError("Server disconnected without sending a response.")
    return _load_group(names)


@pytest.mark.parametrize(
    "get_task_dict",
    [_reject_group, _load_group_with_empty_leaf, _load_group_with_leaf_without_text, _load_group_without_one_leaf],
    ids=["task-load-fails", "one-leaf-has-no-docs", "one-leaf-has-no-text", "one-leaf-is-missing"],
)
def test_lmh_task_failure_does_not_seal_manifest(tmp_path, monkeypatch, get_task_dict):
    """A task that is not in the best-effort table must load all of its pinned leaf tasks."""
    _use_fake_lm_eval(
        monkeypatch, tmp_path, get_task_dict, ["group", "other-task"], best_effort={}, leaf_counts={"group": 2}
    )

    with pytest.raises(ValueError, match="group"):
        prepare_eval_corpus._prepare_lmh()

    assert not (tmp_path / prepare_eval_corpus.EVALS_RELATIVE / prepare_eval_corpus.LMH_MANIFEST_RELATIVE).exists()


def test_lmh_best_effort_task_failure_is_recorded_in_the_sealed_manifest(tmp_path, monkeypatch):
    _use_fake_lm_eval(
        monkeypatch,
        tmp_path,
        _reject_group,
        ["group", "other-task"],
        best_effort={"group": "the Hub drops the connection"},
        leaf_counts={},
    )

    manifest = prepare_eval_corpus._prepare_lmh()

    assert manifest["status"] == "complete_with_failures"
    assert manifest["required_tasks"] == {"other-task": ["other-task"]}
    assert manifest["best_effort_tasks"] == {"group": "the Hub drops the connection"}
    assert [entry["task"] for entry in manifest["failed"]] == ["group"]


def _use_fake_uncheatable(monkeypatch, tmp_path, dumps: dict[str, list[str]], expected: dict[str, int]) -> None:
    monkeypatch.setattr(prepare_eval_corpus, "_output_root", lambda: str(tmp_path))
    monkeypatch.setattr(prepare_eval_corpus, "UNCHEATABLE_RECORDS", expected)
    urls = {prepare_eval_corpus._uncheatable_url(subset): texts for subset, texts in dumps.items()}
    monkeypatch.setattr(prepare_eval_corpus, "_download", lambda url: json.dumps(urls[url]).encode())


def test_uncheatable_suite_writes_each_subset_and_seals_its_manifest(tmp_path, monkeypatch):
    _use_fake_uncheatable(
        monkeypatch, tmp_path, {"bbc_news": ["First article.", "Second article."]}, expected={"bbc_news": 2}
    )

    manifest = prepare_eval_corpus._prepare_uncheatable()

    artifact = tmp_path / "uncheatable" / "bbc_news" / "test.parquet"
    assert pq.read_table(artifact).to_pylist() == [
        {"id": "uncheatable-bbc_news-0", "text": "First article."},
        {"id": "uncheatable-bbc_news-1", "text": "Second article."},
    ]
    assert manifest["status"] == "complete"
    assert manifest["required"] is True
    assert [entry["name"] for entry in manifest["benchmarks"]] == ["Uncheatable Eval bbc_news"]


def test_uncheatable_record_count_mismatch_does_not_seal_manifest(tmp_path, monkeypatch):
    _use_fake_uncheatable(monkeypatch, tmp_path, {"bbc_news": ["Only article."]}, expected={"bbc_news": 2})

    with pytest.raises(ValueError, match="Uncheatable Eval bbc_news: wrote 1 records, expected 2"):
        prepare_eval_corpus._prepare_uncheatable()

    manifest = json.loads((tmp_path / prepare_eval_corpus.UNCHEATABLE_MANIFEST_RELATIVE).read_text())
    assert manifest["status"] == "building"


def test_lmh_import_failure_does_not_seal_manifest(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def reject_lm_eval(name, *args, **kwargs):
        if name == "lm_eval.tasks":
            raise ImportError("lm_eval is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(prepare_eval_corpus, "_output_root", lambda: str(tmp_path))
    monkeypatch.setattr(prepare_eval_corpus, "patch_hf_dataset_loading", lambda: None)
    monkeypatch.setattr(prepare_eval_corpus, "_lmh_task_names", lambda: ["example-task"])
    monkeypatch.setattr(prepare_eval_corpus, "BEST_EFFORT_LMH_TASKS", {})
    monkeypatch.setattr(prepare_eval_corpus, "LMH_LEAF_TASK_COUNTS", {})
    monkeypatch.setattr(builtins, "__import__", reject_lm_eval)

    with pytest.raises(ImportError, match="lm_eval is unavailable"):
        prepare_eval_corpus._prepare_lmh()

    assert not (tmp_path / prepare_eval_corpus.LMH_MANIFEST_RELATIVE).exists()
