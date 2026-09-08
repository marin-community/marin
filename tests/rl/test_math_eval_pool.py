# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-source membership and reproducible manifest behavior."""

import json
from copy import deepcopy

import pyarrow.parquet as pq
import pytest

from experiments.post_training.curriculum_rl.pool import GSM8K_BIN, _pool_record
from experiments.post_training.math_eval.pool import (
    SourceRows,
    build_pool,
    prompt_hash,
    prompt_length_report,
    write_pool,
)
from experiments.post_training.math_eval.sample import sample


def source(questions, *, name="fixture", split="hash"):
    records = [
        _pool_record(question=q, answer="5", pool_bin=GSM8K_BIN, split="train" if split == "hash" else "test", index=i)
        for i, q in enumerate(questions)
    ]
    return SourceRows(name, "a" * 40, "MIT", records, split)


def build(sources, *, cap=1024):
    # A byte tokenizer is a deterministic I/O-boundary stand-in; real template rendering runs.
    return build_pool(
        sources,
        {"qwen": lambda text: list(text.encode()), "snowball": lambda text: list(text.encode())},
        version="1.0.0",
        code_sha="b" * 40,
        tokenizer_hashes={"qwen": "c" * 64, "snowball": "d" * 64},
        max_prompt_tokens=cap,
    )


def test_pool_roundtrip_preserves_split_lock_and_model_template_ids(tmp_path):
    sources = [source([f"Compute unique expression {i} + {i * i}." for i in range(100)])]
    first, second = build(sources), build(sources)
    assert first.selection == second.selection
    locks = [set(ids) for ids in first.selection["rows"].values()]
    assert len(set.union(*locks)) == sum(map(len, locks)) == 100
    assert all(first.selection["rows"][split] for split in ("train", "dev", "heldout"))
    output = tmp_path / "pool"
    write_pool(first, output)
    saved = pq.read_table(output / "manifest.parquet").to_pylist()
    assert saved == first.manifest
    selection = json.loads((output / "selection.json").read_text())
    for model in ("qwen", "snowball"):
        rows = pq.read_table(output / model / "heldout.parquet").to_pylist()
        assert [row["extra_info"]["prompt_sha256"] for row in rows] == selection["rows"]["heldout"]
        assert {row["extra_info"]["prompt_template_id"] for row in rows} == {selection["prompt_template_ids"][model]}


def test_cross_source_duplicates_keep_benchmark_out_of_training():
    question = "How many apples remain after giving two away?"
    result = build([source([question]), source([question.upper()], name="benchmark", split="heldout")])
    assert len(result.manifest) == 1
    assert result.manifest[0]["split"] == "heldout"
    assert result.selection["rows"]["train"] == []
    assert result.selection["dropped"][0]["reason"] == "exact_duplicate"


def test_near_duplicate_is_removed_across_sources():
    question = (
        "A grocery store had exactly forty red apples for sale in its front display this morning. How many apples?"
    )
    result = build([source([question]), source([question + "!"], name="benchmark", split="ood")])
    assert len(result.manifest) == 1
    assert result.manifest[0]["split"] == "ood"
    assert result.selection["dropped"][0]["reason"] == "near_duplicate"


def test_conflicting_gold_for_duplicate_question_fails():
    first, second = source(["What is 2 + 3?"]), source(["What is 2 + 3?"], name="benchmark", split="heldout")
    second.records[0]["reward_spec"]["ground_truth"] = "6"
    second.records[0]["reward_model"]["ground_truth"] = "6"
    with pytest.raises(ValueError, match="Conflicting golds"):
        build([first, second])


def test_test_source_cannot_be_hash_assigned_into_training():
    heldout = source(["What is 2 + 3?"], split="heldout")
    with pytest.raises(ValueError, match="cannot be hash-assigned"):
        build([SourceRows(heldout.source, heldout.revision, heldout.license, heldout.records, "hash")])


def test_eval_length_overflow_fails_instead_of_shrinking_membership():
    with pytest.raises(ValueError, match="exceeds the model prompt cap"):
        build([source(["What is 2 + 3?"], split="heldout")], cap=100)


def test_prompt_identity_normalizes_layout_but_preserves_numbers():
    assert prompt_hash("What is 2 + 3?") == prompt_hash("  WHAT is 2  + 3? ")
    assert prompt_hash("What is 2 + 3?") != prompt_hash("What is 2 + 4?")


def test_sampler_is_seeded_and_refuses_edited_split_locks():
    pool = build([source([f"Compute unique expression {i} + {i * i}." for i in range(100)])])
    kwargs = {"model": "qwen", "split": "train"}
    first = sample(pool.manifest, pool.selection, [GSM8K_BIN.name], 10, 17, **kwargs)
    repeat = sample(pool.manifest, pool.selection, [GSM8K_BIN.name], 10, 17, **kwargs)
    other = sample(pool.manifest, pool.selection, [GSM8K_BIN.name], 10, 29, **kwargs)
    assert first.receipt == repeat.receipt
    assert first.receipt["result_sha256"] != other.receipt["result_sha256"]
    assert {row["split"] for row in first.rows} == {"train"}
    changed = deepcopy(pool.selection)
    changed["rows"]["train"].append(changed["rows"]["heldout"][0])
    with pytest.raises(ValueError, match="Split membership"):
        sample(pool.manifest, changed, [GSM8K_BIN.name], 10, 17, **kwargs)


def test_sampler_never_borrows_a_benchmark_bin_from_another_split():
    pool = build([source(["A unique benchmark question"], split="heldout")])
    with pytest.raises(ValueError, match="no eligible rows"):
        sample(pool.manifest, pool.selection, [GSM8K_BIN.name], 1, 17, model="qwen", split="train")


def test_length_audit_reports_all_rows_without_dropping_evaluation_overflows():
    long_question = "A long problem: " + "x" * 1000
    inputs = [source(["What is 2 + 3?", long_question], split="heldout")]
    report = prompt_length_report(
        inputs, {"qwen": lambda text: list(text.encode()), "snowball": lambda text: list(text.encode())}, cap=700
    )
    assert len(report["overflows"]) == 1
    assert report["overflows"][0]["prompt_sha256"] == prompt_hash(long_question)
    assert all(row["rows"] == 2 and row["over_cap"] == 1 for row in report["sources"].values())
    assert "x" * 1000 not in json.dumps(report)
