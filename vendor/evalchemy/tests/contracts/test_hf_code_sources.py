"""HF adapters preserve prompts, reference programs, ordering, and identities."""

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.chat_benchmarks.BigCodeBench.eval_instruct import (
    BIGCODEBENCH_HARD_REVISION,
    BIGCODEBENCH_REVISION,
    BigCodeBenchBenchmark,
)
from eval.chat_benchmarks.HumanEvalPlus.eval_instruct import HumanEvalPlusBenchmark
from eval.chat_benchmarks.MBPP.eval_instruct import MBPPBenchmark
from eval.chat_benchmarks.MBPPPlus.eval_instruct import MBPPPlusBenchmark
from eval.chat_benchmarks.MultiPLE.eval_instruct import MultipleBenchmark
from eval.sources.common import SourceOptions


def _stage_parquet(root, repo, revision, filename, rows):
    path = root / repo / revision / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    Path(f"{path}.complete").touch()


def test_bigcode_hf_variants_preserve_prompts_references_and_provenance(tmp_path):
    row = {"task_id": "BigCodeBench/7", "test": "assert answer() == 7", "entry_point": "answer",
           "doc_struct": {"description": "seven"}, "code_prompt": "def answer():",
           "canonical_solution": "return 7", "complete_prompt": "# Complete\ndef answer():",
           "instruct_prompt": "Return seven"}
    for repo, revision in (("bigcode/bigcodebench", BIGCODEBENCH_REVISION),
                           ("bigcode/bigcodebench-hard", BIGCODEBENCH_HARD_REVISION)):
        _stage_parquet(tmp_path, repo, revision, "data/v0.1.3-00000-of-00001.parquet", [row])
    source = BigCodeBenchBenchmark()
    source.dataset_cache = str(tmp_path)
    specs = list(source.iter_specs(SourceOptions("BigCodeBench", source.SOURCE)))

    assert [spec.namespace for spec in specs] == ["instruct", "complete", "instruct-hard", "complete-hard"]
    assert [spec.doc["prompt"] for spec in specs] == [row["instruct_prompt"], row["complete_prompt"]] * 2
    assert [spec.source.dataset for spec in specs] == ["bigcode/bigcodebench"] * 2 + ["bigcode/bigcodebench-hard"] * 2
    assert [spec.source.revision for spec in specs] == [BIGCODEBENCH_REVISION] * 2 + [BIGCODEBENCH_HARD_REVISION] * 2
    for spec in specs:
        assert spec.source.split == "v0.1.3"
        assert spec.source_id == "BigCodeBench/7"
        assert spec.target == row["test"]
        assert spec.grader.reference["problem"] == {**{key: row[key] for key in (
            "task_id", "test", "entry_point", "doc_struct", "code_prompt", "canonical_solution")},
            "prompt": spec.doc["prompt"]}


def test_humanevalplus_raw_harness_retains_check_invocation(tmp_path):
    source = HumanEvalPlusBenchmark()
    source.dataset_cache = str(tmp_path)
    row = {"task_id": "HumanEval/0", "entry_point": "answer", "prompt": "def answer():",
           "test": "def check(candidate):\n    assert candidate() == 7", "canonical_solution": "return 7"}
    path = tmp_path / source.SOURCE.dataset / source.SOURCE.revision / "test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row) + "\n")
    Path(f"{path}.complete").touch()

    [spec] = source.iter_specs(SourceOptions("HumanEvalPlus", source.SOURCE))

    assert spec.source_id == "HumanEval/0"
    assert spec.target == row["test"] + "\ncheck(answer)"
    assert spec.grader.reference["problem"] == {**row, "test": spec.target}
    assert spec.requests[0].prompt[0]["content"].endswith("```python\ndef answer():\n```")


@pytest.mark.parametrize("suite, cls", [("MBPP", MBPPBenchmark), ("MBPPPlus", MBPPPlusBenchmark)])
def test_mbpp_hf_few_shot_order_and_native_reference_are_preserved(tmp_path, suite, cls):
    source = cls()
    source.dataset_cache = str(tmp_path)
    is_plus = suite == "MBPPPlus"
    rows = [{"task_id": index + 1, "prompt" if is_plus else "text": f"Problem {index}",
             "code": f"def answer():\n    return {index}", "test_list": [f"assert answer() == {index}"],
             **({"test": f"assert answer() == {index}"} if is_plus else {})} for index in range(14)]
    if is_plus:
        _stage_parquet(tmp_path, source.SOURCE.dataset, source.SOURCE.revision,
                       "data/test-00000-of-00001-d5781c9c51e02795.parquet", rows)
    else:
        # Deliberately interleave task IDs across HF splits; the original file was sorted.
        for index, split in enumerate(("prompt", "train", "test", "validation")):
            _stage_parquet(tmp_path, source.SOURCE.dataset, source.SOURCE.revision,
                           f"full/{split}-00000-of-00001.parquet", rows[index::4])
    [spec] = source.iter_specs(SourceOptions(suite, source.SOURCE, limit=1))

    expected_index = 0 if is_plus else 10
    assert spec.source_id == expected_index + 1
    prompt = spec.requests[0].prompt[0]["content"]
    assert prompt.index("Problem 1\n") < prompt.index("Problem 2\n") < prompt.index("Problem 3\n")
    assert prompt.endswith(f"Problem {expected_index}\n>>> Test Cases:\nassert answer() == {expected_index}\n")
    expected = rows[expected_index] if is_plus else {
        "task_id": 11, "prompt": "Problem 10", "test": ["assert answer() == 10"], "code": rows[10]["code"]}
    assert spec.grader.reference["problem"] == expected
    assert spec.grader.reference["timeout"] == (3.0 if is_plus else 10.0)
    assert spec.target == expected["test"]


def test_multiple_hf_preserves_lua_overlays_and_humaneval_then_mbpp_order(tmp_path):
    source = MultipleBenchmark(languages=["lua"])
    source.dataset_cache = str(tmp_path)
    ids = ["HumanEval_69_search", "HumanEval_142_sum_squares"]
    rows = [{"name": identity, "prompt": "luathon", "tests": "assert(true)", "language": "lua"} for identity in ids]
    _stage_parquet(tmp_path, source.SOURCE.dataset, source.SOURCE.revision,
                   "humaneval-lua/test-00000-of-00001.parquet", rows)
    _stage_parquet(tmp_path, source.SOURCE.dataset, source.SOURCE.revision,
                   "mbpp-lua/test-00000-of-00001.parquet",
                   [{"name": "mbpp_1", "prompt": "```luathon\nfunction answer()", "tests": "assert(true)", "language": "lua"}])

    specs = list(source.iter_specs(SourceOptions("MultiPLE", source.SOURCE)))

    assert [spec.source_id for spec in specs] == ids + ["mbpp_1"]
    assert "greater than \n" in specs[0].doc["prompt"]
    assert "-- >>> lst\n-- {1, 2, 3}\n-- >>> lst\n-- {}\n" in specs[1].doc["prompt"]
    assert specs[2].doc["prompt"] == "```lua\nfunction answer()"
    assert all(spec.target == "assert(true)" for spec in specs)


# Captured from the complete bundled sources before their removal. SourceRef is
# intentionally excluded: provenance now identifies immutable upstream commits.
@pytest.mark.data_integration
@pytest.mark.timeout(300)
@pytest.mark.parametrize("suite, cls, count, digest", [
    ("CruxEval", "CruxEvalBenchmark", 1600, "9bc0b95dcd49aede4bd5d9577592a9c4a54574c03e05e7772bed961a06d6af9a"),
    ("BigCodeBench", "BigCodeBenchBenchmark", 2576, "341291ac2d6341c766b75a21c01c28bcab5b6c4804a0c3530ffc9f9f04d9071b"),
    ("MBPP", "MBPPBenchmark", 500, "3cf7b03b589bfc79994c7f6816f51d2cb844bf3eb2aed286a76a64f795df2a32"),
    ("MBPPPlus", "MBPPPlusBenchmark", 378, "af631f48023e572a3364c609ebc0fc389472ce298e2d1f71efb035bd70f53fad"),
    ("HumanEvalPlus", "HumanEvalPlusBenchmark", 164, "4b0576f35c6505088dbbce8d85d00c80cb747811dae2362ead14f20748222ae8"),
    ("MultiPLE", "MultipleBenchmark", 5477, "9d9d65797608aae439c95788cab5dfa6d8087afcf5f661ca7465b60ff612d40e"),
])
def test_hf_complete_specs_match_original_bundled_source(suite, cls, count, digest):
    source = getattr(importlib.import_module(f"eval.chat_benchmarks.{suite}.eval_instruct"), cls)()
    checksum = hashlib.sha256()
    actual_count = 0
    for spec in source.iter_specs(SourceOptions(suite, source.SOURCE)):
        item = asdict(spec)
        item.pop("source")
        checksum.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode())
        actual_count += 1

    assert actual_count == count
    assert checksum.hexdigest() == digest
