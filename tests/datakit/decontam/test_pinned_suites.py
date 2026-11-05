# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record extraction for the AAII v5 and eval-policy suites, and their sealed manifests."""

import json
from pathlib import Path

import pyarrow.parquet as pq
from marin.datakit.decon import NGramConfig, _validate_required_eval_manifest

from experiments.datakit.decontam import prepare_eval_corpus
from experiments.datakit.decontam.aaii_v5_evals import _automationbench_tasks, gdp_pdf_rows, terminal_bench
from experiments.datakit.decontam.config import ITEM_MATCH
from experiments.datakit.decontam.eval_policy_agentic_evals import Task, records, strip_boilerplate
from experiments.datakit.decontam.pinned_evals import PinnedEval, item_records
from experiments.datakit.reference_pipeline import MIN_MATCHED_FEATURES, NGRAM_LENGTH, OVERLAP_THRESHOLD
from tests.datakit.test_decon import _run_decon_one_shot

_INSTRUCTION = (
    "Write a script that reads the sensor log in the data directory, drops readings taken while the device "
    "was calibrating, and prints the hourly mean temperature as a table."
)
_SOLUTION = (
    "#!/bin/bash\npython3 - <<'EOF'\nimport csv\n"
    "rows = [r for r in csv.DictReader(open('log.csv')) if r['mode'] != 'cal']\nEOF\n"
)
_SHARED_RUNNER = (
    "#!/bin/bash\n# Shared harness: install pytest, run every test under tests/ and write the reward file.\n"
    "pip install pytest && pytest tests/ && echo 1 > /logs/reward.txt\n"
)


def _write_harbor_cache(cache_dir: Path, repo: str, revision: str, files: dict[str, str]) -> None:
    root = cache_dir / repo / revision
    tree = [{"path": path, "type": "blob", "size": len(text)} for path, text in files.items()]
    (root).mkdir(parents=True)
    (root / "_tree.json").write_text(json.dumps({"truncated": False, "tree": tree}))
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)


def test_terminal_bench_indexes_task_views_without_shared_harness_or_data(tmp_path: Path):
    files = {}
    for n in range(3):
        task = f"tasks/sensor-means-{n}"
        files[f"{task}/instruction.md"] = f"{_INSTRUCTION} Use station {n}."
        files[f"{task}/solution/solve.sh"] = f"{_SOLUTION}# station {n} uses a {n * 10} minute warmup window\n"
        # A code file whose name contains "lock" is still part of the solution.
        files[f"{task}/solution/businessClock.ts"] = (
            f"// Station {n} reports in local business hours, so shift every reading into UTC before\n"
            "// grouping it by hour, and skip the readings logged while the clock was being adjusted.\n"
        )
        files[f"{task}/solution/pnpm-lock.yaml"] = "lockfileVersion: '9.0'\n"
        files[f"{task}/tests/test.sh"] = _SHARED_RUNNER
        files[f"{task}/tests/expected.json"] = json.dumps({"hourly_means": [20.5, 21.0]})
        files[f"{task}/environment/Dockerfile"] = "FROM python:3.12"
    _write_harbor_cache(tmp_path, "owner/bench", "abc123", files)

    rows = {row["id"]: row["text"] for row in terminal_bench("bench", "owner/bench", "abc123", 1, tmp_path)}

    assert sorted(rows) == [
        f"bench-sensor-means-{n}{view}" for n in range(3) for view in ("", "-instruction", "-solution")
    ]
    full = rows["bench-sensor-means-1"]
    assert "Use station 1." in full and "10 minute warmup window" in full
    # The runner shared by every task, test data and the environment are not task content.
    assert "Shared harness" not in full
    assert "hourly_means" not in full
    assert "FROM python" not in full
    assert "business hours" in rows["bench-sensor-means-1-solution"]
    assert "lockfileVersion" not in full


def test_gdp_pdf_indexes_prompt_and_rubric_without_empty_criteria():
    item = {
        "task_id": "t1",
        "prompt": (
            "According to the attached filing, how did services segment revenue change between the two most recent "
            "fiscal years, and which line items in the segment note explain most of that change in revenue?"
        ),
        **{f"rubric - {n}. criterion": "None" for n in range(1, 31)},
    }
    item["rubric - 1. criterion"] = (
        "The response should state that services revenue grew by twelve percent year over year, driven by renewals."
    )
    item["rubric - 2. criterion"] = (
        "The response should not treat the one-time licensing gain from the patent settlement as recurring revenue."
    )

    rows = {row["id"]: row["text"] for row in gdp_pdf_rows([item])}

    assert rows["gdp_pdf-t1-instruction"] == item["prompt"]
    assert rows["gdp_pdf-t1-rubric"] == "\n".join([item["rubric - 1. criterion"], item["rubric - 2. criterion"]])
    assert "None" not in rows["gdp_pdf-t1"]


def test_automationbench_tasks_are_read_without_running_the_module():
    source = '''
CRM = "Acme CRM"


def task_follow_up():
    """Docstring."""
    owner = "Dana"
    return {
        "task": "sales_follow_up",
        "prompt": [{"role": "user", "content": f"Ask {owner} to follow up on every open deal in {CRM}."}],
        "info": {"assertions": [{"tool": "send_email", "to": owner}], "initial_state": {"deals": 3 * [1]}},
    }


def get_sales_dataset():
    tasks = [task_follow_up]
    return tasks
'''

    (task,) = _automationbench_tasks(source, "sales")

    assert task["task"] == "sales_follow_up"
    assert task["prompt"][0]["content"] == "Ask Dana to follow up on every open deal in Acme CRM."
    assert task["info"]["initial_state"] == {"deals": [1, 1, 1]}


def test_agentic_records_drop_template_lines_shared_by_most_tasks():
    template = "Run the verifier with uv run pytest and write the reward to /logs/verifier/reward.txt"
    tasks = [
        Task(
            f"task-{n}",
            {
                "instruction:instruction.md": f"{_INSTRUCTION} Report readings for station {n}.",
                "tests:tests/test.sh": (
                    f"{template}\nassert the station {n} mean equals the expected value in the fixture"
                ),
            },
        )
        for n in range(6)
    ]

    rows = {row["id"]: row["text"] for row in records("policy", strip_boilerplate(tasks, 0.5))}

    assert all(template not in text for text in rows.values())
    assert "station 4 mean" in rows["policy-task-4"]
    # The stripped tests view has fewer than MIN_VIEW_WORDS words with a letter, so only the full record holds it.
    assert "policy-task-4-tests" not in rows


def test_pinned_suite_manifest_is_accepted_by_the_bloom_validator(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prepare_eval_corpus, "_output_root", lambda: str(tmp_path))
    texts = [f"{_INSTRUCTION} Station {n} logs every five minutes." for n in range(2)]
    evals = (
        PinnedEval(
            "Example Bench",
            "example_bench",
            "https://example.com/bench",
            "rev1",
            2,
            lambda _cache: [{"id": f"example_bench-{n}", "text": text} for n, text in enumerate(texts)],
        ),
    )

    manifest = prepare_eval_corpus._prepare_pinned_suite("example", "example/_manifest.json", "Example suite", evals)

    assert manifest["status"] == "complete"
    paths = _validate_required_eval_manifest(
        str(tmp_path / "example" / "_manifest.json"),
        prepare_eval_corpus.EVAL_CORPUS_VERSION,
        ("Example Bench",),
        text_field="text",
        ngram=NGramConfig(min_matched_features=2),
    )
    assert [pq.read_table(path).num_rows for path in paths] == [2]


_ISSUE = (
    "Calling `Table.join` with a key column that holds nulls drops every row whose key is null,\n"
    "even when `keep_nulls=True` is passed.\n\n"
    "Steps to reproduce:\n"
    "1. Build two tables that share a nullable `id` column with three null keys each.\n"
    "2. Join them with `keep_nulls=True` and count the rows of the result.\n\n"
    "Expected six null-key rows in the result, but the join returns none of them."
)
_PATCH = (
    "--- a/table.py\n+++ b/table.py\n@@ def join(self, other, keep_nulls=False):\n"
    "-    keys = keys.dropna()\n+    keys = keys if keep_nulls else keys.dropna()\n"
)
_DUMP_FILLER = [
    {"instance_id": f"other__repo-{n}", "problem_statement": f"Unrelated issue {n}: " + " ".join(["the cache"] * 60)}
    for n in range(4)
]


def test_item_rule_marks_a_task_instruction_copied_without_its_solution(tmp_path: Path):
    task = "swebench_verified-example__tables-101"
    eval_records = [
        {"id": task, "text": f"{_ISSUE}\n\n{_PATCH}"},
        {"id": f"{task}-instruction", "text": _ISSUE},
        {"id": f"{task}-solution", "text": _PATCH},
    ]
    # A repository document storing a dataset dump: one JSON line, newlines escaped, issue but no patch.
    dump = json.dumps(
        [*_DUMP_FILLER[:2], {"instance_id": "example__tables-101", "problem_statement": _ISSUE}, *_DUMP_FILLER[2:]]
    )
    document = {"id": "doc", "partition_id": 0, "text": f"Repository: example/eval-dumps\nFile: data/test.json\n{dump}"}
    pipeline = dict(
        ngram_length=NGRAM_LENGTH, overlap_threshold=OVERLAP_THRESHOLD, min_matched_features=MIN_MATCHED_FEATURES
    )

    paragraph_only = _run_decon_one_shot(
        tmp_path / "paragraph", eval_records=eval_records, input_records=[document], ngram=NGramConfig(**pipeline)
    )
    with_items = _run_decon_one_shot(
        tmp_path / "items",
        eval_records=eval_records,
        input_records=[document],
        ngram=NGramConfig(**pipeline, item_match=ITEM_MATCH),
    )

    assert paragraph_only["doc"]["contaminated"] is False
    assert f"{task}-instruction" in with_items["doc"]["matched_eval_ids"]
    assert f"{task}-solution" not in with_items["doc"]["matched_eval_ids"]


def test_item_records_give_a_view_its_own_record_only_when_it_stands_alone():
    question = (
        "A sensor network reports temperature every five minutes, but each station drops readings while it is "
        "being recalibrated, so which hourly mean should the operations dashboard show for that station each hour?"
    )
    short_answer = "The mean of the readings it kept."

    rows = {
        row["id"]: row["text"]
        for row in item_records("qa", "1", [question, short_answer], {"q": question, "a": short_answer})
    }

    # A one-sentence answer holds too few n-grams to mark a document on its own.
    assert sorted(rows) == ["qa-1", "qa-1-q"]
    assert rows["qa-1"] == f"{question}\n\n{short_answer}"
