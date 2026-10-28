# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise runtime eligibility and raw ledgers against real published benchmark corpora."""

import copy
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView
from fray.local_backend import LocalClient
from marin.evaluation.selection import SELECTIONS_TABLE, select_cases
from marin.evaluation.task_driver import EvaluationRunConfig, ModelIdentity, run_evaluation
from rolloutengine.models import ModelRequest, ModelTurn, RolloutLimits, TokenUsage
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.long_context import sources
from experiments.post_training.task_curation.pipeline import source_recipe


class WordTokenizer:
    """A deterministic model-tokenizer boundary, deliberately different from official o200k counts."""

    def __init__(self):
        self.chat_calls: list[dict[str, Any]] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text.split())

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> list[int]:
        self.chat_calls.append(
            {
                "messages": copy.deepcopy(conversation),
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        content = sum(
            len(message["content"].split()) * (3 if message["content"].startswith("costly ") else 1)
            for message in conversation
        )
        return [0] * (content + 2 * len(conversation) + int(add_generation_prompt))


def mrcr_row(index: int, needles: int, words: int, *, costly: bool = False) -> dict:
    content = ("costly " if costly else "") + "word " * words
    nonce = f"nonce-{index}"
    return {
        "prompt": json.dumps(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": f"Repeat {nonce}."},
            ]
        ),
        "answer": nonce + "target",
        "random_string_to_prepend": nonce,
        "n_needles": needles,
    }


def graph_row(prompt: str, nodes: tuple[str, ...], kind: str = "bfs") -> dict:
    return {"prompt": prompt, "prompt_chars": len(prompt), "answer_nodes": list(nodes), "problem_type": kind}


def publish_fixture(tmp_path: Path, benchmark: str, rows: list[dict], *, context_window: int | None = None):
    source = next(source for source in sources() if source.config.protocol.benchmark == benchmark)
    dataset = source.config
    assert dataset is not None
    tmp_path = tmp_path / benchmark
    root = tmp_path / "input"
    table = pa.Table.from_pylist(rows)
    for index, filename in enumerate(dataset.files):
        file = root / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table if index == 0 else table.slice(0, 0), file)
    output = str(tmp_path / "published")
    protocol = dataset.protocol.model_copy(update={"expected_population": len(rows), "context_window": context_window})
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(
                source_recipe(dataset.recipe, {}, None),
                protocol,
                dataset.item_builder,
                context,
                str(root),
                output,
            )
    finally:
        client.shutdown()
    return output, protocol, dataset.files[0]


@pytest.fixture
def interleaved_mrcr(tmp_path):
    rows = [
        mrcr_row(0, 2, 4100, costly=True),
        mrcr_row(1, 2, 4100),
        mrcr_row(2, 2, 9000),
        mrcr_row(3, 4, 4100),
        mrcr_row(4, 4, 9000),
        mrcr_row(5, 8, 4100),
        mrcr_row(6, 8, 9000),
        mrcr_row(7, 2, 17000),
        mrcr_row(8, 2, 10),
    ]
    return (*publish_fixture(tmp_path, "MRCR", rows), rows)


@pytest.fixture
def refill_mrcr(tmp_path):
    rows = [mrcr_row(0, 2, 4100, costly=True), mrcr_row(1, 2, 4100), mrcr_row(2, 4, 4100), mrcr_row(3, 8, 4100)]
    return (*publish_fixture(tmp_path, "MRCR", rows), rows)


@pytest.fixture
def graph_corpus(tmp_path):
    rows = [
        graph_row("word " * 2000, ("a", "b")),
        graph_row("gold exceeds output cap", ("node",) * 100),
        graph_row("longwordwith16chars " * 1000, ("a", "b")),
        graph_row("empty graph", (), "parents"),
        graph_row("word " * 3000, ("a", "b")),
    ]
    return (*publish_fixture(tmp_path, "GraphWalks", rows), rows)


def test_mrcr_selects_interleaved_cohort_and_preserves_history_and_safe_output_cap(interleaved_mrcr):
    root, protocol, filename, rows = interleaved_mrcr
    tokenizer = WordTokenizer()
    decisions = list(
        select_cases(
            root,
            protocol,
            dataset_id="mrcr-fixture",
            context_window=16384,
            max_tokens=4096,
            limit=4,
            tokenizer=tokenizer,
        )
    )
    selected = {decision.record.source_row: decision for decision in decisions if decision.case is not None}
    assert {row: decision.record.rank for row, decision in selected.items()} == {
        f"{filename}:0": 0,
        f"{filename}:3": 1,
        f"{filename}:5": 2,
        f"{filename}:2": 3,
    }
    assert selected[f"{filename}:0"].record.max_tokens == 4006
    assert selected[f"{filename}:0"].record.prompt_tokens == 12314
    assert [
        (decision.record.source_ordinal, decision.record.reason) for decision in decisions if decision.case is None
    ] == [
        (1, "source_limit"),
        (4, "source_limit"),
        (6, "source_limit"),
        (7, "official_bin"),
        (8, "official_bin"),
    ]
    assert len(decisions) == len(rows)
    expected = json.loads(rows[0]["prompt"])
    expected[1]["reasoning_content"] = ""
    assert any(call["messages"] == expected for call in tokenizer.chat_calls)
    assert all(call["kwargs"] == {"enable_thinking": False} for call in tokenizer.chat_calls)
    assert all(call["tokenize"] and call["add_generation_prompt"] for call in tokenizer.chat_calls)


@pytest.mark.parametrize("limit", [None, 2])
def test_mrcr_records_every_exclusion_when_no_official_bin_fits(interleaved_mrcr, limit):
    root, protocol, filename, rows = interleaved_mrcr
    decisions = list(
        select_cases(
            root,
            protocol,
            dataset_id="too-small",
            context_window=4096,
            max_tokens=1024,
            limit=limit,
            tokenizer=WordTokenizer(),
        )
    )
    assert [decision.record.source_row for decision in decisions] == [f"{filename}:{i}" for i in range(len(rows))]
    assert all(decision.case is None and decision.record.reason == "official_bin" for decision in decisions)


def test_mrcr_refills_cohort_after_served_context_exclusion(refill_mrcr):
    root, protocol, filename, _rows = refill_mrcr
    decisions = list(
        select_cases(
            root,
            protocol,
            dataset_id="mrcr-refill",
            context_window=8192,
            max_tokens=4096,
            limit=2,
            tokenizer=WordTokenizer(),
        )
    )
    selected = {decision.record.source_row: decision.record.rank for decision in decisions if decision.case is not None}
    assert selected == {f"{filename}:2": 1, f"{filename}:3": 2}
    assert {
        decision.record.source_ordinal: decision.record.reason for decision in decisions if decision.case is None
    } == {
        0: "context_window",
        1: "source_limit",
    }


@pytest.mark.parametrize("served_window,effective_window,budget", [(20000, 10000, 4096), (8192, 8192, 4017)])
def test_protocol_context_cap_precedes_bin_selection_and_per_prompt_budget(
    tmp_path,
    served_window,
    effective_window,
    budget,
):
    rows = [
        mrcr_row(0, 2, 4100),
        mrcr_row(1, 4, 4100),
        mrcr_row(2, 8, 4100),
        mrcr_row(3, 2, 9000),
        mrcr_row(4, 4, 9000),
        mrcr_row(5, 8, 9000),
    ]
    root, protocol, filename = publish_fixture(tmp_path, "MRCR", rows, context_window=10000)
    decisions = list(
        select_cases(
            root,
            protocol,
            dataset_id="capped",
            context_window=served_window,
            max_tokens=4096,
            limit=3,
            tokenizer=WordTokenizer(),
        )
    )
    assert [
        (decision.record.source_row, decision.record.max_tokens) for decision in decisions if decision.case is not None
    ] == [
        (f"{filename}:0", budget),
        (f"{filename}:1", budget),
        (f"{filename}:2", budget),
    ]
    assert all(decision.record.context_window == effective_window for decision in decisions)
    assert [decision.record.reason for decision in decisions[3:]] == ["official_bin"] * 3


def test_graphwalks_filters_before_limit_and_retains_reference_dependent_budgets(graph_corpus):
    root, protocol, filename, rows = graph_corpus
    tokenizer = WordTokenizer()
    decisions = list(
        select_cases(
            root,
            protocol,
            dataset_id="graphs",
            context_window=10000,
            max_tokens=8300,
            limit=2,
            tokenizer=tokenizer,
        )
    )
    assert [
        (decision.record.source_ordinal, decision.record.reason) for decision in decisions if decision.case is None
    ] == [
        (0, "context_window"),
        (1, "output_cap"),
        (4, "source_limit"),
    ]
    assert [
        (decision.record.source_row, decision.record.rank, decision.record.max_tokens)
        for decision in decisions
        if decision.case is not None
    ] == [
        (f"{filename}:2", 0, 8200),
        (f"{filename}:3", 1, 8198),
    ]
    after_limit = decisions[4].record
    assert after_limit.status == "uninspected"
    assert (after_limit.prompt_tokens, after_limit.reference_tokens, after_limit.max_tokens) == (None, None, None)
    assert any(call["messages"] == [{"role": "user", "content": rows[2]["prompt"]}] for call in tokenizer.chat_calls)
    assert all(call["kwargs"] == {"enable_thinking": True} for call in tokenizer.chat_calls)


@pytest.mark.asyncio
async def test_driver_archives_full_selection_ledger_and_sends_full_prompts(refill_mrcr, graph_corpus, tmp_path):
    mrcr_root, _mrcr_protocol, mrcr_file, mrcr_rows = refill_mrcr
    graph_root, _graph_protocol, graph_file, graph_rows = graph_corpus
    requests = []

    async def model(request: ModelRequest) -> ModelTurn:
        requests.append(request)
        content = request.messages[-1]["content"]
        if match := re.search(r"nonce-\d+", content):
            answer = match.group(0) + "target"
        else:
            answer = "Final Answer: []" if content == "empty graph" else "Final Answer: [a, b]"
        return ModelTurn(
            message={"role": "assistant", "content": answer},
            prompt_token_ids=(),
            response_token_ids=(),
            logprobs=None,
            stop_reason="stop",
            text=answer,
            usage=TokenUsage(10, 2),
        )

    config = EvaluationRunConfig(
        model=ModelIdentity(
            name="word-model",
            revision="model-commit",
            tokenizer="word-tokenizer",
            tokenizer_revision="tokenizer-commit",
            context_window=10000,
            serving={"backend": "fake"},
        ),
        max_tokens=8300,
        concurrency=2,
        source_limit=2,
        limits=RolloutLimits(
            max_turns=1,
            max_total_tokens=None,
            model_turn_timeout=10,
            tool_turn_timeout=10,
            total_turn_timeout=20,
            attempt_timeout=30,
            verifier_timeout=10,
            cleanup_timeout=10,
        ),
    )
    output = str(tmp_path / "results")
    await run_evaluation((mrcr_root, graph_root), output, config, model, machine_factories={}, tokenizer=WordTokenizer())
    reader = ReadView(output)
    ledger = list(reader.iter_rows(SELECTIONS_TABLE))
    assert len(ledger) == len(mrcr_rows) + len(graph_rows)
    selected = [row for row in ledger if row["status"] == "selected"]
    assert {(row["source_row"], row["rank"]) for row in selected} == {
        (f"{mrcr_file}:2", 1),
        (f"{mrcr_file}:3", 2),
        (f"{graph_file}:2", 0),
        (f"{graph_file}:3", 1),
    }
    assert {row["dataset"] for row in selected} == {row["dataset"] for row in ledger}
    assert sum(row["status"] == "uninspected" for row in ledger) == 1
    assert len(requests) == 4
    graph_requests = [request for request in requests if len(request.messages) == 1]
    assert {(request.messages[0]["content"], request.options["max_tokens"]) for request in graph_requests} == {
        (graph_rows[2]["prompt"], 8200),
        (graph_rows[3]["prompt"], 8198),
    }
    mrcr_requests = [request for request in requests if len(request.messages) > 1]
    expected_messages = []
    for index in (2, 3):
        messages = json.loads(mrcr_rows[index]["prompt"])
        messages[1]["reasoning_content"] = ""
        expected_messages.append(messages)
    assert all(list(request.messages) in expected_messages for request in mrcr_requests)
    samples = [sample_from_archive_row(row) for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE)]
    assert len(samples) == 4
    assert all(sample.metrics.get("accuracy", sample.metrics.get("f1")) == 1 for sample in samples)
