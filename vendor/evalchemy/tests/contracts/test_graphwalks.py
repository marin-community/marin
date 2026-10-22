"""GraphWalks selection, budgets, and scores through the shared lifecycle."""

from argparse import Namespace
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from eval.chat_benchmarks.GraphWalks import eval_instruct as graphwalks
from eval.completion_response import CompletionContentPolicy, CompletionText, completion_response_from_chat_choice
from eval.contracts.finestore_output import FineStoreOutput, completed_finestore_output, read_finestore_output
from eval.eval import evaluate
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


class Tokenizer:
    def encode(self, text, **_kwargs):
        return text.split()

    def apply_chat_template(self, messages, *, enable_thinking=False, **_kwargs):
        return {"input_ids": [0] * (len(messages[0]["content"].split()) + 2 + int(enable_thinking))}


class Model:
    world_size = 1
    tokenizer = Tokenizer()
    _evalchemy_chat_template_kwargs: ClassVar[dict[str, bool]] = {"enable_thinking": True}

    def __init__(self):
        self.requests = []

    def tok_encode(self, text):
        return self.tokenizer.encode(text)

    def apply_chat_template(self, messages, **_kwargs):
        return messages

    def generate_until(self, requests):
        self.requests.extend(requests)
        return ["Final Answer: [a, b]" for _request in requests]


def row(words=10, nodes=("a", "b"), kind="bfs"):
    prompt = "word " * words
    return {"prompt": prompt, "prompt_chars": len(prompt), "answer_nodes": list(nodes), "problem_type": kind}


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    prefixes = []

    def download(repo_id, filename, *, revision, cache_prefix):
        assert repo_id == "openai/graphwalks"
        assert revision == graphwalks.GRAPHWALKS_REVISION
        prefixes.append(cache_prefix)
        return str(tmp_path / filename)

    monkeypatch.setattr(graphwalks, "fetch_hf", download)

    def write(rows):
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / graphwalks.GRAPHWALKS_FILES[0])
        pq.write_table(
            pa.Table.from_pylist(rows[:0], schema=pa.Table.from_pylist(rows).schema),
            tmp_path / graphwalks.GRAPHWALKS_FILES[1],
        )
        return prefixes

    return write


@pytest.mark.parametrize(
    "response,nodes,f1,exact",
    [
        ("reasoning\nFinal Answer: [a, a, b]", ("a", "b", "c"), 0.8, 0.0),
        ("Final Answer: []", (), 1.0, 1.0),
        ("Final Answer: [a, b]\n   ", ("a", "b"), 1.0, 1.0),
        ("[a, b]", ("a", "b"), 0.0, 0.0),
        ("Final Answer: [a, b]\ntrailing prose", ("a", "b"), 0.0, 0.0),
    ],
)
def test_set_grading_preserves_last_line_contract(dataset, response, nodes, f1, exact):
    dataset([row(nodes=nodes)])
    source = graphwalks.GraphWalksBenchmark()
    options = SourceOptions(
        "GraphWalks", source.SOURCE, context_length=10000, prompt_token_count=lambda text: len(str(text).split())
    )
    metrics, samples = evaluate_source(source, options, FixedEndpoint(response))
    assert metrics["f1"] == f1
    assert metrics["exact_match"] == exact
    assert samples[0]["f1"] == f1


def test_shared_driver_preserves_selection_budgets_and_coverage(dataset):
    rows = [row(2000), row(), row(nodes=("node",) * 100), row()]
    prefixes = dataset(rows)
    model = Model()
    source = graphwalks.GraphWalksBenchmark()
    source.dataset_cache = "s3://offline-cache/graphwalks"
    result = evaluate(
        model,
        Namespace(get_benchmark=lambda _task: source),
        None,
        ["GraphWalks"],
        {"GraphWalks": "Evalchemy chat benchmark"},
        [1],
        args=Namespace(
            model_args="",
            max_length=10000,
            max_tokens=32768,
            limit=2,
            gen_kwargs={"max_gen_toks": 16384, "temperature": 0.7},
        ),
    )
    assert prefixes == ["s3://offline-cache/graphwalks"] * 2
    assert [request.args[1]["max_gen_toks"] for request in model.requests] == [8200, 8396]
    assert all(request.args[1]["temperature"] == 0.7 for request in model.requests)
    samples = result["samples"]["GraphWalks"]
    assert [sample["doc"]["source_index"] for sample in samples] == [1, 2]
    assert [sample["arguments"][0][1]["max_gen_toks"] for sample in samples] == [8200, 8396]
    assert result["benchmark_metadata"]["GraphWalks"]["n_benchmark"] == 4
    assert result["benchmark_metadata"]["GraphWalks"]["n_attempted"] == 2
    selection = result["results"]["GraphWalks"]["selection"]
    assert selection["skipped_context_by_type"] == {"bfs": 1}
    assert selection["not_inspected_after_limit"] == 1
    assert selection["chat_template_kwargs"] == {"enable_thinking": True}
    assert result["results"]["GraphWalks"]["f1"] == 0.5
    assert result["task_outcomes"]["GraphWalks"]["status"] == "succeeded"


@pytest.mark.parametrize("template,expected", [({"enable_thinking": True}, [1]), ({}, [0, 1])])
def test_thinking_template_controls_context_boundary(dataset, template, expected):
    dataset([row(10), row(9)])
    model = Model()
    model._evalchemy_chat_template_kwargs = template
    result = evaluate(
        model,
        Namespace(get_benchmark=lambda _task: graphwalks.GraphWalksBenchmark()),
        None,
        ["GraphWalks"],
        {"GraphWalks": "Evalchemy chat benchmark"},
        [1],
        args=Namespace(model_args="", max_length=8276, max_tokens=32768),
    )
    assert [sample["doc"]["source_index"] for sample in result["samples"]["GraphWalks"]] == expected


def test_output_cap_skips_without_shortening_gold_budget(dataset):
    dataset([row(nodes=("node",) * 100), row(nodes=())])
    source = graphwalks.GraphWalksBenchmark()
    options = SourceOptions(
        "GraphWalks",
        source.SOURCE,
        context_length=10000,
        max_tokens=8200,
        prompt_token_count=lambda text: len(str(text).split()),
    )
    endpoint = FixedEndpoint("Final Answer: []")
    metrics, samples = evaluate_source(source, options, endpoint)
    assert [sample["doc"]["source_index"] for sample in samples] == [1]
    assert endpoint.requests[0].generation["max_gen_toks"] == 8198
    assert source.selection["skipped_output_cap_by_type"] == {"bfs": 1}
    assert metrics["f1"] == 1.0


def test_length_ended_parse_failure_scores_zero_and_retains_completion(dataset):
    dataset([row()])
    choice = {"message": {"content": "unfinished reasoning"}, "finish_reason": "length"}
    output = CompletionText(
        "unfinished reasoning", completion_response_from_chat_choice({}, choice), CompletionContentPolicy.COMBINE
    )
    source = graphwalks.GraphWalksBenchmark()
    options = SourceOptions(
        "GraphWalks", source.SOURCE, context_length=10000, prompt_token_count=lambda text: len(str(text).split())
    )
    metrics, samples = evaluate_source(source, options, FixedEndpoint(output))
    assert metrics["f1"] == 0.0
    assert samples[0]["completion_responses"][0][0]["finish_reason"] == "length"
    assert samples[0]["answer_extraction_errors"] == ["failed_to_parse"]


def test_no_eligible_examples_persist_selection_diagnostics_without_scores(dataset, tmp_path):
    dataset([row(2000), row(nodes=("node",) * 100)])
    model = Model()
    source = graphwalks.GraphWalksBenchmark()
    root = str(tmp_path / "results")
    with FineStoreOutput(root, "run", ["GraphWalks"]) as output:
        result = evaluate(
            model, Namespace(get_benchmark=lambda _task: source), None,
            ["GraphWalks"], {"GraphWalks": "Evalchemy chat benchmark"}, [1],
            args=Namespace(max_length=10000, max_tokens=8200), output_sink=output,
        )
        output.finish(result)

    saved = read_finestore_output(root, "run")
    assert saved["task_outcomes"]["GraphWalks"]["status"] == "failed"
    assert saved["task_outcomes"]["GraphWalks"]["metrics"] == {}
    assert "f1" not in saved["results"]["GraphWalks"]
    selection = saved["results"]["GraphWalks"]["selection"]
    assert selection["n_benchmark"] == 2
    assert selection["n_attempted"] == 0
    assert selection["skipped_context_by_type"] == {"bfs": 1}
    assert selection["skipped_output_cap_by_type"] == {"bfs": 1}
    assert not completed_finestore_output(root, "run")
    assert model.requests == []
