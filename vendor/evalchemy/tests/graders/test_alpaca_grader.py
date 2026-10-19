"""Alpaca corpus metrics retain their source indices under concurrent completion."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import re
from types import SimpleNamespace

import pandas as pd
import pytest
from alpaca_eval import constants, utils
from alpaca_eval.annotators import SinglePairwiseAnnotator
from alpaca_eval.annotators.pairwise_evaluator import PairwiseAnnotatorLocal
from alpaca_eval.metrics import get_length_controlled_winrate, glm_winrate

from eval.chat_benchmarks.alpaca_eval.eval_instruct import AlpacaBenchmark
from eval.contracts.eval_spec import EvalSpec, GraderSpec, ModelRequest, ReducerSpec, RequestKind, SourceRef
from eval.execution.engine import CollectSamples, Evaluator
from eval.execution.grading import Graders
from eval.execution.metrics import SourceMetrics
from eval.execution.judge import JudgeBackend
from eval.graders.alpaca import alpaca, alpaca_preference
from eval.graders.answer_equivalence import JudgeConfig
from eval.graders.specs import GRADERS
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


@pytest.fixture
def alpaca_calibration(tmp_path, monkeypatch):
    calibration = tmp_path / "df_gamed.csv"
    pd.DataFrame([
        {
            "model": model, "index": index,
            "instruction_difficulty": index / 5 - 2,
            "std_delta_len": length, "not_gamed_baseline": False,
            "preference": 0.2 + index / 50 if length < 0 else 0.8 - index / 50,
        }
        for model, length in (("short", -1), ("long", 1))
        for index in range(20)
    ]).to_csv(calibration, index=False)

    def download(*, filename, **kwargs):
        assert filename == "df_gamed.csv"
        return str(calibration)

    monkeypatch.setattr(glm_winrate, "hf_hub_download", download)


def _ranking(prompt):
    if not isinstance(prompt, str):
        prompt = "\n".join(message["content"] for message in prompt)
    ordinal = int(re.search(r"case-(\d+)", prompt).group(1))
    rank = 1 if ordinal % 2 else 2
    return repr([{"model": "model_1", "rank": rank}, {"model": "model_2", "rank": 3 - rank}])


@pytest.mark.parametrize("reverse_completions", [False, True])
def test_alpaca_corpus_score_matches_original_indexed_annotations(alpaca_calibration, reverse_completions):
    name = "alpaca_eval_gpt4_0613"
    pairs = [
        {
            "index": ordinal, "instruction": f"case-{ordinal}",
            "output_1": "reference " * (ordinal % 5 + 1),
            "output_2": "reference " * (ordinal % 5 + 1) if ordinal == 17 else "candidate " * (ordinal % 7 + 1),
            "generator_1": "reference", "generator_2": "candidate",
        }
        for ordinal in (17, 2, 11, 4, 0, 13, 8, 19, 6, 1)
    ]

    def saved_reply(prompt):
        text = prompt if isinstance(prompt, str) else "\n".join(message["content"] for message in prompt)
        return "invalid" if int(re.search(r"case-(\d+)", text).group(1)) == 2 else _ranking(prompt)

    original = PairwiseAnnotatorLocal(annotators_config=name)
    for annotator in original.annotators.values():
        annotator.fn_completions = lambda prompts, **kwargs: {"completions": [saved_reply(prompt) for prompt in prompts]}
    original_annotations = original(pairs)
    expected = get_length_controlled_winrate(original_annotations, save_weights_dir=None)

    specs = []
    for pair in pairs:
        specs.append(EvalSpec(
            "alpaca_eval", SourceRef("fixture"), pair["index"], pair["index"],
            (ModelRequest(RequestKind.GENERATE, pair["instruction"]),),
            GraderSpec("alpaca", {"config": name, "baseline": "reference"}),
            doc={"instruction": pair["instruction"]}, target=pair["output_1"],
        ))
    pairs_by_index = {pair["index"]: pair for pair in pairs}
    candidate = FixedEndpoint(lambda request: pairs_by_index[request.metadata["source_id"]]["output_2"])
    judge = FixedEndpoint(lambda request: saved_reply(request.prompt))
    sink = CollectSamples()
    grading = Graders(GRADERS)
    engine = Evaluator(candidate, grading, sink, workers=2, judges={"alpaca_eval": judge})

    async def run():
        results = []
        async for completed in engine.run(specs):
            assert completed.error is None
            results.append(completed.result)
        return results

    try:
        results = asyncio.run(run())
    finally:
        grading.close()
    metrics = SourceMetrics(repeated=False, policy=ReducerSpec(name="alpaca", trial_metric=None))
    for result in reversed(results) if reverse_completions else results:
        metrics.add(result)

    reduced = metrics.finish("alpaca_eval")["results"]["alpaca_eval"]
    assert {name: reduced[name] for name in expected} == pytest.approx(expected)
    assert reduced["num_total"] == len(pairs)
    assert reduced["num_judged"] == len(pairs) - 1
    assert reduced["num_judge_failed"] == 1
    annotations = {result.spec.ordinal: result.grade.audit["alpaca_annotation"] for result in results}
    assert annotations[17]["preference"] == 1.5
    assert annotations[2]["preference"] is None
    assert annotations[2]["output_1"] == pairs_by_index[2]["output_1"]
    assert annotations[2]["output_2"] == pairs_by_index[2]["output_2"]
    assert len(judge.requests) == len(pairs) - 1
    assert len(sink.samples["alpaca_eval"]) == len(pairs)


def test_alpaca_all_invalid_judgments_remain_unavailable(monkeypatch):
    rows = [{"instruction": f"case-{index}", "output": "reference", "generator": "reference"} for index in range(2)]
    monkeypatch.setattr(constants, "ALPACAEVAL_REFERENCE_OUTPUTS", lambda: rows)
    source = AlpacaBenchmark(annotator_model="gpt-4")
    monkeypatch.setattr(source, "load_dataset", lambda: rows)

    metrics, samples = evaluate_source(
        source, SourceOptions("alpaca_eval", source.SOURCE), FixedEndpoint("candidate"),
        judges={"alpaca_eval": FixedEndpoint("invalid")},
    )

    assert metrics["win_rate"] is None
    assert metrics["length_controlled_winrate"] is None
    assert metrics["n_total"] == 0
    assert metrics["num_judge_failed"] == 2
    assert len(samples) == 2
    assert all(sample["alpaca_annotation"]["preference"] is None for sample in samples)
    assert all("preference" not in sample["metrics"] for sample in samples)


@pytest.mark.parametrize("model", ["auto", "gpt-4", "gpt-4-1106-preview", "gpt-4o-mini-2024-07-18"])
@pytest.mark.parametrize("rank", [1, 2])
def test_alpaca_wire_model_and_saved_reply_match_original_annotator(monkeypatch, model, rank):
    row = {"instruction": "case-17", "output": "reference response", "generator": "reference"}
    monkeypatch.setattr(constants, "ALPACAEVAL_REFERENCE_OUTPUTS", lambda: [row])
    source = AlpacaBenchmark(annotator_model=model)
    monkeypatch.setattr(source, "load_dataset", lambda: [row])
    spec = next(source.iter_specs(SourceOptions("alpaca_eval", source.SOURCE)))
    config = next(iter(utils.load_configs(source.annotator_conf, relative_to=constants.EVALUATORS_CONFIG_DIR).values()))
    choice = {
        "index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": repr([
            {"model": "model_1", "rank": rank}, {"model": "model_2", "rank": 3 - rank},
        ])},
        "logprobs": {"content": [{"top_logprobs": [
            {"token": "m", "logprob": -0.2 if rank == 1 else -1.6},
            {"token": "M", "logprob": -1.6 if rank == 1 else -0.2},
        ]}]},
    }
    original_requests = []

    def completions(prompts, model_name, **kwargs):
        original_requests.append({"model": model_name, "messages": utils.prompt_to_chatml(prompts[0]), **kwargs})
        return {"completions": [choice["message"]["content"]], "completions_all": [choice]}

    original = SinglePairwiseAnnotator(**{**config, "seed": 0, "fn_completions": completions})
    [expected] = original(pd.DataFrame([{
        "index": 0, "instruction": row["instruction"], "output_1": row["output"], "output_2": "candidate response",
        "generator_1": "reference", "generator_2": "candidate",
    }])).to_dict("records")
    plan = alpaca(spec, ["candidate response"])
    requests = []

    @asynccontextmanager
    async def post(_url, *, headers, json):
        requests.append(json)

        async def response_json():
            return {"choices": [choice]}

        yield SimpleNamespace(raise_for_status=lambda: None, json=response_json)

    async def run():
        backend = JudgeBackend(JudgeConfig("generic-fallback", "http://offline.invalid/v1", "test-key"),
                               SimpleNamespace(post=post))
        return [await backend.execute(request) for request in plan.requests]

    replies = asyncio.run(run())
    grade = alpaca_preference(replace(spec, grader=plan.grader), replies)

    assert requests == original_requests
    assert source.judge_model == original_requests[0]["model"]
    assert grade.metrics["preference"] == pytest.approx(expected["preference"])
