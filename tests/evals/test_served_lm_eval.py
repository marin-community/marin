# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from pathlib import Path
from statistics import mean

import pytest
from marin.evaluation.lm_eval import LmEvalAdapter, LmEvalRun, build_lm_eval_model_args
from marin.inference.types import OpenAIEndpoint, RunningModel

from tests.evals.openai_stub import (
    DeterministicOpenAIStub,
    assert_chat_generation_contract,
    assert_completions_scoring_contract,
    serve_deterministic_openai_stub,
)


@pytest.fixture
def deterministic_openai_stub() -> Iterator[DeterministicOpenAIStub]:
    with serve_deterministic_openai_stub(model="gpt2") as stub:
        yield stub


def test_lm_eval_model_args_selects_completions_endpoint() -> None:
    running_model = RunningModel(
        endpoint=OpenAIEndpoint(base_url="http://127.0.0.1:8000/v1", model="served-model", api_key="test-key"),
        tokenizer="hf-tokenizer",
    )
    run = LmEvalRun(
        tasks=["arc_easy"],
        output_path="out",
        extra_model_args={"num_concurrent": 2, "timeout": 30},
    )

    assert build_lm_eval_model_args(running_model, run) == (
        "model=served-model,"
        "base_url=http://127.0.0.1:8000/v1/completions,"
        "tokenizer_backend=huggingface,"
        "tokenized_requests=False,"
        "api_key=test-key,"
        "tokenizer=hf-tokenizer,"
        "num_concurrent=2,"
        "timeout=30"
    )


def test_lm_eval_model_args_selects_chat_endpoint() -> None:
    running_model = RunningModel(endpoint=OpenAIEndpoint(base_url="http://127.0.0.1:8000/v1/", model="served-model"))
    run = LmEvalRun(
        tasks=["ifeval"],
        output_path="out",
        adapter=LmEvalAdapter.LOCAL_CHAT_COMPLETIONS,
        apply_chat_template=True,
    )

    assert build_lm_eval_model_args(running_model, run).startswith(
        "model=served-model,base_url=http://127.0.0.1:8000/v1/chat/completions,"
    )


def test_lm_eval_model_args_allow_explicit_overrides() -> None:
    running_model = RunningModel(endpoint=OpenAIEndpoint(base_url="http://127.0.0.1:8000/v1", model="gpt2"))
    run = LmEvalRun(
        tasks=["arc_easy"],
        output_path="out",
        extra_model_args={"tokenizer_backend": "tiktoken", "tokenized_requests": False, "timeout": 5},
    )

    assert build_lm_eval_model_args(running_model, run) == (
        "model=gpt2,"
        "base_url=http://127.0.0.1:8000/v1/completions,"
        "tokenizer_backend=tiktoken,"
        "tokenized_requests=False,"
        "timeout=5"
    )


def test_marin_openai_subset_conformance_against_deterministic_stub(
    deterministic_openai_stub: DeterministicOpenAIStub,
) -> None:
    assert_completions_scoring_contract(deterministic_openai_stub.base_url, deterministic_openai_stub.model)
    assert_chat_generation_contract(deterministic_openai_stub.base_url, deterministic_openai_stub.model)


def test_real_lm_eval_local_completions_scoring_against_deterministic_stub(
    deterministic_openai_stub: DeterministicOpenAIStub, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tiktoken")
    pytest.importorskip("lm_eval")

    from lm_eval.api.instance import Instance
    from lm_eval.api.task import Task, TaskConfig
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.loggers import EvaluationTracker

    class TinyScoringTask(Task):
        OUTPUT_TYPE = "loglikelihood"

        def __init__(self) -> None:
            super().__init__()
            self._config = TaskConfig(
                task="marin_tiny_scoring",
                output_type="loglikelihood",
                num_fewshot=0,
                repeats=1,
            )
            self.task_name = "marin_tiny_scoring"

        def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
            self.dataset = {"test": [{"ctx": "A", "target": " B"}]}

        def has_training_docs(self) -> bool:
            return False

        def has_validation_docs(self) -> bool:
            return False

        def has_test_docs(self) -> bool:
            return True

        def test_docs(self) -> list[dict[str, str]]:
            return self.dataset["test"]

        def doc_to_text(self, doc: dict[str, str]) -> str:
            return doc["ctx"]

        def doc_to_target(self, doc: dict[str, str]) -> str:
            return doc["target"]

        def construct_requests(self, doc: dict[str, str], ctx: str, **kwargs: object) -> Instance:
            return Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, self.doc_to_target(doc)),
                idx=0,
                metadata=kwargs["metadata"],
            )

        def process_results(self, doc: dict[str, str], results: list[tuple[float, bool]]) -> dict[str, float]:
            loglikelihood, is_greedy = results[0]
            return {"loglikelihood": loglikelihood, "greedy": float(is_greedy)}

        def aggregation(self) -> dict[str, object]:
            return {"loglikelihood": mean, "greedy": mean}

        def higher_is_better(self) -> dict[str, bool]:
            return {"loglikelihood": True, "greedy": True}

    task = TinyScoringTask()
    running_model = RunningModel(endpoint=OpenAIEndpoint(base_url=deterministic_openai_stub.base_url, model="gpt2"))
    run = LmEvalRun(
        tasks=["marin_tiny_scoring"],
        output_path=str(tmp_path / "lm_eval_results"),
        extra_model_args={
            "tokenizer_backend": "tiktoken",
            "tokenized_requests": False,
            "max_retries": 1,
            "timeout": 5,
        },
    )
    evaluation_tracker = EvaluationTracker(output_path=run.output_path)

    results = simple_evaluate(
        model="local-completions",
        tasks=[task],
        model_args=build_lm_eval_model_args(running_model, run),
        batch_size=1,
        bootstrap_iters=0,
        evaluation_tracker=evaluation_tracker,
        log_samples=True,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    assert results is not None
    samples = results.pop("samples")
    evaluation_tracker.save_results_aggregated(results=results, samples=samples)
    for task_name in results["configs"].keys():
        evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

    completion_requests = deterministic_openai_stub.requests_for("/v1/completions")
    assert len(completion_requests) == 1
    payload = completion_requests[0].payload
    assert payload == {
        "model": "gpt2",
        "prompt": "A B",
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": 1,
        "seed": 1234,
        "echo": True,
    }
    assert "stop" not in payload

    assert results["results"]["marin_tiny_scoring"]["loglikelihood,none"] == pytest.approx(-0.1)
    assert results["results"]["marin_tiny_scoring"]["greedy,none"] == 1.0

    result_dirs = list((tmp_path / "lm_eval_results").glob("gpt2"))
    assert len(result_dirs) == 1
    result_files = list(result_dirs[0].glob("results_*.json"))
    sample_files = list(result_dirs[0].glob("samples_marin_tiny_scoring_*.jsonl"))
    assert len(result_files) == 1
    assert len(sample_files) == 1
