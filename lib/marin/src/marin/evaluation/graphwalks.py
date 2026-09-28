# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the pinned GraphWalks dataset against a served chat model."""

import json
import logging
import math
import re
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypedDict, cast

import requests
from finestore.eval import EvalSample, EvaluationStore, Grading, Message, SampleKind

from marin.evaluation.records import (
    BenchmarkMetadataRef,
    BenchmarkMetricRef,
    EvalTaskRef,
    MetricKind,
    RunStatus,
    TaskCoverage,
)
from marin.evaluation.runner import EvaluationError, EvaluationOutcome
from marin.inference.iris import RemoteInferenceSession

logger = logging.getLogger(__name__)

GRAPHWALKS_DATASET = "openai/graphwalks"
GRAPHWALKS_REVISION = "be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7"
GRAPHWALKS_TASK = "graphwalks"
_MAX_CONCURRENT_REQUESTS = 8
_REQUEST_TIMEOUT = 1800
_MAX_REQUEST_ATTEMPTS = 3
_MIN_COMPLETION_RATE = 0.9
# Allow for small differences between local tokenizer formatting and the vLLM chat template.
_CONTEXT_MARGIN = 64
_PREFIX_TOKEN_MARGIN = 256
_PREFIX_CHARS_PER_TOKEN = 8
_MIN_OUTPUT_TOKENS = 4096
_REASONING_RESERVE = 4096
_FINAL_ANSWER = re.compile(r"\[.*\]")


@dataclass(frozen=True)
class _Example:
    index: int
    prompt: str
    answer_nodes: tuple[str, ...]
    prompt_chars: int
    prompt_tokens: int
    output_tokens: int
    problem_type: str
    date_added: str


class _GraphWalksRow(TypedDict):
    prompt: str
    answer_nodes: list[str]
    prompt_chars: int
    problem_type: str
    date_added: str


@dataclass(frozen=True)
class _Result:
    example: _Example
    output: str | None
    error: str | None


def extract_answer(response: str) -> tuple[list[str], bool]:
    """Apply the GraphWalks last-line parser from the dataset card."""
    line = response.split("\n")[-1]
    if "Final Answer:" not in line:
        return [], True
    match = _FINAL_ANSWER.search(line)
    if match is None:
        return [], True
    return [node.strip() for node in match.group(0).strip("[]").split(",") if node.strip()], False


def grade_answer(response: str, answer_nodes: tuple[str, ...]) -> tuple[dict[str, float], list[str], bool]:
    """Score a predicted node set using GraphWalks precision, recall, and F1."""
    extracted, failed_to_parse = extract_answer(response)
    predicted = set(extracted)
    truth = set(answer_nodes)
    if failed_to_parse:
        precision = recall = f1 = 0.0
    elif not truth and not predicted:
        precision = recall = f1 = 1.0
    else:
        overlap = len(predicted & truth)
        recall = overlap / len(truth) if truth else 0.0
        precision = overlap / len(predicted) if predicted else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return (
        {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "exact_match": float(predicted == truth and not failed_to_parse),
        },
        extracted,
        failed_to_parse,
    )


def _benchmark(n_benchmark: int, n_attempted: int) -> BenchmarkMetadataRef:
    return BenchmarkMetadataRef(
        schema_version=1,
        task=GRAPHWALKS_TASK,
        primary_metric="f1",
        metric_kind=MetricKind.CONTINUOUS,
        metrics=tuple(
            BenchmarkMetricRef(
                name=name,
                source_name=name,
                kind=MetricKind.BINARY if name == "exact_match" else MetricKind.CONTINUOUS,
                higher_is_better=True,
            )
            for name in ("f1", "precision", "recall", "exact_match")
        ),
        n_benchmark=n_benchmark,
        n_attempted=n_attempted,
    )


def _request(example: _Example, endpoint_url: str, model: str, api_key: str | None) -> _Result:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": example.prompt}],
        "max_tokens": example.output_tokens,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    for attempt in range(_MAX_REQUEST_ATTEMPTS):
        try:
            response = requests.post(endpoint_url, json=body, headers=headers, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("chat response content is not text")
            return _Result(example=example, output=content, error=None)
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            if attempt + 1 == _MAX_REQUEST_ATTEMPTS:
                return _Result(example=example, output=None, error=type(exc).__name__)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _sample(result: _Result) -> EvalSample:
    assert result.output is not None
    scores, extracted, failed_to_parse = grade_answer(result.output, result.example.answer_nodes)
    return EvalSample(
        task=GRAPHWALKS_TASK,
        doc_id=str(result.example.index),
        kind=SampleKind.GENERATION,
        prompt_messages=[Message(role="user", content=result.example.prompt)],
        output=result.output,
        extracted=json.dumps(extracted),
        target_text=json.dumps(list(result.example.answer_nodes)),
        grading=Grading(
            method="graphwalks:set_f1",
            metric="f1",
            score=scores["f1"],
            passed=bool(scores["exact_match"]),
            detail=json.dumps(
                {"precision": scores["precision"], "recall": scores["recall"], "failed_to_parse": failed_to_parse}
            ),
        ),
        metrics=scores,
        correct=bool(scores["exact_match"]),
        doc=json.dumps(
            {
                "answer_nodes": list(result.example.answer_nodes),
                "prompt_chars": result.example.prompt_chars,
                "prompt_tokens": result.example.prompt_tokens,
                "max_output_tokens": result.example.output_tokens,
                "problem_type": result.example.problem_type,
                "date_added": result.example.date_added,
            }
        ),
    )


@dataclass(frozen=True)
class GraphWalksExecutor:
    """Run context-eligible GraphWalks examples and publish their graded samples."""

    max_model_len: int
    max_output_tokens: int = 131072
    limit: int | None = None

    def __call__(
        self,
        session: RemoteInferenceSession,
        output_dir: str,
        env_vars: Mapping[str, str],
        *,
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome:
        if self.max_model_len <= _MIN_OUTPUT_TOKENS + _CONTEXT_MARGIN:
            raise ValueError("GraphWalks requires room for a prompt and at least 4096 output tokens")
        if self.max_output_tokens < _MIN_OUTPUT_TOKENS:
            raise ValueError("GraphWalks output cap must be at least 4096 tokens")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("GraphWalks limit must be positive")
        if session.model.tokenizer is None:
            raise ValueError("GraphWalks requires a model tokenizer")

        # The launcher imports evaluation definitions for every CLI operation, while these optional
        # packages are needed only in the GraphWalks worker after the model is serving.
        from datasets import load_dataset  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(session.model.tokenizer, trust_remote_code=True)
        dataset = load_dataset(GRAPHWALKS_DATASET, split="train", revision=GRAPHWALKS_REVISION)
        n_benchmark = len(dataset)
        eligible: list[_Example] = []
        skipped = Counter()
        skipped_output_cap = Counter()
        capped = 0
        for index in range(n_benchmark):
            if self.limit is not None and len(eligible) >= self.limit:
                capped = n_benchmark - index
                break
            row = cast(_GraphWalksRow, dataset[index])
            prompt = row["prompt"]
            answer_text = "Final Answer: [" + ", ".join(row["answer_nodes"]) + "]"
            answer_tokens = len(tokenizer.encode(answer_text, add_special_tokens=False))
            required_output_tokens = max(_MIN_OUTPUT_TOKENS, answer_tokens + _REASONING_RESERVE)
            if required_output_tokens > self.max_output_tokens:
                skipped_output_cap[row["problem_type"]] += 1
                continue
            prompt_budget = self.max_model_len - required_output_tokens - _CONTEXT_MARGIN
            if prompt_budget <= 0:
                skipped[row["problem_type"]] += 1
                continue
            # A long prompt can be ruled out from a tokenized prefix alone. The extra 256-token
            # gap makes boundary changes from completing the prompt irrelevant to that decision.
            if row["prompt_chars"] > prompt_budget * _PREFIX_CHARS_PER_TOKEN:
                prefix = prompt[: prompt_budget * _PREFIX_CHARS_PER_TOKEN]
                prefix_tokens = len(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prefix}], tokenize=True, add_generation_prompt=True
                    )
                )
                if prefix_tokens > prompt_budget + _PREFIX_TOKEN_MARGIN:
                    skipped[row["problem_type"]] += 1
                    continue
            prompt_tokens = len(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
                )
            )
            if prompt_tokens > prompt_budget:
                skipped[row["problem_type"]] += 1
                continue
            eligible.append(
                _Example(
                    index=index,
                    prompt=prompt,
                    answer_nodes=tuple(row["answer_nodes"]),
                    prompt_chars=row["prompt_chars"],
                    prompt_tokens=prompt_tokens,
                    output_tokens=required_output_tokens,
                    problem_type=row["problem_type"],
                    date_added=row["date_added"],
                )
            )

        n_attempted = len(eligible)
        if not n_attempted:
            raise EvaluationError("No GraphWalks examples fit the served context", status=RunStatus.FAILED)
        logger.info(
            "GraphWalks: %d/%d examples eligible for %d token context (output cap %d); "
            "skipped_context=%s skipped_output_cap=%s",
            n_attempted,
            n_benchmark,
            self.max_model_len,
            self.max_output_tokens,
            dict(skipped),
            dict(skipped_output_cap),
        )

        totals: dict[str, list[float]] = {key: [] for key in ("f1", "precision", "recall", "exact_match")}
        errors: Counter[str] = Counter()
        n_unanswered = 0
        with EvaluationStore.open(output_dir, writer_id=f"marin-graphwalks-{uuid.uuid4().hex}") as store:
            store.add_source_artifact(
                "graphwalks-run.json",
                json.dumps(
                    {
                        "dataset": GRAPHWALKS_DATASET,
                        "revision": GRAPHWALKS_REVISION,
                        "n_benchmark": n_benchmark,
                        "n_attempted": n_attempted,
                        "skipped_context_by_type": dict(skipped),
                        "skipped_output_cap_by_type": dict(skipped_output_cap),
                        "not_inspected_after_limit": capped,
                        "max_model_len": self.max_model_len,
                        "max_output_tokens": self.max_output_tokens,
                        "output_budget_policy": "max(4096, tokenized_gold_list_length + 4096)",
                    },
                    sort_keys=True,
                ).encode(),
                content_type="application/json",
            )
            with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_REQUESTS) as pool:
                futures = {
                    pool.submit(
                        _request,
                        example,
                        session.model.endpoint.url("chat/completions"),
                        session.model.endpoint.model,
                        session.model.endpoint.api_key,
                    ): example
                    for example in eligible
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result.error is not None:
                        errors[result.error] += 1
                        continue
                    assert result.output is not None
                    sample = _sample(result)
                    store.add_sample(sample)
                    for key, value in sample.metrics.items():
                        totals[key].append(value)
                    if json.loads(sample.grading.detail)["failed_to_parse"]:
                        n_unanswered += 1
                    if len(totals["f1"]) % 10 == 0:
                        store.flush()
            store.seal()

        n_scored = len(totals["f1"])
        coverage = TaskCoverage(
            n_benchmark=n_benchmark,
            n_attempted=n_attempted,
            n_scored=n_scored,
            n_correct=None,
            n_unanswered=n_unanswered,
            errors=dict(errors),
        )
        if n_scored / n_attempted < _MIN_COMPLETION_RATE:
            raise EvaluationError(
                f"GraphWalks scored {n_scored}/{n_attempted} attempted examples, below {_MIN_COMPLETION_RATE:.0%}",
                status=RunStatus.INFRA_FAILED,
                coverage={GRAPHWALKS_TASK: coverage},
            )
        source_metrics = {key: statistics.mean(values) for key, values in totals.items()}
        source_metrics.update(
            {
                "total_examples": float(n_scored),
                "eligible_examples": float(n_attempted),
                "skipped_context": float(sum(skipped.values())),
                "skipped_output_cap": float(sum(skipped_output_cap.values())),
                "not_inspected_after_limit": float(capped),
            }
        )
        stderr = statistics.stdev(totals["f1"]) / math.sqrt(n_scored) if n_scored > 1 else 0.0
        return EvaluationOutcome(
            metrics={GRAPHWALKS_TASK: source_metrics},
            canonical_metrics={GRAPHWALKS_TASK: {**source_metrics, "f1_stderr": stderr}},
            tasks=(EvalTaskRef(name=GRAPHWALKS_TASK, num_fewshot=None, benchmark=_benchmark(n_benchmark, n_attempted)),),
            coverage={GRAPHWALKS_TASK: coverage},
        )
