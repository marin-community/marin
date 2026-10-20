# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry-wide coverage for the stored benchmark prompt lengths.

The recompute cases cover the benchmarks whose prompts come from data checked
into this repository, so they run without downloading a dataset. Checking the
rest means fetching tens of gigabytes of datasets, so that is
``scripts/benchmarks/compute_prompt_lengths.py --check`` run deliberately rather
than on every PR.
"""

from pathlib import Path

import pytest

from eval.contracts.prompt_corpus import load_reference_tokenizer, render_prompt_corpus
from eval.contracts.prompt_length import (
    BenchmarkPromptLength,
    PromptLengths,
    load_prompt_lengths,
    resolve_task_max_tokens,
)
from eval.task import TaskManager

CUSTOM_BENCHMARK_ROOT = Path("eval/chat_benchmarks")

REPO_LOCAL_BENCHMARKS = (
    "AIME24",
    "AIME25",
    "AIW",
    "AMC23",
    "FinanceBench",
    "GSM8KPerturbed",
    "HumanEval",
    "OlympiadBench",
)
"""Benchmarks whose prompts are reproducible from checked-in data alone."""

# The window a Qwen3-8B slice serves, and the one the OlympiadBench truncations
# were reported against.
CONTEXT_LENGTH = 40_896


def _lengths(longest_prompt_tokens: int, margin: int = 256) -> PromptLengths:
    entry = BenchmarkPromptLength(
        distinct_prompt_count=10,
        longest_prompt_chars=4 * longest_prompt_tokens,
        longest_prompt_tokens=longest_prompt_tokens,
        sha256="0" * 64,
    )
    return PromptLengths("reference", margin, {"Bench": entry}, {"Unmeasured": "needs a context window"})


def test_every_registered_benchmark_has_a_stored_prompt_length():
    registered = {path.parent.name for path in CUSTOM_BENCHMARK_ROOT.glob("*/eval_instruct.py")}
    lengths = load_prompt_lengths()

    assert registered == set(lengths.benchmarks) | set(lengths.unmeasured)


@pytest.mark.parametrize("task_name", REPO_LOCAL_BENCHMARKS)
def test_repo_local_prompt_lengths_match_what_the_benchmark_renders(task_name):
    lengths = load_prompt_lengths()
    stored = lengths.benchmarks.get(task_name)
    assert stored is not None, f"{task_name} is listed as unmeasured but renders from checked-in data"

    corpus = render_prompt_corpus(task_name, load_reference_tokenizer(lengths.reference_tokenizer))

    assert corpus.to_dict() == stored.to_dict()


def test_a_short_prompt_benchmark_spends_the_remaining_window_on_the_response():
    budget = resolve_task_max_tokens(
        "Bench",
        context_length=CONTEXT_LENGTH,
        requested_max_tokens=None,
        prompt_lengths=_lengths(3_392),
        safety_tokens=64,
    )

    assert budget == CONTEXT_LENGTH - 3_392 - 256 - 64


def test_nupa5k_spends_the_remaining_window_on_the_response():
    lengths = load_prompt_lengths()

    budget = resolve_task_max_tokens(
        "NUPA5K-Loose",
        context_length=73_664,
        requested_max_tokens=None,
        prompt_lengths=lengths,
    )

    assert budget == 73_664 - 399 - lengths.prompt_margin_tokens - 64


def test_an_explicit_output_cap_is_kept():
    assert (
        resolve_task_max_tokens(
            "Bench",
            context_length=CONTEXT_LENGTH,
            requested_max_tokens=8_192,
            prompt_lengths=_lengths(3_392),
        )
        == 8_192
    )


def test_an_unmeasured_benchmark_keeps_its_own_default():
    assert (
        resolve_task_max_tokens(
            "Unmeasured",
            context_length=CONTEXT_LENGTH,
            requested_max_tokens=None,
            prompt_lengths=_lengths(3_392),
        )
        is None
    )


def test_a_benchmark_whose_prompts_exceed_the_window_keeps_its_own_default():
    # Capping every request by the longest prompt would also cap the prompts that
    # do fit, so the derivation stands down and the per-request preflight rejects
    # the oversized ones.
    assert (
        resolve_task_max_tokens(
            "Bench",
            context_length=1_024,
            requested_max_tokens=None,
            prompt_lengths=_lengths(3_392),
        )
        is None
    )


def test_a_loaded_benchmark_requests_the_derived_response_budget(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "prompt-length-test")
    manager = TaskManager(task_list=["OlympiadBench"], max_length=CONTEXT_LENGTH)
    benchmark = manager.get_benchmark("OlympiadBench")
    expected = resolve_task_max_tokens(
        "OlympiadBench",
        context_length=CONTEXT_LENGTH,
        requested_max_tokens=None,
        prompt_lengths=load_prompt_lengths(),
    )
    from eval.sources.common import SourceOptions
    from eval.contracts.eval_spec import SourceRef
    options = SourceOptions("OlympiadBench", SourceRef("fixture"), limit=1, max_tokens=expected)
    specs = list(benchmark.iter_specs(options))
    assert expected > 8_192
    assert {spec.requests[0].generation["max_gen_toks"] for spec in specs} == {expected}
