# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RULER's launcher registration: it stays out of the routine suites, and its definition resolves to
an Evalchemy task the client will route through lm-eval's own CLI (metadata-carrying, completion-only).
"""

from marin.evaluation.evalchemy.runner import EvalchemyExecutor
from marin.evaluation.model_config import ModelConfig

from experiments.evaluation.evals import EVALS, SUITES

_MODEL = ModelConfig(name="qwen3-8b", location="Qwen/Qwen3-8B")


def test_ruler_is_registered_but_kept_out_of_routine_suites():
    # RULER needs the eval image rebuilt with the [ruler] deps, so a routine smoke/core launch must
    # never select it; it is reachable only through explicit keys and the longcontext suite.
    assert "ruler" in EVALS
    assert "ruler-smoke" in EVALS
    assert "ruler" not in SUITES["smoke"]
    assert "ruler" not in SUITES["core"]
    assert "ruler" in SUITES["longcontext"]


def test_ruler_resolves_to_a_metadata_completion_task():
    # The definition must carry the metadata that makes the client take the native lm-eval route, and
    # pin the completions route so the haystack is scored as a raw continuation on any model.
    executor = EVALS["ruler"].executor_for(_MODEL, limit=None)
    assert isinstance(executor, EvalchemyExecutor)
    (task,) = executor.config.tasks
    assert task.name == "ruler"
    assert task.num_fewshot == 0
    assert task.generation and task.completion_only
    assert task.metadata is not None and task.metadata["max_seq_lengths"]


def test_ruler_smoke_is_a_single_short_length_with_an_instance_cap():
    # RULER synthesizes a fresh haystack per length, so a cheap smoke needs one short length (not just
    # an instance cap): one 4k length plus the cap keeps the smoke fast.
    executor = EVALS["ruler-smoke"].executor_for(_MODEL, limit=None)
    (task,) = executor.config.tasks
    assert task.metadata["max_seq_lengths"] == [4096]
    assert executor.config.max_eval_instances is not None


def test_limit_override_flows_into_the_ruler_executor():
    executor = EVALS["ruler"].executor_for(_MODEL, limit=5)
    assert executor.config.max_eval_instances == 5
