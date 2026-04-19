# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier A: exact `model_args` / kind encoding for lm-eval over OpenAI HTTP.

Unit tests — they do not launch a model or run a server.
"""

import pytest

from marin.evaluation.api import LmEvalRun
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import (
    _build_lm_eval_model_args,
    _lm_eval_kind,
)
from marin.inference.model_launcher import OpenAIEndpoint, RunningModel


def _make_model(
    url: str = "http://host:8000/v1", model: str = "served-model", tokenizer: str = "hf/tok"
) -> RunningModel:
    return RunningModel(endpoint=OpenAIEndpoint(url=url, model=model), tokenizer_ref=tokenizer)


def _make_run(*, apply_chat_template: bool = False, extra_model_args: tuple[str, ...] = ()) -> LmEvalRun:
    return LmEvalRun(
        evals=[EvalTaskConfig(name="gsm8k_cot", num_fewshot=8)],
        output_path="/tmp/out",
        apply_chat_template=apply_chat_template,
        generation_params={},
        extra_model_args=extra_model_args,
    )


def test_lm_eval_kind_completions_by_default():
    assert _lm_eval_kind(_make_run()) == "local-completions"


def test_lm_eval_kind_chat_completions_when_chat_template():
    assert _lm_eval_kind(_make_run(apply_chat_template=True)) == "local-chat-completions"


def test_model_args_points_to_completions_endpoint():
    args = _build_lm_eval_model_args(_make_model(), _make_run())
    # Positional order matters: lm-eval parses comma-separated k=v pairs.
    assert "base_url=http://host:8000/v1/completions" in args
    assert "/chat/completions" not in args


def test_model_args_points_to_chat_completions_endpoint_when_chat_template():
    args = _build_lm_eval_model_args(_make_model(), _make_run(apply_chat_template=True))
    assert "base_url=http://host:8000/v1/chat/completions" in args


def test_model_args_contains_required_keys():
    args = _build_lm_eval_model_args(_make_model(), _make_run())
    parts = args.split(",")
    # Contract with lm-eval's local-completions wrapper: the first three keys
    # are positional, and `tokenizer_backend=huggingface` + `tokenized_requests=False`
    # are required to route through HF tokenization rather than the server's.
    assert parts[0] == "model=served-model"
    assert parts[1] == "base_url=http://host:8000/v1/completions"
    assert parts[2] == "tokenizer=hf/tok"
    assert "tokenizer_backend=huggingface" in parts
    assert "tokenized_requests=False" in parts


def test_model_args_appends_extra_model_args_verbatim():
    extras = ("max_gen_toks=4096", "num_concurrent=8")
    args = _build_lm_eval_model_args(_make_model(), _make_run(extra_model_args=extras))
    assert args.endswith("max_gen_toks=4096,num_concurrent=8")


def test_model_args_does_not_include_max_eval_instances_or_wandb_tags():
    # These are eval-run scoping concerns, not model-arg concerns; they must not
    # leak into the lm-eval --model_args string (they're passed via different flags).
    run = LmEvalRun(
        evals=[EvalTaskConfig(name="gsm8k_cot", num_fewshot=8)],
        output_path="/tmp/out",
        apply_chat_template=False,
        generation_params={},
        max_eval_instances=5,
        wandb_tags=["smoke"],
    )
    args = _build_lm_eval_model_args(_make_model(), run)
    assert "max_eval_instances" not in args
    assert "wandb" not in args.lower()


@pytest.mark.parametrize(
    "endpoint_url, expected_base",
    [
        ("http://host:8000/v1", "http://host:8000/v1/completions"),
        # The launcher is responsible for not emitting trailing slashes; if it
        # did, lm-eval would probe `/v1//completions` which still works. But the
        # canonical contract is no trailing slash.
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1/completions"),
    ],
)
def test_model_args_preserves_endpoint_url_exactly(endpoint_url, expected_base):
    args = _build_lm_eval_model_args(_make_model(url=endpoint_url), _make_run())
    assert f"base_url={expected_base}" in args
