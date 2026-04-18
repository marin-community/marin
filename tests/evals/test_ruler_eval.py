# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_ruler_eval
from experiments.evals.task_configs import RULER_MAX_GENERATION_TOKENS, ruler_tasks_for_lengths
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import (
    _lm_eval_metadata,
    _model_args_for_lm_eval,
)


def test_ruler_tasks_for_lengths_sets_task_metadata():
    tasks = ruler_tasks_for_lengths((4096, 32768), task_names=("niah_single_1", "ruler_vt"))

    assert tasks == (
        EvalTaskConfig(
            name="niah_single_1",
            num_fewshot=0,
            task_alias="niah_single_1",
            task_kwargs={"max_seq_lengths": (4096, 32768)},
        ),
        EvalTaskConfig(
            name="ruler_vt",
            num_fewshot=0,
            task_alias="ruler_vt",
            task_kwargs={"max_seq_lengths": (4096, 32768)},
        ),
    )


def test_default_ruler_eval_sets_long_context_model_args():
    step = default_ruler_eval(
        "gs://marin-us-central2/checkpoints/tootsie-8b-giraffe-32k/hf/step-2999",
        lengths=(4096, 32768),
        task_names=("niah_single_1",),
        tokenizer="stanford-crfm/marin-tokenizer",
        resource_config=ResourceConfig.with_cpu(cpu=1),
        apply_chat_template=True,
        discover_latest_checkpoint=False,
    )

    config = step.config

    assert config.evaluator == "lm_evaluation_harness"
    assert config.apply_chat_template is True
    assert config.discover_latest_checkpoint is False
    assert config.task_metadata == {"max_seq_lengths": (4096, 32768)}
    assert config.engine_kwargs["tokenizer"] == "stanford-crfm/marin-tokenizer"
    assert config.engine_kwargs["max_model_len"] > 32768
    assert config.engine_kwargs["max_length"] == config.engine_kwargs["max_model_len"]
    assert config.engine_kwargs["max_gen_toks"] == RULER_MAX_GENERATION_TOKENS


def test_lm_eval_metadata_merges_tokenizer_and_task_kwargs():
    model = ModelConfig(
        name="test-model",
        path=None,
        engine_kwargs={},
        task_metadata={"max_seq_lengths": (4096,)},
    )
    task = EvalTaskConfig(
        name="niah_single_1",
        num_fewshot=0,
        task_kwargs={"max_seq_lengths": (4096, 8192)},
    )

    metadata = _lm_eval_metadata(model, task, tokenizer="/tmp/tokenizer")

    assert metadata == {
        "max_seq_lengths": (4096, 8192),
        "pretrained": "/tmp/tokenizer",
        "tokenizer": "/tmp/tokenizer",
    }


def test_lm_eval_model_args_include_length_and_tokenizer():
    model = ModelConfig(
        name="test-model",
        path=None,
        engine_kwargs={"max_length": 32768, "max_model_len": 32768, "tokenizer": "ignored"},
    )

    args = _model_args_for_lm_eval(
        model=model,
        model_id="served-model",
        server_url="http://127.0.0.1:8000/v1",
        tokenizer="/tmp/tokenizer",
        apply_chat_template=False,
    )

    assert args.startswith("model=served-model,base_url=http://127.0.0.1:8000/v1/completions")
    assert "tokenizer=/tmp/tokenizer" in args
    assert "max_length=32768" in args
    assert "max_model_len=32768" in args
    assert "tokenized_requests=False" in args
