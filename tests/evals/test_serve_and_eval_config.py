# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The parent builds the eval child's config; the child turns each task into an ``eval.eval`` argv.

These are the two pure pieces of the serve->eval handoff that do not need a cluster: the JSON payload
the parent hands the eval child, and the lm-eval command the child runs per task. Everything else
(job submission, serving, the eval itself) is exercised by the cluster smoke.
"""

import json

from marin.evaluation.evaluation_config import EvalTaskConfig

from experiments.evals.evalchemy.run_evalchemy_client import build_command, build_model_args
from experiments.evals.evalchemy.serve_and_eval import (
    EvalchemyEvalConfig,
    ServedEndpoint,
    ServeSpec,
    _client_config_json,
)

_ENDPOINT = ServedEndpoint(base_url="http://10.0.0.1:30000/v1", model_id="Qwen/Qwen3-0.6B", tokenizer="Qwen/Qwen3-0.6B")


def _config(**overrides) -> EvalchemyEvalConfig:
    base = dict(
        model="Qwen/Qwen3-0.6B",
        tasks=(EvalTaskConfig("arc_easy", 0), EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_cot")),
        out_path="gs://bucket/evals/qwen3/core",
        serve=ServeSpec(region="us-east5"),
    )
    base.update(overrides)
    return EvalchemyEvalConfig(**base)


def test_client_config_json_encodes_tasks_with_shot_dirs():
    payload = json.loads(_client_config_json(_config(), _ENDPOINT))

    assert payload["base_url"] == _ENDPOINT.base_url
    assert payload["model_id"] == _ENDPOINT.model_id
    # Each task carries its own upload dir; task_alias wins over name, and the fewshot count is in it.
    assert payload["tasks"] == [
        {"name": "arc_easy", "num_fewshot": 0, "dir": "arc_easy_0shot"},
        {"name": "gsm8k", "num_fewshot": 5, "dir": "gsm8k_cot_5shot"},
    ]


def test_build_command_completion_route_with_fewshot_and_limit():
    config = json.loads(_client_config_json(_config(max_eval_instances=7), _ENDPOINT))
    cmd = build_command(config, config["tasks"][1], "/tmp/out", "/opt/py")

    assert cmd[:3] == ["/opt/py", "-m", "eval.eval"]
    assert "local-completions" in cmd and "--apply_chat_template" not in cmd
    # gsm8k is 5-shot; the limit caps evaluated instances.
    assert cmd[cmd.index("--num_fewshot") + 1] == "5"
    assert cmd[cmd.index("--limit") + 1] == "7"
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "base_url=http://10.0.0.1:30000/v1/completions" in model_args
    assert "tokenizer=Qwen/Qwen3-0.6B" in model_args


def test_build_command_chat_route_toggles_model_and_endpoint():
    config = json.loads(_client_config_json(_config(apply_chat_template=True), _ENDPOINT))
    cmd = build_command(config, config["tasks"][0], "/tmp/out", "/opt/py")

    assert "local-chat-completions" in cmd and "--apply_chat_template" in cmd
    assert "/v1/chat/completions" in build_model_args(config)
    # arc_easy is 0-shot, so no --num_fewshot override is emitted.
    assert "--num_fewshot" not in cmd
