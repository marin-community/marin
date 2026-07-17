# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The parent builds the eval child's config; the child turns each task into an ``eval.eval`` argv.

These are the two pure pieces of the serve->eval handoff that do not need a cluster: the JSON payload
the parent hands the eval child (one upload dir per task-config, kept distinct so shot variants of a
task do not collide), and the lm-eval command the child runs per task. Everything else (job
submission, serving, the eval itself) is exercised by the cluster smoke.
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


def test_client_config_json_carries_endpoint_and_per_task_dirs():
    payload = json.loads(_client_config_json(_config(), _ENDPOINT))

    assert payload["base_url"] == _ENDPOINT.base_url
    assert payload["model_id"] == _ENDPOINT.model_id
    assert payload["tokenizer"] == _ENDPOINT.tokenizer
    # Each task carries the bare lm-eval name (what --tasks runs) plus its own upload dir: an alias is
    # used verbatim, an un-aliased task falls back to name_Nshot. The dir is what keeps results apart.
    assert payload["tasks"] == [
        {"name": "arc_easy", "num_fewshot": 0, "dir": "arc_easy_0shot"},
        {"name": "gsm8k", "num_fewshot": 5, "dir": "gsm8k_cot"},
    ]


def test_task_dirs_distinguish_shot_variants_of_one_task():
    # One task at two shot counts (as CORE_TASKS runs hellaswag): the bare name repeats, so the distinct
    # aliases -> distinct dirs are the only thing keeping the two results from overwriting each other.
    config = _config(
        tasks=(
            EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
            EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),
        )
    )
    tasks = json.loads(_client_config_json(config, _ENDPOINT))["tasks"]

    assert [t["name"] for t in tasks] == ["hellaswag", "hellaswag"]
    assert [t["dir"] for t in tasks] == ["hellaswag_0shot", "hellaswag_10shot"]


def test_build_command_completion_route_with_fewshot_and_limit():
    config = json.loads(_client_config_json(_config(max_eval_instances=7), _ENDPOINT))
    cmd = build_command(config, config["tasks"][1], "/tmp/out", "/opt/py")

    assert cmd[:5] == ["/opt/py", "-m", "eval.eval", "--model", "local-completions"]
    assert "--apply_chat_template" not in cmd
    assert cmd[cmd.index("--tasks") + 1] == "gsm8k"
    assert cmd[cmd.index("--output_path") + 1] == "/tmp/out"
    assert cmd[cmd.index("--gen_kwargs") + 1] == "max_gen_toks=2048"
    # gsm8k is 5-shot; the limit caps evaluated instances.
    assert cmd[cmd.index("--num_fewshot") + 1] == "5"
    assert cmd[cmd.index("--limit") + 1] == "7"
    model_args = dict(pair.split("=", 1) for pair in cmd[cmd.index("--model_args") + 1].split(","))
    assert model_args["base_url"] == "http://10.0.0.1:30000/v1/completions"
    assert model_args["model"] == "Qwen/Qwen3-0.6B"
    assert model_args["tokenizer"] == "Qwen/Qwen3-0.6B"
    assert model_args["num_concurrent"] == "16"


def test_build_command_chat_route_toggles_model_and_endpoint():
    config = json.loads(_client_config_json(_config(apply_chat_template=True), _ENDPOINT))
    cmd = build_command(config, config["tasks"][0], "/tmp/out", "/opt/py")

    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    assert "--apply_chat_template" in cmd
    assert "base_url=http://10.0.0.1:30000/v1/chat/completions" in build_model_args(config)
    # arc_easy is 0-shot with no instance cap, so neither --num_fewshot nor --limit is emitted.
    assert "--num_fewshot" not in cmd
    assert "--limit" not in cmd
