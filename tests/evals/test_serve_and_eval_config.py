# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The parent builds the eval child's config; the child turns each task into an ``eval.eval`` argv.

These are the pure pieces of the serve->eval handoff that do not need a cluster: the JSON payload
the parent hands the eval child (one upload dir per task-config, kept distinct so shot variants of a
task do not collide), the lm-eval command the child runs per task (route selection between the
completions and chat APIs included), and the empty-results guard. Everything else (job submission,
serving, the eval itself) is exercised by the cluster smoke.
"""

import json
import os
from typing import cast

from iris.client import Job
from marin.evaluation.evaluation_config import EvalTaskConfig

from experiments.evals.evalchemy.marin_evalchemy_gpu import CANONICAL_AIME_SEEDS, _aime_seeds, _grug_profile
from experiments.evals.evalchemy.run_evalchemy_client import build_command, build_model_args, scored_results
from experiments.evals.evalchemy.serve_and_eval import (
    EvalSession,
    EvalUnit,
    ServedEndpoint,
    _client_config_json,
)

# Only the address fields matter for config-building; the serve-child job handle is never touched.
_ENDPOINT = ServedEndpoint(
    base_url="http://10.0.0.1:30000/v1",
    model_id="Qwen/Qwen3-0.6B",
    tokenizer="Qwen/Qwen3-0.6B",
    job="/test/serve",
    handle=cast(Job, None),
    name="eval-serve-test",
)


def _session(**overrides) -> EvalSession:
    base = dict(model="Qwen/Qwen3-0.6B")
    base.update(overrides)
    return EvalSession(**base)


def _unit(**overrides) -> EvalUnit:
    base = dict(
        name="core",
        tasks=(EvalTaskConfig("arc_easy", 0), EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_cot", generation=True)),
        out_path="gs://bucket/evals/qwen3/core",
    )
    base.update(overrides)
    return EvalUnit(**base)


def _payload(session: EvalSession | None = None, unit: EvalUnit | None = None) -> dict:
    return json.loads(_client_config_json(session or _session(), unit or _unit(), _ENDPOINT))


def test_client_config_json_carries_endpoint_and_per_task_dirs():
    payload = _payload()

    assert payload["base_url"] == _ENDPOINT.base_url
    assert payload["model_id"] == _ENDPOINT.model_id
    assert payload["tokenizer"] == _ENDPOINT.tokenizer
    # Each task carries the bare lm-eval name (what --tasks runs) plus its own upload dir: an alias is
    # used verbatim, an un-aliased task falls back to name_Nshot. The dir is what keeps results apart;
    # the flags drive the child's completions-vs-chat route and unsafe-code opt-in per task.
    assert payload["tasks"] == [
        {
            "name": "arc_easy",
            "num_fewshot": 0,
            "dir": "arc_easy_0shot",
            "generation": False,
            "unsafe_code": False,
            "completion_only": False,
            "seed": None,
        },
        {
            "name": "gsm8k",
            "num_fewshot": 5,
            "dir": "gsm8k_cot",
            "generation": True,
            "unsafe_code": False,
            "completion_only": False,
            "seed": None,
        },
    ]


def test_task_dirs_distinguish_shot_variants_of_one_task():
    # One task at two shot counts: the bare name repeats, so the distinct aliases -> distinct dirs are
    # the only thing keeping the two results from overwriting each other.
    unit = _unit(
        tasks=(
            EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
            EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),
        )
    )
    tasks = _payload(unit=unit)["tasks"]

    assert [t["name"] for t in tasks] == ["hellaswag", "hellaswag"]
    assert [t["dir"] for t in tasks] == ["hellaswag_0shot", "hellaswag_10shot"]


def test_build_command_completion_route_with_fewshot_and_limit():
    config = _payload(unit=_unit(max_eval_instances=7))
    cmd = build_command(config, config["tasks"][1], "/tmp/out", "/opt/py", None)

    assert cmd[:5] == ["/opt/py", "-m", "eval.eval", "--model", "local-completions"]
    assert "--apply_chat_template" not in cmd
    assert cmd[cmd.index("--tasks") + 1] == "gsm8k"
    assert cmd[cmd.index("--output_path") + 1] == "/tmp/out"
    assert cmd[cmd.index("--gen_kwargs") + 1] == "max_gen_toks=2048"
    # Chat-native benchmarks read --max_tokens instead of gen_kwargs; both carry the unit's cap.
    assert cmd[cmd.index("--max_tokens") + 1] == "2048"
    # gsm8k is 5-shot; the limit caps evaluated instances.
    assert cmd[cmd.index("--num_fewshot") + 1] == "5"
    assert cmd[cmd.index("--limit") + 1] == "7"
    model_args = dict(pair.split("=", 1) for pair in cmd[cmd.index("--model_args") + 1].split(","))
    assert model_args["base_url"] == "http://10.0.0.1:30000/v1/completions"
    assert model_args["model"] == "Qwen/Qwen3-0.6B"
    assert model_args["tokenizer"] == "Qwen/Qwen3-0.6B"


def test_build_command_chat_route_needs_template_and_generation():
    config = _payload(session=_session(apply_chat_template=True))
    generative, mcq = config["tasks"][1], config["tasks"][0]

    # A generation task of a chat-template model runs through the chat API...
    cmd = build_command(config, generative, "/tmp/out", "/opt/py", None)
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    assert "--apply_chat_template" in cmd
    assert "base_url=http://10.0.0.1:30000/v1/chat/completions" in build_model_args(config, True)

    # ...but a loglikelihood (MCQ) task always uses completions: chat endpoints cannot echo prompt
    # logprobs, and lm-eval rejects loglikelihood over chat completions.
    cmd = build_command(config, mcq, "/tmp/out", "/opt/py", None)
    assert cmd[cmd.index("--model") + 1] == "local-completions"
    assert "--apply_chat_template" not in cmd


def test_completion_only_pins_completions_route_and_forwards_unsafe_code():
    # humaneval-style code infill: chat formatting breaks the raw-continuation scoring, so the task
    # pins the completions route even for a chat-template model, and code execution needs the opt-in.
    unit = _unit(
        tasks=(
            EvalTaskConfig(
                "humaneval", 0, task_alias="humaneval_0shot", generation=True, unsafe_code=True, completion_only=True
            ),
        )
    )
    config = _payload(session=_session(apply_chat_template=True), unit=unit)
    cmd = build_command(config, config["tasks"][0], "/tmp/out", "/opt/py", None)

    assert cmd[cmd.index("--model") + 1] == "local-completions"
    assert "--apply_chat_template" not in cmd
    assert "--confirm_run_unsafe_code" in cmd


def test_client_forwards_seed_and_extra_generation_kwargs():
    unit = _unit(tasks=(EvalTaskConfig("AIME24", 0, task_kwargs={"seed": 43}, generation=True),))
    config = _payload(
        session=_session(apply_chat_template=True, extra_gen_kwargs={"skip_special_tokens": "false"}), unit=unit
    )
    cmd = build_command(config, config["tasks"][0], "/tmp/out", "/opt/py", None)

    assert cmd[cmd.index("--seed") + 1] == "43"
    assert cmd[cmd.index("--gen_kwargs") + 1] == "max_gen_toks=2048,skip_special_tokens=false"


def test_grug_profile_uses_local_weight_loader_compatible_extra_args():
    """Grug's old RunAI-only loader config must not leak into local-weight GPU serving."""
    args = _grug_profile("penfever/grug-67b-a2b-sft-s2-thinking-step630")["vllm_extra_args"]

    assert args == (
        "--data-parallel-size",
        "8",
        "--enable-expert-parallel",
        "--revision",
        "delphi-v0-think",
    )
    assert "--model-loader-extra-config" not in args


def test_aime_defaults_to_the_three_canonical_campaign_seeds():
    assert _aime_seeds(None) == CANONICAL_AIME_SEEDS == (42, 43, 44)
    assert _aime_seeds("42,44,99") == (42, 44, 99)


def test_build_command_carries_served_max_length_via_evalchemy_flag():
    # Evalchemy owns the canonical request limit.  Keeping this outside lm-eval model_args makes
    # native and custom benchmarks consume exactly the same explicit --max_length contract.
    cmd = build_command(_payload(), _payload()["tasks"][0], "/tmp/out", "/opt/py", 4096)
    args = dict(pair.split("=", 1) for pair in build_model_args(_payload(), False).split(","))
    assert cmd[cmd.index("--max_length") + 1] == "4096"
    assert "max_length" not in args
    assert args["tokenized_requests"] == "False"


def _write_results(local_out: str, results: dict) -> None:
    """Write a results file in lm-eval's nested ``<task_dir>/<model>/results_<ts>.json`` layout."""
    task_dir = os.path.join(local_out, "mmlu_5shot", "local-completions")
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, "results_2026-07-19T00-00-00.json"), "w") as f:
        json.dump({"results": results}, f)


def test_scored_results_rejects_empty_results_dict(tmp_path):
    # eval.eval exits 0 and still writes results_*.json with an empty "results" dict when every
    # endpoint request fails; the client must treat that as an unscored task.
    empty = tmp_path / "empty"
    empty.mkdir()
    _write_results(str(empty), {})
    assert scored_results(str(empty)) is False

    scored = tmp_path / "scored"
    scored.mkdir()
    _write_results(str(scored), {"mmlu": {"acc,none": 0.42}})
    assert scored_results(str(scored)) is True
