# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The parent builds the Evalchemy child's config; the child turns each task into an ``eval.eval`` argv.

These are the pure pieces of the serve->eval handoff that do not need a cluster: the JSON payload
the parent hands the eval child (one upload dir per task-config, kept distinct so shot variants of a
task do not collide), the lm-eval command the child runs per task (route selection between the
completions and chat APIs included), and the empty-results guard. Everything else (job submission,
serving, the eval itself) is exercised by the cluster smoke.
"""

import json
import os
import shlex
from types import SimpleNamespace

import marin.evaluation.evalchemy.runner as evalchemy_runner
from marin.evaluation.evalchemy.client import build_command, build_model_args, scored_results, served_max_length
from marin.evaluation.evalchemy.runner import (
    EvalchemyRunConfig,
    _run_config_json,
)
from marin.evaluation.evalchemy.runtime import EVALCHEMY_EXTRA_PACKAGES, EVALCHEMY_REQUIREMENT
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.serving_config import _auto_serve_overrides_from_config, auto_serve_overrides
from marin.inference.types import OpenAIEndpoint, RunningModel

_MODEL = RunningModel(
    endpoint=OpenAIEndpoint(
        base_url="http://10.0.0.1:30000/v1",
        model="Qwen/Qwen3-0.6B",
    ),
    tokenizer="Qwen/Qwen3-0.6B",
)


def _config(**overrides) -> EvalchemyRunConfig:
    base = dict(
        name="core",
        tasks=(EvalTaskConfig("arc_easy", 0), EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_cot", generation=True)),
    )
    base.update(overrides)
    return EvalchemyRunConfig(**base)


def _payload(config: EvalchemyRunConfig | None = None) -> dict:
    return json.loads(_run_config_json(_MODEL, config or _config(), "gs://bucket/evals/qwen3/core"))


def test_client_config_json_carries_endpoint_and_per_task_dirs():
    payload = _payload()

    assert payload["base_url"] == _MODEL.endpoint.base_url
    assert payload["model_id"] == _MODEL.endpoint.model
    assert payload["tokenizer"] == _MODEL.tokenizer
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
        },
        {
            "name": "gsm8k",
            "num_fewshot": 5,
            "dir": "gsm8k_cot",
            "generation": True,
            "unsafe_code": False,
            "completion_only": False,
        },
    ]


def test_task_dirs_distinguish_shot_variants_of_one_task():
    # One task at two shot counts: the bare name repeats, so the distinct aliases -> distinct dirs are
    # the only thing keeping the two results from overwriting each other.
    config = _config(
        tasks=(
            EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
            EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),
        )
    )
    tasks = _payload(config)["tasks"]

    assert [t["name"] for t in tasks] == ["hellaswag", "hellaswag"]
    assert [t["dir"] for t in tasks] == ["hellaswag_0shot", "hellaswag_10shot"]


def test_build_command_completion_route_with_fewshot_and_limit():
    config = _payload(_config(max_eval_instances=7))
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


def test_extra_gen_kwargs_ride_on_gen_kwargs():
    # A thinking model (snowball-sft) needs skip_special_tokens=false so its delimiters survive scoring
    # plus a light repetition penalty; both ride on --gen_kwargs alongside the budget, on every task.
    config = _payload(_config(extra_gen_kwargs={"skip_special_tokens": "false", "repetition_penalty": "1.1"}))
    cmd = build_command(config, config["tasks"][1], "/tmp/out", "/opt/py", None)
    gen_kwargs = cmd[cmd.index("--gen_kwargs") + 1]
    assert gen_kwargs == "max_gen_toks=2048,skip_special_tokens=false,repetition_penalty=1.1"


def test_no_extra_gen_kwargs_leaves_gen_kwargs_at_budget_only():
    cmd = build_command(_payload(), _payload()["tasks"][1], "/tmp/out", "/opt/py", None)
    assert cmd[cmd.index("--gen_kwargs") + 1] == "max_gen_toks=2048"


def test_build_command_chat_route_needs_template_and_generation():
    config = _payload(_config(apply_chat_template=True))
    generative, mcq = config["tasks"][1], config["tasks"][0]

    # A generation task of a chat-template model runs through the chat API...
    cmd = build_command(config, generative, "/tmp/out", "/opt/py", None)
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    assert "--apply_chat_template" in cmd
    assert "base_url=http://10.0.0.1:30000/v1/chat/completions" in build_model_args(config, True, None)

    # ...but a loglikelihood (MCQ) task always uses completions: chat endpoints cannot echo prompt
    # logprobs, and lm-eval rejects loglikelihood over chat completions.
    cmd = build_command(config, mcq, "/tmp/out", "/opt/py", None)
    assert cmd[cmd.index("--model") + 1] == "local-completions"
    assert "--apply_chat_template" not in cmd


def test_completion_only_pins_completions_route_and_forwards_unsafe_code():
    # humaneval-style code infill: chat formatting breaks the raw-continuation scoring, so the task
    # pins the completions route even for a chat-template model, and code execution needs the opt-in.
    config = _config(
        apply_chat_template=True,
        tasks=(
            EvalTaskConfig(
                "humaneval", 0, task_alias="humaneval_0shot", generation=True, unsafe_code=True, completion_only=True
            ),
        ),
    )
    config = _payload(config)
    cmd = build_command(config, config["tasks"][0], "/tmp/out", "/opt/py", None)

    assert cmd[cmd.index("--model") + 1] == "local-completions"
    assert "--apply_chat_template" not in cmd
    assert "--confirm_run_unsafe_code" in cmd


def test_model_args_carry_served_max_length():
    # lm-eval assumes a 2048-token window unless told otherwise, silently left-truncating few-shot
    # prompts; the client reads the served max_model_len and passes it through.
    args = dict(pair.split("=", 1) for pair in build_model_args(_payload(), False, 4096).split(","))
    assert args["max_length"] == "4096"
    assert args["tokenized_requests"] == "False"


def test_auto_overrides_clamps_context_and_leaves_plain_model_alone():
    # A vanilla dense model needs no derived vLLM flags, and max_model_len clamps down to the model's
    # own context so lm-eval never asks for more window than the checkpoint was trained for.
    config = {"architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 8192}
    extra_args, max_model_len = _auto_serve_overrides_from_config("Qwen/Qwen3-1.7B", config, 40960, ())
    assert extra_args == ()
    assert max_model_len == 8192


def test_auto_overrides_derives_gdn_and_vision_flags():
    # A linear-attention (GDN) multimodal checkpoint needs the triton prefill backend and, since these
    # evals are text-only, an explicit zero multimodal limit so vLLM does not reserve image/video slots.
    config = {
        "architectures": ["Qwen3NextForConditionalGeneration"],
        "vision_config": {"depth": 24},
        "text_config": {"max_position_embeddings": 262144},
    }
    extra_args, max_model_len = _auto_serve_overrides_from_config("Qwen/Qwen3-Next-Thinking", config, 32768, ())
    assert "--gdn-prefill-backend" in extra_args
    assert extra_args[extra_args.index("--gdn-prefill-backend") + 1] == "triton"
    assert "--limit-mm-per-prompt" in extra_args
    assert "--reasoning-parser" in extra_args
    # The nested text_config context wins, and 32768 is already below it, so the cap is left untouched.
    assert max_model_len == 32768


def test_auto_overrides_never_overrides_explicit_flags():
    # An explicitly configured backend must win over the derived default; the merge only fills gaps.
    # The org/model name is deliberately non-Qwen so only the GDN rule (via architecture) fires.
    config = {"architectures": ["Qwen3NextForCausalLM"], "max_position_embeddings": 262144}
    existing = ("--gdn-prefill-backend", "cuda")
    extra_args, _ = _auto_serve_overrides_from_config("org/gdn-model", config, None, existing)
    assert extra_args == existing
    # A None cap stays None: vLLM then falls back to the model's own default context.
    _, max_model_len = _auto_serve_overrides_from_config("org/gdn-model", config, None, ())
    assert max_model_len is None


def test_auto_serve_overrides_reads_local_model_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "max_position_embeddings": 16384,
            }
        )
    )

    extra_args, max_model_len = auto_serve_overrides(str(tmp_path), 32768)

    assert extra_args == ("--gdn-prefill-backend", "triton")
    assert max_model_len == 16384


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


def test_served_max_length_does_not_log_capability_url(monkeypatch, capsys):
    base_url = "https://iris.example/proxy/task/secret-capability/inference/v1"

    def fail_with_url(url: str, timeout: int):
        raise RuntimeError(f"request failed for {url} after {timeout}s")

    monkeypatch.setattr("urllib.request.urlopen", fail_with_url)

    assert served_max_length(base_url) is None
    assert "secret-capability" not in capsys.readouterr().out


def test_evalchemy_child_runs_pinned_uvx_environment_on_default_task_image(monkeypatch):
    submitted: dict = {}

    class Job:
        job_id = "/eval/client"

        def wait(self, timeout: float) -> None:
            assert timeout == float("inf")

    class Client:
        def submit(self, **kwargs):
            submitted.update(kwargs)
            return Job()

    monkeypatch.setattr(evalchemy_runner, "iris_ctx", lambda: SimpleNamespace(client=Client()))

    evalchemy_runner._run_evalchemy_child(_MODEL, _config(), "gs://bucket/evals/qwen3/core", {})

    assert submitted["task_image"] is None
    command = [
        "exec",
        "uvx",
        "--no-config",
        "--python",
        "3.12",
        "--from",
        EVALCHEMY_REQUIREMENT,
    ]
    for package in EVALCHEMY_EXTRA_PACKAGES:
        command.extend(("--with", package))
    command.extend(("python", "$IRIS_WORKDIR/lib/marin/src/marin/evaluation/evalchemy/client.py"))

    assert shlex.split(submitted["entrypoint"].command[2]) == command
