# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The parent builds the Evalchemy child's config; the child turns each task into an ``evalchemy`` argv.

These are the pure pieces of the serve->eval handoff that do not need a cluster: the JSON payload
the parent hands the eval child (one upload dir per task-config, kept distinct so shot variants of a
task do not collide), the lm-eval command the child runs per task (route selection between the
completions and chat APIs included), and the FineStore output target. Everything else (job submission,
serving, the eval itself) is exercised by the cluster smoke.
"""

import base64
import io
import json
import os
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest
from iris.cluster.client.bundle import create_workspace_zip
from marin.evaluation.evalchemy import runtime
from marin.evaluation.evalchemy.client import build_command, build_model_args
from marin.evaluation.evalchemy.config import EvalchemyConfig, EvalchemyJudgeConfig
from marin.evaluation.evalchemy.runner import (
    EvalchemyRunConfig,
    EvalchemyRuntimeConfig,
    _run_config_json,
    _run_evalchemy_child,
)
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
        max_gen_toks=2048,
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


def test_file_config_fields_reach_the_evalchemy_command():
    config = _payload(
        _config(
            tasks=(EvalTaskConfig("ifeval", 0, generation=True),),
            batch_size=1,
            seed=1234,
            extra_model_args={"timeout": 900},
            max_length=32768,
            runtime=EvalchemyRuntimeConfig(dataset_cache="s3://eval-data/cache"),
        )
    )

    command = build_command(config, config["tasks"][0], "/opt/py", 32768)

    assert command[command.index("--batch_size") + 1] == "1"
    assert command[command.index("--seed") + 1] == "1234"
    assert command[command.index("--dataset_cache") + 1] == "s3://eval-data/cache"
    model_args = dict(pair.split("=", 1) for pair in command[command.index("--model_args") + 1].split(","))
    assert model_args["timeout"] == "900"
    assert model_args["max_length"] == "32768"


def test_dataset_cache_follows_output_region_instead_of_launcher_region(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-west1")
    monkeypatch.delenv("MARIN_TEMP_PREFIX", raising=False)
    config = json.loads(_run_config_json(_MODEL, _config(), "gs://marin-us-central2/evals/run"))

    command = build_command(config, config["tasks"][0], "/opt/py", 32768)

    assert command[command.index("--dataset_cache") + 1].startswith("gs://marin-us-central2/")


def test_explicit_thinking_mode_is_serialized_for_chat_endpoint():
    template_kwargs = {"enable_thinking": True, "add_generation_prompt": False}
    config = _payload(
        _config(
            tasks=(EvalTaskConfig("MATH500", 0, generation=True),),
            apply_chat_template=True,
            chat_template_kwargs=template_kwargs,
        )
    )

    command = build_command(config, config["tasks"][0], "/opt/py", 32768)
    assert command[command.index("--model") + 1] == "local-chat-completions"
    model_args = dict(pair.split("=", 1) for pair in command[command.index("--model_args") + 1].split(","))
    assert (
        json.loads(base64.urlsafe_b64decode(model_args["chat_template_kwargs"].removeprefix("base64:")))
        == template_kwargs
    )


def test_parent_rejects_endpoint_model_arg_overrides():
    # Endpoint identity is Marin-owned; a config must not reroute the isolated child to another model.
    with pytest.raises(ValueError):
        _payload(_config(extra_model_args={"model": "other"}))


def test_financebench_requires_an_explicit_external_judge():
    with pytest.raises(ValueError, match="requires an explicit judge"):
        EvalchemyConfig(tasks=("FinanceBench",))

    judge = EvalchemyJudgeConfig(
        base_url="https://judge.example/v1",
        model="judge-model",
        api_key=("env:JUDGE_KEY",),
    )
    with pytest.raises(ValueError, match="only for a single FinanceBench"):
        EvalchemyConfig(tasks=("FinanceBench", "gsm8k"), judge=judge)
    with pytest.raises(ValueError, match="only for a single FinanceBench"):
        EvalchemyConfig(tasks=("gsm8k",), judge=judge)


@pytest.mark.parametrize(
    "judge",
    [
        {
            "base_url": "https://user:password@judge.example/v1",
            "model": "judge-model",
            "api_key": ["env:JUDGE_KEY"],
        },
        {
            "base_url": "https://judge.example/v1?api_key=secret",
            "model": "judge-model",
            "api_key": ["env:JUDGE_KEY"],
        },
        {
            "base_url": "https://judge.example/v1",
            "model": "judge-model",
            "api_key": ["literal-secret"],
        },
    ],
)
def test_financebench_rejects_inline_judge_credentials(judge):
    with pytest.raises(ValueError):
        EvalchemyConfig(tasks=("FinanceBench",), judge=judge)


def test_evalchemy_child_keeps_candidate_and_judge_credentials_separate(monkeypatch):
    submitted = []

    class FakeJob:
        job_id = "/eval/financebench"

        def wait(self, timeout):
            assert timeout == float("inf")

    client = SimpleNamespace(submit=lambda **kwargs: submitted.append(kwargs) or FakeJob())
    monkeypatch.setattr("marin.evaluation.evalchemy.runner.iris_ctx", lambda: SimpleNamespace(client=client))
    config = _config(
        name="financebench",
        tasks=(EvalTaskConfig("FinanceBench", 0, generation=True),),
        judge=EvalchemyJudgeConfig(
            base_url="https://judge.example/v1",
            model="judge-model",
            api_key=("env:TOGETHER_API_KEY",),
        ),
    )

    _run_evalchemy_child(
        _MODEL,
        config,
        "gs://bucket/evals/qwen3/financebench",
        {"JUDGE_API_KEY": "judge-key", "HF_TOKEN": "candidate-runtime-token"},
    )

    env = submitted[0]["environment"].env_vars
    assert env["OPENAI_API_KEY"] == "local-endpoint"
    assert env["JUDGE_API_KEY"] == "judge-key"
    assert env["JUDGE_BASE_URL"] == "https://judge.example/v1"
    assert env["JUDGE_MODEL"] == "judge-model"
    assert env["HF_TOKEN"] == "candidate-runtime-token"


def test_evalchemy_child_rejects_a_missing_judge_secret():
    config = _config(
        name="financebench",
        tasks=(EvalTaskConfig("FinanceBench", 0, generation=True),),
        judge=EvalchemyJudgeConfig(
            base_url="https://judge.example/v1",
            model="judge-model",
            api_key=("env:TOGETHER_API_KEY",),
        ),
    )

    with pytest.raises(ValueError, match="requires JUDGE_API_KEY"):
        _run_evalchemy_child(_MODEL, config, "gs://bucket/evals/qwen3/financebench", {})


def test_evalchemy_child_uses_its_project_environment_instead_of_the_parent(tmp_path, monkeypatch):
    executable = tmp_path / "bin" / "uv"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "Path(os.environ['PROBE_OUTPUT']).write_text(json.dumps({\n"
        "    'environment': os.environ['UV_PROJECT_ENVIRONMENT'],\n"
        "    'cwd': os.getcwd(),\n"
        "}))\n"
    )
    executable.chmod(0o755)
    output = tmp_path / "runtime.json"

    def submit(*, entrypoint, environment, **_kwargs):
        assert not environment.to_proto().setup_scripts
        subprocess.run(
            entrypoint.command,
            env={
                **os.environ,
                "IRIS_WORKDIR": str(tmp_path),
                "UV_PROJECT_ENVIRONMENT": "/parent/.venv",
                "PROBE_OUTPUT": str(output),
                "PATH": f"{executable.parent}:{os.environ['PATH']}",
                **environment.env_vars,
            },
            check=True,
        )
        return SimpleNamespace(job_id="/eval/isolated", wait=lambda **_kwargs: None)

    client = SimpleNamespace(submit=submit)
    monkeypatch.setattr("marin.evaluation.evalchemy.runner.iris_ctx", lambda: SimpleNamespace(client=client))
    _run_evalchemy_child(_MODEL, _config(), "gs://bucket/evals/qwen3/core", {})

    assert json.loads(output.read_text()) == {
        "environment": str(tmp_path / "vendor/evalchemy/.venv"),
        "cwd": str(tmp_path),
    }


@pytest.fixture
def evaluator_sources(tmp_path, monkeypatch):
    for directory in runtime._RUNTIME_DIRECTORIES:
        source = tmp_path / directory / "source.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n")
    for filename in runtime._RUNTIME_FILES:
        source = tmp_path / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("source input\n")
    monkeypatch.setattr(runtime, "find_project_root", lambda _start: tmp_path)
    return tmp_path


@pytest.fixture
def git_evaluator_sources(evaluator_sources):
    (evaluator_sources / ".gitignore").write_text("ignored.py\nbuild/\ndist/\n")
    subprocess.run(["git", "init", "--quiet"], cwd=evaluator_sources, check=True)
    subprocess.run(["git", "add", "."], cwd=evaluator_sources, check=True)
    return evaluator_sources


@pytest.mark.parametrize(
    "filename",
    [
        "vendor/evalchemy/eval/source.py",
        "vendor/evalchemy/uv.lock",
        "lib/finestore/src/source.py",
        "lib/rigging/src/source.py",
        "lib/marin/src/marin/evaluation/evalchemy/client.py",
    ],
)
def test_evalchemy_provenance_changes_when_runtime_inputs_change(evaluator_sources, filename):
    before = runtime.evalchemy_runtime_source_hash()
    source = evaluator_sources / filename
    source.write_text(source.read_text() + "changed input\n")
    assert runtime.evalchemy_runtime_source_hash() != before


def test_evalchemy_provenance_ignores_build_metadata_and_bytecode(evaluator_sources):
    before = runtime.evalchemy_runtime_source_hash()
    for filename in (
        "vendor/evalchemy/eval/__pycache__/source.cpython-312.pyc",
        "vendor/evalchemy/eval/benchmark.egg-info/PKG-INFO",
        "vendor/evalchemy/eval/chat_benchmarks/alpaca_eval/build/lib/alpaca_eval/__init__.py",
        "vendor/evalchemy/eval/chat_benchmarks/MTBench/dist/wheel.whl",
        "vendor/evalchemy/eval/.pytest_cache/v/cache/nodeids",
    ):
        path = evaluator_sources / filename
        path.parent.mkdir(parents=True)
        path.write_text("generated output\n")
    assert runtime.evalchemy_runtime_source_hash() == before


@pytest.mark.parametrize(
    "filename",
    ["vendor/evalchemy/eval/source.py", "vendor/evalchemy/packages/source.py", "vendor/evalchemy/uv.lock"],
)
def test_evalchemy_evaluator_identity_changes_with_evaluator_inputs(git_evaluator_sources, filename):
    before = runtime.evalchemy_evaluator_source_hash()
    source = git_evaluator_sources / filename
    source.write_text(source.read_text() + "changed evaluator\n")

    assert runtime.evalchemy_evaluator_source_hash() != before


@pytest.mark.parametrize(
    "filename",
    [
        "lib/finestore/src/source.py",
        "lib/rigging/src/source.py",
        "lib/marin/src/marin/evaluation/evalchemy/client.py",
    ],
)
def test_evalchemy_dependency_changes_affect_provenance_only(git_evaluator_sources, filename):
    evaluator = runtime.evalchemy_evaluator_source_hash()
    provenance = runtime.evalchemy_runtime_source_hash()
    source = git_evaluator_sources / filename
    source.write_text(source.read_text() + "changed dependency\n")

    assert runtime.evalchemy_evaluator_source_hash() == evaluator
    assert runtime.evalchemy_runtime_source_hash() != provenance


def test_evalchemy_identity_ignores_gitignored_scratch_and_tracks_shipped_new_code(git_evaluator_sources):
    evaluator = runtime.evalchemy_evaluator_source_hash()
    provenance = runtime.evalchemy_runtime_source_hash()
    (git_evaluator_sources / "vendor/evalchemy/eval/ignored.py").write_text("scratch = True\n")

    assert runtime.evalchemy_evaluator_source_hash() == evaluator
    assert runtime.evalchemy_runtime_source_hash() == provenance

    (git_evaluator_sources / "vendor/evalchemy/eval/new_source.py").write_text("scoring = 2\n")

    assert runtime.evalchemy_evaluator_source_hash() != evaluator
    assert runtime.evalchemy_runtime_source_hash() != provenance


def test_evalchemy_identities_match_extracted_iris_bundle(git_evaluator_sources, tmp_path, monkeypatch):
    (git_evaluator_sources / "vendor/evalchemy/eval/new_source.py").write_text("scoring = 2\n")
    (git_evaluator_sources / "vendor/evalchemy/eval/ignored.py").write_text("scratch = True\n")
    build = git_evaluator_sources / "vendor/evalchemy/eval/build/lib/generated.py"
    build.parent.mkdir(parents=True)
    build.write_text("generated = True\n")
    evaluator = runtime.evalchemy_evaluator_source_hash()
    provenance = runtime.evalchemy_runtime_source_hash()

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(io.BytesIO(create_workspace_zip(git_evaluator_sources))) as bundle:
        bundle.extractall(extracted)
    monkeypatch.setattr(runtime, "find_project_root", lambda _start: extracted)

    assert runtime.evalchemy_evaluator_source_hash() == evaluator
    assert runtime.evalchemy_runtime_source_hash() == provenance


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
    cmd = build_command(config, config["tasks"][1], "/opt/py", None)

    assert cmd[:3] == ["/opt/evalchemy", "--model", "local-completions"]
    assert "--apply_chat_template" not in cmd
    assert cmd[cmd.index("--tasks") + 1] == "gsm8k"
    assert cmd[cmd.index("--finestore_output_path") + 1] == config["out_path"]
    assert cmd[cmd.index("--finestore_output_prefix") + 1] == "gsm8k_cot"
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
    assert float(model_args["transport_retry_budget"]) > float(model_args["timeout"])


def test_build_command_uses_evaluator_default_when_fewshot_is_unset():
    config = _payload(_config(tasks=(EvalTaskConfig("ifeval", None, generation=True),)))

    command = build_command(config, config["tasks"][0], "/opt/py", None)

    assert "--num_fewshot" not in command


def test_extra_gen_kwargs_ride_on_gen_kwargs():
    # A thinking model (snowball-sft) needs skip_special_tokens=false so its delimiters survive scoring
    # plus a light repetition penalty; both ride on --gen_kwargs alongside the budget, on every task.
    config = _payload(_config(extra_gen_kwargs={"skip_special_tokens": "false", "repetition_penalty": "1.1"}))
    cmd = build_command(config, config["tasks"][1], "/opt/py", None)
    gen_kwargs = cmd[cmd.index("--gen_kwargs") + 1]
    assert gen_kwargs == "max_gen_toks=2048,skip_special_tokens=false,repetition_penalty=1.1"


def test_no_extra_gen_kwargs_leaves_gen_kwargs_at_budget_only():
    cmd = build_command(_payload(), _payload()["tasks"][1], "/opt/py", None)
    assert cmd[cmd.index("--gen_kwargs") + 1] == "max_gen_toks=2048"


def test_unset_budget_lets_evalchemy_size_its_own_benchmark(monkeypatch):
    # Without a configured cap Evalchemy sizes a chat benchmark's responses from the served context
    # and its stored longest prompt; an explicit max_gen_toks or --max_tokens would override that.
    monkeypatch.setattr("marin.evaluation.evalchemy.client.is_evalchemy_benchmark", lambda name: name == "AIME24")
    config = _payload(_config(max_gen_toks=None, tasks=(EvalTaskConfig("AIME24", 0, generation=True),)))
    cmd = build_command(config, config["tasks"][0], "/opt/py", 40960)

    assert "--max_tokens" not in cmd
    assert "--gen_kwargs" not in cmd
    model_args = dict(pair.split("=", 1) for pair in cmd[cmd.index("--model_args") + 1].split(","))
    assert model_args["max_length"] == "40960"


@pytest.mark.parametrize(
    ("max_length", "expected_gen_kwargs"),
    [
        # Served context known: the native task gets the context minus the prompt reserve.
        (40960, "max_gen_toks=36864,skip_special_tokens=false"),
        # Unknown context: nothing to derive from, only the model's extra kwargs go out.
        (None, "skip_special_tokens=false"),
    ],
)
def test_unset_budget_sizes_native_generation_task_from_served_context(monkeypatch, max_length, expected_gen_kwargs):
    # lm-eval-native tasks only read max_gen_toks (default 256); --max_tokens stays off because it is
    # an Evalchemy-only knob.
    monkeypatch.setattr("marin.evaluation.evalchemy.client.is_evalchemy_benchmark", lambda name: False)
    config = _payload(_config(max_gen_toks=None, extra_gen_kwargs={"skip_special_tokens": "false"}))
    cmd = build_command(config, config["tasks"][1], "/opt/py", max_length)

    assert cmd[cmd.index("--gen_kwargs") + 1] == expected_gen_kwargs
    assert "--max_tokens" not in cmd


def test_unset_budget_loglikelihood_task_gets_no_gen_kwargs(monkeypatch):
    monkeypatch.setattr("marin.evaluation.evalchemy.client.is_evalchemy_benchmark", lambda name: False)
    config = _payload(_config(max_gen_toks=None))
    cmd = build_command(config, config["tasks"][0], "/opt/py", 40960)  # arc_easy, loglikelihood

    assert "--gen_kwargs" not in cmd
    assert "--max_tokens" not in cmd


def test_build_command_chat_route_needs_template_and_generation():
    config = _payload(_config(apply_chat_template=True))
    generative, mcq = config["tasks"][1], config["tasks"][0]

    # A generation task of a chat-template model runs through the chat API...
    cmd = build_command(config, generative, "/opt/py", None)
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    assert "--apply_chat_template" in cmd
    chat_args = dict(pair.split("=", 1) for pair in build_model_args(config, True, None).split(","))
    assert chat_args["base_url"] == "http://10.0.0.1:30000/v1/chat/completions"
    # The endpoint applies the chat template; a client-side tokenizer would reject custom tokenizer
    # code (Kimi-Linear) or metadata the client's Transformers cannot read (Gemma 4).
    assert chat_args["tokenizer_backend"] == "none"
    assert "tokenizer" not in chat_args
    completion_args = dict(pair.split("=", 1) for pair in build_model_args(config, False, None).split(","))
    assert completion_args["tokenizer_backend"] == "huggingface"
    assert completion_args["tokenizer"] == _MODEL.tokenizer
    assert completion_args["trust_remote_code"] == "True"

    # ...but a loglikelihood (MCQ) task always uses completions: chat endpoints cannot echo prompt
    # logprobs, and lm-eval rejects loglikelihood over chat completions.
    cmd = build_command(config, mcq, "/opt/py", None)
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
    cmd = build_command(config, config["tasks"][0], "/opt/py", None)

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
