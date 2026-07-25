# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model YAML validation and serving-option rendering."""

import textwrap

import pytest
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import (
    ModelConfig,
    ResourceHint,
    ServeBackend,
    ServeConfig,
    load_model_config,
    scan_model_configs,
    serve_config_vllm_args,
)
from marin.evaluation.serving_config import inference_config_for_model

from experiments.evaluation.fleet import MARIN_EVAL_HARDWARE
from experiments.evaluation.models import models

_CATALOG_YAML = textwrap.dedent(
    """
    name: qwen3-32b
    location: Qwen/Qwen3-32B
    apply_chat_template: true
    resource_hint:
      gpu:
        H100: 2
    serve:
      tensor_parallel_size: 2
      max_model_len: 32768
      tool_call_parser: hermes
      reasoning_parser: qwen3
      vllm_extra_args: ["--enable-prefix-caching"]
    generation:
      extra_gen_kwargs:
        skip_special_tokens: "false"
    agent:
      agent_kwargs:
        extra_body: '{"chat_template_kwargs":{"enable_thinking":true}}'
    """
)


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_load_model_config_round_trips_the_catalog_shape(tmp_path):
    config = load_model_config(_write(tmp_path, "Qwen/Qwen3-32B.yaml", _CATALOG_YAML))

    assert config.name == "qwen3-32b"
    assert config.serve.backend is ServeBackend.VLLM
    assert config.serve.tensor_parallel_size == 2
    assert config.resource_hint.gpu == {"H100": 2}
    assert config.serve.vllm_extra_args == ("--enable-prefix-caching",)
    assert dict(config.generation.extra_gen_kwargs) == {"skip_special_tokens": "false"}
    assert "enable_thinking" in config.agent.agent_kwargs["extra_body"]


def test_load_rejects_unknown_field_at_load_time(tmp_path):
    # A typo in a serve field name is the exact class of bug the schema exists to catch: it must fail
    # at load, not surface as a dropped serve flag hours into a run. draccus wraps the field-level
    # decoding error, so the offending name rides on the exception chain rather than the top message.
    bad = "name: x\nlocation: org/x\nserve:\n  swp_space: 8\n"
    with pytest.raises(Exception) as excinfo:
        load_model_config(_write(tmp_path, "bad.yaml", bad))
    chain = " ".join(str(exc) for exc in (excinfo.value, excinfo.value.__cause__, excinfo.value.__context__) if exc)
    assert "swp_space" in chain


@pytest.mark.parametrize(
    ("field", "value"),
    [("swap_space_gb", "32"), ("trust_remote_code", "true")],
)
def test_load_rejects_removed_serve_knob(tmp_path, field, value):
    # swap_space_gb and trust_remote_code were removed from the schema: the CUDA vLLM fork rejects
    # --swap-space, and the native server already forces --trust-remote-code on for every model, so a
    # per-model knob only duplicated the flag. A catalog file that still carries either must fail at
    # load rather than lower into a flag the server rejects or warns on.
    body = f"name: x\nlocation: org/x\nserve:\n  {field}: {value}\n"
    with pytest.raises(Exception) as excinfo:
        load_model_config(_write(tmp_path, "removed.yaml", body))
    chain = " ".join(str(exc) for exc in (excinfo.value, excinfo.value.__cause__, excinfo.value.__context__) if exc)
    assert field in chain


def test_serve_config_vllm_args_renders_typed_knobs():
    serve = ServeConfig(
        data_parallel_size=2,
        limit_mm_per_prompt='{"image":0,"video":0}',
        reasoning_parser="qwen3",
        tool_call_parser="hermes",
        vllm_extra_args=("--enable-prefix-caching",),
    )
    args = serve_config_vllm_args(serve)

    assert args[args.index("--data-parallel-size") + 1] == "2"
    assert args[args.index("--limit-mm-per-prompt") + 1] == '{"image":0,"video":0}'
    assert args[args.index("--reasoning-parser") + 1] == "qwen3"
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "hermes"
    assert "--enable-prefix-caching" in args
    # The removed knobs are never reintroduced by rendering.
    assert "--swap-space" not in args
    assert "--trust-remote-code" not in args


def test_serve_config_vllm_args_explicit_flag_wins_over_typed_knob():
    # An operator who hand-wrote --data-parallel-size in the escape hatch must not get a second one
    # from the typed knob; the typed knob fills a gap, it does not duplicate.
    serve = ServeConfig(data_parallel_size=2, vllm_extra_args=("--data-parallel-size", "4"))
    args = serve_config_vllm_args(serve)
    assert args.count("--data-parallel-size") == 1
    assert args[args.index("--data-parallel-size") + 1] == "4"


def test_gpu_lowering_emits_no_swap_space_or_trust_remote_code():
    # Regression for the Qwen3-32B catalog failure: the lowering path rendered --swap-space (which the
    # CUDA vLLM fork rejects) and a second --trust-remote-code (the native server already passes one).
    # A catalog-shaped GPU model must lower to engine args carrying neither.
    model = ModelConfig(
        name="qwen3-32b",
        location="Qwen/Qwen3-32B",
        resource_hint=ResourceHint(gpu={"H100": 2}),
        serve=ServeConfig(
            tensor_parallel_size=2,
            max_model_len=32768,
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            vllm_extra_args=("--enable-prefix-caching",),
            auto_overrides=False,  # skip the config.json fetch; this test covers flag rendering
        ),
    )
    choice = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=2)
    engine_args = inference_config_for_model(model, choice, env_vars={}).engine.extra_args
    assert "--swap-space" not in engine_args
    assert "--trust-remote-code" not in engine_args


def test_scan_model_configs_keys_by_name_and_skips_underscored(tmp_path):
    _write(tmp_path, "Qwen/Qwen3-8B.yaml", "name: qwen3-8b\nlocation: Qwen/Qwen3-8B\n")
    _write(tmp_path, "org/other.yaml", "name: other\nlocation: org/other\n")
    _write(tmp_path, "_patterns.yaml", "name: ignored\nlocation: x\n")
    configs = scan_model_configs(tmp_path)
    assert set(configs) == {"qwen3-8b", "other"}


def test_scan_model_configs_rejects_duplicate_names(tmp_path):
    _write(tmp_path, "a/dup.yaml", "name: dup\nlocation: a/x\n")
    _write(tmp_path, "b/dup.yaml", "name: dup\nlocation: b/x\n")
    with pytest.raises(ValueError, match="duplicate model name"):
        scan_model_configs(tmp_path)


def test_every_registered_model_resolves_on_the_marin_fleet():
    for name, config in models().items():
        platform = Platform.GPU if config.resource_hint.gpu else Platform.TPU
        choice = MARIN_EVAL_HARDWARE.select(config, platform, override=None)
        if config.resource_hint.gpu:
            gpu_type = choice.gpu_type
            assert gpu_type is not None and gpu_type in config.resource_hint.gpu, name
            assert choice.gpu_count == config.resource_hint.gpu[gpu_type], name
        else:
            assert choice.platform is Platform.TPU, name


def test_model_registry_callers_cannot_mutate_the_cached_catalog():
    registry = models()
    removed_name = next(iter(registry))
    del registry[removed_name]

    assert removed_name in models()


def test_gpu_required_model_rejects_tpu_override():
    model = ModelConfig(name="gpu-model", location="org/model", resource_hint=ResourceHint(gpu={"H100": 2}))

    with pytest.raises(ValueError, match="requires GPU"):
        MARIN_EVAL_HARDWARE.select(model, Platform.GPU, override="v6e-4")


def test_gpu_required_model_rejects_types_absent_from_fleet():
    model = ModelConfig(name="gpu-model", location="org/model", resource_hint=ResourceHint(gpu={"A100": 2}))

    with pytest.raises(ValueError, match="absent from this fleet"):
        MARIN_EVAL_HARDWARE.select(model, Platform.GPU, override=None)


def test_explicit_gpu_override_respects_fleet_limits():
    model = ModelConfig(name="portable", location="org/model", resource_hint=ResourceHint(hbm_gb=20))

    selected = MARIN_EVAL_HARDWARE.select(model, Platform.TPU, override="H100x4")
    assert selected.label == "H100x4"
    assert selected.target_cluster == "cw-us-east-02a"

    with pytest.raises(ValueError, match="positive power of two"):
        MARIN_EVAL_HARDWARE.select(model, Platform.GPU, override="H100x3")
