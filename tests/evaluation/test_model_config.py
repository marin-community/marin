# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The model/serve config protocol: YAML round-trip, schema validation, and the flag rendering.

These are the pure pieces of the ``ModelConfig`` contract that need no cluster: draccus loads a
catalog YAML into the dataclass and rejects unknown fields at load; the typed serve knobs render to
``vllm serve`` flags with the explicit escape hatch winning; per-hardware variants overlay; the
directory scan keys by name; and the migrated registry preserves the serve options the earlier
per-launcher registry produced.
"""

import textwrap

import pytest
from marin.evaluation.model_config import (
    GenerationConfig,
    ModelConfig,
    ServeBackend,
    ServeConfig,
    load_model_config,
    resolve_serve_variant,
    scan_model_configs,
    serve_config_vllm_args,
)

_CATALOG_YAML = textwrap.dedent(
    """
    name: qwen3-32b
    location: Qwen/Qwen3-32B
    apply_chat_template: true
    serve:
      hbm_gb: 84
      tensor_parallel_size: 2
      max_model_len: 32768
      swap_space_gb: 32
      trust_remote_code: true
      tool_call_parser: hermes
      reasoning_parser: qwen3
      vllm_extra_args: ["--enable-prefix-caching"]
      variants:
        gh200:
          tensor_parallel_size: 1
          vllm_extra_args: ["--enable-prefix-caching", "--max-num-seqs", "512"]
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
    assert config.serve.swap_space_gb == 32
    # YAML lists decode to the tuple field, and nested variants to a recursive ServeConfig.
    assert config.serve.vllm_extra_args == ("--enable-prefix-caching",)
    assert config.serve.variants["gh200"].tensor_parallel_size == 1
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


def test_serve_config_vllm_args_renders_typed_knobs():
    serve = ServeConfig(
        trust_remote_code=True,
        swap_space_gb=32,
        data_parallel_size=2,
        limit_mm_per_prompt='{"image":0,"video":0}',
        reasoning_parser="qwen3",
        tool_call_parser="hermes",
        vllm_extra_args=("--enable-prefix-caching",),
    )
    args = serve_config_vllm_args(serve)

    assert "--trust-remote-code" in args
    assert args[args.index("--swap-space") + 1] == "32"
    assert args[args.index("--data-parallel-size") + 1] == "2"
    assert args[args.index("--reasoning-parser") + 1] == "qwen3"
    # A tool-call parser is inert in vLLM without auto tool choice, so both flags are emitted.
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "hermes"
    # The explicit escape-hatch flag rides through unchanged.
    assert "--enable-prefix-caching" in args


def test_serve_config_vllm_args_explicit_flag_wins_over_typed_knob():
    # An operator who hand-wrote --swap-space in the escape hatch must not get a second one from the
    # typed knob; the typed knob fills a gap, it does not duplicate.
    serve = ServeConfig(swap_space_gb=32, vllm_extra_args=("--swap-space", "8"))
    args = serve_config_vllm_args(serve)
    assert args.count("--swap-space") == 1
    assert args[args.index("--swap-space") + 1] == "8"


def test_resolve_serve_variant_overlays_only_changed_fields():
    serve = ServeConfig(
        tensor_parallel_size=2,
        swap_space_gb=32,
        variants={"gh200": ServeConfig(tensor_parallel_size=1)},
    )
    resolved = resolve_serve_variant(serve, "gh200")
    assert resolved.tensor_parallel_size == 1
    # A field the variant left at its default is untouched, so the base swap space survives.
    assert resolved.swap_space_gb == 32
    # An unmatched label is a no-op (the marin slices never match gh200).
    assert resolve_serve_variant(serve, "H100x8") is serve


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


def test_registry_migration_preserves_snowball_serve_options():
    # The migrated factory entries must produce the same serve options the earlier registry did: the
    # 256-expert MoE serves data-parallel + expert-parallel at tensor_parallel_size=1 on a pinned
    # 8xH100 node, and the thinking SFT carries the special-token gen kwargs that keep its CoT scored.
    from experiments.evaluation.models import MODELS

    snow = MODELS["snowball"]
    assert snow.serve.fixed_gpu == ("H100", 8)
    assert snow.serve.tensor_parallel_size == 1
    assert snow.serve.data_parallel_size == 8
    assert snow.serve.gpu_only is True
    assert snow.serve.target_cluster == "cw-us-east-02a"
    assert serve_config_vllm_args(snow.serve) == (
        "--data-parallel-size",
        "8",
        "--enable-expert-parallel",
        "--model-loader-extra-config",
        '{"distributed":true}',
    )

    sft = MODELS["snowball-sft"]
    assert dict(sft.generation.extra_gen_kwargs) == {"skip_special_tokens": "false", "repetition_penalty": "1.1"}

    base = MODELS["llama-3.1-8b-base"]
    assert base.revision == "d04e592"
    assert base.apply_chat_template is False


def test_model_config_defaults_are_serve_and_chat_ready():
    config = ModelConfig(name="x", location="org/x")
    assert config.apply_chat_template is True
    assert config.serve == ServeConfig()
    assert config.generation == GenerationConfig()
    assert config.serve.auto_overrides is True
