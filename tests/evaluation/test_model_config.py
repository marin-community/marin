# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model YAML validation and serving-option rendering."""

import textwrap

import pytest
from draccus.utils import ParsingError
from iris.rpc import job_pb2
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


def test_load_model_config_rejects_name_with_job_path_separator(tmp_path):
    body = "name: Qwen/Qwen3-8B\nlocation: Qwen/Qwen3-8B\n"

    with pytest.raises(ParsingError):
        load_model_config(_write(tmp_path, "model.yaml", body))


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
        # The memory hint keeps the lowering offline: it skips the checkpoint measurement the same
        # way auto_overrides=False skips the config.json fetch. This test covers flag rendering.
        resource_hint=ResourceHint(gpu={"H100": 2}, memory="128g"),
        serve=ServeConfig(
            tensor_parallel_size=2,
            max_model_len=32768,
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            vllm_extra_args=("--enable-prefix-caching",),
            auto_overrides=False,
        ),
    )
    choice = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=2)
    engine_args = inference_config_for_model(
        model,
        choice,
        env_vars={},
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    ).engine.extra_args
    assert "--swap-space" not in engine_args
    assert "--trust-remote-code" not in engine_args


def _write_checkpoint(root, files: dict[str, int]):
    """Materialize a checkpoint layout as sparse files, so a 70 GiB shard costs no disk."""
    for name, size in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(size)
    return root


def _lowered_ram_bytes(model: ModelConfig, gpu_count: int) -> int:
    choice = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=gpu_count)
    resources = inference_config_for_model(
        model,
        choice,
        env_vars={},
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    ).iris.worker_resources
    assert resources.ram.endswith("g")
    return int(resources.ram.removesuffix("g")) * 1024**3


def _sized_model(location, **hint) -> ModelConfig:
    return ModelConfig(
        name="sized",
        location=str(location),
        resource_hint=ResourceHint(gpu={"H100": 2}, **hint),
        serve=ServeConfig(auto_overrides=False),
    )


def test_serve_host_memory_covers_a_checkpoint_larger_than_the_flat_default(tmp_path):
    # Regression for the H100 serve killed with SIGKILL while loading: every model got a flat 64g
    # host-memory request, so a checkpoint bigger than that was OOM-killed mid-load. The request
    # has to leave room for the checkpoint the loader pulls through host memory.
    weights = 67 * 1024**3
    checkpoint = _write_checkpoint(
        tmp_path / "qwen3.5-35b-a3b",
        {f"model-0000{shard}-of-00002.safetensors": weights // 2 for shard in (1, 2)},
    )

    assert _lowered_ram_bytes(_sized_model(checkpoint), gpu_count=4) > weights


def test_serve_host_memory_grows_with_the_rank_count(tmp_path):
    # Every rank on the host carries its own CUDA context and torch runtime, so the same checkpoint
    # spread over more devices needs more host memory than it does on one.
    checkpoint = _write_checkpoint(tmp_path / "moe", {"model.safetensors": 40 * 1024**3})
    model = _sized_model(checkpoint)

    assert _lowered_ram_bytes(model, gpu_count=8) > _lowered_ram_bytes(model, gpu_count=1)


def test_serve_host_memory_ignores_a_duplicate_subdirectory_copy(tmp_path):
    # The Llama repos ship a second copy of the weights under original/ that vLLM never reads.
    # Counting it would roughly double every request sized from such a checkpoint.
    shard = 15 * 1024**3
    plain = _write_checkpoint(tmp_path / "plain", {"model.safetensors": shard})
    duplicated = _write_checkpoint(
        tmp_path / "duplicated",
        {"model.safetensors": shard, "original/consolidated.00.safetensors": shard},
    )

    assert _lowered_ram_bytes(_sized_model(duplicated), gpu_count=1) == _lowered_ram_bytes(
        _sized_model(plain), gpu_count=1
    )


def test_unmeasurable_checkpoint_fails_the_launch_instead_of_under_sizing(tmp_path):
    # A layout with no weight file to measure would otherwise fall through to the floor and hand a
    # large model a small request -- the failure this sizing exists to prevent. Fail at lowering,
    # and say which knob resolves it.
    checkpoint = _write_checkpoint(tmp_path / "opaque", {"model.gguf": 40 * 1024**3})

    with pytest.raises(ValueError, match=r"resource_hint\.memory"):
        _lowered_ram_bytes(_sized_model(checkpoint), gpu_count=1)


def test_explicit_memory_hint_wins_without_measuring_the_checkpoint():
    # The override is the escape hatch for a model that needs more than its weights imply, and it
    # has to work for a location the launcher cannot list (an object-store export needing creds).
    model = _sized_model("s3://marin-us-east-02a/marin/exports/absent/", memory="512g")

    assert _lowered_ram_bytes(model, gpu_count=8) == 512 * 1024**3


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
