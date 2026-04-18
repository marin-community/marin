# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import hashlib
import json
import math
import os
import sys

import pytest

EXPECTED_SUBSET_SIZES = {
    "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": 226313,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": 44800,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": 89300,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": 5690,
}


def test_qwen3_32b_tokenizer_vocab_size_is_cached():
    data_configs = importlib.import_module("marin.processing.tokenize.data_configs")

    assert data_configs._KNOWN_VOCAB_SIZES["Qwen/Qwen3-32B"] == 151936


def test_qwen3_32b_hf_matches_official_model_shape():
    qwen3 = importlib.import_module("experiments.qwen3")

    cfg = qwen3.qwen3_32b_hf

    assert qwen3.qwen3_32b_tokenizer == "Qwen/Qwen3-32B"
    assert cfg.reference_checkpoint == "Qwen/Qwen3-32B"
    assert cfg.max_seq_len == 40960
    assert cfg.hidden_dim == 5120
    assert cfg.intermediate_dim == 25600
    assert cfg.num_heads == 64
    assert cfg.num_kv_heads == 8
    assert cfg.num_layers == 64
    assert cfg.head_dim == 128
    assert cfg.tie_word_embeddings is False
    assert cfg.layer_norm_epsilon == 1e-6
    assert cfg.initializer_range == 0.02


def test_exp4307_matches_full_corpus_recipe():
    module = importlib.import_module("experiments.exp4307_sft_nemotron_terminal_corpus_qwen3_32b")

    assert module.SUBSET_SIZES == EXPECTED_SUBSET_SIZES
    assert module.DATASETS == {name: name for name in EXPECTED_SUBSET_SIZES}
    assert module.EFFECTIVE_EXAMPLES == sum(EXPECTED_SUBSET_SIZES.values())
    assert module.TARGET_EPOCHS == 2
    assert module.TRAIN_BATCH_SIZE == 128
    assert module.NUM_TRAIN_STEPS == math.ceil(2 * module.EFFECTIVE_EXAMPLES / module.TRAIN_BATCH_SIZE)
    assert module.RESOURCES.device.variant == "v5p-256"
    assert module.sft_config.initialize_from_hf == "Qwen/Qwen3-32B"
    assert module.sft_config.max_seq_len == 32768
    assert module.qwen3_32b_32k.max_seq_len == 32768


def test_exp4760_matches_15pct_marin32b_recipe():
    module = importlib.import_module("experiments.exp4760_sft_marin_32b_base_terminal_corpus")

    expected_effective_examples = sum(EXPECTED_SUBSET_SIZES.values()) * 0.15

    assert module.SUBSET_SIZES == EXPECTED_SUBSET_SIZES
    assert module.DATASETS == {name: name for name in EXPECTED_SUBSET_SIZES}
    assert module.TRAIN_FRACTION == pytest.approx(0.15)
    assert module.EFFECTIVE_EXAMPLES_FLOAT == pytest.approx(expected_effective_examples)
    assert module.TARGET_EPOCHS == 2
    assert module.TRAIN_BATCH_SIZE == 128
    assert module.NUM_TRAIN_STEPS == math.ceil(2 * expected_effective_examples / module.TRAIN_BATCH_SIZE)
    assert module.RESOURCES.device.variant == "v4-512"
    assert module.sft_config.initialize_from_hf == "marin-community/marin-32b-base"
    assert module.sft_config.tokenizer == "marin-community/marin-tokenizer"
    assert module.sft_config.tensor_parallel_size == 2
    assert module.sft_config.max_seq_len == 32768
    assert module.marin_32b_32k.max_seq_len == 32768


def test_exp4307_released_eval_targets_tb2_first():
    module = importlib.import_module("experiments.exp4307_eval_released_nemotron_terminal_32b_tb2")

    assert module.MODEL_NAME == "nemotron-terminal-32b"
    assert module.RELEASED_MODEL_NAME == "nvidia/Nemotron-Terminal-32B"
    assert module.RELEASED_MODEL_PATH == "nvidia/Nemotron-Terminal-32B"
    assert module.DATASET == "terminal-bench"
    assert module.VERSION == "2.0"
    assert module.ENV_TYPE == "daytona"
    assert module.DEFAULT_VLLM_MODE == "native"
    assert module.VLLM_MODE == "native"
    assert module.HARBOR_AGENT == "terminus-2"
    assert module.RUN_VARIANT is None
    assert module.HARBOR_N_CONCURRENT == 8
    assert module.HARBOR_MAX_INSTANCES is None
    assert module.HARBOR_TASK_NAMES is None
    assert module.RESOURCES.device.variant == "v5p-8"
    assert "us-east5" in module.RESOURCES.regions
    assert module.BASE_OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2"
    assert module.RUN_OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/full"
    assert module.TASK_SHARD_HASH is None
    assert module.OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/full"
    assert module.HARBOR_AGENT_KWARGS["temperature"] == 0.6
    assert module.HARBOR_AGENT_KWARGS["top_p"] == 0.95
    assert module.HARBOR_AGENT_KWARGS["top_k"] == 20
    assert module.HARBOR_AGENT_KWARGS["model_info"]["max_input_tokens"] == 40960
    assert module.VLLM_ENGINE_KWARGS == {
        "max_model_len": 40960,
        "max_num_seqs": 8,
        "tensor_parallel_size": 4,
    }


def test_exp4307_released_eval_defaults_to_native_vllm(monkeypatch):
    module_name = "experiments.exp4307_eval_released_nemotron_terminal_32b_tb2"

    monkeypatch.delenv("MARIN_VLLM_MODE", raising=False)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)

    assert module.VLLM_MODE == "native"
    assert os.environ["MARIN_VLLM_MODE"] == "native"


def test_exp4307_smoke_eval_uses_distinct_output_dir(monkeypatch):
    module_name = "experiments.exp4307_eval_released_nemotron_terminal_32b_tb2"

    monkeypatch.setenv("HARBOR_MAX_INSTANCES", "1")
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)

    assert module.HARBOR_MAX_INSTANCES == 1
    assert module.RUN_OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/smoke-1"
    assert module.OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/smoke-1"


def test_exp4307_sharded_eval_uses_distinct_output_dir(monkeypatch):
    module_name = "experiments.exp4307_eval_released_nemotron_terminal_32b_tb2"

    task_names = ["build-pov-ray", "sam-cell-seg"]
    task_names_json = json.dumps(task_names)
    expected_hash = hashlib.sha256(
        json.dumps(task_names, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]

    monkeypatch.setenv("HARBOR_TASK_NAMES_JSON", task_names_json)
    monkeypatch.delenv("HARBOR_MAX_INSTANCES", raising=False)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)

    assert module.HARBOR_TASK_NAMES == task_names
    assert module.RUN_OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/full"
    assert module.TASK_SHARD_HASH == expected_hash
    assert (
        module.OUTPUT_DIR
        == f"evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/full/shard_{expected_hash}"
    )


def test_exp4307_variant_uses_distinct_output_dir(monkeypatch):
    module_name = "experiments.exp4307_eval_released_nemotron_terminal_32b_tb2"

    monkeypatch.setenv("RUN_VARIANT", "latency-v1")
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)

    assert module.RUN_VARIANT == "latency-v1"
    assert module.BASE_OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/latency-v1"
    assert module.RUN_OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/latency-v1/full"
    assert module.OUTPUT_DIR == "evaluation/harbor/terminal-bench/nemotron-terminal-32b/terminus-2/latency-v1/full"


def test_vllm_binary_resolution_prefers_active_python_bin(tmp_path):
    vllm_server = importlib.import_module("marin.inference.vllm_server")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vllm_path = bin_dir / "vllm"
    vllm_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    vllm_path.chmod(0o755)

    resolved = vllm_server._resolve_vllm_binary(path_env="", python_executable=str(bin_dir / "python"))

    assert resolved == str(vllm_path)


def test_vllm_cli_command_falls_back_to_python_module_when_binary_missing(tmp_path):
    vllm_server = importlib.import_module("marin.inference.vllm_server")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_path = bin_dir / "python"
    python_path.write_text("", encoding="utf-8")

    resolved = vllm_server._resolve_vllm_cli_command(path_env="", python_executable=str(python_path))

    assert resolved == [str(python_path), "-m", "vllm.entrypoints.cli.main"]


def test_evaluate_harbor_local_model_uses_direct_iris_execution():
    evals = importlib.import_module("experiments.evals.evals")

    resources = evals.ResourceConfig.with_tpu("v5p-8")
    step = evals.evaluate_harbor(
        model_name="nemotron-terminal-32b",
        model_path="nvidia/Nemotron-Terminal-32B",
        dataset="terminal-bench",
        version="2.0",
        resource_config=resources,
        env="daytona",
        agent="terminus-2",
        task_names=["build-pov-ray", "sam-cell-seg"],
        engine_kwargs={"tensor_parallel_size": 4, "max_num_seqs": 8},
    )

    assert step.fn.resources == resources
    assert step.fn.pip_dependency_groups == ["harbor", "vllm", "tpu"]
    assert step.config.launch_with_ray is False
    assert step.config.resource_config == resources
    assert step.config.engine_kwargs["harbor_config"]["env"] == "daytona"
    assert step.config.engine_kwargs["harbor_config"]["task_names"] == ["build-pov-ray", "sam-cell-seg"]
    assert step.config.engine_kwargs["tensor_parallel_size"] == 4
    assert step.config.engine_kwargs["max_num_seqs"] == 8


def test_vllm_engine_kwargs_to_cli_args_supports_parallelism_flags():
    vllm_server = importlib.import_module("marin.inference.vllm_server")

    args = vllm_server._engine_kwargs_to_cli_args(
        {
            "load_format": "runai_streamer",
            "max_model_len": 40960,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "tensor_parallel_size": 4,
        }
    )

    assert args == [
        "--load-format",
        "runai_streamer",
        "--max-model-len",
        "40960",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "8",
        "--tensor-parallel-size",
        "4",
    ]


def test_evaluate_harbor_api_model_stays_cpu_only():
    evals = importlib.import_module("experiments.evals.evals")

    step = evals.evaluate_harbor(
        model_name="claude-opus-4",
        model_path=None,
        dataset="aime",
        version="1.0",
        env="daytona",
        agent="claude-code",
    )

    assert step.fn.resources == evals.ResourceConfig.with_cpu()
    assert step.fn.pip_dependency_groups == ["harbor"]
    assert step.config.launch_with_ray is False
    assert step.config.resource_config is None


def test_evaluate_harbor_local_model_requires_iris_resources():
    evals = importlib.import_module("experiments.evals.evals")

    with pytest.raises(ValueError, match="resource_config must be provided"):
        evals.evaluate_harbor(
            model_name="nemotron-terminal-32b",
            model_path="nvidia/Nemotron-Terminal-32B",
            dataset="terminal-bench",
            version="2.0",
        )


def test_evaluate_harbor_forwards_daytona_and_vllm_env(monkeypatch):
    evals = importlib.import_module("experiments.evals.evals")

    monkeypatch.setenv("DAYTONA_API_KEY", "daytona-test-token")
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")
    monkeypatch.setenv("MARIN_VLLM_MODE", "native")

    step = evals.evaluate_harbor(
        model_name="nemotron-terminal-32b",
        model_path="nvidia/Nemotron-Terminal-32B",
        dataset="terminal-bench",
        version="2.0",
        resource_config=evals.ResourceConfig.with_tpu("v5p-8"),
        env="daytona",
        agent="terminus-2",
    )

    assert step.fn.env_vars["DAYTONA_API_KEY"] == "daytona-test-token"
    assert step.fn.env_vars["MARIN_PREFIX"] == "gs://marin-us-east5"
    assert step.fn.env_vars["MARIN_VLLM_MODE"] == "native"
