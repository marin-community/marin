# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
import yaml
from click.testing import CliRunner

from experiments.post_training.mochi_three_rack import main


@pytest.mark.parametrize("resume_step", [None, 5])
@pytest.mark.parametrize("response_tokens", [None, 1024])
def test_three_rack_cli_renders_required_placement_and_continuations(
    tmp_path, monkeypatch, resume_step, response_tokens
):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-08a")
    (tmp_path / "train.parquet").write_bytes(b"pinned mixed tasks")
    digest = hashlib.sha256((tmp_path / "train.parquet").read_bytes()).hexdigest()
    output = tmp_path / "launch.yaml"
    arguments = f"""--version 2026.11.15.1 --input-version 2026.11.15.0
        --model-uri "{tmp_path / 'model'}" --tokenizer-revision pinned
        --data-root "{tmp_path}" --data-sha256 {digest} --updates 2 --config-output "{output}"
    """
    if resume_step is not None:
        arguments += f' --resume-uri "{tmp_path / f"global_step_{resume_step}"}"'
    if response_tokens is not None:
        arguments += f" --max-new-tokens {response_tokens}"
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 0, result.output
    launch = yaml.safe_load(output.read_text())
    trainer = launch["skyrl"]["trainer"]
    assert launch["iris"]["role_topology"] == "nvlink.domain"
    assert launch["iris"]["allocation"]["num_nodes"] * launch["iris"]["allocation"]["gpus_per_node"] == 192
    assert trainer["resume_mode"] == "latest" and trainer["max_steps"] == (resume_step or 0) + 2
    assert launch["skyrl"]["context_budget"] == {
        "request_window_tokens": 262144,
        "max_new_tokens_per_turn": response_tokens or 8192,
        "max_turns": 4,
    }
    policy = trainer["policy"]
    assert policy["nccl_buffer_size_bytes"] == 262144
    memory = policy["megatron_config"]["transformer_config_kwargs"]
    assert memory["bias_activation_fusion"] and memory["grug_checkpoint_cpu_offload"]
    assert launch["skyrl"]["generator"]["engine_init_kwargs"]["compilation_config"]["cudagraph_mode"] == "NONE"
    if resume_step is not None:
        assert launch["inputs"]["starting_checkpoint"]["uri"] == str(tmp_path / f"global_step_{resume_step}")
    else:
        assert "starting_checkpoint" not in launch["inputs"]
