# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from experiments.grug.moe import real_checkpoint_vllm_smoke as smoke
from experiments.grug.moe.real_checkpoint_vllm_smoke import (
    SmokeConfig,
    join_path,
    legacy_split_expert_inference_state_dict,
    real_checkpoint_model_config,
    require_local_or_europe_west4,
    stage_artifact_for_vllm,
    validate_locality,
)


def test_require_local_or_europe_west4_accepts_local_and_europe_west4_paths():
    require_local_or_europe_west4("local", "/tmp/grug-artifact")
    require_local_or_europe_west4("relative", "local-output")
    require_local_or_europe_west4("gcs", "gs://marin-eu-west4/grug/checkpoint")


@pytest.mark.parametrize(
    "path",
    [
        "gs://marin-us-central2/grug/checkpoint",
        "gs://some-other-bucket/grug/checkpoint",
        "mirror://tokenizers/meta-llama/Meta-Llama-3.1-8B",
        "hf://meta-llama/Meta-Llama-3.1-8B",
    ],
)
def test_require_local_or_europe_west4_rejects_nonlocality_paths(path: str):
    with pytest.raises(ValueError):
        require_local_or_europe_west4("asset", path)


def test_validate_locality_includes_artifact_as_vllm_model_path():
    config = SmokeConfig(
        phase="all",
        checkpoint_path="gs://marin-eu-west4/grug/checkpoint",
        tokenizer_path="gs://marin-eu-west4/tokenizers/llama",
        output_dir="gs://marin-eu-west4/tmp/smoke",
        cache_dir="gs://marin-eu-west4/compilation-cache/smoke",
        prompt="prompt",
        expected_output="42",
        calibrate_output=False,
        overwrite=False,
        max_shard_size=1024,
        max_model_len=128,
        max_tokens=1,
        server_port=8000,
        server_timeout_seconds=1,
        vllm_dtype="bfloat16",
    )

    paths = validate_locality(config)

    assert paths["artifact_dir"] == "gs://marin-eu-west4/tmp/smoke/artifact"
    assert paths["vllm_model_path"] == paths["artifact_dir"]


def test_join_path_preserves_gcs_scheme():
    assert join_path("gs://marin-eu-west4/tmp/smoke", "artifact", "config.json") == (
        "gs://marin-eu-west4/tmp/smoke/artifact/config.json"
    )


def test_stage_artifact_for_vllm_passes_local_artifact_through(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    vllm_model_path, staging = stage_artifact_for_vllm(str(artifact_dir))

    assert vllm_model_path == str(artifact_dir)
    assert staging == {
        "staged": False,
        "source_artifact_dir": str(artifact_dir),
        "vllm_model_path": str(artifact_dir),
        "copied_files": None,
    }


def test_serve_artifact_calibration_records_observed_output_without_default_assertion(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text("{}\n")

    class FakeResponse:
        ok = True
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"text": " observed"}], "usage": {"completion_tokens": 1}}

    class FakeVllmEnvironment:
        def __init__(self, **kwargs):
            del kwargs
            self.model_id = "fake-model"
            self.server_url = "http://127.0.0.1:8000/v1"
            self.vllm_server = SimpleNamespace(log_dir="/tmp/fake-vllm-logs")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def logs_tail(self, *, max_lines: int):
            del max_lines
            return "fake logs"

    monkeypatch.setattr(smoke.requests, "post", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(smoke, "time", SimpleNamespace(time=lambda: 0.0))
    monkeypatch.setattr("marin.inference.vllm_server.VllmEnvironment", FakeVllmEnvironment)
    config = SmokeConfig(
        phase="serve",
        checkpoint_path="gs://marin-eu-west4/grug/checkpoint",
        tokenizer_path="gs://marin-eu-west4/tokenizers/llama",
        output_dir=str(tmp_path),
        cache_dir="gs://marin-eu-west4/compilation-cache/smoke",
        prompt="prompt",
        expected_output=" default",
        calibrate_output=True,
        overwrite=False,
        max_shard_size=1024,
        max_model_len=128,
        max_tokens=1,
        server_port=8000,
        server_timeout_seconds=1,
        vllm_dtype="bfloat16",
    )

    result = smoke.serve_artifact(config)

    assert result["completion"] == " observed"
    assert result["expected_output"] is None
    assert result["configured_expected_output"] == " default"
    assert result["calibrated_expected_output"] == " observed"
    assert result["calibration_mode"] is True
    assert result["passed"] is True


def test_real_checkpoint_model_config_matches_may_compute_opt_checkpoint():
    cfg = real_checkpoint_model_config()

    assert cfg.hidden_dim == 512
    assert cfg.intermediate_dim == 256
    assert cfg.shared_expert_intermediate_dim == 512
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_token == 4
    assert cfg.num_layers == 6
    assert cfg.num_heads == 4
    assert cfg.num_kv_heads == 1
    assert cfg.max_seq_len == 4096
    assert cfg.sliding_window == 2048
    assert cfg.router_z_loss_coef == 0.0


def test_runtime_snapshot_uses_marin_git_sha_env(monkeypatch):
    monkeypatch.setenv("MARIN_GIT_SHA", "abc123")

    snapshot = smoke.runtime_snapshot(include_jax_devices=False)

    assert snapshot["marin_sha"] == "abc123"


def test_legacy_split_expert_state_dict_exports_canonical_expert_tensors():
    cfg = real_checkpoint_model_config()
    cfg = cfg.__class__(
        vocab_size=3,
        hidden_dim=2,
        intermediate_dim=1,
        shared_expert_intermediate_dim=0,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=4,
        sliding_window=2,
    )
    gate = jnp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]])
    up = gate + 10.0
    down = jnp.asarray([[[21.0, 22.0]], [[23.0, 24.0]]])
    model = SimpleNamespace(
        token_embed=jnp.ones((3, 2)),
        embed_norm=SimpleNamespace(weight=jnp.ones((2,))),
        embed_gated_norm=SimpleNamespace(w_down=jnp.ones((2, 1)), w_up=jnp.ones((1, 2))),
        final_norm=SimpleNamespace(weight=jnp.ones((2,))),
        final_gated_norm=SimpleNamespace(w_down=jnp.ones((2, 1)), w_up=jnp.ones((1, 2))),
        output_proj=jnp.ones((2, 3)),
        blocks=(
            SimpleNamespace(
                rms_attn=SimpleNamespace(weight=jnp.ones((2,))),
                attn_gated_norm=SimpleNamespace(w_down=jnp.ones((2, 1)), w_up=jnp.ones((1, 2))),
                attn=SimpleNamespace(
                    w_q=jnp.ones((2, 2)),
                    w_k=jnp.ones((2, 2)),
                    w_v=jnp.ones((2, 2)),
                    w_o=jnp.ones((2, 2)),
                    attn_gate=jnp.ones((2, 1)),
                ),
                rms_mlp=SimpleNamespace(weight=jnp.ones((2,))),
                mlp_gated_norm=SimpleNamespace(w_down=jnp.ones((2, 1)), w_up=jnp.ones((1, 2))),
                mlp=SimpleNamespace(
                    router=jnp.ones((2, 2)),
                    router_bias=jnp.ones((2,)),
                    w_gate=gate,
                    w_up=up,
                    w_down=down,
                ),
                shared=None,
            ),
        ),
    )

    state_dict = legacy_split_expert_inference_state_dict(model, cfg)

    assert set(state_dict) == {
        "model.embed_tokens.weight",
        "model.embed_norm.weight",
        "model.embed_gated_norm.down_proj.weight",
        "model.embed_gated_norm.up_proj.weight",
        "model.norm.weight",
        "model.final_gated_norm.down_proj.weight",
        "model.final_gated_norm.up_proj.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.attn_gated_norm.down_proj.weight",
        "model.layers.0.attn_gated_norm.up_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.attn_gate.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp_gated_norm.down_proj.weight",
        "model.layers.0.mlp_gated_norm.up_proj.weight",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
    }
    assert state_dict["model.layers.0.mlp.experts.gate_proj.weight"].tolist() == jnp.swapaxes(gate, -1, -2).tolist()
    assert state_dict["model.layers.0.mlp.experts.up_proj.weight"].tolist() == jnp.swapaxes(up, -1, -2).tolist()
    assert state_dict["model.layers.0.mlp.experts.down_proj.weight"].tolist() == jnp.swapaxes(down, -1, -2).tolist()
