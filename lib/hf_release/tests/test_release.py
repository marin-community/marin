# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import ml_dtypes
import numpy as np
import pytest
from marin_hf_release import (
    PrecisionMode,
    ReleasePolicy,
    ReleaseRequest,
    TensorSpec,
    canonical_grug_config,
    grug_tensor_specs,
    persist_request,
    resolve_metadata,
    validate_release_metadata,
    validate_release_weights,
    write_completion_manifest,
)
from marin_hf_release.request import MANIFEST_NAME
from safetensors.numpy import save_file


def _config():
    return {
        "model_type": "grug_moe",
        "vocab_size": 4,
        "hidden_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "max_position_embeddings": 16,
        "rope_theta": 10000.0,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "shared_expert_intermediate_size": 2,
        "sliding_window": 8,
        "grugmoe_attention_mode": "production",
        "grugmoe_artifact_schema_version": 1,
        "qk_mult": 1.0,
    }


def _request(mode: PrecisionMode):
    policy = ReleasePolicy(mode)
    specs = grug_tensor_specs(
        _config(),
        {
            "model.embed_tokens.weight": TensorSpec((4, 2), "BF16"),
            "lm_head.weight": TensorSpec((4, 2), "BF16"),
            "model.layers.0.mlp.router.bias": TensorSpec((2,), "F32"),
        },
    )
    tokenizer, generation = resolve_metadata(
        {"eos_token_ids": [1, 2]},
        {
            "vocab_size": 4,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "chat_template": "{{ bos_token }}",
            "special_token_ids": {"bos_token": 0, "eos_token": 1},
        },
        {},
        require_chat_template=True,
    )
    return ReleaseRequest(
        policy=policy,
        schema_id="grug-moe-v1",
        tensors=specs,
        config={**_config(), "dtype": policy.config_dtype(specs)},
        generation=generation,
        tokenizer=tokenizer,
        source_checkpoint="gs://checkpoints/step-12",
        source_step=12,
        checkpoint_identity="sha256:checkpoint",
        source_model="org/model",
        source_model_commit="a" * 40,
        exporter_revision="b" * 40,
    )


def _metadata(root, request):
    (root / "config.json").write_text(json.dumps(request.config))
    (root / "generation_config.json").write_text(json.dumps(request.generation))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": request.tokenizer.chat_template,
                "bos_token": "0",
                "eos_token": "1",
            }
        )
    )
    (root / "chat_template.jinja").write_text(request.tokenizer.chat_template)
    (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {str(i): i for i in range(4)}}}))


@pytest.mark.parametrize("mode", [PrecisionMode.BF16, PrecisionMode.FP32])
def test_grug_release_validates_tensor_precision_and_metadata(tmp_path, mode):
    request = _request(mode)
    persist_request(tmp_path, request)
    dtype = ml_dtypes.bfloat16 if mode is PrecisionMode.BF16 else np.float32
    tensors = {
        "model.embed_tokens.weight": np.ones((4, 2), dtype=dtype),
        "lm_head.weight": np.ones((4, 2), dtype=dtype),
        "model.layers.0.mlp.router.bias": np.asarray([0.25, -0.5], dtype=np.float32),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    _metadata(tmp_path, request)

    hashes = validate_release_weights(tmp_path, request.tensors, request.policy)
    hashes.update(validate_release_metadata(tmp_path, request))
    write_completion_manifest(tmp_path, request, hashes)

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["request_sha256"] == request.digest()
    assert manifest["files"]["model.safetensors"] == hashes["model.safetensors"]
    assert (
        request.policy.target_dtype("model.layers.0.mlp.router.bias", request.tensors["model.layers.0.mlp.router.bias"])
        == "F32"
    )


def test_release_retry_rejects_changed_request_and_bad_index(tmp_path):
    request = _request(PrecisionMode.FP32)
    persist_request(tmp_path, request)
    save_file(
        {
            "model.embed_tokens.weight": np.ones((4, 2), dtype=np.float32),
            "lm_head.weight": np.ones((4, 2), dtype=np.float32),
            "model.layers.0.mlp.router.bias": np.asarray([0.25, -0.5], dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {name: "model.safetensors" for name in request.tensors},
            }
        )
    )
    with pytest.raises(ValueError, match="total_size"):
        validate_release_weights(tmp_path, request.tensors, request.policy)
    (tmp_path / "model.safetensors.index.json").unlink()
    _metadata(tmp_path, request)
    persist_request(tmp_path, request)
    with pytest.raises(FileExistsError, match="request changed"):
        persist_request(tmp_path, replace(request, source_step=13))
    hashes = validate_release_weights(tmp_path, request.tensors, request.policy)
    hashes.update(validate_release_metadata(tmp_path, request))
    write_completion_manifest(tmp_path, request, hashes)
    assert (tmp_path / MANIFEST_NAME).is_file()


def test_grug_alias_conflict_and_missing_stop_metadata_are_rejected():
    with pytest.raises(ValueError, match="conflicting Grug aliases"):
        canonical_grug_config({**_config(), "num_local_experts": 3})
    with pytest.raises(ValueError, match="eos_token_ids"):
        resolve_metadata(
            {}, {"vocab_size": 4, "eos_token_id": 1, "chat_template": "template"}, {}, require_chat_template=True
        )


def test_release_rejects_mismatched_tokenizer_special_id(tmp_path):
    request = _request(PrecisionMode.FP32)
    _metadata(tmp_path, request)
    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    tokenizer_config["eos_token"] = "2"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    with pytest.raises(ValueError, match="eos_token differs"):
        validate_release_metadata(tmp_path, request)
