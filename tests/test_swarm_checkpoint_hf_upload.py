# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.domain_phase_mix.upload_swarm_checkpoints_to_hf import _has_required_model_files


def test_hf_upload_accepts_single_safetensors_file() -> None:
    rels = {"config.json", "model.safetensors", "tokenizer_config.json"}

    assert _has_required_model_files(rels)


def test_hf_upload_accepts_sharded_safetensors_files() -> None:
    rels = {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer_config.json",
    }

    assert _has_required_model_files(rels)


def test_hf_upload_rejects_missing_model_weights() -> None:
    rels = {"config.json", "model.safetensors.index.json", "tokenizer_config.json"}

    assert not _has_required_model_files(rels)
