# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from levanter.compat import hf_checkpoints
from levanter.models.qwen import Qwen3Config
from marin.evaluation.olmo_base_eval.run import _model_config_from_checkpoint
from transformers import Qwen3Config as HfQwen3Config


def test_model_config_from_checkpoint_does_not_load_reference_tokenizer(tmp_path, monkeypatch):
    hf_config = HfQwen3Config(
        hidden_size=896,
        intermediate_size=3584,
        num_hidden_layers=10,
        num_attention_heads=7,
        num_key_value_heads=7,
        max_position_embeddings=4096,
        head_dim=128,
        attention_bias=False,
    )
    (tmp_path / "config.json").write_text(json.dumps(hf_config.to_dict()))

    def reject_tokenizer_load(*args, **kwargs):
        raise AssertionError("model-config probing must not load a tokenizer")

    monkeypatch.setattr(hf_checkpoints, "load_tokenizer", reject_tokenizer_load)
    model_config = _model_config_from_checkpoint(str(tmp_path))

    assert isinstance(model_config, Qwen3Config)
    assert model_config.hidden_dim == 896
    assert model_config.intermediate_dim == 3584
    assert model_config.num_layers == 10
