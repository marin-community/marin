# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


@pytest.mark.manual
def test_cpu_preflight_tokenizer_loads_without_staged_model(tmp_path):
    # Run this optional-dependency regression in the frozen MarinSkyRL CPU environment.
    from omegaconf import OmegaConf  # noqa: PLC0415 - optional frozen runtime dependency

    from experiments.post_training.snowball_inventory_evaluate_runtime import preflight_tokenizer  # noqa: PLC0415

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "[EOS]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    source = tmp_path / "pinned-tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(source)
    staged = tmp_path / "not-yet-staged-model"
    config = OmegaConf.create(
        {
            "inputs": {"model": {"tokenizer_uri": str(source), "tokenizer_revision": "a" * 40}},
            "skyrl": {
                "trainer": {
                    "disable_fast_tokenizer": False,
                    "policy": {"model": {"tokenizer_path": str(staged), "tokenizer_revision": "a" * 40}},
                }
            },
        }
    )
    original = OmegaConf.to_container(config, resolve=True)

    tokenizer = preflight_tokenizer(config)

    assert tokenizer.encode("hello world", add_special_tokens=False) == [1, 2]
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id == 3
    assert not staged.exists()
    assert OmegaConf.to_container(config, resolve=True) == original
