"""Megatron configuration remains independent of GPU package imports."""

import sys

from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_megatron_cfg


def test_flash_attention_config_validates_without_fa2(monkeypatch):
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.flash_attn = True
    config.trainer.policy.model.path = "Qwen/Qwen3-0.6B"
    config.trainer.logger = "console"
    monkeypatch.setitem(sys.modules, "flash_attn", None)

    # A valid FA3 configuration must not require FA2's separate Python module.
    assert validate_megatron_cfg(config) is None
