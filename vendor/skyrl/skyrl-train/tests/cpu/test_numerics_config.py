import pytest

from skyrl_train.config.numerics import validate_numerics_config
from skyrl_train.config.utils import get_default_config


def test_numerics_validation_refuses_batch_invariant_numerics_on_sglang_engines():
    cfg = get_default_config()
    cfg.trainer.numerics.mode = "batch_invariant"
    cfg.generator.backend = "sglang"

    with pytest.raises(ValueError):
        validate_numerics_config(cfg)
