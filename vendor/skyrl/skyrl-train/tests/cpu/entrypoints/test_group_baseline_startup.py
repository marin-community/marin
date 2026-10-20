from unittest.mock import Mock

import pytest

from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.main_base import run_ray_driver
from skyrl_train.utils import utils as trainer_utils


class _RayInitializationReached(RuntimeError):
    pass


@pytest.mark.parametrize("operation", [EntrypointOperation.TRAIN, EntrypointOperation.GENERATE])
def test_group_estimator_training_needs_two_samples_per_prompt(monkeypatch, operation):
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.advantage_estimator = "grpo"
    cfg.generator.n_samples_per_prompt = 1
    monkeypatch.setattr(trainer_utils, "initialize_ray", Mock(side_effect=_RayInitializationReached()))
    if operation is EntrypointOperation.TRAIN:
        with pytest.raises(ValueError):
            run_ray_driver(cfg, None, operation=operation)
    else:
        with pytest.raises(_RayInitializationReached):
            run_ray_driver(cfg, None, operation=operation)
