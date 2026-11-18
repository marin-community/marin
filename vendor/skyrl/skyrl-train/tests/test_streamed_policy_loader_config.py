# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for streamed policy loader configuration."""

import sys
from types import ModuleType

import pytest

from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config


@pytest.fixture
def engine_factory(monkeypatch):
    module = ModuleType("skyrl_train.inference_engines.ray_wrapped_inference_engine")
    module.MODEL_METADATA_PATH_KEY = "model_metadata_path"
    module.create_ray_wrapped_inference_engines = lambda **kwargs: kwargs
    module.create_ray_wrapped_inference_engines_with_retry = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize(
    ("configured_loader", "expected_loader"),
    [
        ({}, {"distributed": True, "memory_limit": 8 * 1024**3}),
        (
            {"distributed": False, "concurrency": 4, "memory_limit": 2 * 1024**3},
            {"distributed": False, "concurrency": 4, "memory_limit": 2 * 1024**3},
        ),
    ],
)
def test_streamed_policy_loader_has_finite_budget(engine_factory, configured_loader, expected_loader):
    cfg = get_default_config()
    cfg.trainer.policy.model.source_uri = "gs://models/policy"
    cfg.generator.engine_init_kwargs = {"model_loader_extra_config": configured_loader}
    launch_kwargs = create_ray_wrapped_inference_engines_from_config(cfg, None)

    assert launch_kwargs["engine_init_kwargs"]["model_loader_extra_config"] == expected_loader


def test_streamed_policy_rejects_unbounded_budget(engine_factory):
    cfg = get_default_config()
    cfg.trainer.policy.model.source_uri = "gs://models/policy"
    cfg.generator.engine_init_kwargs = {"model_loader_extra_config": {"memory_limit": -1}}

    with pytest.raises(ValueError, match="memory_limit"):
        create_ray_wrapped_inference_engines_from_config(cfg, None)
