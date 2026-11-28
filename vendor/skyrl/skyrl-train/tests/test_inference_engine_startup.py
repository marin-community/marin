"""Rendezvous port selection and staged-model inference gang recovery."""

from pathlib import Path

import ray
from hydra import compose, initialize_config_dir

from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines import ray_wrapped_inference_engine


def test_staged_dp_engine_restarts_after_actor_death(monkeypatch):
    directory = Path(__file__).parents[1] / "skyrl_train" / "config"
    with initialize_config_dir(config_dir=str(directory.resolve()), version_base=None):
        config = compose(config_name="ppo_base_config")
    config.generator.model_loading = "stage_local"
    config.generator.inference_engine_data_parallel_size = 8
    config.trainer.model_load_retry.backoff_base_seconds = 0.001
    config.trainer.model_load_retry.backoff_cap_seconds = 0.001
    engines = [object()]
    first_attempt = True

    def allocate_gpu_gang(**kwargs):
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            raise ray.exceptions.ActorDiedError()
        return engines

    monkeypatch.setattr(ray_wrapped_inference_engine, "create_ray_wrapped_inference_engines", allocate_gpu_gang)
    assert create_ray_wrapped_inference_engines_from_config(config, None) is engines
