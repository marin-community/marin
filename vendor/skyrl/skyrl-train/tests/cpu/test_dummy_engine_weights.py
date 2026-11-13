import asyncio

import pytest
import requests
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints import main_base
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.weight_sync.expert_block.driver import SyncTimings


def _dummy_config():
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy.model.path = "/tmp/model-metadata"
    cfg.trainer.policy.model.source_uri = "s3://models/policy"
    cfg.generator.weight_sync_transport = "expert_block"
    OmegaConf.update(cfg, "generator.engine_init_kwargs.load_format", "dummy", force_add=True)
    return cfg


class _Engine:
    def __init__(self, url=None):
        self.url, self.http_up_during_pause, self.paused = url, None, False

    async def pause_generation(self):
        try:
            self.http_up_during_pause = self.url is not None and requests.get(self.url, timeout=1).ok
        except requests.ConnectionError:
            self.http_up_during_pause = False
        self.paused = True

    async def resume_generation(self, policy_step):
        self.paused = False


def test_dummy_engine_client_pauses_engines_before_serving_http(unused_tcp_port, monkeypatch):
    engine = _Engine(f"http://127.0.0.1:{unused_tcp_port}/v1/models")
    cfg = _dummy_config()
    cfg.generator.enable_http_endpoint = True
    cfg.generator.http_endpoint_port = unused_tcp_port
    experiment = main_base.BasePPOExp.__new__(main_base.BasePPOExp)
    experiment.cfg, experiment.tokenizer, experiment.colocate_pg = cfg, object(), None
    monkeypatch.setattr(main_base, "create_ray_wrapped_inference_engines_from_config", lambda *a, **k: [engine])
    client = experiment.create_inference_engine_client()
    try:
        assert engine.paused and client.generation_paused_event.is_set()
        assert engine.http_up_during_pause is False
        assert requests.get(engine.url, timeout=1).ok
    finally:
        client.shutdown_http_endpoint()


class _Sync:
    def __init__(self, client, error):
        self.client, self.error, self.verified = client, error, []

    async def sync(self, version):
        assert self.client.generation_paused_event.is_set()
        return SyncTimings(0.0, 0.0, 0.0, 0.0, 0.0)

    async def verify(self, version):
        self.verified.append(version)
        if self.error is not None:
            raise self.error
        return {}


class _Policy:
    def async_run_ray_method(self, dispatch_type, method_name):
        return []


@pytest.mark.parametrize("verify_error", [None, RuntimeError("1 of 9 replayed bytes differ")])
def test_initial_dummy_sync_keeps_startup_pause_and_forces_verification(verify_error):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = _dummy_config()
    trainer.cfg.generator.expert_block_sync.verify = False
    trainer.colocate_all = False
    trainer.policy_model = _Policy()
    trainer.inference_engine_client = InferenceEngineClient(
        [_Engine()], object(), trainer.cfg, pause_before_serving=True
    )
    trainer._expert_block_sync = sync = _Sync(trainer.inference_engine_client, verify_error)
    trainer._inference_policy_step = None
    trainer.global_step = 0
    trainer.all_timings, trainer.all_startup_timings, trainer.all_metrics = {}, {}, {}
    if verify_error is not None:
        with pytest.raises(RuntimeError, match="replayed bytes differ"):
            asyncio.run(trainer._sync_policy_for_rollouts(reason="initial"))
        assert trainer.inference_engine_client.engines[0].paused
        return
    asyncio.run(trainer._sync_policy_for_rollouts(reason="initial"))
    trainer.global_step = 1
    asyncio.run(trainer._sync_policy_for_rollouts(reason="training_step"))
    assert sync.verified == [0]
    assert not trainer.inference_engine_client.engines[0].paused
