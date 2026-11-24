"""Rendezvous port selection and staged-model inference gang recovery."""

from pathlib import Path

import pytest
import ray
from hydra import compose, initialize_config_dir
from iris.cluster.worker.port_allocator import DEFAULT_TASK_PORT_RANGE

from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines import ray_wrapped_inference_engine
from skyrl_train.inference_engines import utils as engine_ports


@pytest.mark.parametrize("ephemeral_range", [(10240, 65535), (14000, 65535), (32768, 60999)])
def test_rendezvous_ports_avoid_ephemeral_and_named_ranges(tmp_path, monkeypatch, ephemeral_range):
    kernel_range = tmp_path / "ip_local_port_range"
    kernel_range.write_text(f"{ephemeral_range[0]} {ephemeral_range[1]}\n")
    reserved_ports = tmp_path / "ip_local_reserved_ports"
    reserved_ports.write_text("10050-10100,10200\n")
    monkeypatch.setattr(engine_ports, "_EPHEMERAL_PORT_RANGE_PATH", kernel_range)
    monkeypatch.setattr(engine_ports, "_RESERVED_PORTS_PATH", reserved_ports)
    task_start, task_stop = DEFAULT_TASK_PORT_RANGE
    excluded = range(task_start, task_stop)
    sockets = engine_ports._reserve_available_rendezvous_ports(6, excluded)
    try:
        ports = [listener.getsockname()[1] for listener in sockets]
        assert len(set(ports)) == len(ports)
        for port in ports:
            assert not ephemeral_range[0] <= port <= ephemeral_range[1]
            assert not task_start <= port < task_stop
            assert not 10050 <= port <= 10100
            assert port != 10200
    finally:
        for listener in sockets:
            listener.close()


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
