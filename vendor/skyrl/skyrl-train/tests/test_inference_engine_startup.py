"""Recovery of staged-model inference gangs after a rendezvous bind failure."""

import errno
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import ray
from hydra import compose, initialize_config_dir
from iris.cluster.worker.port_allocator import DEFAULT_TASK_PORT_RANGE

from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines import ray_wrapped_inference_engine
from skyrl_train.inference_engines import utils as engine_ports


@pytest.mark.parametrize("ephemeral_range", [(10240, 65535), (14000, 65535), (32768, 60999)])
def test_concurrent_gangs_keep_rendezvous_ports_out_of_ephemeral_and_named_ranges(tmp_path, monkeypatch, ephemeral_range):
    kernel_range = tmp_path / "ip_local_port_range"
    kernel_range.write_text(f"{ephemeral_range[0]} {ephemeral_range[1]}\n")
    reserved_ports = tmp_path / "ip_local_reserved_ports"
    reserved_ports.write_text("10050-10100,10200\n")
    monkeypatch.setattr(engine_ports, "_EPHEMERAL_PORT_RANGE_PATH", kernel_range)
    monkeypatch.setattr(engine_ports, "_RESERVED_PORTS_PATH", reserved_ports)
    task_start, task_stop = DEFAULT_TASK_PORT_RANGE
    excluded = range(task_start, task_stop)
    (occupied_socket,) = engine_ports._reserve_available_rendezvous_ports(1, excluded)
    sockets = [occupied_socket]
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            groups = list(executor.map(engine_ports._reserve_available_rendezvous_ports, [6] * 8, [excluded] * 8))
        sockets.extend(listener for group in groups for listener in group)
        ports = [listener.getsockname()[1] for listener in sockets]
        assert len(set(ports)) == len(ports)
        for port in ports:
            assert not ephemeral_range[0] <= port <= ephemeral_range[1]
            assert not task_start <= port < task_stop
            assert not 10050 <= port <= 10100
            assert port != 10200
            with socket.socket() as contender, pytest.raises(OSError) as failure:
                contender.bind(("", port))
            assert failure.value.errno == errno.EADDRINUSE
        # The exact reserved port can be handed to a listener once its owner
        # releases it; no second free-port probe chooses a different endpoint.
        port = sockets[-1].getsockname()[1]
        sockets.pop().close()
        with socket.socket() as consumer:
            consumer.bind(("", port))
            consumer.listen()
            assert consumer.getsockname()[1] == port
    finally:
        for listener in sockets:
            listener.close()


def test_staged_dp_engine_restarts_after_rendezvous_collision(monkeypatch):
    directory = Path(__file__).parents[1] / "skyrl_train" / "config"
    with initialize_config_dir(config_dir=str(directory.resolve()), version_base=None):
        config = compose(config_name="ppo_base_config")
    config.generator.model_loading = "stage_local"
    config.generator.inference_engine_data_parallel_size = 8
    config.trainer.model_load_retry.backoff_base_seconds = 0.001
    config.trainer.model_load_retry.backoff_cap_seconds = 0.001
    listeners = []
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        address = occupied.getsockname()
        collision_observed = False

        def allocate_gpu_gang(**kwargs):
            nonlocal collision_observed
            if not collision_observed:
                with socket.socket() as contender:
                    try:
                        contender.bind(address)
                    except OSError as error:
                        collision_observed = True
                        raise ray.exceptions.ActorDiedError() from error
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listeners.append(listener)
            return [listener]

        # Replace the GPU/Ray allocation boundary; config resolution and the
        # coordinated startup/recovery path execute normally.
        monkeypatch.setattr(ray_wrapped_inference_engine, "create_ray_wrapped_inference_engines", allocate_gpu_gang)
        try:
            (listener,) = create_ray_wrapped_inference_engines_from_config(config, None)
            with socket.create_connection(listener.getsockname(), timeout=1) as client:
                connection, _ = listener.accept()
                with connection:
                    client.sendall(b"ready")
                    assert connection.recv(5) == b"ready"
        finally:
            for listener in listeners:
                listener.close()
    assert collision_observed
