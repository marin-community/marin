import socket

import pytest

from skyrl_train.inference_engines import utils


@pytest.mark.parametrize("ephemeral_start", [10_240, 32_768])
def test_reservation_avoids_host_client_range_and_live_listeners(tmp_path, monkeypatch, ephemeral_start):
    kernel_range = tmp_path / "ip_local_port_range"
    kernel_range.write_text(f"{ephemeral_start}\t65535\n")
    monkeypatch.setattr(utils, "_LINUX_EPHEMERAL_PORT_RANGE", kernel_range)
    monkeypatch.setattr(utils.sys, "platform", "linux")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", utils._find_available_rendezvous_port()))
        listener.listen()
        occupied = listener.getsockname()[1]
        excluded = utils._find_available_rendezvous_port()
        sockets = utils._reserve_available_rendezvous_ports(6, [excluded])
        try:
            ports = [reserved.getsockname()[1] for reserved in sockets]
            assert len(set(ports)) == 6
            assert all(1_024 <= port < ephemeral_start for port in ports)
            assert occupied not in ports
            assert excluded not in ports
        finally:
            for reserved in sockets:
                reserved.close()
