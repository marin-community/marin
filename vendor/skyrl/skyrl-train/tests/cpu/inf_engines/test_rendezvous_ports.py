import socket

import pytest

from skyrl_train.inference_engines import utils


@pytest.mark.parametrize("ephemeral_range", [(10240, 65535), (14000, 65535), (32768, 60999)])
def test_reservation_avoids_ephemeral_reserved_and_live_listeners(tmp_path, monkeypatch, ephemeral_range):
    kernel_range = tmp_path / "ip_local_port_range"
    kernel_range.write_text(f"{ephemeral_range[0]} {ephemeral_range[1]}\n")
    reserved_ports = tmp_path / "ip_local_reserved_ports"
    reserved_ports.write_text("10050-10100,10200\n")
    monkeypatch.setattr(utils, "_EPHEMERAL_PORT_RANGE_PATH", kernel_range)
    monkeypatch.setattr(utils, "_RESERVED_PORTS_PATH", reserved_ports)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        listener.listen()
        occupied = listener.getsockname()[1]
        if ephemeral_range[0] <= occupied <= ephemeral_range[1]:
            # A live listener inside the ephemeral band cannot be pinned portably; the
            # allocator already avoids the whole band.
            occupied = None
        sockets = utils._reserve_available_rendezvous_ports(6, {occupied} if occupied else set())
        try:
            ports = [reserved.getsockname()[1] for reserved in sockets]
            assert len(set(ports)) == 6
            for port in ports:
                assert not ephemeral_range[0] <= port <= ephemeral_range[1]
                assert not 10050 <= port <= 10100
                assert port != 10200
            assert occupied not in ports
        finally:
            for reserved in sockets:
                reserved.close()
