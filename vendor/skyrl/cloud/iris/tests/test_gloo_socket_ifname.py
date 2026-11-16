"""Regression guard: GLOO_SOCKET_IFNAME is a per-node name, never a cluster-wide one.

Gloo reads GLOO_SOCKET_IFNAME as a literal interface name, and NIC names differ across
the nodes of a gang. Job 20260729-102429-52af30 had the head derive ``enp90s0np0`` and
hand it to the training driver, which forwards the variable into ray.init's job-level
runtime_env; the node at 10.168.206.93 names its NIC ``enp90s0f0np0`` and megatron's
gloo group creation there died with ``Unable to find address for: enp90s0np0``.

Run:
    python -m pytest cloud/iris/tests/test_gloo_socket_ifname.py -v
"""

from __future__ import annotations

import fcntl
import os
import socket
import struct
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.task_runtime import (  # noqa: E402
    _iface_for_ip,
    pin_socket_ifname,
    training_driver_env,
)

_HEAD_NICS = {"lo": "127.0.0.1", "enp90s0np0": "10.168.206.11"}
_WORKER_NICS = {"lo": "127.0.0.1", "enp90s0f0np0": "10.168.206.93"}


def _fake_host(monkeypatch, nics: dict[str, str]) -> None:
    """Make the interface table look like a host with exactly ``nics``."""
    monkeypatch.setattr(socket, "if_nameindex", lambda: [(i, name) for i, name in enumerate(nics, start=1)])

    def fake_ioctl(_fd, _request, arg):
        name = arg[:16].split(b"\0", 1)[0].decode()
        address = nics.get(name)
        if address is None:
            raise OSError(19, "No such device")
        return struct.pack("20s4s8s", b"", socket.inet_aton(address), b"")

    monkeypatch.setattr(fcntl, "ioctl", fake_ioctl)


def _unset_gloo_ifname(monkeypatch) -> None:
    """Unset GLOO_SOCKET_IFNAME so monkeypatch restores it once the test ends.

    ``pin_socket_ifname`` writes the variable as a side effect, and a bare
    ``delenv`` on an already-absent name records nothing to undo.
    """
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "")
    monkeypatch.delenv("GLOO_SOCKET_IFNAME")


def test_each_host_resolves_its_own_nic_name(monkeypatch):
    _fake_host(monkeypatch, _HEAD_NICS)
    assert _iface_for_ip("10.168.206.11") == "enp90s0np0"

    _fake_host(monkeypatch, _WORKER_NICS)
    assert _iface_for_ip("10.168.206.93") == "enp90s0f0np0"


def test_no_interface_holds_the_address_yields_no_pin(monkeypatch):
    _fake_host(monkeypatch, _WORKER_NICS)
    _unset_gloo_ifname(monkeypatch)
    monkeypatch.setenv("IRIS_ADVERTISE_HOST", "10.168.206.250")

    assert _iface_for_ip("10.168.206.250") is None
    assert pin_socket_ifname() is None
    assert "GLOO_SOCKET_IFNAME" not in os.environ


def test_worker_pins_its_own_nic_not_the_head_name(monkeypatch):
    _fake_host(monkeypatch, _WORKER_NICS)
    _unset_gloo_ifname(monkeypatch)
    monkeypatch.setenv("IRIS_ADVERTISE_HOST", "10.168.206.93")

    assert pin_socket_ifname() == "enp90s0f0np0"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "enp90s0f0np0"


def test_node_derived_name_is_withheld_from_the_training_driver(monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "enp90s0np0")
    assert "GLOO_SOCKET_IFNAME" not in training_driver_env("enp90s0np0")


def test_operator_supplied_name_reaches_the_training_driver(monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "bond0")
    assert training_driver_env(None)["GLOO_SOCKET_IFNAME"] == "bond0"
    assert training_driver_env(None)["SKYRL_RAY_CLUSTER_OWNER"] == "iris-task-runtime"
