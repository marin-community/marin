# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct host<->host byte channel for the pipeline's cross-host activation hop.

The device-group pipeline's only data movement that leaves a host is the activation
crossing the one stage boundary that splits the two processes. ``broadcast_one_to_all``
does that as a global psum over every device -- a fixed per-call cost (hundreds of ms,
not bandwidth: a 48 MiB activation is ~ms over the fabric) that dominates the multi-host
step. :class:`HostChannel` replaces it with a point-to-point TCP transfer: the activation
is pulled to the host once, sent directly to the peer over the (IB-backed) socket, and
rebuilt there -- no global collective.

Two processes only (one boundary). Rendezvous uses the JAX distributed key-value store
(already up after ``jax.distributed.initialize``): the lower-rank process binds an
ephemeral port, publishes its address, and accepts; the higher-rank process reads the
address and connects. The resulting socket is full-duplex, so one connection carries both
the forward activation (low->high) and the backward cotangent (high->low) streams.

Send/recv pair by FIFO order: callers on both hosts issue hops in identical 1f1b
``op_order``, so each ``send`` on one host lines up with the matching ``recv`` on the other.
"""

from __future__ import annotations

import logging
import socket
import time

import jax
import numpy as np
from iris.cluster.client.job_info import get_job_info
from jax._src import distributed

logger = logging.getLogger(__name__)

_KV_TIMEOUT_MS = 120_000
_SOCK_BUF = 64 << 20  # 64 MiB, comfortably above one 48 MiB activation
_CONNECT_ATTEMPTS = 120
_CONNECT_DELAY = 1.0


def _kv_client():
    client = distributed.global_state.client
    if client is None:
        raise RuntimeError("jax.distributed is not initialized; HostChannel needs the KV store for rendezvous")
    return client


def _advertise_host() -> str:
    """This process's address as peers reach it (the Iris job's advertised host)."""
    info = get_job_info()
    if info is None:
        raise RuntimeError("no Iris job info; cannot determine this host's advertised address")
    return info.advertise_host


def _recv_exactly(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError(f"peer closed mid-message after {got}/{n} bytes")
        got += r
    return buf


def _connect_retry(host: str, port: int) -> socket.socket:
    last: Exception | None = None
    for _ in range(_CONNECT_ATTEMPTS):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            return sock
        except OSError as exc:  # listener not up yet -- retry
            last = exc
            time.sleep(_CONNECT_DELAY)
    raise RuntimeError(f"could not connect to {host}:{port} after {_CONNECT_ATTEMPTS} attempts: {last}")


class HostChannel:
    """Full-duplex TCP channel to one peer process, for raw activation bytes.

    The receiver knows each message's shape/dtype a priori (the pipeline's placeholder
    carries them), so messages are bare bytes with no header.
    """

    def __init__(self, peer: int, *, tag: str = "moe_pp_p2p") -> None:
        if jax.process_count() != 2:
            raise NotImplementedError(f"HostChannel is two-process only, got {jax.process_count()}")
        me = jax.process_index()
        self.peer = peer
        kv = _kv_client()
        key = f"{tag}/{min(me, peer)}-{max(me, peer)}"
        if me < peer:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", 0))
            listener.listen(1)
            addr = f"{_advertise_host()}:{listener.getsockname()[1]}"
            kv.key_value_set(key, addr)
            logger.info("HostChannel: process %d listening at %s for peer %d", me, addr, peer)
            self._sock, _ = listener.accept()
            listener.close()
        else:
            addr = kv.blocking_key_value_get(key, _KV_TIMEOUT_MS)
            host, port = addr.rsplit(":", 1)
            logger.info("HostChannel: process %d connecting to %s for peer %d", me, addr, peer)
            self._sock = _connect_retry(host, int(port))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCK_BUF)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCK_BUF)

    def send(self, buf: np.ndarray) -> None:
        self._sock.sendall(np.ascontiguousarray(buf).tobytes())

    def recv(self, shape: tuple[int, ...], dtype) -> np.ndarray:
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        data = _recv_exactly(self._sock, nbytes)
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def close(self) -> None:
        self._sock.close()
