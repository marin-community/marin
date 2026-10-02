# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, separate-process Torch sender for weight-transfer validation."""

import argparse
import multiprocessing
import os
import socket
import subprocess
import sys
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.connection import Connection
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _new_process_group_helper, _world


@dataclass
class BroadcastSource:
    connection: Connection
    port: int
    timeout: float

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError("Torch sender did not acknowledge the broadcast")
        result = self.connection.recv()
        if isinstance(result, str):
            raise RuntimeError(result)
        return result

    def stop(self) -> None:
        """Signal sender teardown before the receiving rank destroys its NCCL group."""
        if not self.connection.closed:
            self.connection.send(None)
            self.connection.close()

    def broadcast(self, dtype: str, values: list) -> None:
        self.connection.send((dtype, values))
        assert self.receive() is None


@contextmanager
def broadcast_source(
    log_path: Path,
    *,
    backend: str = "gloo",
    timeout: float = 15,
    environment: Mapping[str, str] | None = None,
) -> Iterator[BroadcastSource]:
    """Run a rank-zero sender; the caller initializes serving rank one and receives."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    connection, child_connection = multiprocessing.Pipe()
    with log_path.open("wb") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "levanter.testing.weight_broadcast",
                "--port",
                str(port),
                "--pipe-fd",
                str(child_connection.fileno()),
                "--backend",
                backend,
                "--timeout",
                str(timeout),
            ],
            pass_fds=(child_connection.fileno(),),
            env={**os.environ, **(environment or {})},
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    child_connection.close()
    try:
        source = BroadcastSource(connection, port, timeout)
        yield source
    finally:
        if process.poll() is None:
            source.stop()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        connection.close()
    if process.returncode:
        raise RuntimeError(f"Torch sender failed: {log_path.read_text()}")


def _send(port: int, connection: Connection, backend: str, timeout: float) -> None:
    group = None
    try:
        device = torch.device("cuda:0" if backend == "nccl" else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        store = dist.TCPStore("127.0.0.1", port, 2, True, timeout=timedelta(seconds=timeout))
        group, _ = _new_process_group_helper(
            2, 0, [], backend, dist.PrefixStore("skyrl", store), group_name="skyrl", timeout=timedelta(seconds=timeout)
        )
        assert isinstance(group, dist.ProcessGroup)
        _world.pg_group_ranks[group] = {0: 0, 1: 1}
        connection.send(
            {
                "torch": torch.__version__,
                "device": str(device),
                "uuid": str(torch.cuda.get_device_properties(device).uuid) if device.type == "cuda" else None,
            }
        )
        while (command := connection.recv()) is not None:
            dtype, values = command
            tensor = torch.tensor(values, dtype=getattr(torch, dtype.removeprefix("torch.")), device=device)
            dist.broadcast(tensor, src=0, group=group)
            connection.send(None)
    except Exception:
        connection.send(traceback.format_exc())
        raise
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pipe-fd", type=int, required=True)
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    _send(args.port, Connection(args.pipe_fd), args.backend, args.timeout)
