# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bring up the vendored SkyRL Ray gang on every task and run one benchmark driver command on the head.

Gang start-up, rendezvous, log capture and teardown are the vendored task runtime's; only the driver command
differs. Ports use the Kubernetes zero placeholders, which the runtime resolves to free ports.
"""

import argparse
import os
from pathlib import Path

from cloud.iris import task_runtime as runtime
from cloud.iris.ray_ports import RayPort, resolve_ray_ports
from cloud.iris.ray_storage import RaySpillBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendezvous-dir", required=True)
    parser.add_argument("--ray-log-dir", required=True)
    parser.add_argument("--rendezvous-timeout", type=int, default=3600)
    parser.add_argument("--cluster-join-timeout", type=int, default=3600)
    parser.add_argument("--driver-liveness-timeout", type=int, default=2400)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a driver command is required")
    args.ray_spill_dir = "/tmp/mochi-ray-spill"
    args.ray_spill_backend = RaySpillBackend.LOCAL
    for port in RayPort:
        os.environ.setdefault(f"IRIS_PORT_{port.value.upper()}", "0")
    runtime._pin_boto3_s3_addressing_style()
    interface = runtime.pin_socket_ifname()
    runtime.ensure_fr_dump_dir()
    resolve_ray_ports()
    runtime.training_driver_command = lambda _config_path: command
    if runtime._rank() == 0:
        return runtime.run_head(args, Path("/tmp/mochi-driver-config.yaml"), interface)
    return runtime.run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
