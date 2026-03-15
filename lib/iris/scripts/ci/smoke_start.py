# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone CI script that starts a cloud smoke cluster and keeps it alive until killed.

Usage:
    uv run python lib/iris/scripts/ci/smoke_start.py \
        --config infra/iris-smoke.yaml \
        --label-prefix ci-smoke-123 \
        --url-file /tmp/iris-url.txt
"""

import argparse
import logging
import signal
import sys
import time
import threading

import fsspec

from iris.cli.cluster import _build_cluster_images, _pin_latest_images
from iris.cluster.config import IrisConfig, load_config
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

logger = logging.getLogger(__name__)

WORKER_WAIT_TIMEOUT = 600.0
WORKER_POLL_INTERVAL = 2.0


def _clear_remote_state(remote_state_dir: str) -> None:
    """Remove all files under the remote state dir so the controller starts fresh."""
    fs, path = fsspec.core.url_to_fs(remote_state_dir)
    if fs.exists(path):
        fs.rm(path, recursive=True)


def _wait_for_workers(url: str, min_workers: int = 1, timeout: float = WORKER_WAIT_TIMEOUT) -> None:
    """Poll controller RPC until at least min_workers healthy workers are registered."""
    controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
    try:
        deadline = time.monotonic() + timeout
        healthy_count = 0
        while time.monotonic() < deadline:
            try:
                request = cluster_pb2.Controller.ListWorkersRequest()
                response = controller_client.list_workers(request)
                healthy = [w for w in response.workers if w.healthy]
                healthy_count = len(healthy)
                if healthy_count >= min_workers:
                    print(f"Got {healthy_count} healthy worker(s)", file=sys.stderr)
                    return
            except Exception:
                pass
            time.sleep(WORKER_POLL_INTERVAL)
        raise TimeoutError(f"Only {healthy_count} of {min_workers} workers registered in {timeout}s")
    finally:
        controller_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a cloud smoke cluster for CI")
    parser.add_argument("--config", required=True, help="Path to Iris cluster config YAML")
    parser.add_argument("--label-prefix", help="Label prefix override for cluster isolation")
    parser.add_argument("--url-file", required=True, help="Path to write the tunnel URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    config = load_config(args.config)

    if args.label_prefix:
        config.platform.label_prefix = args.label_prefix
        config.storage.remote_state_dir = f"gs://marin-tmp-eu-west4/ttl=7d/iris/state/{args.label_prefix}"

    print("Pinning and building cluster images...", file=sys.stderr)
    _pin_latest_images(config)
    _build_cluster_images(config, verbose=False)

    iris_config = IrisConfig(config)
    platform = iris_config.platform()

    print("Stopping any existing cluster...", file=sys.stderr)
    try:
        platform.stop_all(config)
    except Exception:
        print("No existing cluster to stop (or stop failed), continuing", file=sys.stderr)

    remote_state_dir = config.storage.remote_state_dir
    if remote_state_dir:
        print(f"Clearing remote state dir: {remote_state_dir}", file=sys.stderr)
        _clear_remote_state(remote_state_dir)

    print("Starting fresh controller...", file=sys.stderr)
    address = platform.start_controller(config)
    print(f"Controller started at {address}", file=sys.stderr)

    with platform.tunnel(address) as url:
        print(f"Tunnel ready: {url}", file=sys.stderr)

        print("Waiting for workers...", file=sys.stderr)
        _wait_for_workers(url, min_workers=1)

        with open(args.url_file, "w") as f:
            f.write(url)
        print(f"URL written to {args.url_file}", file=sys.stderr)

        # Block until SIGINT/SIGTERM
        shutdown = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: shutdown.set())
        signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
        print("Cluster ready. Blocking until killed (SIGINT/SIGTERM)...", file=sys.stderr)
        shutdown.wait()
        print("Shutting down.", file=sys.stderr)


if __name__ == "__main__":
    main()
