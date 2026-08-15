# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for a persistent Zephyr memory-store subprocess shard."""

import argparse

from rigging.log_setup import configure_logging

from zephyr.memory_store import run_memory_store_subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("actor_index", type=int)
    parser.add_argument("local_shard_index", type=int)
    parser.add_argument("endpoint_name")
    parser.add_argument("port", type=int)
    args = parser.parse_args()
    configure_logging()
    run_memory_store_subprocess(args.actor_index, args.local_shard_index, args.endpoint_name, args.port)


if __name__ == "__main__":
    main()
