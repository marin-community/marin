# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator entrypoint for the GLM synthesis serving pool.

`run_iris_service` must run inside an Iris job: it creates the broker actor, submits
one engine worker job per instance, registers the OpenAI endpoint, and blocks until
the service timeout. `launch_pool.py` submits this as the coordinator job's
entrypoint; the `__main__` path supports manual `iris job run` invocations.
"""

import argparse
from dataclasses import replace

from marin.inference.iris import run_iris_service

from experiments.glm_synthesis_serving.pool import POOL_INSTANCES, service


def serve(
    instances: int = POOL_INSTANCES,
    extra_args: tuple[str, ...] = (),
    endpoint_name: str | None = None,
    worker_env_vars: tuple[tuple[str, str], ...] = (),
) -> None:
    """Run the brokered serving service for `instances` engines until its timeout.

    `extra_args` extends the engine arguments (warm-run variants such as MTP
    speculative decoding); `endpoint_name` overrides the registered endpoint so a
    variant can bench on its own URL; `worker_env_vars` adds engine-worker
    environment variables (stability variants).
    """
    service_config = service(instances, engine_extra_args=extra_args, worker_env_vars=worker_env_vars)
    if endpoint_name is not None:
        service_config = replace(service_config, endpoint_name=endpoint_name)
    run_iris_service(service_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=POOL_INSTANCES)
    parser.add_argument(
        "--mtp",
        type=int,
        default=None,
        help="Enable MTP speculative decoding with this many speculative tokens.",
    )
    args = parser.parse_args()
    if args.instances < 1:
        raise ValueError("--instances must be positive")
    extra_args: tuple[str, ...] = ()
    if args.mtp is not None:
        if args.mtp < 1:
            raise ValueError("--mtp must be positive")
        extra_args = ("--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{args.mtp}}}')
    serve(args.instances, extra_args)
