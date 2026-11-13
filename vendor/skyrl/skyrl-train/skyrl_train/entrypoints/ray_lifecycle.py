import atexit
import os
import sys

import ray
from loguru import logger


def shutdown_ray() -> None:
    owner = os.environ.get("SKYRL_RAY_CLUSTER_OWNER")
    if not owner:
        ray.shutdown()
        return

    atexit.unregister(ray.shutdown)
    logger.info(f"Leaving Ray cluster teardown to {owner}")


def exit_without_ray_destructors(exit_code: int = 0) -> None:
    owner = os.environ.get("SKYRL_RAY_CLUSTER_OWNER")
    if not owner:
        return

    logger.info(f"Exiting after handing Ray cluster teardown to {owner}")
    # os._exit intentionally skips interpreter teardown, including normal stream flushing.
    # Flush explicitly so the exception immediately preceding this call survives in task logs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
