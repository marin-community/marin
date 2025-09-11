import argparse
import logging
import sys

import ray
from ray.exceptions import RayError

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")


def kill_status_actor(address: str):
    """Connect to a Ray cluster and kill the detached `status_actor` if it exists.

    This is a convenience util when the global StatusActor is pinned to a node
    that no longer exists (see executor.py).
    """
    ray.init(address=address, namespace="marin", ignore_reinit_error=True)

    try:
        actor = ray.get_actor("status_actor")
    except ValueError:
        logger.info("`status_actor` not found – nothing to do.")
        return
    except RayError as err:
        logger.error(f"Error while looking up `status_actor`: {err}")
        sys.exit(1)

    try:
        ray.kill(actor, no_restart=True)
        logger.info("Successfully killed `status_actor`.")
    except RayError as err:
        logger.error(f"Failed to kill `status_actor`: {err}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kill stale detached status_actor from Marin executor.")
    parser.add_argument(
        "--address",
        default="auto",
        help="Ray cluster address; default connects to an existing local or remote cluster via 'auto'.",
    )

    args = parser.parse_args()
    kill_status_actor(args.address) 