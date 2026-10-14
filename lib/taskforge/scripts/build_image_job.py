# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a directory's Dockerfile with ``taskforge.sandbox.image_builder.IrisImageBuilder`` and print the pinned image.

    uv run scripts/build_image_job.py --context DIR --repository capability-infra/taskforge-smoke

The jobs go through the named cluster's controller (``--cluster``, default ``marin``) and run on
``--target-cluster``; the builder's module docstring describes the build and push jobs and where
the registry credential travels.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from taskforge.sandbox.image_builder import TARGET_CLUSTER, IrisImageBuilder
from taskforge.sandbox.images import BUILD_LIMITS, docker_build_from_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--context", type=Path, required=True, help="build context directory with a Dockerfile")
    parser.add_argument("--repository", required=True, help="repository under the registry, e.g. capability-infra/x")
    parser.add_argument("--cluster", default="marin")
    parser.add_argument("--target-cluster", default=TARGET_CLUSTER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # blob redirects carry presigned object-store URLs
    build = docker_build_from_directory(args.context, BUILD_LIMITS)
    builder = IrisImageBuilder(cluster=args.cluster, target_cluster=args.target_cluster)
    published = asyncio.run(builder.publish_with_timings(build, args.repository))
    print(
        json.dumps(
            {
                "image": published.image,
                "build_digest": published.build_digest,
                "build_job_id": published.build_job_id,
                "push_job_id": published.push_job_id,
                "seconds_to_built": round(published.seconds_to_built, 1),
                "seconds_total": round(published.seconds_total, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
