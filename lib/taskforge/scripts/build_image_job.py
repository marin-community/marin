# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a directory's Dockerfile with ``taskforge.sandbox.image_builder.IrisImageBuilder`` and print the pinned image.

    uv run scripts/build_image_job.py --context DIR --repository capability-infra/example \
        --priority batch --timeout 259200 --scheduling-timeout 259200

The jobs go through the named cluster's controller (``--cluster``, default ``marin``) and run on
``--target-cluster``. The script reads the registry credential from Secret Manager with this
laptop's ``gcloud`` and hands it to the builder in process; the builder's module docstring
describes the build and push jobs and where the credential travels.
"""

import asyncio
import json
import logging
from pathlib import Path

import click
from iris.rpc import job_pb2
from rigging.timing import Duration

from taskforge.sandbox.image_builder import (
    TARGET_CLUSTER,
    ImageJobPolicy,
    IrisImageBuilder,
    RegistryCredential,
    registry_credential_json,
)
from taskforge.sandbox.images import BUILD_LIMITS, docker_build_from_directory


@click.command(help=__doc__.split("\n\n")[0])
@click.option(
    "--context",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="build context directory with a Dockerfile",
)
@click.option("--repository", required=True, help="repository under the registry, e.g. capability-infra/x")
@click.option("--cluster", default="marin", show_default=True)
@click.option("--target-cluster", default=TARGET_CLUSTER, show_default=True)
@click.option("--priority", type=click.Choice(["batch", "production", "interactive"]), required=True)
@click.option("--timeout", type=click.IntRange(min=1), required=True, help="Execution deadline in seconds")
@click.option("--scheduling-timeout", type=click.IntRange(min=1), required=True, help="Scheduling deadline in seconds")
@click.option("--no-wait", is_flag=True, help="Submit the build job and print its id and destination without waiting")
def main(
    context: Path,
    repository: str,
    cluster: str,
    target_cluster: str,
    priority: str,
    timeout: int,
    scheduling_timeout: int,
    no_wait: bool,
    memory_gb: int | None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # blob redirects carry presigned object-store URLs
    build = docker_build_from_directory(context, BUILD_LIMITS)
    credential = RegistryCredential.from_json(registry_credential_json())
    policy = ImageJobPolicy(
        priority_band=job_pb2.PriorityBand.Value(f"PRIORITY_BAND_{priority.upper()}"),
        timeout=Duration.from_seconds(timeout),
        scheduling_timeout=Duration.from_seconds(scheduling_timeout),
    )
    builder = IrisImageBuilder(cluster=cluster, target_cluster=target_cluster, credential=credential, policy=policy)
    if no_wait:
        memory_bytes = memory_gb * 1024**3 if memory_gb else None
        submitted = asyncio.run(builder.submit_build(build, repository, memory_bytes=memory_bytes))
        print(
            json.dumps(
                {
                    "submitted": True,
                    "build_job_id": submitted.job_id,
                    "destination": submitted.destination,
                    "build_digest": submitted.build_digest,
                    "tag": submitted.tag,
                },
                indent=2,
            )
        )
        return
    published = builder.publish_with_timings(build, repository)
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
