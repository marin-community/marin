# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate the vLLM per-position reference for the Snowball training-forward parity check.

Submits one H100x8 job that serves the Snowball export with the pinned Marin vLLM fork, scores every position of
the 64 representative prompts with ``prompt_logprobs``, and writes the reference JSON to ``--output``. The job log
prints the file's sha256 and the content-addressed public URL to stage it at; staging and pinning
``VLLM_POSITION_REFERENCE_SHA256`` in ``marin.testing.inference.snowball`` are manual. Without ``--run`` it only
prints the job it would submit::

    uv run python tests/cluster/vllm/generate_snowball_position_reference.py \\
      --output s3://marin-us-east-02a/<scratch>/snowball-position-reference.json --run
"""

import argparse
import logging
import uuid
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.testing.inference.snowball_backend_parity_jobs import GPU_COUNT, write_vllm_position_reference

logger = logging.getLogger(__name__)

# A script run outside pytest cannot import tests.cluster (a site-packages "tests" shadows it), so the
# cluster names of tests/cluster/conftest.py are repeated here.
MARIN_ROOT = Path(__file__).resolve().parents[3]
MARIN_HUB_CLUSTER = "marin"
MARIN_GPU_CLUSTER = "cw-us-east-02a"


def reference_job(output_uri: str, attention_backend: str) -> JobRequest:
    return JobRequest(
        name=f"snowball-position-reference-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(write_vllm_position_reference, args=[output_uri, attention_backend]),
        resources=ResourceConfig.with_gpu(
            "H100", count=GPU_COUNT, cpu=64, ram="512g", disk="128g", target_cluster=MARIN_GPU_CLUSTER
        ),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars={"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="where the job writes the reference JSON")
    parser.add_argument("--attention-backend", choices=("FLASH_ATTN", "TRITON_ATTN"), default="FLASH_ATTN")
    parser.add_argument("--run", action="store_true", help="submit; without it only print the job")
    args = parser.parse_args()

    request = reference_job(args.output, args.attention_backend)
    logger.info("Reference job: name=%s output=%s attention=%s", request.name, args.output, args.attention_backend)
    if not args.run:
        return
    with connect_controller(cluster_name=MARIN_HUB_CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, workspace=MARIN_ROOT, credentials=endpoint.credentials) as client:
            job = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
            logger.info("Submitted %s", job.job_id)


if __name__ == "__main__":
    main()
