# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise a federated CPU coordinator, H100 inference child, and capability URL.

Run the CPU coordinator through marin-dev and pin the whole job tree to the
cw-us-west-04a CI cluster:

    uv run iris --cluster=marin-dev job run --target-cluster cw-us-west-04a \
      --job-name federated-inference-proxy-demo --cpu 2 --memory 8g --disk 20g \
      --enable-extra-resources --extra cpu --priority interactive --no-wait \
      -- python -m experiments.evals.federated_inference_proxy_demo

The coordinator starts a one-H100 vLLM child, asks the child controller to mint
a capability URL, and calls ``/v1/models`` through marin-dev's public proxy.
"""

import json
import logging
import subprocess
from urllib.parse import urlsplit, urlunsplit

from fray.types import ANY_REGION, ResourceConfig, create_environment
from marin.inference.config import (
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.inference.iris import remote_inference
from rigging.log_setup import configure_logging

PARENT_ORIGIN = "https://iris-dev.oa.dev"
CHILD_CLUSTER = "cw-us-west-04a"
MODEL = "Qwen/Qwen3-0.6B-Base"
REQUEST_TIMEOUT_SECONDS = 60
CURL_METADATA_MARKER = "\n__IRIS_CURL_METADATA__"

logger = logging.getLogger(__name__)


def _redacted_url(url: str) -> str:
    parsed = urlsplit(url)
    segments = parsed.path.split("/")
    if "t" in segments:
        token_index = segments.index("t") + 1
        if segments[token_index].startswith("cluster="):
            token_index += 1
        segments[token_index] = "<redacted>"
    return urlunsplit((parsed.scheme, parsed.netloc, "/".join(segments), "", ""))


def main() -> None:
    configure_logging()
    config = RemoteInferenceConfig(
        model=ServedModelConfig(
            weights=MODEL,
            tokenizer="Qwen/Qwen3-0.6B",
            max_model_len=2048,
        ),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            startup_timeout_seconds=1800,
        ),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu(
                "H100",
                count=1,
                cpu=8,
                ram="64g",
                disk="100g",
                regions=[ANY_REGION],
            ),
            worker_environment=create_environment(),
        ),
        capability_origin=PARENT_ORIGIN,
    )

    with remote_inference(config) as session:
        base_url = session.model.endpoint.base_url
        expected_prefix = f"{PARENT_ORIGIN}/proxy/t/cluster={CHILD_CLUSTER}/"
        if not base_url.startswith(expected_prefix):
            raise RuntimeError(
                f"Minted capability URL did not use the federated parent route: {_redacted_url(base_url)}"
            )

        response = subprocess.run(
            [
                "curl",
                "--fail-with-body",
                "--location",
                "--max-time",
                str(REQUEST_TIMEOUT_SECONDS),
                "--request",
                "GET",
                "--show-error",
                "--silent",
                "--write-out",
                f"{CURL_METADATA_MARKER}%{{http_code}}\t%{{content_type}}\t%{{url_effective}}",
                session.model.endpoint.url("models"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        body, marker, metadata = response.stdout.rpartition(CURL_METADATA_MARKER)
        if not marker:
            raise RuntimeError(f"curl returned no response metadata: {response.stderr.strip()}")
        status_text, content_type, final_url = metadata.split("\t", maxsplit=2)
        status_code = int(status_text)
        diagnostics = {
            "bytes": len(body.encode()),
            "content_type": content_type,
            "curl_exit_code": response.returncode,
            "final_url": _redacted_url(final_url),
            "status_code": status_code,
        }
        if response.returncode != 0:
            raise RuntimeError(
                f"Unauthenticated curl failed: {json.dumps(diagnostics)}; stderr={response.stderr.strip()}"
            )
        if not body or "application/json" not in content_type:
            raise RuntimeError(f"Federated inference returned a non-JSON response: {json.dumps(diagnostics)}")
        payload = json.loads(body)
        logger.info(
            "FEDERATED_INFERENCE_PROXY_DEMO %s",
            json.dumps(
                {
                    "capability_url": _redacted_url(base_url),
                    "client": "curl",
                    "inference_job": str(session.jobs[0].job_id),
                    "model": payload["data"][0]["id"],
                    "status_code": status_code,
                },
                sort_keys=True,
            ),
        )


if __name__ == "__main__":
    main()
