# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM inference-service discovery shared by post-training experiments."""

from iris.client.client import iris_ctx
from iris.cluster.types import JobName

GLM_BULK_TOKEN_ENV = "GLM_BULK_TOKEN"
GLM_MODEL = "glm-5.3"
DEFAULT_GLM_RELAY_JOB = "/muchanem/glm53-relay-08a"


def resolve_glm_base_url(relay_job: str) -> str:
    """Resolve the OpenAI-compatible GLM endpoint registered by an Iris relay job."""

    client = iris_ctx().client
    if client is None:
        raise RuntimeError("Iris client is unavailable inside the task")
    endpoints = client.resolver_for_job(JobName.from_string(relay_job)).resolve(GLM_MODEL).endpoints
    if not endpoints:
        raise RuntimeError("The GLM relay has no registered endpoint")
    base_url = endpoints[0].url.rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"
