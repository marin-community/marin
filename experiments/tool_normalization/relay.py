# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover the authenticated shared GLM relay."""

import json
import os
import urllib.request
from pathlib import Path

from iris.client.client import iris_ctx
from iris.cluster.types import JobName

RELAY_JOB = "/muchanem/glm53-relay-rno2a-g-20261027T0414Z"


def resolve_relay_base_url(token: str, job: str = RELAY_JOB) -> str:
    """Resolve the relay's registered glm-5.3 endpoint and confirm it serves the model."""
    client = iris_ctx().client
    if client is None:
        raise RuntimeError("Iris client is unavailable inside the task")
    endpoints = client.resolver_for_job(JobName.from_string(job)).resolve("glm-5.3").endpoints
    if not endpoints:
        raise RuntimeError("The GLM relay has no registered endpoint")
    base_url = endpoints[0].url.rstrip("/")
    base_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
    request = urllib.request.Request(base_url + "/models", headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=30) as response:
        models = {entry["id"] for entry in json.load(response)["data"]}
    if "glm-5.3" not in models:
        raise RuntimeError(f"Relay does not serve glm-5.3: {sorted(models)}")
    return base_url


def read_bulk_token() -> str:
    secret_file = Path(os.environ.get("RIGGING_SECRETS_DIR", "/run/rigging/secrets")) / "GLM_BULK_TOKEN"
    if secret_file.exists():
        token = secret_file.read_text().strip()
        if token:
            return token
    token = os.environ.get("GLM_BULK_TOKEN", "").strip()
    if token:
        return token
    raise RuntimeError("GLM_BULK_TOKEN not provided via secret file or environment")
