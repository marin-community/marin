# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marina: one Cloud Run service serving every app under infra/marina/apps.

The image is built from the repository root so it can install lib/rigging. IAP is the
front door; the container verifies IAP's signed assertion against the service's audience
so each request carries the caller's email. IAM grants (deploy pushes, IAP viewers) are
declared in the ``marin`` stack by iac.gcp.marina.
"""

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs

PROJECT = "hai-gcp-models"
REGION = "us-central1"
SERVICE = "marina"


def iap_audience(project_number: str) -> str:
    """The ``aud`` claim IAP signs into X-Goog-IAP-JWT-Assertion for a Cloud Run service."""
    return f"/projects/{project_number}/locations/{REGION}/services/{SERVICE}"


def main() -> None:
    gcp_provider = gcp.Provider("gcp", project=PROJECT)
    project_number = gcp.organizations.get_project(
        project_id=PROJECT, opts=pulumi.InvokeOptions(provider=gcp_provider)
    ).number

    service = CloudRunService(
        "service",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=SERVICE,
            build_context="../..",
            dockerfile="infra/marina/Dockerfile",
            env={"MARINA_IAP_AUDIENCE": iap_audience(project_number)},
            min_instances=1,
            max_instances=2,
            max_instance_request_concurrency=40,
            cpu="1",
            memory="1Gi",
        ),
        gcp_provider=gcp_provider,
    )
    pulumi.export("uri", service.uri)


main()
