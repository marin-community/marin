# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for the Marin infra status dashboard.

Deploys this directory (the Hono + React status page) as an IAP-gated Cloud Run service
through the reusable ``iac.gcp.cloud_run.CloudRunService`` component. The dashboard's fixed
shape — project, region, one warm instance, Direct VPC egress to the iris controller — lives
here; the list of people admitted through IAP is stack config (``marin-infra-dashboard:viewers``).

Runs on the shared repo venv (plain ``python`` runtime), which is where ``iac`` and the Pulumi
GCP/Docker providers live; ``uv sync --all-packages`` first. See README.md.
"""

import os

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv

# Kept in lockstep with the dashboard's controller discovery, which resolves the iris
# controller's internal IP in this project/zone; deploying elsewhere would silently break
# the Iris and Probes panels.
PROJECT = "hai-gcp-models"
REGION = "us-central1"
SERVICE = "marin-infra-dashboard"

# The runtime service account predates the service rename (GCP cannot rename one in place),
# so it keeps its original id and is adopted rather than replaced.
SERVICE_ACCOUNT_ID = "marin-status-page"

# The GitHub token lifts the API rate limit from 60/hr to 5000/hr and is required for the
# Build panel's GraphQL calls. The secret holds the value; Pulumi only grants the runtime
# account read access to it.
GITHUB_TOKEN_SECRET = "marin-status-page-github-token"

# Controller discovery (GCE label -> internal URL) and the Iris panel's display name.
CONTROLLER_ZONE = "us-central1-a"
CONTROLLER_LABEL = "iris-marin-controller"
CONTROLLER_PORT = 10000
CLUSTER_NAME = "marin"

# This file sits beside the Dockerfile and the server/web sources; the whole directory is the
# image build context.
BUILD_CONTEXT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    config = pulumi.Config()
    # IAM members admitted through IAP, e.g. group:marin@…; set with
    #   pulumi config set --path 'viewers[0]' group:someone@example.com
    viewers = config.get_object("viewers") or []

    provider = gcp.Provider("gcp", project=PROJECT)
    service = CloudRunService(
        "status-page",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=SERVICE,
            service_account_id=SERVICE_ACCOUNT_ID,
            build_context=BUILD_CONTEXT,
            cpu="1",
            memory="1Gi",
            # The dashboard lists the iris controller VM's internal IP through the Compute API.
            service_account_roles=("roles/compute.viewer",),
            env={
                "GCP_PROJECT": PROJECT,
                "CONTROLLER_ZONE": CONTROLLER_ZONE,
                "CONTROLLER_LABEL": CONTROLLER_LABEL,
                "CONTROLLER_PORT": str(CONTROLLER_PORT),
                "CLUSTER_NAME": CLUSTER_NAME,
            },
            secrets=(SecretEnv(name="GITHUB_TOKEN", secret=GITHUB_TOKEN_SECRET),),
            iap_members=tuple(viewers),
        ),
        gcp_provider=provider,
    )
    pulumi.export("url", service.uri)
    pulumi.export("image", service.image_ref)


main()
