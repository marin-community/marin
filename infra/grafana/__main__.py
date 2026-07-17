# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for the Grafana Cloud Run service.

Deploys this directory (Grafana + the finelog bridge) as an IAP-gated Cloud Run service
through the reusable `iac.gcp.cloud_run.CloudRunService` component. Grafana's fixed shape
— project, region, one warm instance — lives here; the list of people admitted through
IAP is stack config (`marin-grafana:viewers`).

Runs on the shared repo venv (plain `python` runtime), which is where `iac` and the Pulumi
GCP/Docker providers live; `uv sync --all-packages` first. See README.md.
"""

import os

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs

# Kept in lockstep with the bridge's config.py, which pins the same project for the finelog
# VM lookup — deploying elsewhere while still reading hai-gcp-models would silently break.
PROJECT = "hai-gcp-models"
REGION = "us-central1"
SERVICE = "marin-grafana"

# This file sits beside the Dockerfile, dashboards, and bridge source; the whole directory
# is the image build context.
BUILD_CONTEXT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    config = pulumi.Config()
    # IAM members admitted through IAP, e.g. group:marin@…; set with
    #   pulumi config set --path 'viewers[0]' group:someone@example.com
    viewers = config.get_object("viewers") or []

    provider = gcp.Provider("gcp", project=PROJECT)
    service = CloudRunService(
        "grafana",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=SERVICE,
            build_context=BUILD_CONTEXT,
            # The bridge lists finelog VM internal IPs through the Compute API.
            service_account_roles=("roles/compute.viewer",),
            iap_members=tuple(viewers),
        ),
        gcp_provider=provider,
    )
    pulumi.export("url", service.uri)
    pulumi.export("image", service.image_ref)


main()
