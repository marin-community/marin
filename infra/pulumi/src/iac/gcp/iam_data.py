# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed view of the machine-readable GCP IAM declaration."""

from iac.gcp.iam_config import IAM_DATA_PATH, load_iam_config

CONFIG = load_iam_config(IAM_DATA_PATH)

KMS_LOCATION = CONFIG.kms_location
KMS_KEY_RING = CONFIG.kms_key_ring
KMS_KEY = CONFIG.kms_key
CUSTOM_ROLES = CONFIG.custom_roles
OWNED_SERVICE_ACCOUNTS = CONFIG.owned_service_accounts
PROJECT_GRANTS = CONFIG.project_grants
KMS_GRANTS = CONFIG.kms_grants
SECRETS = CONFIG.secrets
BUCKETS = CONFIG.buckets
ARTIFACT_REPOSITORIES = CONFIG.artifact_repositories
SERVICE_ACCOUNTS = CONFIG.service_accounts
