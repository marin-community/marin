# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable identity for resource schema v2."""

from typing import Final

RESOURCE_SCHEMA_EPOCH: Final[int] = 2
RESOURCE_SCHEMA_NAME: Final[str] = "resource_schema_v2"

# Sealed against the exact merge-base schema by the resource migration tests.
# Values are updated only when the accepted source checkout deliberately moves.
MERGE_BASE_SCHEMA_FINGERPRINT: Final[str] = "28b0960a84c65a88cc7510e6bc6745a0113e350589b6a189d99a887c23df3722"
MERGE_BASE_MIGRATION_NAMES: Final[tuple[str, ...]] = (
    "0001_baseline.py",
    "0027_attempt_uid.py",
    "0028_drop_api_keys_key_hash.py",
    "0029_drop_reservations.py",
    "0030_container_profile.py",
    "0031_endpoint_lease.py",
    "0032_worker_provenance.py",
    "0033_backend_id.py",
    "0034_federation.py",
    "0035_federation_unify.py",
    "0036_endpoint_access.py",
    "0037_federation_fixup.py",
    "0038_derive_task_counts.py",
    "0039_drop_api_keys.py",
    "0040_drop_users.py",
    "0041_job_submitting_user.py",
    "0042_drop_backends_table.py",
    "0043_endpoint_peer_id.py",
    "0044_federated_handoff_nonce.py",
    "0045_task_status_message.py",
    "0046_mirrored_job_config_backfill.py",
    "0047_attempt_backend_identity.py",
    "0048_endpoints_drop_job_fk.py",
    "0049_resolve_priority_band.py",
    "0050_drop_controller_secrets.py",
)
