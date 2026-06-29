# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ducky configuration, resolved once at startup from the task environment.

DuckDB's ``httpfs`` cannot consume GCP application-default credentials, so GCS is
reached through the S3-compatible interop API with HMAC keys. R2 and CoreWeave are
native S3 providers; ducky keys each backend to a distinct URL scheme so a single
DuckDB ``SECRET`` per backend disambiguates without bucket-level scoping:

- ``gs://`` → GCS (HMAC interop key/secret)
- ``r2://`` → R2 (account id + S3 key/secret)
- ``s3://`` → CoreWeave object store (endpoint + S3 key/secret)
"""

from __future__ import annotations

import dataclasses
import os

_ENV_PREFIX = "DUCKY_"


@dataclasses.dataclass(frozen=True)
class DuckyConfig:
    """Resolved ducky configuration. Construct directly, or via :meth:`from_environment`."""

    region: str
    """Service region (e.g. ``us-east5``). Operational pin only; not enforced in the runner."""

    scratch_bucket: str
    """``gs://`` prefix where full results spill. Lives in ``region``; carries a lifecycle TTL rule."""

    gcs_hmac_key_id: str
    gcs_hmac_secret: str
    r2_account_id: str
    r2_access_key: str
    r2_secret_key: str
    cw_endpoint: str
    cw_access_key: str
    cw_secret_key: str

    preview_row_cap: int = 10_000
    """Max rows returned inline to the browser. The full result always spills to parquet."""

    memory_fraction: float = 0.8
    """DuckDB ``memory_limit`` = this fraction of host RAM, leaving headroom for Python/Arrow/OS."""

    result_ttl_days: int = 7
    """Informational — enforced by the scratch bucket's lifecycle rule, not by ducky (ducky only writes)."""

    port_name: str = "ducky"
    """Iris named port, bound via ``ctx.get_port(port_name)``."""

    @classmethod
    def from_environment(cls) -> DuckyConfig:
        """Build from ``DUCKY_*`` env vars. Raise ``ValueError`` listing any missing required keys."""
        required = {
            "region": "DUCKY_REGION",
            "scratch_bucket": "DUCKY_SCRATCH_BUCKET",
            "gcs_hmac_key_id": "DUCKY_GCS_HMAC_KEY_ID",
            "gcs_hmac_secret": "DUCKY_GCS_HMAC_SECRET",
            "r2_account_id": "DUCKY_R2_ACCOUNT_ID",
            "r2_access_key": "DUCKY_R2_ACCESS_KEY",
            "r2_secret_key": "DUCKY_R2_SECRET_KEY",
            "cw_endpoint": "DUCKY_CW_ENDPOINT",
            "cw_access_key": "DUCKY_CW_ACCESS_KEY",
            "cw_secret_key": "DUCKY_CW_SECRET_KEY",
        }
        values = {field: os.environ.get(env_key) for field, env_key in required.items()}
        missing = [required[field] for field, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing required ducky env vars: {', '.join(sorted(missing))}")

        return cls(
            preview_row_cap=int(os.environ.get(f"{_ENV_PREFIX}PREVIEW_ROW_CAP", cls.preview_row_cap)),
            memory_fraction=float(os.environ.get(f"{_ENV_PREFIX}MEMORY_FRACTION", cls.memory_fraction)),
            result_ttl_days=int(os.environ.get(f"{_ENV_PREFIX}RESULT_TTL_DAYS", cls.result_ttl_days)),
            **values,  # pyrefly: ignore  # required keys are non-None after the check above
        )
