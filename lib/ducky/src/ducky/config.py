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

Each backend is optional: it is enabled only when its full credential set is
present, which lets a cred-free smoke deploy query DuckDB built-ins and spill to a
local scratch dir. ``region`` and ``scratch_bucket`` are always required.
"""

from __future__ import annotations

import dataclasses
import os

_ENV_PREFIX = "DUCKY_"

# field name -> env var, grouped per backend. A backend is enabled iff every var in
# its group is set; a partially-set group is a misconfiguration (raises).
_BACKEND_ENV = {
    "gcs": {"gcs_hmac_key_id": "DUCKY_GCS_HMAC_KEY_ID", "gcs_hmac_secret": "DUCKY_GCS_HMAC_SECRET"},
    "r2": {
        "r2_account_id": "DUCKY_R2_ACCOUNT_ID",
        "r2_access_key": "DUCKY_R2_ACCESS_KEY",
        "r2_secret_key": "DUCKY_R2_SECRET_KEY",
    },
    "cw": {
        "cw_endpoint": "DUCKY_CW_ENDPOINT",
        "cw_access_key": "DUCKY_CW_ACCESS_KEY",
        "cw_secret_key": "DUCKY_CW_SECRET_KEY",
    },
}


@dataclasses.dataclass(frozen=True)
class DuckyConfig:
    """Resolved ducky configuration. Construct directly, or via :meth:`from_environment`."""

    region: str
    """Service region (e.g. ``us-east5``). Operational pin only; not enforced in the runner."""

    scratch_bucket: str
    """Prefix where full results spill (``gs://…`` in prod, a local dir for smoke). Carries a lifecycle TTL rule."""

    # Optional per-backend credentials. A backend is enabled only when its full set is present.
    gcs_hmac_key_id: str | None = None
    gcs_hmac_secret: str | None = None
    r2_account_id: str | None = None
    r2_access_key: str | None = None
    r2_secret_key: str | None = None
    cw_endpoint: str | None = None
    cw_access_key: str | None = None
    cw_secret_key: str | None = None

    cw_url_style: str = "vhost"
    """S3 addressing for CoreWeave: ``vhost`` (bucket in host) or ``path``. CoreWeave's
    endpoints reject path-style, so default to virtual-hosted."""

    preview_row_cap: int = 10_000
    """Max rows returned inline to the browser. The full result always spills to parquet."""

    memory_fraction: float = 0.8
    """DuckDB ``memory_limit`` = this fraction of host RAM, leaving headroom for Python/Arrow/OS."""

    result_ttl_days: int = 7
    """Informational — enforced by the scratch bucket's lifecycle rule, not by ducky (ducky only writes)."""

    port_name: str = "ducky"
    """Iris named port, bound via ``ctx.get_port(port_name)``."""

    endpoint_name: str = "/ducky"
    """Endpoint registry name. A leading slash registers a cluster-global (non-namespaced)
    endpoint so the dashboard is reachable at ``/proxy/ducky/`` instead of a per-job path."""

    @property
    def gcs_enabled(self) -> bool:
        return bool(self.gcs_hmac_key_id and self.gcs_hmac_secret)

    @property
    def r2_enabled(self) -> bool:
        return bool(self.r2_account_id and self.r2_access_key and self.r2_secret_key)

    @property
    def cw_enabled(self) -> bool:
        return bool(self.cw_endpoint and self.cw_access_key and self.cw_secret_key)

    @classmethod
    def from_environment(cls) -> DuckyConfig:
        """Build from ``DUCKY_*`` env vars.

        ``DUCKY_REGION`` and ``DUCKY_SCRATCH_BUCKET`` are required. Each backend's
        credentials are optional but all-or-nothing: a partially-configured backend
        raises ``ValueError`` rather than silently disabling itself.
        """
        region = os.environ.get(f"{_ENV_PREFIX}REGION")
        scratch_bucket = os.environ.get(f"{_ENV_PREFIX}SCRATCH_BUCKET")
        missing = [
            env_key
            for env_key, value in [(f"{_ENV_PREFIX}REGION", region), (f"{_ENV_PREFIX}SCRATCH_BUCKET", scratch_bucket)]
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required ducky env vars: {', '.join(missing)}")

        creds: dict[str, str | None] = {}
        for backend, env_map in _BACKEND_ENV.items():
            present = {field: os.environ.get(env_key) for field, env_key in env_map.items()}
            set_count = sum(1 for value in present.values() if value)
            if set_count == 0:
                creds.update(dict.fromkeys(env_map, None))
            elif set_count == len(env_map):
                creds.update(present)
            else:
                raise ValueError(
                    f"Backend {backend!r} is partially configured; set all or none of {list(env_map.values())}"
                )

        return cls(
            region=region,  # pyrefly: ignore  # non-None after the check above
            scratch_bucket=scratch_bucket,  # pyrefly: ignore  # non-None after the check above
            preview_row_cap=int(os.environ.get(f"{_ENV_PREFIX}PREVIEW_ROW_CAP", cls.preview_row_cap)),
            memory_fraction=float(os.environ.get(f"{_ENV_PREFIX}MEMORY_FRACTION", cls.memory_fraction)),
            result_ttl_days=int(os.environ.get(f"{_ENV_PREFIX}RESULT_TTL_DAYS", cls.result_ttl_days)),
            endpoint_name=os.environ.get(f"{_ENV_PREFIX}ENDPOINT_NAME", cls.endpoint_name),
            cw_url_style=os.environ.get(f"{_ENV_PREFIX}CW_URL_STYLE", cls.cw_url_style),
            **creds,
        )
