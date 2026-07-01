# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ducky configuration, resolved once at startup from the task environment.

DuckDB's ``httpfs`` cannot consume GCP application-default credentials, so GCS is
reached through the S3-compatible interop API with HMAC keys. R2 and CoreWeave are
both addressed as ``s3://`` with different endpoints; ducky creates one DuckDB
``SECRET`` per backend, each S3 secret ``SCOPE``-d to its bucket prefix so DuckDB
picks the right endpoint per URI:

- ``gs://``            → GCS (HMAC interop key/secret)
- ``s3://<r2-bucket>`` → R2 (S3 secret: endpoint + key/secret, scoped)
- ``s3://<cw-bucket>`` → CoreWeave (S3 secret: endpoint + key/secret, scoped, virtual-host)

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
        "r2_endpoint": "DUCKY_R2_ENDPOINT",
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
    """Service region (e.g. ``us-east5``). Operational pin; the same-region guardrail is
    the allowlist below, since GCS HMAC keys are region-agnostic and can't enforce it."""

    scratch_bucket: str
    """Prefix where full results spill (``gs://…`` in prod, a local dir for smoke). Carries a lifecycle TTL rule."""

    allowed_buckets: tuple[str, ...] = ()
    """Object-store URI prefixes a query may read, e.g. ``("gs://marin-us-east5", "r2://")``.
    A query referencing a ``gs://``/``s3://``/``r2://`` URI outside the allowlist is refused
    before execution — the same-region guardrail. Empty disables enforcement (allow all).
    Catches literal URIs in the SQL, not paths hidden behind views/macros."""

    # Optional per-backend credentials. A backend is enabled only when its full set is present.
    gcs_hmac_key_id: str | None = None
    gcs_hmac_secret: str | None = None
    # R2 and CoreWeave are S3-compatible, addressed as s3:// with their own endpoint; the
    # secret is SCOPE-d to the bucket so DuckDB routes each s3:// URI to the right endpoint.
    r2_endpoint: str | None = None
    r2_access_key: str | None = None
    r2_secret_key: str | None = None
    cw_endpoint: str | None = None
    cw_access_key: str | None = None
    cw_secret_key: str | None = None

    r2_scope: str = "s3://marin-na"
    """DuckDB SECRET scope for the R2 backend (the s3:// bucket prefix it serves)."""
    r2_url_style: str = "path"
    """S3 addressing for R2: ``path`` (R2 account endpoint serves the bucket in the path)."""
    cw_scope: str = "s3://marin-us-east-02a"
    """DuckDB SECRET scope for the CoreWeave backend."""
    cw_url_style: str = "vhost"
    """S3 addressing for CoreWeave: ``vhost``. CoreWeave endpoints reject path-style."""

    preview_row_cap: int = 10_000
    """Max rows returned inline to the browser. The full result always spills to parquet."""

    memory_fraction: float = 0.6
    """DuckDB ``memory_limit`` = this fraction of host RAM, a hard self-cap. Leaves generous
    headroom (~40%) for concurrent Arrow previews, httpfs read buffers, and untracked
    allocations so the container isn't cgroup-OOM-killed under load — a query that needs more
    fails per-query instead of taking down the service."""

    spill_directory: str = "/var/tmp/ducky-spill"
    """Local disk path DuckDB spills to when a query exceeds ``memory_limit`` (out-of-core
    execution). ``/var/tmp`` rather than ``/tmp`` because ``/tmp`` is often tmpfs — spilling
    there consumes RAM and defeats the point."""

    spill_limit: str = "60GB"
    """Cap on DuckDB's on-disk spill (``max_temp_directory_size``). Bounded well under the
    ~100 GB boot disk so a runaway spill fails *that query* cleanly instead of filling the
    disk and crashing the whole container."""

    max_concurrent_queries: int = 8
    """How many queries run at once. Each gets its own DuckDB cursor (sharing the instance's
    secrets/settings); they share the host's thread pool and memory budget."""

    query_timeout: int = 600
    """Hard per-query wall-clock limit (seconds). A query exceeding it is interrupted and
    fails — so a runaway (e.g. a recursive glob over millions of objects) frees its slot
    instead of holding it forever."""

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
        return bool(self.r2_endpoint and self.r2_access_key and self.r2_secret_key)

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

        allowed = os.environ.get(f"{_ENV_PREFIX}ALLOWED_BUCKETS", "")
        allowed_buckets = tuple(b.strip() for b in allowed.split(",") if b.strip())

        return cls(
            region=region,  # pyrefly: ignore  # non-None after the check above
            scratch_bucket=scratch_bucket,  # pyrefly: ignore  # non-None after the check above
            allowed_buckets=allowed_buckets,
            preview_row_cap=int(os.environ.get(f"{_ENV_PREFIX}PREVIEW_ROW_CAP", cls.preview_row_cap)),
            memory_fraction=float(os.environ.get(f"{_ENV_PREFIX}MEMORY_FRACTION", cls.memory_fraction)),
            max_concurrent_queries=int(
                os.environ.get(f"{_ENV_PREFIX}MAX_CONCURRENT_QUERIES", cls.max_concurrent_queries)
            ),
            spill_directory=os.environ.get(f"{_ENV_PREFIX}SPILL_DIR", cls.spill_directory),
            spill_limit=os.environ.get(f"{_ENV_PREFIX}SPILL_LIMIT", cls.spill_limit),
            query_timeout=int(os.environ.get(f"{_ENV_PREFIX}QUERY_TIMEOUT", cls.query_timeout)),
            result_ttl_days=int(os.environ.get(f"{_ENV_PREFIX}RESULT_TTL_DAYS", cls.result_ttl_days)),
            endpoint_name=os.environ.get(f"{_ENV_PREFIX}ENDPOINT_NAME", cls.endpoint_name),
            r2_scope=os.environ.get(f"{_ENV_PREFIX}R2_SCOPE", cls.r2_scope),
            r2_url_style=os.environ.get(f"{_ENV_PREFIX}R2_URL_STYLE", cls.r2_url_style),
            cw_scope=os.environ.get(f"{_ENV_PREFIX}CW_SCOPE", cls.cw_scope),
            cw_url_style=os.environ.get(f"{_ENV_PREFIX}CW_URL_STYLE", cls.cw_url_style),
            **creds,
        )
