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
local scratch dir. ``scratch_bucket`` is always required.
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


# Fixed Iris identifiers (not configurable): the named port ducky binds/publishes, and the
# cluster-global endpoint it registers — the leading slash makes it reachable at
# ``/proxy/ducky/`` rather than a per-job path.
PORT_NAME = "ducky"
ENDPOINT_NAME = "/ducky"

# Default object-store roots for the pre-baked catalog (see catalog.py), applied by
# `from_environment` when the DUCKY_* env var is unset; set `DUCKY_FINELOG_ROOT=`/
# `DUCKY_DATAKIT_ROOT=` (empty) to disable a source's views. Catalog views over a root
# outside `allowed_buckets` are skipped, so these defaults never silently egress across
# regions — the allowlist gates them (see QueryRunner._create_catalog_views).
#
# finelog's marin deployment writes to us-central2. datakit normalized parquet lives under
# `<MARIN_PREFIX>/normalized`, so its root is derived from the MARIN_PREFIX env when set
# (the datakit corpus is canonical in eu-west4); the constant is only a last-resort fallback.
DEFAULT_FINELOG_ROOT = "gs://marin-us-central2/finelog/marin"
DEFAULT_DATAKIT_ROOT = "gs://marin-us-east5/normalized"
_MARIN_PREFIX_ENV = "MARIN_PREFIX"


def _resolve_datakit_root() -> str | None:
    """Datakit normalized-parquet root: explicit DUCKY_DATAKIT_ROOT wins (empty disables),
    else ``<MARIN_PREFIX>/normalized`` when MARIN_PREFIX is set, else the fallback constant."""
    explicit = os.environ.get(f"{_ENV_PREFIX}DATAKIT_ROOT")
    if explicit is not None:
        return explicit or None
    marin_prefix = os.environ.get(_MARIN_PREFIX_ENV)
    if marin_prefix:
        return f"{marin_prefix.rstrip('/')}/normalized"
    return DEFAULT_DATAKIT_ROOT


@dataclasses.dataclass(frozen=True)
class DuckyConfig:
    """Resolved ducky configuration. Construct directly, or via :meth:`from_environment`."""

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
    cw_scope: str = "s3://marin-us-east-02a"
    """DuckDB SECRET scope for the CoreWeave backend."""

    finelog_root: str | None = None
    """Object-store root of the finelog store (``<root>/<namespace>/seg_L*.parquet``).
    Pre-baked ``finelog.*`` views are created over it. ``None``/empty skips them.
    `from_environment` supplies :data:`DEFAULT_FINELOG_ROOT`; direct construction defaults
    to ``None`` so unit tests don't reach the network binding views."""
    datakit_root: str | None = None
    """Object-store root of datakit normalized parquet
    (``<root>/<name>_<hash>/outputs/main/*.parquet``). Curated ``datakit.*`` views are
    created over it. ``None``/empty skips them; see :attr:`finelog_root` on the default."""

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

    @property
    def effective_allowed_buckets(self) -> tuple[str, ...]:
        """Object-store prefixes a query may read: the configured allowlist plus the catalog
        roots. Configuring a catalog root (``finelog_root``/``datakit_root``) declares that
        prefix readable, so a pre-baked view and a literal ``read_parquet`` of the same prefix
        behave identically — the view can't reach a bucket a literal URI couldn't. An empty
        allowlist means allow-all and stays empty (adding roots would turn it into a
        restrictive list)."""
        if not self.allowed_buckets:
            return ()
        roots = tuple(root for root in (self.finelog_root, self.datakit_root) if root)
        return self.allowed_buckets + roots

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

        ``DUCKY_SCRATCH_BUCKET`` is required. Each backend's credentials are optional but
        all-or-nothing: a partially-configured backend raises ``ValueError`` rather than
        silently disabling itself.
        """
        scratch_bucket = os.environ.get(f"{_ENV_PREFIX}SCRATCH_BUCKET")
        if not scratch_bucket:
            raise ValueError(f"Missing required ducky env var: {_ENV_PREFIX}SCRATCH_BUCKET")

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
            scratch_bucket=scratch_bucket,
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
            r2_scope=os.environ.get(f"{_ENV_PREFIX}R2_SCOPE", cls.r2_scope),
            cw_scope=os.environ.get(f"{_ENV_PREFIX}CW_SCOPE", cls.cw_scope),
            finelog_root=os.environ.get(f"{_ENV_PREFIX}FINELOG_ROOT", DEFAULT_FINELOG_ROOT) or None,
            datakit_root=_resolve_datakit_root(),
            **creds,
        )
