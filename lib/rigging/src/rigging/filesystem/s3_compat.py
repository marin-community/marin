# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fsspec/boto setup for S3-compatible object stores: credentials, endpoints, and
request bounds.

CoreWeave AI Object Storage (and R2) speak the S3 protocol but need three things plain AWS
environment variables cannot fully express together: a custom endpoint, virtual-hosted
addressing, and region-less signing. Their connections also need finite request bounds so a
wedged object-store call can fail into the task retry path.

Two ways to apply that setup, for two different callers:

* :func:`configure_fsspec_s3` / :func:`configure_coreweave_s3` configure endpoint
  variables and the ``FSSPEC_S3`` block. Explicit credentials stay in fsspec's
  in-memory configuration. One endpoint per process suits a task pinned to one cluster.
* :func:`s3_credentials` and :func:`s3_endpoint` resolve one backend's settings without touching
  the environment, so a single process can hold connections to several backends at once. This is
  what :mod:`rigging.filesystem.buckets` builds per-bucket filesystems from.
  Routed CoreWeave clients can select the SDK's refreshable container provider through
  :func:`coreweave_federation`.

Endpoints and credential variables are not hardcoded here: each backend declares them under
``data.stores`` in its cluster config (``config/*.yaml``), read through
:func:`rigging.filesystem.cluster_config.store_config`. For Marin that means ``CW_KEY_ID`` / ``CW_KEY_SECRET``
for CoreWeave and ``R2_KEY_ID`` / ``R2_KEY_SECRET`` for R2, each falling back to the generic
``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` pair. A process talking to both backends at
once must use the namespaced pairs, since the two have distinct keys.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp
import fsspec
import s3fs
from aiobotocore.credentials import AioContainerProvider, AioCredentialResolver
from aiobotocore.httpsession import AIOHTTPSession
from aiobotocore.session import AioSession
from botocore.credentials import SharedCredentialProvider

from rigging.filesystem.cluster_config import BucketSpec, StoreType, data_buckets, store_config
from rigging.filesystem.listing_cache import DEFAULT_LISTINGS_EXPIRY_TIME, configured_listing_cache_options
from rigging.secrets import get_secret

# Endpoint domains that reject path-style requests outright.
VIRTUAL_HOST_ONLY_S3_DOMAINS = ("cwobject.com", "cwlota.com")
_COREWEAVE_API_HOST = "api.coreweave.com"
_CONTAINER_CREDENTIALS_URI = "AWS_CONTAINER_CREDENTIALS_FULL_URI"
_CONTAINER_AUTHORIZATION_FILE = "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"

_S3_CONNECT_TIMEOUT = 30
_S3_READ_TIMEOUT = 120
S3_RETRY_MAX_ATTEMPTS = 5

# botocore's default pool of 10 serializes any caller fanning listing or transfer
# threads over one filesystem. Connections are created on demand, so a high cap
# costs nothing when idle.
_S3_MAX_POOL_CONNECTIONS = 128
# Whole-request ceiling. It spans pool wait, connect, body upload, and response
# download, so it bounds every S3 request rather than only a stalled one: a
# genuine transfer slower than 600s now fails and is retried from byte zero, up
# to S3_RETRY_MAX_ATTEMPTS times. 600 leaves ample headroom for the largest
# request we issue (a 50 MiB multipart part needs ~90 KiB/s to fit), while still
# failing a wedged request inside a shard's useful lifetime. Single-object reads
# of many GB through this filesystem are the case to re-check if this bites.
_S3_TOTAL_TIMEOUT = 600


class TotalDeadlineAIOHTTPSession(AIOHTTPSession):
    """aiobotocore HTTP session that also bounds the request as a whole.

    aiobotocore builds ``ClientTimeout(sock_connect=..., sock_read=...)`` and
    never sets ``total``. botocore sends ``Expect: 100-continue`` on
    ``UploadPart``, so aiohttp waits for the interim response before it writes
    the body. None of the scalar bounds covers that wait: ``sock_read`` cannot
    arm because no read is in flight. A peer that withholds ``100 Continue``
    therefore leaves no timer scheduled at all -- the loop parks in
    ``epoll_wait(timeout=-1)`` and the fsspec caller polls forever, with the
    body still unsent (#6719). ``total`` is the only bound that ends it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # aiobotocore annotates the ``timeout`` constructor parameter as a scalar
        # but stores the assembled ClientTimeout in the same attribute.
        # pyrefly: ignore[bad-assignment]
        self._timeout = aiohttp.ClientTimeout(
            sock_connect=self._timeout.sock_connect,
            sock_read=self._timeout.sock_read,
            total=_S3_TOTAL_TIMEOUT,
        )


def s3_request_bounds_config_kwargs() -> dict[str, Any]:
    """Return finite botocore timeouts and retries for fsspec S3 filesystems.

    s3fs/aiobotocore otherwise permit upload requests to wait indefinitely when a
    connection wedges. Finite bounds turn that stall into an error handled by the
    task retry path (#6487).

    JSON-safe by construction: this feeds the ``FSSPEC_S3`` environment block the
    Iris controller exports to every task. Callers building a filesystem in
    Python want :func:`s3_python_config_kwargs` instead.
    """
    return {
        "connect_timeout": _S3_CONNECT_TIMEOUT,
        "read_timeout": _S3_READ_TIMEOUT,
        "retries": {"max_attempts": S3_RETRY_MAX_ATTEMPTS, "mode": "standard"},
        "max_pool_connections": _S3_MAX_POOL_CONNECTIONS,
    }


def s3_python_config_kwargs() -> dict[str, Any]:
    """Return :func:`s3_request_bounds_config_kwargs` plus the whole-request deadline.

    For callers that build a filesystem in Python and can therefore pass a class.
    Not JSON-serializable, so it must not reach the ``FSSPEC_S3`` env block.
    """
    return {
        **s3_request_bounds_config_kwargs(),
        "http_session_cls": TotalDeadlineAIOHTTPSession,
    }


def needs_virtual_host_addressing(endpoint_url: str) -> bool:
    """True when *endpoint_url* is served by a domain that only accepts virtual-hosted requests."""
    hostname = urlparse(endpoint_url).hostname or ""
    return any(hostname == domain or hostname.endswith("." + domain) for domain in VIRTUAL_HOST_ONLY_S3_DOMAINS)


def fsspec_s3_conf(endpoint: str) -> dict:
    """The ``FSSPEC_S3`` config block for *endpoint*: virtual-hosted addressing where the domain
    demands it, region-less ("auto") signing, finite request bounds, and fresh listings.

    Non-AWS S3-compatible endpoints (R2, CoreWeave Object Storage) don't honor the AWS region
    scheme; signing with the wrong region surfaces as 400 Bad Request. "auto" tells boto3 to skip
    region validation and let the endpoint route the request itself.
    """
    config_kwargs = s3_request_bounds_config_kwargs()
    if needs_virtual_host_addressing(endpoint):
        config_kwargs["s3"] = {"addressing_style": "virtual"}
    return {
        "endpoint_url": endpoint,
        "client_kwargs": {"region_name": "auto"},
        "config_kwargs": config_kwargs,
        "listings_expiry_time": DEFAULT_LISTINGS_EXPIRY_TIME,
    }


def configure_fsspec_s3(endpoint: str, key: str | None = None, secret: str | None = None) -> None:
    """Point ``s3://`` access (fsspec and boto alike) at an S3-compatible *endpoint*.

    Sets endpoint variables and ``FSSPEC_S3`` without overwriting existing
    settings. Credentials stay in memory.
    """
    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)
    os.environ.setdefault("AWS_REGION", "auto")
    os.environ.setdefault("AWS_DEFAULT_REGION", "auto")

    if "FSSPEC_S3" not in os.environ:
        conf = fsspec_s3_conf(endpoint)
        configured_cache = configured_listing_cache_options(("s3", "s3a"))
        if configured_cache:
            conf.pop("listings_expiry_time")
            conf.update(configured_cache)
        os.environ["FSSPEC_S3"] = json.dumps(conf)

    # Flush fsspec/s3fs cached instances so they pick up the new config.
    fsspec.config.set_conf_env(fsspec.config.conf)
    if key and secret:
        config = fsspec.config.conf.setdefault("s3", {})
        config.setdefault("key", key)
        config.setdefault("secret", secret)
    s3fs.S3FileSystem.clear_instance_cache()


def configure_coreweave_s3() -> None:
    """Configure ``s3://`` access using named CoreWeave credentials from rigging.

    For processes outside a CoreWeave cluster (a dev box, a dashboard) that read or write CW
    buckets. No-op when the keys are absent or an ``FSSPEC_S3`` block is already exported (a CW
    pod's runtime config wins).
    """
    credentials = s3_credentials(StoreType.COREWEAVE)
    if credentials is None:
        return
    key, secret = credentials
    configure_fsspec_s3(s3_endpoint(StoreType.COREWEAVE), key=key, secret=secret)


def s3_credential_options() -> dict[str, str]:
    """Read a complete configured credential set or the named AWS secrets."""
    configured = fsspec.config.conf.get("s3") or {}
    if configured.get("key") and configured.get("secret"):
        return {name: configured[name] for name in ("key", "secret", "token") if configured.get(name)}
    return {
        option: value
        for option, name in (
            ("key", "AWS_ACCESS_KEY_ID"),
            ("secret", "AWS_SECRET_ACCESS_KEY"),
            ("token", "AWS_SESSION_TOKEN"),
        )
        if (value := get_secret(name))
    }


@dataclass(frozen=True)
class CoreWeaveFederation:
    """Settings for a native SDK to exchange a rotating OIDC token for S3 credentials."""

    uri: str
    token_file: str


def coreweave_federation() -> CoreWeaveFederation | None:
    """Read the provider configuration when the CoreWeave API origin is selected."""
    uri = os.environ.get(_CONTAINER_CREDENTIALS_URI)
    if uri is None:
        return None
    parsed = urlparse(uri)
    if parsed.hostname != _COREWEAVE_API_HOST:
        return None
    if parsed.scheme != "https" or parsed.username is not None or parsed.port not in (None, 443):
        raise ValueError(f"{_CONTAINER_CREDENTIALS_URI} must use the CoreWeave API HTTPS origin")
    token_file = os.environ.get(_CONTAINER_AUTHORIZATION_FILE)
    if not token_file:
        raise ValueError(f"{_CONTAINER_AUTHORIZATION_FILE} is required for CoreWeave federation")
    return CoreWeaveFederation(uri, os.path.abspath(token_file))


def coreweave_federated_bucket(bucket: str) -> tuple[BucketSpec, CoreWeaveFederation] | None:
    """Resolve federation settings for a bucket declared as CoreWeave."""
    if not os.environ.get(_CONTAINER_CREDENTIALS_URI):
        return None
    spec = data_buckets().get(bucket)
    if spec is None or spec.store is not StoreType.COREWEAVE:
        return None
    federation = coreweave_federation()
    return (spec, federation) if federation is not None else None


def coreweave_federated_s3_options(spec: BucketSpec, federation: CoreWeaveFederation) -> dict[str, Any]:
    """Build a routed CoreWeave filesystem configuration with refreshable credentials."""
    endpoint = s3_endpoint(spec.store)
    conf = fsspec_s3_conf(endpoint)
    return {
        "endpoint_url": endpoint,
        "client_kwargs": {"region_name": spec.signing_region},
        "config_kwargs": {**conf["config_kwargs"], **s3_python_config_kwargs()},
        "session": _coreweave_federated_session(os.getpid(), federation),
        # Ambient fsspec options can bypass the scoped credential provider.
        "key": None,
        "secret": None,
        "token": None,
        "anon": False,
    }


@functools.lru_cache
def _coreweave_federated_session(_pid: int, federation: CoreWeaveFederation) -> AioSession:
    # Keep fsspec's instance cache effective and never reuse SDK state across a fork.
    # The provider reopens the token file on refresh, so rotating its contents needs no new session.
    session = AioSession(session_vars={"profile": (None, None, None, None)})
    provider = AioContainerProvider(
        environ={_CONTAINER_CREDENTIALS_URI: federation.uri, _CONTAINER_AUTHORIZATION_FILE: federation.token_file}
    )
    session.register_component("credential_provider", AioCredentialResolver([provider]))
    return session


def s3_credentials(store: StoreType) -> tuple[str, str] | None:
    """Resolve a store's key/secret pair, or return ``None`` when unavailable.

    Reads the secrets the backend's :class:`~rigging.filesystem.cluster_config.StoreConfig` names, then
    the generic ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` and the native
    ``AWS_SHARED_CREDENTIALS_FILE`` profile. Callers that cannot
    proceed without credentials raise their own error, so a browser can list the buckets it
    *can* reach instead of failing on the first unconfigured backend.

    Raises:
        ValueError: if no cluster config declares *store* as an S3-compatible backend.
    """
    config = store_config(store)
    key = get_secret(config.key_id_env) or get_secret("AWS_ACCESS_KEY_ID")
    secret = get_secret(config.key_secret_env) or get_secret("AWS_SECRET_ACCESS_KEY")
    if key and secret:
        return key, secret
    if filename := os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
        credentials = SharedCredentialProvider(
            creds_filename=filename, profile_name=os.environ.get("AWS_PROFILE", "default")
        ).load()
        if credentials is not None:
            frozen = credentials.get_frozen_credentials()
            if frozen.access_key and frozen.secret_key:
                return frozen.access_key, frozen.secret_key
    return None


def s3_endpoint(store: StoreType) -> str:
    """The endpoint URL for *store*: its config's override variable if set, else the configured
    default.

    Raises:
        ValueError: if no cluster config declares *store* as an S3-compatible backend.
    """
    config = store_config(store)
    return os.environ.get(config.endpoint_env) or config.endpoint


def credentials_hint(store: StoreType) -> str:
    """Name the secrets needed to authenticate to *store*."""
    config = store_config(store)
    return f"provide {config.key_id_env} and {config.key_secret_env} secrets (or the generic AWS_* pair)"
