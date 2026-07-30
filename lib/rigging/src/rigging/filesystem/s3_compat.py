# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fsspec/boto setup for S3-compatible object stores: credentials, endpoints, and
request bounds.

CoreWeave AI Object Storage (and R2) speak the S3 protocol but need three things plain AWS
environment variables cannot fully express together: a custom endpoint, virtual-hosted
addressing, and region-less signing. Their connections also need finite request bounds so a
wedged object-store call can fail into the task retry path.

Two ways to apply that setup, for two different callers:

* :func:`configure_fsspec_s3` / :func:`configure_coreweave_s3` write the process-wide
  ``AWS_*`` variables and the ``FSSPEC_S3`` config block that :mod:`rigging.filesystem.factory`
  and every plain ``fsspec`` caller read. One endpoint per process — right for a task pinned to
  one cluster. Processes inside a cluster usually arrive with ``FSSPEC_S3`` already exported by
  the runtime, in which case these are no-ops.
* :func:`s3_credentials` and :func:`s3_endpoint` resolve one backend's settings without touching
  the environment, so a single process can hold connections to several backends at once. This is
  what :mod:`rigging.filesystem.buckets` builds per-bucket filesystems from.

Endpoints and credential variables are not hardcoded here: each backend declares them under
``data.stores`` in its cluster config (``config/*.yaml``), read through
:func:`rigging.filesystem.store_config`. For Marin that means ``CW_KEY_ID`` / ``CW_KEY_SECRET``
for CoreWeave and ``R2_KEY_ID`` / ``R2_KEY_SECRET`` for R2, each falling back to the generic
``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` pair. A process talking to both backends at
once must use the namespaced pairs, since the two have distinct keys.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

import aiohttp
import fsspec
import s3fs
from aiobotocore.httpsession import AIOHTTPSession

from rigging.filesystem.cluster_config import StoreType, store_config

# Endpoint domains that reject path-style requests outright.
VIRTUAL_HOST_ONLY_S3_DOMAINS = ("cwobject.com", "cwlota.com")

_S3_CONNECT_TIMEOUT = 30
_S3_READ_TIMEOUT = 120
_S3_RETRY_MAX_ATTEMPTS = 5
# Whole-request ceiling, well above any healthy transfer: a 50 MiB multipart
# part needs ~437 KiB/s to finish inside it. Only a wedged request reaches it.
_S3_TOTAL_TIMEOUT = 600


class TotalDeadlineAIOHTTPSession(AIOHTTPSession):
    """aiobotocore HTTP session that also bounds the request as a whole.

    aiobotocore builds ``ClientTimeout(sock_connect=..., sock_read=...)`` and
    never sets ``total``. ``sock_read`` only limits the gap between received
    bytes *after* a response has started, so a peer that accepts a request body
    and then answers nothing leaves the caller with no asyncio timer armed at
    all: the event loop parks in ``epoll_wait(timeout=-1)`` and the fsspec
    caller polls forever. ``total`` is the only bound that covers the wait for
    the first response byte (#6719).
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

    JSON-safe by construction: this feeds the ``FSSPEC_S3`` environment block
    that the Iris controller exports to every task. The whole-request deadline
    cannot travel that way, because it needs a class rather than a scalar — see
    :func:`s3_python_config_kwargs`.
    """
    return {
        "connect_timeout": _S3_CONNECT_TIMEOUT,
        "read_timeout": _S3_READ_TIMEOUT,
        "retries": {"max_attempts": _S3_RETRY_MAX_ATTEMPTS, "mode": "standard"},
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
    demands it, region-less ("auto") signing, and finite request bounds.

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
    }


def configure_fsspec_s3(endpoint: str, key: str | None = None, secret: str | None = None) -> None:
    """Point ``s3://`` access (fsspec and boto alike) at an S3-compatible *endpoint*.

    Sets the ``AWS_*`` credential/endpoint variables and the ``FSSPEC_S3`` config block, never
    overwriting values already present, then flushes fsspec's config and s3fs's instance cache so
    already-imported modules pick the settings up.
    """
    if key and secret:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret)

    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)
    os.environ.setdefault("AWS_REGION", "auto")
    os.environ.setdefault("AWS_DEFAULT_REGION", "auto")

    if "FSSPEC_S3" not in os.environ:
        os.environ["FSSPEC_S3"] = json.dumps(fsspec_s3_conf(endpoint))

    # Flush fsspec/s3fs cached instances so they pick up the new config.
    fsspec.config.set_conf_env(fsspec.config.conf)
    s3fs.S3FileSystem.clear_instance_cache()


def configure_coreweave_s3() -> None:
    """Configure ``s3://`` access to CoreWeave object storage from ambient ``CW_KEY_*`` credentials.

    For processes outside a CoreWeave cluster (a dev box, a dashboard) that read or write CW
    buckets. No-op when the keys are absent or an ``FSSPEC_S3`` block is already exported (a CW
    pod's runtime config wins).
    """
    credentials = s3_credentials(StoreType.COREWEAVE)
    if credentials is None:
        return
    key, secret = credentials
    configure_fsspec_s3(s3_endpoint(StoreType.COREWEAVE), key=key, secret=secret)


def s3_credentials(store: StoreType) -> tuple[str, str] | None:
    """The ``(key_id, secret)`` pair for *store*, or ``None`` when neither pair is complete.

    Reads the variables the backend's :class:`~rigging.filesystem.StoreConfig` names, then
    the generic ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``. Callers that cannot
    proceed without credentials raise their own error, so a browser can list the buckets it
    *can* reach instead of failing on the first unconfigured backend.

    Raises:
        ValueError: if no cluster config declares *store* as an S3-compatible backend.
    """
    config = store_config(store)
    key = os.environ.get(config.key_id_env) or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get(config.key_secret_env) or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        return None
    return key, secret


def s3_endpoint(store: StoreType) -> str:
    """The endpoint URL for *store*: its config's override variable if set, else the configured
    default.

    Raises:
        ValueError: if no cluster config declares *store* as an S3-compatible backend.
    """
    config = store_config(store)
    return os.environ.get(config.endpoint_env) or config.endpoint


def credentials_hint(store: StoreType) -> str:
    """A one-line "set these variables" hint naming *store*'s credential environment pair."""
    config = store_config(store)
    return f"set {config.key_id_env} and {config.key_secret_env} (or the generic AWS_* pair)"
