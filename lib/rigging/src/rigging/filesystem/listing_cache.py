# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-wide fsspec directory-listing cache defaults."""

import fsspec

DEFAULT_LISTINGS_EXPIRY_TIME = 0

_CLOUD_PROTOCOL_GROUPS = (("gs", "gcs"), ("s3", "s3a"))
_LISTING_CACHE_OPTIONS = frozenset({"use_listings_cache", "listings_expiry_time", "cache_timeout"})


def configure_listing_cache_defaults() -> None:
    """Expire cloud listings immediately unless the process configured a cache.

    fsspec applies protocol config to raw constructors as well as Marin's guarded
    factories. Long-lived GCS and S3 instances have otherwise hidden writes from
    other processes (#1632, #7975). A zero-second expiry bounds staleness while
    allowing callers to opt into a finite cache with ``listings_expiry_time`` or
    gcsfs's ``cache_timeout``.
    """
    for protocols in _CLOUD_PROTOCOL_GROUPS:
        configured = any(
            _LISTING_CACHE_OPTIONS.intersection(fsspec.config.conf.get(protocol) or {}) for protocol in protocols
        )
        if configured:
            continue
        fsspec.config.conf.setdefault(protocols[0], {})["listings_expiry_time"] = DEFAULT_LISTINGS_EXPIRY_TIME
