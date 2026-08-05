# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-wide fsspec directory-listing cache defaults."""

import fsspec

DEFAULT_LISTINGS_EXPIRY_TIME = 0

_CLOUD_PROTOCOL_GROUPS = (("gs", "gcs"), ("s3", "s3a"))
_LISTING_CACHE_OPTIONS = frozenset({"use_listings_cache", "listings_expiry_time", "cache_timeout"})


def configure_listing_cache_defaults() -> None:
    """Set zero-second GCS/S3 listing expiry unless cache options already exist."""
    for protocols in _CLOUD_PROTOCOL_GROUPS:
        configured = any(
            _LISTING_CACHE_OPTIONS.intersection(fsspec.config.conf.get(protocol) or {}) for protocol in protocols
        )
        if configured:
            continue
        # fsspec otherwise caches external writes indefinitely (#1632, #7975).
        fsspec.config.conf.setdefault(protocols[0], {})["listings_expiry_time"] = DEFAULT_LISTINGS_EXPIRY_TIME
