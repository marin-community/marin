# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-wide fsspec directory-listing cache defaults."""

from typing import Any

import fsspec

DEFAULT_LISTINGS_EXPIRY_TIME = 0

_CLOUD_PROTOCOL_GROUPS = (("gs", "gcs"), ("s3", "s3a"))
_LISTING_CACHE_OPTIONS = frozenset({"use_listings_cache", "listings_expiry_time", "cache_timeout"})


def configured_listing_cache_options(protocols: tuple[str, ...]) -> dict[str, Any]:
    """Return configured listing-cache options for the given protocol aliases."""
    options = {}
    for protocol in protocols:
        protocol_config = fsspec.config.conf.get(protocol) or {}
        options.update({key: protocol_config[key] for key in _LISTING_CACHE_OPTIONS if key in protocol_config})
    return options


def configure_listing_cache_defaults() -> None:
    """Set zero-second GCS/S3 listing expiry unless cache options already exist."""
    for protocols in _CLOUD_PROTOCOL_GROUPS:
        if configured_listing_cache_options(protocols):
            continue
        # fsspec otherwise caches external writes indefinitely (#1632, #7975).
        fsspec.config.conf.setdefault(protocols[0], {})["listings_expiry_time"] = DEFAULT_LISTINGS_EXPIRY_TIME
