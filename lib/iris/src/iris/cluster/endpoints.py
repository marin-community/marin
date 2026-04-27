# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve cluster endpoint URIs declared in the cluster config.

Endpoints map logical names like ``/system/log-server`` to concrete URLs.
The cluster config carries them as ``EndpointSpec`` entries with a scheme-tagged
URI; this module dispatches by scheme and returns a concrete ``http(s)://`` URL.

Example YAML:

    endpoints:
      /system/log-server:
        uri: http://log-server.iris.svc.cluster.local:10001
"""

from __future__ import annotations


def resolve_endpoint_uri(uri: str, metadata: dict[str, str] | None = None) -> str:
    """Resolve a scheme-tagged endpoint URI to a concrete ``http(s)://`` URL.

    Schemes:
      - ``http://`` / ``https://`` — returned as-is.
      - ``gcp://<service>`` — not yet wired; raises ``NotImplementedError``.
      - ``k8s://<service>[.<namespace>]`` — not yet wired; raises ``NotImplementedError``.

    Unknown schemes raise ``ValueError``.
    """
    del metadata  # reserved for future scheme resolvers
    if uri.startswith(("http://", "https://")):
        return uri
    if uri.startswith("gcp://"):
        raise NotImplementedError("gcp:// scheme requires GCP resolver wiring; use http:// for now")
    if uri.startswith("k8s://"):
        raise NotImplementedError("k8s:// scheme requires in-cluster DNS resolver wiring; use http:// for now")
    raise ValueError(f"unknown endpoint scheme in {uri!r}")
