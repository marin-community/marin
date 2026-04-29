# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve cluster endpoint URIs declared in the cluster config.

Endpoints map logical names like ``/system/log-server`` to concrete URLs.
The cluster config carries them as ``EndpointSpec`` entries with a scheme-tagged
URI; this module dispatches by scheme and returns a concrete ``http(s)://`` URL.

Supported schemes:

- ``http://`` / ``https://`` — returned as-is.
- ``gcp://<instance>[:port]`` — resolves a GCE instance to its internal IP via
  ``gcloud compute instances describe``. Metadata keys consumed:
  ``project`` (falls back to ADC default project), ``zone`` (required),
  ``port`` (required if not in URI).
- ``k8s://<service>[.<namespace>][:port]`` — pure string templating to the
  in-cluster Service DNS name. Metadata keys consumed: ``namespace`` (falls
  back to the pod's mounted ``serviceaccount/namespace`` file),
  ``port`` (required if not in URI).

Unknown schemes raise ``ValueError``. Resolution failures (gcloud nonzero exit,
missing fields) raise ``RuntimeError``; bad inputs raise ``ValueError``.

Example YAML:

    endpoints:
      /system/log-server:
        uri: gcp://my-controller-vm
        metadata:
          zone: us-central1-a
          port: "10001"
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable

import google.auth

logger = logging.getLogger(__name__)

SchemeResolver = Callable[[str, dict[str, str]], str]

_REGISTRY: dict[str, SchemeResolver] = {}

_NS_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
_GCLOUD_TIMEOUT = 30.0


def register_scheme(scheme: str, handler: SchemeResolver) -> None:
    """Register ``handler`` as the resolver for ``scheme://...`` URIs."""
    _REGISTRY[scheme] = handler


def resolve_endpoint_uri(uri: str, metadata: dict[str, str] | None = None) -> str:
    """Resolve a scheme-tagged endpoint URI to a concrete ``http(s)://`` URL.

    ``http://`` and ``https://`` URIs are returned unchanged. Other schemes are
    dispatched through the registry; unknown schemes raise ``ValueError``.
    """
    md = metadata or {}
    if uri.startswith(("http://", "https://")):
        return uri
    scheme, sep, _ = uri.partition("://")
    if not sep:
        raise ValueError(f"endpoint URI missing scheme separator: {uri!r}")
    handler = _REGISTRY.get(scheme)
    if handler is None:
        raise ValueError(f"unknown endpoint scheme {scheme!r} in {uri!r}")
    return handler(uri, md)


def _split_host_port(authority: str) -> tuple[str, str | None]:
    """Split ``host[:port]`` into ``(host, port_or_None)``."""
    if ":" in authority:
        host, _, port = authority.partition(":")
        return host, port or None
    return authority, None


def _resolve_gcp(uri: str, metadata: dict[str, str]) -> str:
    """Resolve ``gcp://<instance>[:port]`` to ``http://<internal_ip>:<port>``."""
    authority = uri[len("gcp://") :]
    if not authority:
        raise ValueError(f"gcp:// URI missing instance name: {uri!r}")
    instance, port_from_uri = _split_host_port(authority)
    if not instance:
        raise ValueError(f"gcp:// URI missing instance name: {uri!r}")

    port = port_from_uri or metadata.get("port")
    if not port:
        raise ValueError(f"gcp:// endpoint requires a port (URI suffix or metadata.port): {uri!r}")

    zone = metadata.get("zone")
    if not zone:
        raise ValueError(f"gcp:// endpoint requires metadata.zone: {uri!r}")

    project = metadata.get("project")
    if not project:
        try:
            _, project = google.auth.default()
        except Exception as exc:
            raise RuntimeError(f"gcp:// endpoint could not resolve project via ADC: {exc}") from exc
    if not project:
        raise ValueError(f"gcp:// endpoint requires metadata.project (ADC default unavailable): {uri!r}")

    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        instance,
        f"--project={project}",
        f"--zone={zone}",
        "--format=json",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GCLOUD_TIMEOUT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gcloud describe failed for {instance!r} in {project}/{zone}: {exc.stderr or exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"gcloud describe timed out for {instance!r}: {exc}") from exc

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gcloud describe returned non-JSON for {instance!r}: {exc}") from exc

    interfaces = payload.get("networkInterfaces") or []
    if not interfaces:
        raise RuntimeError(f"gcp:// instance {instance!r} has no networkInterfaces")
    ip = interfaces[0].get("networkIP")
    if not ip:
        raise RuntimeError(f"gcp:// instance {instance!r} has no networkInterfaces[0].networkIP")

    return f"http://{ip}:{port}"


def _read_pod_namespace() -> str | None:
    """Return the namespace from the in-pod serviceaccount mount, or ``None``."""
    try:
        with open(_NS_FILE) as fh:
            ns = fh.read().strip()
    except OSError:
        return None
    return ns or None


def _resolve_k8s(uri: str, metadata: dict[str, str]) -> str:
    """Resolve ``k8s://<service>[.<namespace>][:port]`` to in-cluster DNS."""
    authority = uri[len("k8s://") :]
    if not authority:
        raise ValueError(f"k8s:// URI missing service name: {uri!r}")
    host, port_from_uri = _split_host_port(authority)
    if "." in host:
        service, _, ns_from_uri = host.partition(".")
    else:
        service, ns_from_uri = host, ""
    if not service:
        raise ValueError(f"k8s:// URI missing service name: {uri!r}")

    namespace = ns_from_uri or metadata.get("namespace") or _read_pod_namespace()
    if not namespace:
        raise ValueError(
            f"k8s:// endpoint requires a namespace (URI suffix, metadata.namespace, "
            f"or in-pod serviceaccount file): {uri!r}"
        )

    port = port_from_uri or metadata.get("port")
    if not port:
        raise ValueError(f"k8s:// endpoint requires a port (URI suffix or metadata.port): {uri!r}")

    return f"http://{service}.{namespace}.svc.cluster.local:{port}"


register_scheme("gcp", _resolve_gcp)
register_scheme("k8s", _resolve_k8s)
