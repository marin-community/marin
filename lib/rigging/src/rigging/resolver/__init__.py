# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""URL-scheme resolver for rigging service references.

Public surface:

- :func:`resolve` — turn ``gcp://``, registered ``<scheme>://``, or bare
  ``host:port`` into ``(host, port)``.
- :func:`register_scheme` — install a handler for a new scheme. Used by
  package-level plugins (e.g. ``iris.client.resolver_plugin``).
- :func:`vm_address` — VM-name → ``(host, port)`` for a given provider.
- :class:`ServiceURL` — parsed reference for callers that want the pieces.
"""

from rigging.resolver.providers import DEFAULT_GCP_PORT, vm_address
from rigging.resolver.resolver import register_scheme, resolve
from rigging.resolver.url import ServiceURL

__all__ = [
    "DEFAULT_GCP_PORT",
    "ServiceURL",
    "register_scheme",
    "resolve",
    "vm_address",
]
