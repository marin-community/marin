# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""URL-scheme dispatch for rigging service references.

:func:`resolve` turns a textual reference into a concrete ``(host, port)``
pair:

- ``host:port`` (no scheme) — short-circuited as a plain literal.
- ``gcp://<vm-name>`` — direct VM-by-name lookup via :func:`vm_address`,
  registered eagerly at module-load time.
- Other schemes — must be installed by the owning package via
  :func:`register_scheme`. ``iris://`` is registered by
  ``iris.client.resolver_plugin`` on import.

Unknown schemes raise ``ValueError``. Per the log-store extraction plan §D5
the registry is a plain dict; later registrations overwrite earlier ones,
matching dict semantics.
"""

from collections.abc import Callable

from rigging.resolver.providers import vm_address
from rigging.resolver.url import ServiceURL

SchemeHandler = Callable[[ServiceURL], tuple[str, int]]

_HANDLERS: dict[str, SchemeHandler] = {}


def register_scheme(scheme: str, handler: SchemeHandler) -> None:
    """Register ``handler`` as the resolver for ``scheme://...`` URLs.

    Re-registering the same scheme overwrites the prior handler — this is
    intentional and matches dict semantics, so plugins / tests can swap
    implementations without juggling unregister calls.
    """
    _HANDLERS[scheme] = handler


def resolve(ref: str) -> tuple[str, int]:
    """Resolve ``ref`` to a concrete ``(host, port)``.

    Bare ``host:port`` short-circuits before any registry lookup. Otherwise
    the URL is parsed and the registered handler for its scheme is invoked;
    schemes with no handler raise :class:`ValueError`.
    """
    if "://" not in ref:
        host, port = ref.rsplit(":", 1)
        return host, int(port)
    url = ServiceURL.parse(ref)
    handler = _HANDLERS.get(url.scheme)
    if handler is None:
        raise ValueError(f"unsupported scheme: {url.scheme!r}")
    return handler(url)


def _resolve_gcp(url: ServiceURL) -> tuple[str, int]:
    return vm_address(url.host, provider="gcp")


register_scheme("gcp", _resolve_gcp)
