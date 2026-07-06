# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a connection from a transport and auth, then resolve it once.

A connection has two composable parts — a *transport* (how bytes reach the box:
direct, an SSH tunnel, or a k8s port-forward) and *auth* (which interceptors a
caller must attach) — plus an opaque URL path appended after the transport
resolves. This module knows nothing about clusters, config files, or service
registries: it consumes generic transport URLs and hands the caller's own
``factory(endpoint) -> client`` a resolved :class:`Endpoint`.

``connect`` returns the genuine client the factory built, not a wrapper, and
binds the transport's lifetime to *that client's* garbage collection (with
``disconnect`` for deterministic teardown). The "extra hop" to a service behind
a controller is encoded entirely as the URL path ``/proxy/<name>``, resolved
server-side; rigging never inspects it.
"""

import weakref
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol, TypeVar
from urllib.parse import parse_qsl, urlsplit

from rigging.auth import BearerTokenInjector, IapServiceAccountTokenProvider, TokenProvider
from rigging.tunnel import GcpSshForwardTarget, K8sPortForwardTarget, open_tunnel

T = TypeVar("T")


@dataclass(frozen=True)
class Endpoint:
    """A reachable URL plus the auth interceptors a caller must attach to use it."""

    url: str
    interceptors: tuple = ()


class Transport(Protocol):
    """Establishes the network path to a service and yields its base Endpoint.

    ``open`` registers any background resource (e.g. a tunnel subprocess) on
    ``stack`` so the connection's lifetime owns it, and returns the base
    Endpoint the client should target.
    """

    def open(self, stack: ExitStack, timeout: float) -> Endpoint: ...


class DirectTransport:
    """Targets a URL directly, with no tunnel (public HTTPS, loopback, in-cluster DNS)."""

    def __init__(self, url: str):
        self._url = url

    def open(self, stack: ExitStack, timeout: float) -> Endpoint:
        return Endpoint(self._url)


class TunnelTransport:
    """Backs a transport with a :func:`rigging.tunnel.open_tunnel` child process.

    The tunnel kind — SSH ``-L`` forward or ``kubectl port-forward`` — follows
    from the target type ``open_tunnel`` dispatches on (a ``GcpSshForwardTarget``
    or ``K8sPortForwardTarget``), so one transport covers both. ``spawn`` is
    injectable for tests.
    """

    def __init__(self, target: GcpSshForwardTarget | K8sPortForwardTarget, *, spawn=None):
        self._target = target
        self._spawn = spawn

    def open(self, stack: ExitStack, timeout: float) -> Endpoint:
        url = stack.enter_context(
            open_tunnel(self._target, timeout=timeout, **({"spawn": self._spawn} if self._spawn else {}))
        )
        return Endpoint(url)


class Auth(Protocol):
    """Returns the client interceptors needed to authenticate to a target."""

    def interceptors(self) -> tuple: ...


class NoAuth:
    """Contributes no interceptors (loopback / SSH-tunnel-trust)."""

    def interceptors(self) -> tuple:
        return ()


class JwtAuth:
    """App auth: attaches a bearer token in ``Authorization``."""

    def __init__(self, provider: TokenProvider):
        self._provider = provider

    def interceptors(self) -> tuple:
        return (BearerTokenInjector(self._provider, "authorization"),)


class IapAuth:
    """Edge auth: attaches an IAP OIDC token in ``proxy-authorization``."""

    def __init__(self, provider: TokenProvider):
        self._provider = provider

    def interceptors(self) -> tuple:
        return (BearerTokenInjector(self._provider, "proxy-authorization"),)


class ChainedAuth:
    """Concatenates the interceptors of several auths, attached as a unit.

    Composing edge and app auth here (rather than at the call site) is what
    prevents a caller from attaching one and forgetting the other.
    """

    def __init__(self, *auths: Auth):
        self._auths = auths

    def interceptors(self) -> tuple:
        result: tuple = ()
        for auth in self._auths:
            result += tuple(auth.interceptors())
        return result


def proxy_path(name: str) -> str:
    """Build the controller proxy path for an endpoint name (``/`` → ``.``)."""
    return "/proxy/" + name.strip("/").replace("/", ".")


def capability_path(name: str, token: str) -> str:
    """Build the controller capability-URL path for an endpoint (scoped token in the path).

    Targets the controller's ``@public`` ``/proxy/t/<token>/<name>/...`` route:
    the token rides in the URL, so possession of the URL is the credential and no
    auth header is needed. ``name`` is encoded the same way as :func:`proxy_path`.
    """
    return f"/proxy/t/{token}/" + name.strip("/").replace("/", ".")


@dataclass(frozen=True)
class ParsedTransport:
    """A transport URL resolved into its transport, scheme-implied auth, and path."""

    transport: Transport
    auth: Auth
    path: str


def parse_transport(url: str) -> ParsedTransport:
    """Resolve a generic transport URL into a :class:`ParsedTransport`.

    Dispatches on the URL scheme:

    - ``http`` / ``https``: a :class:`DirectTransport`, no edge auth.
    - ``iap+https``: a :class:`DirectTransport` over HTTPS to an IAP-fronted
      endpoint. With an ``audience`` query param the scheme self-provisions the
      IAP token from service-account credentials (:class:`IapAuth` over
      :class:`~rigging.auth.IapServiceAccountTokenProvider`); without one it
      attaches no token and the caller must supply the IAP provider via
      ``connect(..., auth=IapAuth(provider))`` (e.g. a human desktop-OAuth token).
    - ``ssh+gcp``: a :class:`TunnelTransport` over SSH; ``project`` and ``zone``
      query params and a port are required, ``sa`` selects impersonation,
      ``iap=true`` tunnels SSH through IAP.
    - ``k8s``: a :class:`TunnelTransport` over ``kubectl port-forward``;
      ``namespace`` query param and a port are required, ``context`` is optional.

    Raises:
        ValueError: on an unsupported scheme or a missing required parameter.
    """
    parts = urlsplit(url)
    scheme = parts.scheme
    q = dict(parse_qsl(parts.query))
    host = parts.hostname

    if scheme in ("http", "https"):
        if not host:
            raise ValueError(f"{scheme} transport requires a host: {url!r}")
        return ParsedTransport(DirectTransport(f"{scheme}://{parts.netloc}"), NoAuth(), parts.path)

    if scheme == "iap+https":
        if not host:
            raise ValueError(f"iap+https transport requires a host: {url!r}")
        # With an audience, self-provision the IAP token from service-account
        # creds; without one, leave edge auth to the caller's auth= provider.
        audience = q.get("audience")
        edge_auth: Auth = IapAuth(IapServiceAccountTokenProvider(audience)) if audience else NoAuth()
        return ParsedTransport(DirectTransport(f"https://{parts.netloc}"), edge_auth, parts.path)

    if scheme == "ssh+gcp":
        project = q.get("project")
        zone = q.get("zone")
        if not host or not project or not zone or parts.port is None:
            raise ValueError(f"ssh+gcp transport requires a host, 'project', 'zone', and a port: {url!r}")
        target = GcpSshForwardTarget(
            project=project,
            zone=zone,
            instance=host,
            port=parts.port,
            impersonate_service_account=q.get("sa"),
            tunnel_through_iap=q.get("iap") == "true",
        )
        return ParsedTransport(TunnelTransport(target), NoAuth(), parts.path)

    if scheme == "k8s":
        namespace = q.get("namespace")
        if not host or not namespace or parts.port is None:
            raise ValueError(f"k8s transport requires a host, a 'namespace' query param, and a port: {url!r}")
        target = K8sPortForwardTarget(
            namespace=namespace,
            service=host,
            port=parts.port,
            context=q.get("context"),
        )
        return ParsedTransport(TunnelTransport(target), NoAuth(), parts.path)

    raise ValueError(f"unsupported transport scheme: {scheme!r}")


# Keyed by id(client), not the client itself: clients are arbitrary caller
# objects that may be unhashable or compare equal to one another, either of
# which would corrupt a WeakKeyDictionary. The finalize callback owns the only
# strong reference to the ExitStack, so the transport lives exactly as long as
# the client is reachable.
_FINALIZERS: dict[int, weakref.finalize] = {}


def _normalize_path(path: str) -> str:
    """Validate an appended URL path: empty, or a single-rooted ``/...`` segment."""
    if path == "":
        return ""
    if not path.startswith("/"):
        raise ValueError(f"path must be empty or start with '/': {path!r}")
    if path.startswith("//"):
        raise ValueError(f"path must not start with '//': {path!r}")
    return path


def _register_finalizer(client: object, stack: ExitStack) -> None:
    key = id(client)

    def _teardown() -> None:
        _FINALIZERS.pop(key, None)
        stack.close()

    # A present entry under this key can only be a live finalizer (its own
    # callback pops the key on teardown, and a live object's id is never
    # reused), so the factory handed back a client that already anchors a
    # connection. That client's GC fires once, so the prior transport would
    # leak — close it now before this one takes over the slot.
    existing = _FINALIZERS.get(key)
    if existing is not None:
        existing()

    try:
        _FINALIZERS[key] = weakref.finalize(client, _teardown)
    except TypeError as exc:
        stack.close()
        raise TypeError(f"connect() requires a weak-referenceable client; {type(client).__name__} is not") from exc


def _resolve_transport(transport: Transport | str) -> ParsedTransport:
    """Normalize a Transport-or-URL into a :class:`ParsedTransport`.

    The single place the ``str`` convenience surface is turned into a concrete
    transport: a URL contributes its scheme-implied auth and path, a Transport
    object contributes neither.
    """
    if isinstance(transport, str):
        return parse_transport(transport)
    return ParsedTransport(transport, NoAuth(), "")


def connect(
    transport: Transport | str,
    factory: Callable[[Endpoint], T],
    *,
    path: str = "",
    auth: Auth = NoAuth(),
    connect_timeout: float = 60.0,
) -> T:
    """Open ``transport``, build an :class:`Endpoint`, and return ``factory(endpoint)``.

    ``transport`` is either a :class:`Transport` or a generic transport URL
    string (see :func:`parse_transport`). A URL may carry its own path; ``path``
    is the explicit override. Either way the effective path must be empty or a
    single-rooted ``/...`` segment. The scheme-implied auth and the ``auth``
    argument are attached together as a :class:`ChainedAuth`. ``connect_timeout``
    bounds establishing the transport (e.g. an SSH tunnel warm-up); the per-RPC
    deadline lives in the client.

    The transport is torn down when the returned client is garbage-collected, via
    :func:`disconnect`, or at interpreter exit — whichever comes first. The client
    is the lifetime anchor: anything derived from it that strong-references it
    keeps the transport alive.

    Raises:
        ValueError: if both ``path`` and the URL path are set and disagree, or
            the effective path is malformed.
        TypeError: if the factory returns a client that cannot be weak-referenced.
    """
    resolved = _resolve_transport(transport)
    if path and resolved.path and path != resolved.path:
        raise ValueError(f"path {path!r} conflicts with the URL path {resolved.path!r}")
    effective_path = _normalize_path(path or resolved.path)

    final_auth = ChainedAuth(resolved.auth, auth)

    stack = ExitStack()
    try:
        base = resolved.transport.open(stack, connect_timeout)
        final_url = base.url.rstrip("/") + effective_path
        endpoint = Endpoint(final_url, tuple(base.interceptors) + final_auth.interceptors())
        client = factory(endpoint)
        _register_finalizer(client, stack)
    except BaseException:
        stack.close()
        raise

    return client


def disconnect(client) -> None:
    """Tear down ``client``'s transport now (idempotent; no-op if already gone)."""
    fin = _FINALIZERS.pop(id(client), None)
    if fin is not None:
        fin()
