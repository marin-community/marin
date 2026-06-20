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

import contextlib
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol, TypeVar
from urllib.parse import parse_qsl, urlsplit

from rigging.auth import (
    AuthTokenInjector,
    IapUserIdTokenProvider,
    ProxyAuthTokenInjector,
    TokenProvider,
)
from rigging.tunnel import GcpSshForwardTarget, K8sPortForwardTarget, open_tunnel

T = TypeVar("T")


@dataclass(frozen=True)
class Endpoint:
    """A reachable URL plus the auth interceptors a caller must attach to use it."""

    url: str
    interceptors: tuple = ()

    def socket_address(self) -> tuple[str, int]:
        """Return ``(host, port)`` for socket-level callers.

        Valid only for a bare ``http``/``https`` origin URL — no path, query, or
        fragment — and raises otherwise, so a proxy-prefixed or parameterized URL
        is never silently truncated to host:port.
        """
        parts = urlsplit(self.url)
        if (
            parts.scheme not in ("http", "https")
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or parts.hostname is None
        ):
            raise ValueError(f"endpoint {self.url!r} is not a bare host:port")
        return parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)


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


class _TunnelTransport:
    """Backs a transport with a :func:`rigging.tunnel.open_tunnel` child process.

    The tunnel kind is decided by the target type ``open_tunnel`` dispatches on,
    so the only thing the concrete transports add is a typed constructor.
    """

    def __init__(self, target, *, spawn=None):
        self._target = target
        self._spawn = spawn

    def open(self, stack: ExitStack, timeout: float) -> Endpoint:
        url = stack.enter_context(
            open_tunnel(self._target, timeout=timeout, **({"spawn": self._spawn} if self._spawn else {}))
        )
        return Endpoint(url)


class SshTunnel(_TunnelTransport):
    """Opens an SSH ``-L`` port forward via :func:`rigging.tunnel.open_tunnel`."""

    def __init__(self, target: GcpSshForwardTarget, *, spawn=None):
        super().__init__(target, spawn=spawn)


class K8sPortForward(_TunnelTransport):
    """Opens a ``kubectl port-forward`` via :func:`rigging.tunnel.open_tunnel`."""

    def __init__(self, target: K8sPortForwardTarget, *, spawn=None):
        super().__init__(target, spawn=spawn)


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
        return (AuthTokenInjector(self._provider),)


class IapAuth:
    """Edge auth: attaches an IAP OIDC token in ``proxy-authorization``."""

    def __init__(self, provider: TokenProvider):
        self._provider = provider

    def interceptors(self) -> tuple:
        return (ProxyAuthTokenInjector(self._provider),)


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
    - ``iap+https``: a :class:`DirectTransport` over HTTPS plus :class:`IapAuth`;
      the IAP OAuth client id comes from the required ``audience`` query param.
    - ``ssh+gcp``: an :class:`SshTunnel`; ``project`` and ``zone`` query params
      and a port are required, ``sa`` selects impersonation, ``iap=true`` tunnels
      SSH through IAP.
    - ``k8s``: a :class:`K8sPortForward`; ``namespace`` query param and a port are
      required, ``context`` is optional.

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
        audience = q.get("audience")
        if not audience:
            raise ValueError(f"iap+https transport requires an 'audience' query param: {url!r}")
        return ParsedTransport(
            DirectTransport(f"https://{parts.netloc}"),
            IapAuth(IapUserIdTokenProvider(audience)),
            parts.path,
        )

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
        return ParsedTransport(SshTunnel(target), NoAuth(), parts.path)

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
        return ParsedTransport(K8sPortForward(target), NoAuth(), parts.path)

    raise ValueError(f"unsupported transport scheme: {scheme!r}")


@dataclass(frozen=True)
class ConnectionOptions:
    """Tunables for establishing a connection. The per-RPC deadline lives in the client."""

    connect_timeout: float = 60.0


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


def _resolve_transport(transport: Transport | str) -> tuple[Transport, Auth, str]:
    """Normalize a Transport-or-URL into ``(transport, scheme-implied auth, path)``.

    The single place the ``str`` convenience surface is turned into a concrete
    transport: a URL contributes its scheme-implied auth and path, a Transport
    object contributes neither.
    """
    if isinstance(transport, str):
        parsed = parse_transport(transport)
        return parsed.transport, parsed.auth, parsed.path
    return transport, NoAuth(), ""


def connect(
    transport: Transport | str,
    factory: Callable[[Endpoint], T],
    *,
    path: str = "",
    auth: Auth = NoAuth(),
    options: ConnectionOptions = ConnectionOptions(),
) -> T:
    """Open ``transport``, build an :class:`Endpoint`, and return ``factory(endpoint)``.

    ``transport`` is either a :class:`Transport` or a generic transport URL
    string (see :func:`parse_transport`). A URL may carry its own path; ``path``
    is the explicit override. Either way the effective path must be empty or a
    single-rooted ``/...`` segment. The scheme-implied auth and the ``auth``
    argument are attached together as a :class:`ChainedAuth`.

    The transport (e.g. an SSH tunnel subprocess) is torn down when the returned
    client is garbage-collected, via :func:`disconnect`, or at interpreter exit
    — whichever comes first. The client is the lifetime anchor: anything derived
    from it that strong-references it keeps the transport alive.

    Raises:
        ValueError: if both ``path`` and the URL path are set and disagree, or
            the effective path is malformed.
        TypeError: if the factory returns a client that cannot be weak-referenced.
    """
    tp, scheme_auth, url_path = _resolve_transport(transport)
    if path and url_path and path != url_path:
        raise ValueError(f"path {path!r} conflicts with the URL path {url_path!r}")
    effective_path = _normalize_path(path or url_path)

    final_auth = ChainedAuth(scheme_auth, auth)

    stack = ExitStack()
    try:
        base = tp.open(stack, options.connect_timeout)
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


@contextlib.contextmanager
def closing_connection(
    transport: Transport | str,
    factory: Callable[[Endpoint], T],
    *,
    path: str = "",
    auth: Auth = NoAuth(),
    options: ConnectionOptions = ConnectionOptions(),
):
    """Scoped :func:`connect` that calls :func:`disconnect` on exit."""
    client = connect(transport, factory, path=path, auth=auth, options=options)
    try:
        yield client
    finally:
        disconnect(client)


def _transport_summary(transport: Transport) -> str:
    if isinstance(transport, DirectTransport):
        return f"DirectTransport(url={transport._url})"
    if isinstance(transport, SshTunnel):
        t = transport._target
        return f"SshTunnel(host={t.instance}, project={t.project}, zone={t.zone}, port={t.port})"
    if isinstance(transport, K8sPortForward):
        t = transport._target
        return f"K8sPortForward(service={t.service}, namespace={t.namespace}, port={t.port})"
    return type(transport).__name__


def explain(transport: Transport | str, auth: Auth = NoAuth()) -> str:
    """Describe how a connection resolves — transport, auth, base URL — secret-free.

    Reports the transport class and its target summary, the auth provider class
    names, and the base URL, so "why SSH / why IAP / why no JWT" is answerable
    without leaking any token value.
    """
    transport_obj, scheme_auth, path = _resolve_transport(transport)
    final_auth = ChainedAuth(scheme_auth, auth)
    injector_names = ", ".join(type(i).__name__ for i in final_auth.interceptors()) or "none"

    base_url = transport_obj._url if isinstance(transport_obj, DirectTransport) else "<established at connect time>"
    return (
        f"transport: {_transport_summary(transport_obj)}\n"
        f"auth interceptors: {injector_names}\n"
        f"base url: {base_url}{path}"
    )
