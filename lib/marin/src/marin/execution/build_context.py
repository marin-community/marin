# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deferred artifact versions via an ambient build-time context.

Every :class:`~marin.execution.lazy.ArtifactStep` is addressed by an explicit ``name@version``.
Hardcoding a version on each builder call is tedious while iterating: a version is really a
*run-wide policy* ("build everything at ``2026.07.16``, but pin this one dataset") that today has
to be spelled out per artifact. This module lets a builder **defer** its version to a shared
policy that rides on an ambient context:

- A :class:`VersionCodex` maps an artifact ``name`` to a version — one ``default`` plus per-name
  ``overrides``.
- A :class:`BuildContext` carries the codex (and is the seam for future build-time properties).
  It is set as an ambient :class:`~contextvars.ContextVar` while an experiment's handles are
  *constructed*, via :func:`build_context`.
- :func:`resolve_version` is what a builder calls: an explicit ``version=`` always wins; otherwise
  the version comes from the ambient codex. A builder invoked with no explicit version and no
  active context is an error — a deferred version never silently defaults.

The version is resolved to a concrete string at *construction* time, before the handle is lowered.
Nothing about identity changes: the fingerprint, the ``{name}/{version}`` path, and the drift check
all see an ordinary version string, and resolving to the ``dev`` default reproduces today's
mutable, per-user-namespaced behavior. The whole handle graph a driver builds must be materialized
inside the ``with build_context(...)`` block — a builder (or a lazily-invoked factory) called after
the block exits has no context to draw from. Construction is expected to be synchronous; a
:class:`~contextvars.ContextVar` isolates concurrent async tasks and nests correctly, but does not
propagate into freshly spawned OS threads.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from marin.execution.artifact import validate_version


@dataclass(frozen=True)
class VersionCodex:
    """Resolves an artifact ``name`` to a version: a ``default`` plus per-name ``overrides``.

    ``overrides`` are keyed by the ``name`` passed to the builder (before any per-user
    namespacing), matched exactly. Both the default and every override value are validated as
    versions at construction, so a malformed ``--override`` fails before anything is built.
    ``unused_overrides`` reports override names that no builder ever asked for — typically a typo,
    or an override aimed at an artifact that hardcodes its version.
    """

    default: str
    overrides: Mapping[str, str] = field(default_factory=dict)
    # Names actually consulted via version_for, for the unused-override check. Instrumentation
    # only (compare=False, init=False); mutating the set on a frozen instance is allowed.
    _used: set[str] = field(default_factory=set, compare=False, repr=False, init=False)

    def __post_init__(self) -> None:
        validate_version(self.default)
        for name, version in self.overrides.items():
            try:
                validate_version(version)
            except ValueError as e:
                raise ValueError(f"override {name!r}: {e}") from e

    def version_for(self, name: str) -> str:
        """The version for artifact ``name``: its override if one is set, else the default."""
        if name in self.overrides:
            self._used.add(name)
            return self.overrides[name]
        return self.default

    def unused_overrides(self) -> frozenset[str]:
        """Override names never consulted since construction — a likely typo or a no-op override."""
        return frozenset(self.overrides) - frozenset(self._used)


@dataclass(frozen=True)
class BuildContext:
    """Ambient context that rides while an experiment's handles are constructed.

    Carries the :class:`VersionCodex` today; the extension point for other build-time properties
    (e.g. shared tags or a region hint) later, so those can ride the context instead of being
    threaded through every builder.
    """

    versions: VersionCodex


_ACTIVE: ContextVar["BuildContext | None"] = ContextVar("marin_build_context", default=None)


@contextmanager
def build_context(ctx: BuildContext) -> Iterator[BuildContext]:
    """Make ``ctx`` the ambient build context for the duration of the ``with`` block.

    Construct the experiment's handle graph inside the block so deferring builders can resolve
    their versions from ``ctx``. Nests correctly and is restored on exit (including on exception).
    """
    token = _ACTIVE.set(ctx)
    try:
        yield ctx
    finally:
        _ACTIVE.reset(token)


def current_build_context() -> BuildContext | None:
    """The active :class:`BuildContext`, or ``None`` outside any :func:`build_context` block."""
    return _ACTIVE.get()


def resolve_version(name: str, version: str | None) -> str:
    """The version for artifact ``name``: ``version`` if given, else the ambient codex's choice.

    An explicit ``version`` always wins (deferral is opt-in per call). With no explicit version and
    no active :class:`BuildContext`, this raises rather than guessing — a deferred version never
    silently defaults.
    """
    if version is not None:
        return version
    ctx = current_build_context()
    if ctx is None:
        raise ValueError(
            f"{name}: no version given and no active BuildContext. Pass version=..., or construct "
            "this handle inside build_context(BuildContext(versions=VersionCodex(default=...)))."
        )
    return ctx.versions.version_for(name)
