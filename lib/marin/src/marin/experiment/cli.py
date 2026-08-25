# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click bootstrap for experiment ``main``s that defer their artifact versions.

An experiment's driver constructs a handle graph and then either prints its plan or builds it.
This module makes that a one-liner while wiring up the deferred-version machinery
(:mod:`marin.execution.build_context`):

- :func:`build_options` is a Click decorator that adds ``--version`` (the run-wide default),
  ``--override NAME=VERSION`` (per-artifact, repeatable), ``--run`` (build vs. print the plan), and
  ``--max-concurrent``. It builds the wrapped function's handle graph **inside** a
  :class:`~marin.execution.build_context.BuildContext` assembled from those options, so any builder
  that deferred its version (omitted ``version=``) resolves against them.
- :func:`experiment_main` wraps a nullary ``build()`` into a ready-to-run command.

``--version`` is required — there is no silent ``dev`` default, because a deferred *dataset* at a
mutable version rebuilds its (often multi-TB) cache on every run and, unlike a training checkpoint,
is not namespaced per user. To iterate, pass ``--version dev`` explicitly; the printed plan flags
every artifact that resolved to a mutable version and will rebuild.

An experiment with its own options composes with :func:`build_options` by stacking it *below* the
experiment's own ``@click.option``s and ``@click.command()`` on top::

    @click.command()
    @click.option("--device", ...)
    @build_options
    def main(device):
        return build(device=device)   # version now deferred to --version / --override

    if __name__ == "__main__":
        main()

Construct the whole handle graph inside ``build()``; a builder invoked lazily after it returns has
no active context and raises. Handles are frozen at construction, so reusing one across two
invocations keeps its first resolved version.
"""

import functools
from collections.abc import Callable, Mapping
from typing import Any

import click

from marin.execution.artifact import is_mutable_version, validate_version
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, lower, run

BuildResult = ArtifactStep | Mapping[str, ArtifactStep] | list[ArtifactStep] | tuple[ArtifactStep, ...]


def _version_callback(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Reject a malformed ``--version`` at parse time, as a Click error rather than a traceback."""
    try:
        validate_version(value)
    except ValueError as e:
        raise click.BadParameter(str(e), ctx=ctx, param=param) from e
    return value


def _override_callback(ctx: click.Context, param: click.Parameter, value: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated ``NAME=VERSION`` into a validated ``{name: version}`` map, as Click errors."""
    overrides: dict[str, str] = {}
    for item in value:
        name, sep, version = item.partition("=")
        if not sep or not name:
            raise click.BadParameter(f"{item!r} is not NAME=VERSION", ctx=ctx, param=param)
        if name in overrides:
            raise click.BadParameter(f"duplicate override for {name!r}", ctx=ctx, param=param)
        try:
            validate_version(version)
        except ValueError as e:
            raise click.BadParameter(f"override {name!r}: {e}", ctx=ctx, param=param) from e
        overrides[name] = version
    return overrides


def _as_handles(result: BuildResult) -> list[ArtifactStep]:
    """Normalize a ``build()`` return — one handle, a list/tuple, or a name→handle mapping."""
    if isinstance(result, ArtifactStep):
        return [result]
    if isinstance(result, Mapping):
        return list(result.values())
    if isinstance(result, list | tuple):
        return list(result)
    raise TypeError(
        f"build() must return an ArtifactStep, a list/tuple, or a mapping of them, not {type(result).__name__}"
    )


def _graph_handles(handles: list[ArtifactStep]) -> list[ArtifactStep]:
    """Every handle reachable from ``handles`` (deps included), deduped by identity, deps first."""
    seen: set[int] = set()
    order: list[ArtifactStep] = []

    def walk(handle: ArtifactStep) -> None:
        if id(handle) in seen:
            return
        seen.add(id(handle))
        for dep in handle.deps:
            walk(dep)
        order.append(handle)

    for handle in handles:
        walk(handle)
    return order


def _print_plan(handles: list[ArtifactStep]) -> None:
    """Print the lowered plan, then a summary of every artifact's resolved ``name@version``.

    The summary flags each artifact that resolved to a mutable version inline, since a mutable
    version rebuilds on every run — the thing a driver most wants to catch before ``--run``.
    """
    for handle in handles:
        click.echo(lower(handle))
    click.echo("\nResolved versions:")
    for handle in _graph_handles(handles):
        flag = "  (mutable — rebuilds every run)" if is_mutable_version(handle.version) else ""
        click.echo(f"  {handle.name}@{handle.version}{flag}")


def build_options(fn: Callable[..., BuildResult]) -> Callable[..., None]:
    """Add the shared version/run options to a command that returns the experiment's handle(s).

    ``fn`` returns one :class:`~marin.execution.lazy.ArtifactStep`, a list/tuple of them, or a
    name→handle mapping; it is called inside a :class:`~marin.execution.build_context.BuildContext`
    built from ``--version`` and ``--override``. Without ``--run`` the lowered plan is printed;
    with it the handles are built. An ``--override`` naming an artifact that never deferred (a typo,
    or one that hardcodes its version) is rejected, so a silently-ignored override cannot slip a run
    under the wrong version.
    """

    @click.option(
        "--version",
        "default_version",
        required=True,
        callback=_version_callback,
        help="Run-wide default version for artifacts that defer (omit version=). "
        "A calendar version YYYY.MM.DD, or 'dev' to iterate.",
    )
    @click.option(
        "--override",
        "overrides",
        multiple=True,
        metavar="NAME=VERSION",
        callback=_override_callback,
        help="Per-artifact version override, keyed by the artifact's name. Repeatable.",
    )
    @click.option("--run", "do_run", is_flag=True, help="Build the handles (default: print the plan).")
    @click.option("--max-concurrent", type=int, default=8, show_default=True, help="Max steps built concurrently.")
    @functools.wraps(fn)
    def wrapper(
        *args: Any, default_version: str, overrides: dict[str, str], do_run: bool, max_concurrent: int, **kwargs: Any
    ) -> None:
        codex = VersionCodex(default=default_version, overrides=overrides)
        with build_context(BuildContext(versions=codex)):
            handles = _as_handles(fn(*args, **kwargs))
        unused = codex.unused_overrides()
        if unused:
            raise click.BadParameter(
                f"never applied to any artifact: {sorted(unused)}. The name is a typo, or that "
                "artifact hardcodes its version (an explicit version wins over an override).",
                param_hint="--override",
            )
        if not do_run:
            _print_plan(handles)
            return
        run(*handles, max_concurrent=max_concurrent)

    return wrapper


def experiment_main(build: Callable[[], BuildResult]) -> Callable[..., None]:
    """Turn a nullary ``build()`` into a ready-to-run experiment command with the shared options.

    ``experiment_main(build)()`` is the whole ``if __name__ == "__main__":`` body for an experiment
    that takes no options of its own. For one that does, use :func:`build_options` directly (see the
    module docstring for the stacking order).
    """
    return click.command()(build_options(build))
