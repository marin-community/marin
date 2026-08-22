# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint declarations for Iris actors."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar, cast

_WEB_ENDPOINT_ATTRIBUTE = "__iris_actor_web_endpoints__"

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class _ActorWebEndpoint:
    path: str
    method: str


def web_endpoint(path: str, *, method: str = "GET") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Expose an actor method through the actor HTTP application.

    Args:
        path: Starlette route path.
        method: HTTP method for the route.
    """
    if not path.startswith("/"):
        raise ValueError("Actor web endpoint paths must start with '/'")
    normalized_method = method.upper()
    if not normalized_method:
        raise ValueError("Actor web endpoint methods cannot be empty")
    endpoint = _ActorWebEndpoint(path=path, method=normalized_method)

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        target = cast(Any, function)
        endpoints = getattr(target, _WEB_ENDPOINT_ATTRIBUTE, ())
        setattr(target, _WEB_ENDPOINT_ATTRIBUTE, (*endpoints, endpoint))
        return function

    return decorator


def _actor_web_endpoints(function: Callable[..., Any]) -> tuple[_ActorWebEndpoint, ...]:
    target = getattr(function, "__func__", function)
    return cast(tuple[_ActorWebEndpoint, ...], getattr(target, _WEB_ENDPOINT_ATTRIBUTE, ()))
