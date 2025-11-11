# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


# This whole file is copied from Equinox.
# (c) 2023, Google LLC. and/or Patrick Kidger. Apache 2.0 licensed.
# Patrick doesn't like that I depend on Equinox internals, so I copied this stuff
import functools as ft
import types
import warnings
import weakref
from typing import Any

import jax.tree_util as jtu


def _strip_wrapped_partial(fun):
    if hasattr(fun, "__wrapped__"):  # ft.wraps
        return _strip_wrapped_partial(fun.__wrapped__)
    if isinstance(fun, ft.partial):
        return _strip_wrapped_partial(fun.func)
    return fun


internal_caches = []  # type: ignore
internal_lru_caches = []  # type: ignore


def clear_caches():
    """Clears internal Equinox caches.

    Best used before calling `jax.clear_caches()` or `jax.clear_backends()`.

    **Arguments:**

    None.

    **Returns:**

    None.
    """
    for cache in internal_caches:
        cache.clear()
    for cache in internal_lru_caches:
        cache.cache_clear()


def _default_cache_key():
    assert False


def compile_cache(cacheable_fn):
    cache = weakref.WeakKeyDictionary()  # type: ignore
    internal_caches.append(cache)

    def cached_fn_impl(leaves, treedef):
        user_fn_names, args, kwargs = jtu.tree_unflatten(treedef, leaves)
        return cacheable_fn(user_fn_names, *args, **kwargs)

    @ft.wraps(cacheable_fn)
    def wrapped_cacheable_fn(user_fn, *args, **kwargs):
        user_fn = _strip_wrapped_partial(user_fn)
        # Best-effort attempt to clear the cache of old and unused entries.
        cache_key: Any
        if type(user_fn) is types.FunctionType:  # noqa: E721
            cache_key = user_fn
        else:
            cache_key = _default_cache_key

        try:
            user_fn_names = user_fn.__name__, user_fn.__qualname__
        except AttributeError:
            user_fn_names = type(user_fn).__name__, type(user_fn).__qualname__
        leaves, treedef = jtu.tree_flatten((user_fn_names, args, kwargs))
        leaves = tuple(leaves)

        try:
            cached_fn = cache[cache_key]
        except KeyError:
            cached_fn = cache[cache_key] = ft.lru_cache(maxsize=None)(cached_fn_impl)
        return cached_fn(leaves, treedef)

    def delete(user_fn):
        user_fn = _strip_wrapped_partial(user_fn)
        if type(user_fn) is types.FunctionType:  # noqa: E721
            try:
                del cache[user_fn]
            except KeyError:
                warnings.warn(f"Could not delete cache for function {user_fn}. Has it already been deleted?")
        else:
            warnings.warn("Could not delete non-function from cache.")

    wrapped_cacheable_fn.delete = delete  # type: ignore
    return wrapped_cacheable_fn
