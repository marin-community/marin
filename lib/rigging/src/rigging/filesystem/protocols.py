# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalization for fsspec filesystem protocols."""

from collections.abc import Iterable


def normalize_protocols(protocol: str | Iterable[str]) -> tuple[str, ...]:
    """Return fsspec's scalar or iterable protocol declaration as a tuple."""
    if isinstance(protocol, str):
        return (protocol,)
    return tuple(protocol)
