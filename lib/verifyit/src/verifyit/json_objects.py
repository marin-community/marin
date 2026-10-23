# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject ambiguous duplicate keys while decoding trusted JSON objects."""


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result
