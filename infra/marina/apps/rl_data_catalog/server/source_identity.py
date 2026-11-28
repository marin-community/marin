# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Permanent aliases for published source identities, read from the pinned registry.

The alias records are catalog data, not code: the loader reads the versioned
alias table the catalog lanes publish to the pinned registry revision named by
the committed ``source_id_aliases.json`` reference.
"""

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Annotated

from finestore.rl.registry_reference import read_registry_json
from pydantic import BeforeValidator

ALIASES_REFERENCE = Path(__file__).with_name("source_id_aliases.json")
"""The committed pin: which registry revision holds the alias records this loader reads."""


@cache
def source_id_aliases() -> dict[str, str]:
    """The validated historical-to-catalog identity aliases of one pinned revision."""
    document = read_registry_json(ALIASES_REFERENCE)
    aliases = document["aliases"]
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("Source alias registry maps two historical IDs to one identity")
    return aliases


@cache
def original_source_id_map() -> dict[str, str]:
    """The reverse alias map, keeping artifact identities tied to the original IDs."""
    return {new: old for old, new in source_id_aliases().items()}


def canonical_source_id(source_id: str) -> str:
    """Resolve a published historical ID to its generated catalog identity."""
    return source_id_aliases().get(source_id, source_id)


def original_source_id(source_id: str) -> str:
    """Keep artifact identities, sampling seeds and ordering tied to the original ID."""
    return original_source_id_map().get(source_id, source_id)


SourceId = Annotated[str, BeforeValidator(canonical_source_id)]


def canonical_source_map[T](records: Mapping[str, T]) -> dict[str, T]:
    """Resolve source-keyed records without silently collapsing two historical entries."""
    result = {}
    for source_id, record in records.items():
        key = canonical_source_id(source_id)
        if key in result:
            raise ValueError(f"Duplicate aliased source identity: {key}")
        result[key] = record
    return result
