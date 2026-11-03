# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invariants: validation evidence belongs to the exact task (digests do not depend on the hash seed)."""

import os
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

from taskforge.content_hash import canonical_json

# A value whose sets would serialize in hash-seed order: a module-level frozenset, as a build step's
# module holds, and a set inside a dataclass inside a dict.
SEEDED_VALUE = """
from dataclasses import dataclass
from taskforge.content_hash import canonical_json

@dataclass(frozen=True)
class Rules:
    labels: frozenset[str]

print(canonical_json({"categories": frozenset({"known_correct", "empty_or_malformed", "plausible_wrong"}),
                      "rules": Rules(frozenset({"alpha", "beta", "gamma", "delta"}))}))
"""


def test_a_set_encodes_the_same_under_every_hash_seed():
    encodings = {
        subprocess.run(
            [sys.executable, "-c", SEEDED_VALUE],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3", "4")
    }
    assert encodings == {
        '{"categories":["empty_or_malformed","known_correct","plausible_wrong"],'
        '"rules":{"labels":["alpha","beta","delta","gamma"]}}\n'
    }


class Shade(StrEnum):
    DARK = "dark"


@dataclass(frozen=True)
class Inner:
    blob: bytes
    shade: Shade


@dataclass(frozen=True)
class Outer:
    name: str
    parts: tuple[Inner, ...]
    weights: dict[str, float]


def test_a_value_without_sets_encodes_with_sorted_keys_base64_bytes_and_enum_values():
    value = Outer("x", (Inner(b"\x00\x01", Shade.DARK),), {"b": 0.5, "a": 1.0})
    assert canonical_json(value) == '{"name":"x","parts":[{"blob":"AAE=","shade":"dark"}],"weights":{"a":1.0,"b":0.5}}'
