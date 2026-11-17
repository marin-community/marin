# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned restriction-tier rulings; Python packaging preserves this registry in hosted backends.

An override forces a source's restriction tier regardless of what a stored declaration
says: it is authoritative for declarations written before the tier existed and for a
stored row whose tag was tampered with. Every entry records its user ruling; a private
override must never be removed to "downgrade" a source — retiring a private source is a
declaration change in code, reviewed like any other.
"""

import json

RESTRICTION_OVERRIDES = r"""{
  "version": "2026.11.17.1",
  "restrictions": []
}"""


def restriction_registry_version() -> str:
    """The loaded registry's version; part of every cache key the redaction touches."""
    return json.loads(RESTRICTION_OVERRIDES)["version"]
