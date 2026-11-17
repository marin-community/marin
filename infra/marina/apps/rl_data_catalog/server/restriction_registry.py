# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned restriction-tier rulings; Python packaging preserves this registry in hosted backends.

An override forces a source private regardless of what a stored declaration
says: it is authoritative for declarations written before the tier existed and
for a stored row whose tag was tampered with or erased. It can only strengthen
the tier — it never lowers an explicit ``PRIVATE_ALWAYS`` declaration, because a
private source stops being private only through a reviewed declaration change in
code. Every entry records its user ruling.
"""

import json

RESTRICTION_OVERRIDES = r"""{
  "version": "2026.11.17.1",
  "restrictions": []
}"""


def restriction_registry_version() -> str:
    """The loaded registry's version; part of every cache key the redaction touches."""
    return json.loads(RESTRICTION_OVERRIDES)["version"]
