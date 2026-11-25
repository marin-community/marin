# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content identities shared by fixed-cohort preparation and saved-generation audits."""

import hashlib
import json
from collections.abc import Iterable


def prompt_hash(token_ids: list[int]) -> str:
    """Hash exact prompt IDs encoded as a compact JSON array."""
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def cohort_hash(prompt_hashes: Iterable[str]) -> str:
    """Hash the concatenated, sorted prompt digests without changing cohort membership."""
    return hashlib.sha256("".join(sorted(prompt_hashes)).encode()).hexdigest()
