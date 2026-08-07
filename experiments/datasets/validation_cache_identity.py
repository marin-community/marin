# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable artifact identities for tokenizer-specific validation caches."""

import hashlib

from experiments.llama import llama3_tokenizer
from experiments.marin_tokenizer import marin_tokenizer

VALIDATION_CACHE_VERSION = "2026.07.23"


def validation_tokenizer_suffix(tokenizer: str) -> str:
    """Return the artifact-name suffix that distinguishes a validation tokenizer."""
    if tokenizer == llama3_tokenizer:
        return "llama3"
    if tokenizer == marin_tokenizer:
        return "marin"
    digest = hashlib.sha256(tokenizer.encode()).hexdigest()[:8]
    return f"tokenizer-{digest}"
