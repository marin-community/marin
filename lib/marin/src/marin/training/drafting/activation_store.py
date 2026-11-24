# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Address the per-row safetensors shared by prefill and Speculators."""

from rigging.filesystem.storage_path import StoragePath

OBJECT_STORE_BACKEND = "object-store"


def activation_path(root: StoragePath, row_index: int) -> StoragePath:
    """Return the activation object for a row in the prepared Arrow dataset."""
    return root / f"hs_{row_index}.safetensors"
