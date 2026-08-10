# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for selected allenai/dolma3.5_pool sources."""

HF_DATASET_ID = "allenai/dolma3.5_pool"
HF_REVISION = "d2bf6ae"
STAGED_ROOT = f"raw/dolma3_5_pool-{HF_REVISION}"
DATA_FILE_EXTENSION = ".jsonl.zst"
DATA_FILE_GLOB = f"**/*{DATA_FILE_EXTENSION}"
