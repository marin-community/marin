# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import BaseModel

from harbor.models.registry import RegistryTaskId


class DownloadedDatasetItem(BaseModel):
    id: RegistryTaskId
    downloaded_path: Path
