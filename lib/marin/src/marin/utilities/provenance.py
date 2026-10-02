# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset provenance sidecars for raw downloads."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)


def write_provenance_json(output_path: str, metadata: dict[str, Any]) -> None:
    logger.info("Writing Dataset `.provenance.json` to `%s`", output_path)
    metadata["access_time"] = datetime.now(UTC).isoformat()

    # Dot-prefix keeps the sidecar out of data-discovery passes that match by
    # extension (e.g. ``normalize._discover_files`` would otherwise read it as
    # JSONL — see #5864).
    StoragePath(f"{output_path}/.provenance.json").write_text(json.dumps(metadata, indent=4, sort_keys=True))
