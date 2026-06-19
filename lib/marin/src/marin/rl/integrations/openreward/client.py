# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any


def load_openreward_client() -> type[Any]:
    try:
        openreward_module = importlib.import_module("openreward")
    except ImportError as exc:
        raise ImportError(
            "OpenReward SDK is not installed. Install `openreward` before preparing task manifests."
        ) from exc
    return openreward_module.OpenReward
