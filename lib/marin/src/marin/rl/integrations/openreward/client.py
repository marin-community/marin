# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


def load_openreward_client() -> type:
    try:
        from openreward import OpenReward
    except ImportError as exc:
        raise ImportError(
            "OpenReward SDK is not installed. Install `openreward` before preparing task manifests."
        ) from exc
    return OpenReward
