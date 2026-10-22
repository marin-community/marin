# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave task storage credential requirements."""

from collections.abc import Mapping

COREWEAVE_STORAGE_ENV_VARS = ("CW_KEY_ID", "CW_KEY_SECRET")
COREWEAVE_ACCESS_KEYS_URL = "https://console.coreweave.com/object-storage/access-keys"


def coreweave_storage_error(env: Mapping[str, str]) -> str | None:
    """Return an actionable error when a job lacks personal storage credentials."""
    missing = [name for name in COREWEAVE_STORAGE_ENV_VARS if not env.get(name)]
    if not missing:
        return None
    return (
        f"CoreWeave jobs require your own object-storage credentials. Missing: {', '.join(missing)}. "
        f"Create an access key at {COREWEAVE_ACCESS_KEYS_URL}, "
        "then export CW_KEY_ID and CW_KEY_SECRET before submitting, or set both in the job environment."
    )
