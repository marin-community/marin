# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged native verifier and retain its component rewards."""

import json
import math
import subprocess
from pathlib import Path

NATIVE_REWARD = Path("/logs/verifier/reward.json")
ADAPTED_REWARD = Path("/logs/verifier/curation-reward.json")


def component_reward(rewards: dict[str, float]) -> dict:
    """Return the arithmetic mean as ``reward`` and native components in ``detail.native_rewards``."""
    if not isinstance(rewards, dict) or not rewards:
        raise ValueError("Expected a nonempty native reward mapping")
    if any(
        isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1
        for value in rewards.values()
    ):
        raise ValueError("Native reward components must be finite numbers in [0, 1]")
    return {
        "reward": math.fsum(rewards.values()) / len(rewards),
        "detail": {"native_rewards": rewards},
    }


def main() -> None:
    NATIVE_REWARD.parent.mkdir(parents=True, exist_ok=True)
    NATIVE_REWARD.unlink(missing_ok=True)
    ADAPTED_REWARD.unlink(missing_ok=True)
    # Harbor reads a written reward even when the test command exits nonzero.
    subprocess.run(["bash", "/tests/test.sh"], check=False)
    ADAPTED_REWARD.write_text(json.dumps(component_reward(json.loads(NATIVE_REWARD.read_text()))) + "\n")


if __name__ == "__main__":
    main()
