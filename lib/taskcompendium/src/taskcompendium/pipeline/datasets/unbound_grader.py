# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An explicitly unavailable source evaluator with its private binding contract."""

import json
import os
from pathlib import Path


def main() -> None:
    tests = Path(os.environ["VERIFYIT_TESTS_DIR"])
    logs = Path(os.environ["VERIFYIT_LOGS_DIR"])
    contract = json.loads((tests / "config.json").read_text())
    detail = {
        "error": "source evaluator is unbound",
        "evaluator": contract["evaluator"],
        "source_revision": contract["source_revision"],
        "runtime_requirements": contract["runtime_requirements"],
    }
    (logs / "verdict.json").write_text(json.dumps({"status": "infra_error", "reward": 0.0, "detail": detail}))


if __name__ == "__main__":
    main()
