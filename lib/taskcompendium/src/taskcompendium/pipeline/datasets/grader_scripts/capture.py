# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the source's output-capture comparison to a captured file."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from verifyit.grade import scored
from verifyit.modes.grade_nl2bash import score_capture


def main() -> None:
    config = json.loads((Path(os.environ["VERIFYIT_TESTS_DIR"]) / "config.json").read_text())
    workspace = Path(os.environ["VERIFYIT_WORKSPACE"])
    capture = workspace / "captured" / config["output_path"].lstrip("/")
    if capture.exists():
        reward, errors = score_capture(capture.read_text(errors="replace"), config["expected_output"])
        verdict = scored(float(reward), errors=errors)
    else:
        verdict = scored(0.0, error="Missing capture file")
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "verdict.json").write_text(json.dumps(asdict(verdict)))


if __name__ == "__main__":
    main()
