# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a Harbor answer file with TaskSpec's in-process candidate semantics."""

import argparse
import json
from pathlib import Path

from verifyit.candidate import grade_candidate
from verifyit.grade import DEFAULT_LOGS_DIR, local_output_path, scored, write_reward
from verifyit.spec import DEFAULT_WORKSPACE, parse_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("answer")
    parser.add_argument("--workspace", type=Path, default=Path(DEFAULT_WORKSPACE))
    parser.add_argument("--logs-dir", type=Path, default=Path(DEFAULT_LOGS_DIR))
    args = parser.parse_args()
    spec = parse_spec(args.spec.read_text())
    paths = json.loads((args.spec.parent / "taskcompendium-resources.json").read_text())
    resources = {path: (args.spec.parent / path).read_bytes() for path in paths}
    answer = local_output_path(args.answer, args.workspace)
    reward = (
        grade_candidate(spec, answer.read_text(), resources) if answer.is_file() else scored(0.0, reason="no_output")
    )
    write_reward(args.logs_dir, reward)


if __name__ == "__main__":
    main()
