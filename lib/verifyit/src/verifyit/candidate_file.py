# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an answer file with in-process candidate semantics."""

import argparse
import json
from pathlib import Path

from verifyit.candidate import grade_candidate
from verifyit.grade import local_output_path, scored, write_reward
from verifyit.spec import parse_spec


def main(argv: list[str] | None = None) -> int:
    """Grade the full answer text using explicitly supplied task resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--resources-manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = parse_spec(args.spec.read_text())
    paths = json.loads(args.resources_manifest.read_text())
    resources = {path: (args.resources_manifest.parent / path).read_bytes() for path in paths}
    answer = local_output_path(args.answer, args.workspace)
    reward = (
        grade_candidate(spec, answer.read_text(), resources) if answer.is_file() else scored(0.0, reason="no_output")
    )
    write_reward(args.logs_dir, reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
