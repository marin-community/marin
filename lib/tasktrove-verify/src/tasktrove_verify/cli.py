# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``tasktrove-verify SPEC``: grade the task and write reward.json; never fail without a reward."""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from tasktrove_verify.grade import grade
from tasktrove_verify.reward import Reward, infra_error, invalid_task, write_reward
from tasktrove_verify.spec import DEFAULT_WORKSPACE, parse_spec

DEFAULT_LOGS_DIR = "/logs/verifier"

logger = logging.getLogger("tasktrove_verify")


def run(spec_path: Path, workspace: Path) -> Reward:
    try:
        spec = parse_spec(spec_path.read_text())
    except (OSError, ValueError, KeyError) as error:
        return invalid_task(f"cannot read verifier spec {spec_path}: {error}")
    try:
        return grade(spec, tests_dir=spec_path.parent, workspace=workspace)
    except Exception as error:
        logger.error("grader crashed: %s", traceback.format_exc())
        return infra_error(f"{type(error).__name__}: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="path to verifier.toml")
    parser.add_argument("--logs-dir", type=Path, default=Path(DEFAULT_LOGS_DIR))
    parser.add_argument("--workspace", type=Path, default=Path(DEFAULT_WORKSPACE))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    reward = run(args.spec, args.workspace)
    write_reward(args.logs_dir, reward)
    logger.info("reward=%s status=%s", reward.reward, reward.status.value)
    return 0
