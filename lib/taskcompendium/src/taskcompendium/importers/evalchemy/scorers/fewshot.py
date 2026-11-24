# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score canonical completion benchmarks independently of retained catalog packages."""

# Optional scorer dependencies load only for the selected environment.

import json
import sys
from dataclasses import asdict
from pathlib import Path

from verifyit.grade import GradingInfraError, InvalidTask, Reward, infra_error, invalid_task


def score(scorer: str, config: dict, response: str) -> Reward:
    if scorer == "gsm8k_cot":
        from .math import grade_gsm8k_cot  # noqa: PLC0415 - optional scorer resources

        return grade_gsm8k_cot(config["document"], response)
    if scorer == "minerva_math":
        from .math import grade_minerva_completion  # noqa: PLC0415 - optional scorer resources

        return grade_minerva_completion(config["document"], response)
    if scorer == "bbh_cot":
        from .choices import grade_bbh_cot  # noqa: PLC0415 - optional scorer resources

        return grade_bbh_cot(config["document"], response)
    if scorer == "harness_generation":
        from .harness_generation import grade_harness_generation_candidate  # noqa: PLC0415 - optional scorer

        return grade_harness_generation_candidate(config["profile"], config["references"], response, config["question"])
    raise ValueError(f"Unknown completion scorer: {scorer}")


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    try:
        result = score(sys.argv[1], config, Path("/app/answer.txt").read_text())
    except InvalidTask as error:
        result = invalid_task(str(error))
    except GradingInfraError as error:
        result = infra_error(f"{type(error).__name__}: {error}", **error.detail)
    Path("/app/verdict.json").write_text(json.dumps(asdict(result), allow_nan=False))


if __name__ == "__main__":
    main()
