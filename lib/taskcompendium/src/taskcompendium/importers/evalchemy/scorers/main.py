# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a dataset's source scorer and retain its structured verdict."""

# Each task ships only its scorer's modules and optional dependencies.
# ruff: noqa: PLC0415

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verifyit.grade import GradingInfraError, InvalidTask, Reward, Status, infra_error, invalid_task
from verifyit.spec import Constraint, JudgeSpec, parse_spec


def score(scorer: str, config: dict[str, Any], response: str) -> Reward:
    # Scorers load only the dependencies of the selected dataset's environment.
    if scorer == "boxed_math":
        from .math import grade_boxed_math

        return grade_boxed_math(config["expected"], response)
    if scorer == "amc":
        from .amc import grade_amc

        return grade_amc(config["expected"], response)
    if scorer == "olympiad":
        from .olympiad import grade_olympiad

        result = grade_olympiad(tuple(config["expected"]), response)
        if not config["judged"] or result.status is not Status.SCORED or result.reward == 1:
            return result
        answer = result.detail["extracted"]
        if not answer.strip():
            return result
        from verifyit.modes.grade_judge import grade_judge_candidate

        judge = parse_spec(Path("/tests/judge.toml").read_text())
        assert isinstance(judge, JudgeSpec)
        verdict = grade_judge_candidate(judge, answer)
        return Reward(verdict.reward, verdict.status, {**verdict.detail, "extracted": answer, "method": "llm_judge"})
    if scorer in {"gpqa", "mmlu_pro"}:
        from .choices import grade_gpqa_answer, grade_mmlu_answer

        grade = grade_gpqa_answer if scorer == "gpqa" else grade_mmlu_answer
        return grade(config["expected"], response)
    if scorer in {"mrcr", "graphwalks"}:
        from .long_context import grade_graphwalks, grade_nonce_similarity

        if scorer == "mrcr":
            return grade_nonce_similarity(config["expected"], config["nonce"], response)
        return grade_graphwalks(tuple(config["expected"]), response)
    if scorer == "nupa":
        from .nupa import grade_nupa_candidate

        return grade_nupa_candidate(config["expected"], config["answer_format"], response)
    if scorer == "jee":
        from .jee import grade_jee_candidate

        return grade_jee_candidate(config["expected"], config["question_type"], response)
    if scorer == "harness_generation":
        from .harness_generation import grade_harness_generation_candidate

        return grade_harness_generation_candidate(config["profile"], config["references"], response)
    if scorer in {"ifeval", "ifbench"}:
        from .instruction_spec import InstructionProfile, InstructionSpec
        from .instructions import grade_source_instructions

        spec = InstructionSpec(
            constraints=tuple(Constraint(**constraint) for constraint in config["constraints"]),
            profile=InstructionProfile(config["profile"]),
            prompt=config["prompt"],
        )
        return grade_source_instructions(spec, response)
    raise ValueError(f"Unknown source scorer: {scorer}")


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
