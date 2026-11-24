# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a dataset's source scorer and retain its structured verdict."""

# Each task ships only its scorer's modules and optional dependencies.
# ruff: noqa: PLC0415

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verifyit.grade import GradingInfraError, InvalidTask, Reward, Status, infra_error, invalid_task
from verifyit.spec import Constraint, JudgeSpec, parse_spec


def score(scorer: str, config: dict[str, Any], response: str, judge: JudgeSpec | None = None) -> Reward:
    """Grade ``response`` with the named source scorer; judged scorers require ``judge``."""
    # Scorers load only the dependencies of the selected dataset's environment.
    if scorer == "pdbthink":
        from .pdbthink import grade_pdbthink

        return grade_pdbthink(config, response)
    if scorer == "genomeqa":
        from .genomeqa import grade_genomeqa

        return grade_genomeqa(config["expected"], config["options"], config["cell_weight"], response)
    if scorer == "gsm8k_cot":
        from .math import grade_gsm8k_cot

        return grade_gsm8k_cot(config["document"], response)
    if scorer == "minerva_math":
        from .math import grade_minerva_completion

        return grade_minerva_completion(config["document"], response)
    if scorer == "bbh_cot":
        from .choices import grade_bbh_cot

        return grade_bbh_cot(config["document"], response)
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

        assert judge is not None, "A judged olympiad scorer requires --judge-spec"
        verdict = grade_judge_candidate(judge, answer)
        return Reward(verdict.reward, verdict.status, {**verdict.detail, "extracted": answer, "method": "llm_judge"})
    if scorer == "facts_grounding":
        from verifyit.modes.grade_judge import judge_reply

        from .facts_grounding import grade_facts_grounding

        assert judge is not None, "The FACTS Grounding scorer requires --judge-spec"
        return grade_facts_grounding(
            config["full_prompt"],
            judge.question,
            judge.references[0],
            response,
            lambda messages: judge_reply(judge, messages),
        )
    if scorer in {"gpqa", "mmlu_pro"}:
        from .choices import grade_gpqa_answer, grade_mmlu_answer

        grade = grade_gpqa_answer if scorer == "gpqa" else grade_mmlu_answer
        return grade(config["expected"], response)
    if scorer == "zeroeval_mcqa":
        from .zeroeval import grade_zeroeval_mcqa

        return grade_zeroeval_mcqa(config["expected"], response)
    if scorer == "lexam_mcq":
        from .lexam import grade_lexam_mcq

        return grade_lexam_mcq(config["expected"], response)
    if scorer in {"mrcr", "graphwalks"}:
        from .long_context import grade_graphwalks, grade_nonce_similarity

        if scorer == "mrcr":
            return grade_nonce_similarity(config["expected"], config["nonce"], response)
        return grade_graphwalks(tuple(config["expected"]), response)
    if scorer == "longbench_v2":
        from .longbench_v2 import grade_longbench_v2

        return grade_longbench_v2(config["expected"], response)
    if scorer == "nupa":
        from .nupa import grade_nupa_candidate

        return grade_nupa_candidate(config["expected"], config["answer_format"], response)
    if scorer == "jee":
        from .jee import grade_jee_candidate

        return grade_jee_candidate(config["expected"], config["question_type"], response)
    if scorer == "harness_generation":
        from .harness_generation import grade_harness_generation_candidate

        return grade_harness_generation_candidate(config["profile"], config["references"], response, config["question"])
    if scorer == "arc_agi":
        from .arc_agi import grade_arc_agi

        return grade_arc_agi(config["expected"], config["weight"], response)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scorer")
    parser.add_argument("--judge-spec", type=Path, help="Judge contract TOML for scorers that call an LLM judge.")
    args = parser.parse_args()
    judge = None
    if args.judge_spec is not None:
        judge = parse_spec(args.judge_spec.read_text())
        assert isinstance(judge, JudgeSpec)
    config = json.loads(Path("/tests/config.json").read_text())
    try:
        result = score(args.scorer, config, Path("/app/answer.txt").read_text(), judge)
    except InvalidTask as error:
        result = invalid_task(str(error))
    except GradingInfraError as error:
        result = infra_error(f"{type(error).__name__}: {error}", **error.detail)
    Path("/app/verdict.json").write_text(json.dumps(asdict(result), allow_nan=False))


if __name__ == "__main__":
    main()
