# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score canonical completion benchmarks independently of retained catalog packages."""

# Optional scorer dependencies load only for the selected environment.
# ruff: noqa: PLC0415

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from verifyit.grade import GradingInfraError, InvalidTask, Reward, infra_error, invalid_task, scored


def normalize_numeric_answer(value: str) -> str:
    return re.sub(r",|\$|(?s:.*#### )|\.$", "", value).lower()


def grade_gsm8k_cot(document: dict, response: str) -> Reward:
    expected = document["answer"].rsplit("####", 1)[-1].strip()
    strict = re.findall(r"The answer is (\-?[0-9\.\,]+).", response)
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
    extracted = {
        "strict-match": strict[0] if strict else "[invalid]",
        "flexible-extract": next((value for value in flexible[-1] if value), "[invalid]") if flexible else "[invalid]",
    }
    metrics = {
        f"exact_match,{name}": float(normalize_numeric_answer(value) == normalize_numeric_answer(expected))
        for name, value in extracted.items()
    }
    return scored(metrics["exact_match,flexible-extract"], metrics=metrics, extracted=extracted)


def grade_minerva_completion(document: dict, response: str) -> Reward:
    from math_verify import parse, verify

    from .amc import last_boxed_answer
    from .minerva import is_equiv, normalize_final_answer

    matches = re.findall(
        r"Final Answer: The final answer is(.*?). I hope it is correct.", response + "I hope it is correct."
    )
    answer = normalize_final_answer(matches[0].strip() if matches else "[invalidanswer]")
    expected = normalize_final_answer(last_boxed_answer(document["solution"]))
    metrics = {
        "exact_match,none": float(is_equiv(answer, expected)),
        "math_verify,none": float(bool(verify(gold=parse(document["solution"]), target=parse(response)))),
    }
    return scored(metrics["exact_match,none"], metrics=metrics, extracted=answer)


def grade_bbh_cot(document: dict, response: str) -> Reward:
    matches = re.findall(r"(?<=the answer is )(.*)(?=.)", response)
    answer = matches[0] if matches else "[invalid]"
    value = float(answer == document["target"])
    return scored(value, metrics={"exact_match,get-answer": value}, extracted=answer)


def score(scorer: str, config: dict, response: str) -> Reward:
    if scorer == "gsm8k_cot":
        return grade_gsm8k_cot(config["document"], response)
    if scorer == "minerva_math":
        return grade_minerva_completion(config["document"], response)
    if scorer == "bbh_cot":
        return grade_bbh_cot(config["document"], response)
    if scorer == "harness_generation":
        from .harness_completion import grade_harness_generation_candidate

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
