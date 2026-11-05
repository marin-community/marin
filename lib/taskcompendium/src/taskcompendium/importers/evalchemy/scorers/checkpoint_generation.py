# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source extraction and metrics for checkpoint generation protocols (lm-eval v0.4.12)."""

import re

from math_verify import parse, verify
from verifyit.grade import Reward, scored

from .amc import last_boxed_answer
from .minerva import is_equiv, normalize_final_answer


def normalize_numeric_answer(value: str) -> str:
    return re.sub(r",|\$|(?s:.*#### )|\.$", "", value).lower()


def grade_checkpoint_generation(profile: str, document: dict, response: str) -> Reward:
    if profile == "bbh_cot_fewshot":
        matches = re.findall(r"(?<=the answer is )(.*)(?=.)", response)
        answer = matches[0] if matches else "[invalid]"
        value = float(answer == document["target"])
        return scored(value, metrics={"exact_match,get-answer": value}, extracted=answer)
    if profile == "gsm8k_cot":
        expected = document["answer"].rsplit("####", 1)[-1].strip()
        strict = re.findall(r"The answer is (\-?[0-9\.\,]+).", response)
        flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
        extracted = {
            "strict-match": strict[0] if strict else "[invalid]",
            "flexible-extract": (
                next((value for value in flexible[-1] if value), "[invalid]") if flexible else "[invalid]"
            ),
        }

        metrics = {
            f"exact_match,{name}": float(normalize_numeric_answer(value) == normalize_numeric_answer(expected))
            for name, value in extracted.items()
        }
        return scored(metrics["exact_match,strict-match"], metrics=metrics, extracted=extracted)
    if profile == "minerva_math":
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
    raise ValueError(f"Unsupported checkpoint generation profile: {profile}")
