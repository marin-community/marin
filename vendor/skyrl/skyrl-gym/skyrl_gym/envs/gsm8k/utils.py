# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verifyit.grade import InvalidTask, Status
from verifyit.adapters.skyrl import grade_gsm8k_final_line, grade_gsm8k_extracted

import re

FINAL_ANSWER = re.compile(r"#### (-?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?)")


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible", "final_line"]

    if method == "final_line":
        lines = solution_str.strip().splitlines()
        match = FINAL_ANSWER.fullmatch(lines[-1]) if lines else None
        return match.group(1).replace(",", "") if match is not None else None

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search(r"#### \$?(-?[0-9.,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(1).replace(",", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: 'strict', 'flexible', or 'final_line'. The final-line mode requires
            a standalone final #### number line and compares decimal values.
        format_score: the score for the format
        score: the score for the correct answer
    """
    if method == "final_line":
        try:
            verdict = grade_gsm8k_final_line(ground_truth, solution_str)
        except InvalidTask:
            return 0
        if verdict.status != Status.SCORED or verdict.detail.get("reason") == "missing_final_answer":
            return 0
        return score if verdict.reward == 1.0 else format_score
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    try:
        verdict = grade_gsm8k_extracted(ground_truth, answer)
    except InvalidTask:
        return 0
    if verdict.status != Status.SCORED:
        return 0
    return score if verdict.reward else format_score
