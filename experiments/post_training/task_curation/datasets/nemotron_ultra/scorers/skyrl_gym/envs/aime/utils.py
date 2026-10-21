# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
# https://github.com/volcengine/verl/blob/1a62568f801ba35ac1f5387e27232a2df7eac488/verl/utils/reward_score/math_dapo.py

import math
import re
from typing import Optional, Dict, Any


BOXED_ANSWER_INSTRUCTION = (
    " Please reason step by step. At the very end, output your final answer on its "
    "own line in the exact format: 'Answer: \\boxed{ANSWER}'."
)


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n<]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)

    NOTE(shu): parsing the answer before the last character "<"
    Later refactor this into the tokenizer
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_length_penalty(
    correct: bool,
    response_length: Optional[int],
    truncated: bool,
    *,
    length_penalty_weight: float = 0.0,
    target_length: int = 0,
    max_gen_length: int = 0,
    truncated_penalty: float = -2.0,
    min_response_length: int = 16,
) -> tuple[float, Dict[str, Any]]:
    """Compute a tunable, backward-compatible length-shaped reward.

    Design (cosine length-scaled reward, gated on correctness; Yeo et al.,
    "Demystifying Long CoT", DeepSeek / Kimi Long2Short family):

    Let w = length_penalty_weight (the single tunable grid axis). When w == 0,
    this returns the *exact* legacy reward (+1.0 correct / -1.0 incorrect) with
    NO length term and NO truncation distinction, so weight=0 is a clean A/B
    superset of the current reward.

    For w > 0, define a normalized "over-budget" position
        f = clamp( (L - target) / (max_gen - target), 0, 1 )
    where L is the response token length, `target` is the desired concise length,
    and `max_gen` is the generation cap. f = 0 at/under target, f = 1 at the cap.

    Three regimes:
      1. CORRECT & complete (not truncated):
            base = +1.0
            shaped = +1.0 - w * (1 - cos(pi * f)) / 2
         i.e. a cosine ramp: full +1.0 reward when L <= target, smoothly
         decaying by up to w as L -> max_gen. Concise-correct >= verbose-correct
         (requirement (a)), and it never drops below (1 - w), so a correct answer
         is never punished below an incorrect one as long as w <= 2.
      2. INCORRECT & complete (not truncated):
            base = -1.0  (unchanged; we do not length-shape wrong-but-complete
            answers, preserving the "tried and wrong" signal)
      3. TRUNCATED (hit length cap / no terminal box): a DISTINCT, more negative
         penalty `truncated_penalty` (default -2.0), scaled by w:
            shaped = -1.0 + w * (truncated_penalty - (-1.0))
         At w=1 this is exactly `truncated_penalty`; at w=0 it collapses to -1.0
         (legacy). This separates "rambled into truncation" from "wrong-but-
         complete" (requirement (b)).

    Reward-hacking guard (requirement (d)): a correct answer shorter than
    `min_response_length` tokens (degenerate / no-CoT collapse) does NOT receive
    the concision bonus — it is clamped to the verbose-correct floor (1 - w).
    This removes the degenerate optimum of emitting an instant boxed guess.

    Returns (reward, info) where info carries the shaping diagnostics.
    """
    info: Dict[str, Any] = {
        "length_shaped": False,
        "truncated": bool(truncated),
        "response_length": response_length,
        "length_frac": None,
    }

    # weight=0  ->  exact legacy reward.
    if length_penalty_weight == 0.0:
        return (1.0 if correct else -1.0), info

    info["length_shaped"] = True

    # Truncated / over-budget with no terminal answer: distinct penalty regime.
    if truncated:
        reward = -1.0 + length_penalty_weight * (truncated_penalty - (-1.0))
        info["regime"] = "truncated"
        return reward, info

    if not correct:
        # Wrong-but-complete: keep the legacy -1.0 (no length shaping).
        info["regime"] = "incorrect_complete"
        return -1.0, info

    # Correct & complete: cosine concision ramp.
    if response_length is None or max_gen_length <= target_length:
        # Missing length signal -> fall back to legacy +1.0 (no shaping possible).
        info["length_shaped"] = False
        info["regime"] = "correct_no_length_signal"
        return 1.0, info

    f = (response_length - target_length) / float(max_gen_length - target_length)
    f = min(1.0, max(0.0, f))
    info["length_frac"] = f

    # Reward-hacking guard: too-short "correct" answers get the verbose floor,
    # NOT the full concision bonus.
    if response_length < min_response_length:
        info["regime"] = "correct_degenerate_short"
        return 1.0 - length_penalty_weight, info

    cosine_decay = (1.0 - math.cos(math.pi * f)) / 2.0  # 0 at f=0, 1 at f=1
    reward = 1.0 - length_penalty_weight * cosine_decay
    info["regime"] = "correct_complete"
    return reward, info


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    response_length: Optional[int] = None,
    stop_reason: Optional[str] = None,
    length_penalty_weight: float = 0.0,
    target_length: int = 0,
    max_gen_length: int = 0,
    truncated_penalty: float = -2.0,
    min_response_length: int = 16,
    end_think_token: str = "<|end_think|>",
) -> Dict[str, Any]:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string (full response text)
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens
        response_length: Response length in TOKENS (plumbed from the generator).
            None when unavailable -> length shaping is skipped.
        stop_reason: The generation stop reason ("stop"/"length"/...). "length"
            (or a missing terminal box/end-think marker) marks truncation.
        length_penalty_weight: Tunable weight w (the grid axis). 0.0 == legacy
            +1/-1 reward (backward compatible default).
        target_length: Target concise length in tokens (full reward at/under).
        max_gen_length: Generation cap in tokens (the reference for f).
        truncated_penalty: Reward assigned to truncated/no-box rollouts at w=1.
        min_response_length: Reward-hacking guard floor (tokens).
        end_think_token: String marker whose ABSENCE in a length-stopped
            response is a truncation proxy.

    Returns:
        Reward score and other information. With length_penalty_weight=0.0 this
        is identical to the legacy reward (1.0 correct / -1.0 incorrect).
    """
    # Verify on the tail of the FULL response (legacy behavior).
    verify_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters
    correct, pred = verify(verify_str, ground_truth, strict_box_verify, pause_tokens_index)
    acc = correct

    # Truncation detection: the generation hit the length cap (stop_reason ==
    # "length"). We additionally treat a length-stop without an end-think marker
    # as the "rambled into truncation, never reached Answer:" case. If stop_reason
    # is unavailable, the end-think marker absence is used as a proxy only when a
    # length signal is present (otherwise we cannot tell truncation from a short
    # non-thinking answer, so we do NOT flag it).
    truncated = False
    if length_penalty_weight != 0.0:
        if stop_reason is not None:
            truncated = stop_reason == "length"
        elif response_length is not None and max_gen_length > 0:
            # Proxy: at/over the cap and missing the end-think marker.
            truncated = response_length >= max_gen_length and (end_think_token not in solution_str)

    reward, length_info = compute_length_penalty(
        correct=correct,
        response_length=response_length,
        truncated=truncated,
        length_penalty_weight=length_penalty_weight,
        target_length=target_length,
        max_gen_length=max_gen_length,
        truncated_penalty=truncated_penalty,
        min_response_length=min_response_length,
    )

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
        **length_info,
    }
