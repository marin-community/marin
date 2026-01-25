# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AIME RL environment following the DeepMath approach.

This environment uses the zwhe99/DeepMath-103K dataset for training and
evaluates using the DeepMath grading approach (dual verification with
openmathinst math_equal and math_verify).

The grading logic is copied directly from:
- DeepMath/utils/openmathinst_utils.py (extract_answer, math_equal)
- DeepMath/utils/reward_utils/reward_func.py (reward_func)

Reference: https://github.com/zwhe99/DeepMath
"""

from __future__ import annotations

import contextlib
import logging
import re
import signal
from dataclasses import dataclass, field
from math import isclose
from typing import Any, ClassVar

import jax
import numpy as np
from datasets import load_dataset

from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import Rollout, RolloutGroup

from .base import MarinEnv

logger = logging.getLogger(__name__)


# ============================================================
# Grading utilities copied from DeepMath/utils/openmathinst_utils.py
# (NVIDIA / Microsoft / OpenAI / Hendrycks licenses apply to this section)
# ============================================================


class _TimeoutException(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise _TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _fix_fracs(string):
    while "\\frac " in string:
        string = string.replace("\\frac ", "\\frac")
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _str_is_int(x: str) -> bool:
    try:
        stripped = _strip_properly_formatted_commas(x)
        val = float(stripped)
        return abs(val - round(val)) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    if "_" in x:
        x = x.split("_")[0]
    return int(float(x))


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _remove_right_units(expr):
    if "\\text" in expr:
        try:
            splits = re.split(r"\\text\s*{\s*", expr)
            assert len(splits) == 2 and splits[0] not in ("", "(")
            return splits[0]
        except AssertionError:
            pass

    if "\\text{" in expr:
        return re.sub(r"\\text{([^}]+)}", r"\1", expr)
    elif "\\mbox{" in expr:
        splits = expr.split("\\mbox{")
        if len(splits) == 2:
            return splits[0]
        else:
            return expr
    else:
        return expr


def _process_and_or_inside_text(string):
    string = re.sub(r"\s*\\text{\s*(or|and)\s*}\s*", ",", string)
    string = re.sub(r",\s*,", ",", string)
    return string


def _remove_left_and_right(expr):
    expr = re.sub(r"\\left", "", expr)
    expr = re.sub(r"\\right", "", expr)
    return expr


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\s*\w+)", r"\\sqrt{\1}", string)
    return _string


def _fix_interval(expr):
    if "\\in " in expr:
        return expr.split("\\in ")[1].strip()
    return expr


def _inject_implicit_mixed_fraction(step: str):
    p1 = re.compile(r"(\d+) *\\frac{(\d+)}{(\d+)}")

    def replacer(match):
        whole_part = match.group(1)
        numerator = match.group(2)
        denominator = match.group(3)
        if whole_part:
            return f"{whole_part} + {numerator}/{denominator}"
        else:
            return f"{numerator}/{denominator}"

    step = p1.sub(replacer, step)
    return step


def _normalize_answer_string(expr: str | None) -> str | None:
    if expr is None:
        return None

    expr = _remove_left_and_right(expr)
    expr = _process_and_or_inside_text(expr)
    expr = _remove_right_units(expr)
    expr = _fix_interval(expr)
    for surround_str in ["\\\\text", "\\\\mathrm", "\\\\mathcal", "\\\\textbf", "\\\\textit"]:
        expr = expr.replace(surround_str, "")
        pattern = f"^{surround_str}" + r"\{(?P<text>.+?)\}$"
        m = re.search(pattern, expr)
        if m is not None:
            expr = m.group("text")

    expr = expr.replace(r"\!", "")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace("^{\\circ}", "")

    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "p.m.",
        "PM",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)

    if "day" in expr:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_expressed = any(day in expr for day in days)
        if not weekday_expressed:
            expr = re.sub(r"day(s)?", "", expr)

    expr = re.sub(r"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = _fix_sqrt(expr)
    expr = _fix_fracs(expr)

    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = _inject_implicit_mixed_fraction(expr)
    expr = expr.replace(" ", "")

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def _is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num
        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def _normalize(answer) -> str:
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer)) or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")
    return answer


def _format_intervals(prediction):
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }
    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)
            if key == "Interval(":
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":
                return f"({inner_content}]"
            elif key == "Interval.open(":
                return f"({inner_content})"
    return prediction


def _symbolic_equal(a, b, tolerance, timeout=10.0):
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                with _time_limit(timeout):
                    return f(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with _time_limit(timeout):
            if sympy.simplify(a - b) == 0:
                return True
    except Exception:
        pass

    try:
        with _time_limit(timeout):
            if isclose(sympy.N(a), sympy.N(b), rel_tol=tolerance):
                return True
    except Exception:
        pass
    return False


def math_equal(
    prediction: bool | float | str,
    reference: float | str,
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    check_antlr_version: bool = True,
) -> bool:
    """Exact match of math expressions (from DeepMath/utils/openmathinst_utils.py)."""
    from sympy.parsing.sympy_parser import parse_expr

    prediction = _normalize(prediction)
    reference = _normalize(reference)

    prediction_normalized = _normalize_answer_string(prediction)
    reference_normalized = _normalize_answer_string(reference)
    if prediction_normalized is None or reference_normalized is None:
        return False
    prediction = prediction_normalized
    reference = reference_normalized

    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]

    # 0. string comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:  # 1. numerical equal
        pred_digit = _is_digit(prediction)
        ref_digit = _is_digit(reference)
        if pred_digit[0] and ref_digit[0]:
            pred_num = pred_digit[1]
            ref_num = ref_digit[1]
            assert pred_num is not None and ref_num is not None
            if include_percentage:
                gt_result = [ref_num / 100, ref_num, ref_num * 100]
            else:
                gt_result = [ref_num]
            for item in gt_result:
                try:
                    if isclose(item, pred_num, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    prediction = _format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_pt, ref_pt, include_percentage, tolerance, check_antlr_version=False)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts, strict=True)
            ):
                return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance, check_antlr_version=False)
                for i in range(len(pred_parts))
            ):
                return True
            else:
                return False

    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_pt, ref_pt, include_percentage, tolerance, check_antlr_version=False)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts, strict=True)
            ):
                return True

    if reference.startswith("\\begin{pmatrix}") and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items):
                if all(
                    math_equal(ref, pred, include_percentage, tolerance, check_antlr_version=False)
                    for ref, pred in zip(ref_matrix_items, pred_matrix, strict=True)
                ):
                    return True
        except Exception:
            pass

    return _symbolic_equal(prediction, reference, tolerance, timeout)


def extract_answer(string: str, extract_from_boxed: bool = True, extract_regex: str = r"The final answer is (.+)$"):
    """Extract answer string from \\boxed expression (from DeepMath/utils/openmathinst_utils.py)."""
    if not extract_from_boxed:
        match = re.search(extract_regex, string)
        if match:
            return match.group(1)
        return None

    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
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

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


# ============================================================
# End of grading utilities from DeepMath
# ============================================================


@dataclass
class DeepMathDataExample:
    """Single DeepMath data example."""

    raw_prompt: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    example_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AimeEnv(MarinEnv):
    """AIME RL environment following DeepMath-Zero approach.

    Uses zwhe99/DeepMath-103K for training data. Evaluates using
    the exact DeepMath grading logic:
    1. Check for </think> tag (format_correct)
    2. Extract \\boxed{} answer from post-think portion
    3. Verify with BOTH openmathinst math_equal AND math_verify
    4. Correct if either verifier agrees
    5. Score: +1.0 correct, -1.0 incorrect
    """

    def __init__(
        self,
        tokenizer=None,
        max_train_examples: int | None = None,
        max_eval_examples: int | None = None,
        seed: int | None = None,
        train_dataset_name: str = "zwhe99/DeepMath-103K",
        eval_dataset_name: str | None = None,
        overlong_buffer_len: int = 2048,
        overlong_penalty_factor: float = 1.0,
    ) -> None:
        """Initialize the AIME environment.

        Args:
            tokenizer: Tokenizer for the model (optional).
            max_train_examples: Maximum number of training examples to use.
            max_eval_examples: Maximum number of evaluation examples to use.
            seed: Random seed for sampling.
            train_dataset_name: HuggingFace dataset name for training.
            eval_dataset_name: HuggingFace dataset name for evaluation.
            overlong_buffer_len: Buffer length for overlong penalty (DeepMath uses 2048).
            overlong_penalty_factor: Penalty factor for overlong responses (DeepMath uses 1.0).
        """
        self.tokenizer = tokenizer
        self.max_train_examples = max_train_examples
        self.max_eval_examples = max_eval_examples
        self._rng = np.random.default_rng(seed)
        self.train_dataset_name = train_dataset_name
        self.eval_dataset_name = eval_dataset_name
        self.overlong_buffer_len = overlong_buffer_len
        self.overlong_penalty_factor = overlong_penalty_factor

        # Load training data
        self.train_examples = self._load_dataset(self.train_dataset_name, "train", max_train_examples)

        # Load eval data (use a held-out portion of train if no separate eval)
        if eval_dataset_name:
            self.eval_examples = self._load_dataset(eval_dataset_name, "eval", max_eval_examples)
        else:
            # Use last 10% of training data for evaluation
            eval_start = int(len(self.train_examples) * 0.9)
            self.eval_examples = self.train_examples[eval_start:]
            self.train_examples = self.train_examples[:eval_start]
            if max_eval_examples and len(self.eval_examples) > max_eval_examples:
                self.eval_examples = self.eval_examples[:max_eval_examples]

        logger.info(
            "Initialized AimeEnv with %d train examples and %d eval examples from %s.",
            len(self.train_examples),
            len(self.eval_examples),
            train_dataset_name,
        )

    # Known dataset schemas: maps dataset name -> (split, problem_key, answer_key)
    DATASET_SCHEMAS: ClassVar[dict[str, dict[str, str]]] = {
        "zwhe99/DeepMath-103K": {"split": "train", "problem_key": "question", "answer_key": "final_answer"},
        "math-ai/aime25": {"split": "test", "problem_key": "problem", "answer_key": "answer"},
    }

    def _load_dataset(
        self,
        dataset_name: str,
        split_name: str,
        limit: int | None,
    ) -> list[DeepMathDataExample]:
        """Load and process the dataset."""
        logger.info("Loading dataset %s...", dataset_name)

        schema = self.DATASET_SCHEMAS.get(dataset_name, {})
        hf_split = schema.get("split", "train")
        problem_key = schema.get("problem_key", "question")
        answer_key = schema.get("answer_key", "final_answer")

        dataset = load_dataset(dataset_name, split=hf_split)

        examples = []
        for idx, item in enumerate(dataset):
            example_id = f"{split_name}_{idx}"

            raw_prompt = item[problem_key]
            raw_answer = item[answer_key]

            metadata = {}
            if "difficulty" in item:
                metadata["difficulty"] = item["difficulty"]
            if "topic" in item:
                metadata["topic"] = item["topic"]

            example = DeepMathDataExample(
                raw_prompt=raw_prompt,
                raw_answer=raw_answer,
                processed_prompt=raw_prompt,
                processed_answer=raw_answer,
                example_id=example_id,
                metadata=metadata,
            )
            examples.append(example)

            if limit is not None and len(examples) >= limit:
                break

        return examples

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample prompts, evaluate responses, and create rollouts."""

        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported mode: {mode}")

        available_examples = self.train_examples if mode == "train" else self.eval_examples
        if not available_examples:
            raise ValueError(f"No examples available for mode '{mode}'")

        n_to_sample = min(n_examples, len(available_examples))
        if isinstance(prng_key, int):
            seed = prng_key
        else:
            seed = jax.random.randint(prng_key, (), 0, 1_000_000).item()
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(available_examples), size=n_to_sample, replace=False)
        sampled_examples = [available_examples[int(idx)] for idx in indices]

        # Build prompts as chat messages with system prompt
        prompts = []
        for example in sampled_examples:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": example.processed_prompt})
            prompts.append(messages)
        completions = inference_ctx.batch_completions(
            prompts=prompts,
            temperature=temperature,
            n=n_generations,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            stop=stop,
            system_prompt=system_prompt,
        )

        rollout_groups: list[RolloutGroup] = []
        total_choices = 0
        reward_sum = 0.0
        format_sum = 0.0
        correct_sum = 0.0
        response_lengths: list[int] = []
        truncated_count = 0

        for example, completion in zip(sampled_examples, completions, strict=True):
            group_rollouts: list[Rollout] = []

            for choice in completion.choices:
                reward, fmt_score, correct_score = self._score_choice(
                    example=example,
                    response_text=choice.message.content,
                    finish_reason=choice.finish_reason,
                    max_tokens=max_tokens,
                )

                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=example.processed_prompt,
                    choice=choice,
                    env_name="aime",
                    env_example_id=example.example_id,
                    reward=reward,
                    correctness_reward=correct_score,
                    temperature=temperature,
                    top_k=top_k,
                    system_prompt=system_prompt,
                )

                group_rollouts.append(rollout)
                total_choices += 1
                reward_sum += reward
                format_sum += fmt_score
                correct_sum += correct_score
                response_lengths.append(rollout.response_tokens.size)

                if choice.finish_reason == "length":
                    truncated_count += 1

            if group_rollouts:
                rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices; cannot compute metrics")

        prefix = f"aime.{mode}"
        metrics = {
            f"{prefix}_mean_reward": reward_sum / total_choices,
            f"{prefix}_format_accuracy": format_sum / total_choices,
            f"{prefix}_correct_accuracy": correct_sum / total_choices,
            f"{prefix}_mean_response_tokens": float(np.mean(response_lengths)),
            f"{prefix}_std_response_tokens": float(np.std(response_lengths)),
            f"{prefix}_min_response_tokens": float(np.min(response_lengths)),
            f"{prefix}_max_response_tokens": float(np.max(response_lengths)),
            f"{prefix}_total_responses": float(total_choices),
            f"{prefix}_sampled_examples": float(len(sampled_examples)),
            f"{prefix}_truncated_percentage": float(truncated_count) / total_choices,
        }

        return rollout_groups, metrics

    def _score_choice(
        self,
        example: DeepMathDataExample,
        response_text: str,
        finish_reason: str,
        max_tokens: int | None = None,
    ) -> tuple[float, float, float]:
        """Score a single generated response following the exact DeepMath reward logic.

        From DeepMath/utils/reward_utils/reward_func.py:
        1. Check for </think> tag -> format_correct
        2. If format_correct, extract answer from post-think portion
        3. Verify with openmathinst math_equal (omi_correct)
        4. Verify with math_verify parse+verify (mathv_correct)
        5. acc = format_correct and (omi_correct or mathv_correct)
        6. score = +1.0 if acc else -1.0

        Returns:
            Tuple of (reward, format_score, correct_score).
        """
        from math_verify import parse, verify

        solution_str = response_text.strip()
        ground_truth = example.processed_answer

        # Check format: DeepMath requires </think> tag
        if "</think>" in solution_str:
            # Only look at post-think portion for answer extraction
            solution_str_for_grading = solution_str.split("</think>")[-1]
            format_correct = True
        else:
            solution_str_for_grading = solution_str
            format_correct = False

        format_score = float(format_correct)

        omi_correct = False
        mathv_correct = False

        if format_correct:
            # Method 1: openmathinst extract_answer + math_equal
            try:
                omi_pred = extract_answer(solution_str_for_grading, extract_from_boxed=True)
                if omi_pred is not None:
                    omi_correct = math_equal(omi_pred, ground_truth, check_antlr_version=False)
            except Exception:
                omi_correct = False

            # Method 2: math_verify parse + verify
            try:
                mathv_pred = parse(solution_str_for_grading)
                mathv_correct = verify(
                    parse(f"\\boxed{{${ground_truth}$}}"),
                    mathv_pred,
                    float_rounding=6,
                    numeric_precision=15,
                    strict=True,
                    timeout_seconds=3,
                )
            except Exception:
                mathv_correct = False

        acc = format_correct and (omi_correct or mathv_correct)
        correct_score = float(acc)

        # DeepMath reward: +1 for correct, -1 for incorrect
        reward = 1.0 if acc else -1.0

        # Apply overlong penalty (from DeepMath's overlong_buffer config)
        if finish_reason == "length" and max_tokens is not None:
            reward = min(reward, -self.overlong_penalty_factor)

        return reward, format_score, correct_score

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Sample evaluation examples deterministically."""
        if not self.eval_examples:
            return []

        eval_key = jax.random.PRNGKey(42)
        n_to_sample = min(n_examples, len(self.eval_examples))
        indices = jax.random.choice(eval_key, len(self.eval_examples), shape=(n_to_sample,), replace=False)
        return [
            {
                "prompt": self.eval_examples[int(idx)].processed_prompt,
                "answer": self.eval_examples[int(idx)].processed_answer,
                "example_id": self.eval_examples[int(idx)].example_id,
            }
            for idx in indices
        ]
