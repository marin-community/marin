# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal lm-evaluation-harness grading for the embedded Micro210 grader.

Reproduces, for the two ``lm_eval:*`` tasks a Micro210 row selects (gsm8k, triviaqa), the
filter pipelines and metrics the harness applies to one complete document — without the
``ConfigurableTask``/``TaskManager`` machinery. The vendored task utils come from the same
runtime-v4 wheels the frozen table pins; the filter and metric semantics replicate
lm-evaluation-harness v0.4.12 (regex/custom filters with ``take_first``, HF-evaluate
``exact_match`` normalization, and the multiple-target any-match rule). The differential
harness compares this module against the real wheel on every selected document.
"""

import ast
import hashlib
import json
import re
import string
from collections.abc import Sequence
from typing import Any

import numpy as np
from eval.contracts.eval_spec import EvalSpec, Grade
from eval.lm_eval_tasks.gsm8k.utils import gsm8k_flexible_extraction_filter
from eval.lm_eval_tasks.gsm8k.utils import process_results as gsm8k_process_results
from eval.lm_eval_tasks.short_answer_extraction import (
    marked_short_answer_filter,
    short_answer_filter,
)

GSM8K_STRICT_PATTERN = r"#### (\-?[0-9\.\,]+)"
INVALID = "[invalid]"


def regex_filter(pattern: str, resps: Sequence[Sequence[str]], fallback: str = INVALID) -> list[list[str]]:
    """lm-eval's RegexFilter with its default group_select=0, applied per response."""
    regex = re.compile(pattern)
    filtered = []
    for inst in resps:
        inner = []
        for resp in inst:
            resp = resp if isinstance(resp, str) else ""
            match = regex.findall(resp)
            if match:
                value = match[0]
                if isinstance(value, tuple):
                    non_empty = [part for part in value if part]
                    value = non_empty[0] if non_empty else fallback
                inner.append(value.strip())
            else:
                inner.append(fallback)
        filtered.append(inner)
    return filtered


def take_first(filtered: list[list[str]]) -> list[list[str]]:
    return [[inst[0]] if inst else [""] for inst in filtered]


def custom_filter(function, resps: Sequence[Sequence[str]], docs: Sequence[dict[str, Any]]) -> list[list[str]]:
    return function(list(resps), list(docs))


def exact_match_hf_evaluate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """HF-evaluate exact_match with the triviaqa task's normalization flags."""
    prediction_array = np.asarray(predictions, dtype=object).astype(str)
    reference_array = np.asarray(references, dtype=object).astype(str)
    prediction_array = np.char.lower(prediction_array)
    reference_array = np.char.lower(reference_array)
    table = string.punctuation.maketrans("", "", string.punctuation)
    prediction_array = np.char.translate(prediction_array, table=table)
    reference_array = np.char.translate(reference_array, table=table)
    return float(np.mean(prediction_array == reference_array))


def hash_string(item: Any) -> str:
    return hashlib.sha256(str(item).encode("utf-8")).hexdigest()


def _variants(spec: EvalSpec, responses: Sequence[Any]) -> dict[str, list[str]]:
    """Apply each configured filter pipeline; return {filter_name: [filtered values]}."""
    task = spec.grader.name.removeprefix("lm_eval:")
    values = [str(response) for response in responses]
    if task == "gsm8k":
        strict = take_first(regex_filter(GSM8K_STRICT_PATTERN, [values]))
        flexible = take_first(custom_filter(gsm8k_flexible_extraction_filter, [values], [spec.doc]))
        return {"strict-match": strict[0], "flexible-extract": flexible[0]}
    if task == "triviaqa":
        strict = take_first(custom_filter(marked_short_answer_filter, [values], [spec.doc]))
        extract = take_first(custom_filter(short_answer_filter, [values], [spec.doc]))
        return {"strict_answer": strict[0], "extract_answer": extract[0]}
    raise ValueError(f"No vendored lm-eval grading for task {task!r}")


def _metrics(spec: EvalSpec, filter_name: str, filtered: list[str]) -> dict[str, float]:
    task = spec.grader.name.removeprefix("lm_eval:")
    if task == "gsm8k":
        return gsm8k_process_results(spec.doc, filtered)
    # lm-eval's doc_to_target parses a rendered "[...]" target back into its list, which
    # sets the multiple-target any-match rule the triviaqa task relies on.
    target = spec.target
    if isinstance(target, str) and len(target) >= 2 and target[0] == "[" and target[-1] == "]":
        try:
            target = ast.literal_eval(target)
        except (SyntaxError, ValueError):
            pass
    golds = target if isinstance(target, list) else [target]
    score = exact_match_hf_evaluate([filtered[0]] * len(golds), [str(gold) for gold in golds])
    return {"exact_match": 1.0 if score > 0.0 else 0.0}


def native_grade(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    """Grade one complete lm-eval document through its filters and metric definitions."""
    variants = _variants(spec, responses)
    metrics: dict[str, float] = {}
    reducer_inputs: dict[str, float] = {}
    primary_filtered: list[str] = []
    for order, (filter_name, filtered) in enumerate(variants.items()):
        raw = _metrics(spec, filter_name, filtered)
        metrics.update({f"{metric},{filter_name}": float(value) for metric, value in raw.items()})
        reducer_inputs.update({f"{metric},{filter_name}": value for metric, value in raw.items()})
        if order == 0:
            primary_filtered = filtered
    audit = {
        "filter": next(iter(variants)),
        "doc_hash": hash_string(json.dumps(spec.doc, indent=2, default=str, ensure_ascii=False)),
        "prompt_hash": hash_string(
            spec.requests[0].prompt if isinstance(spec.requests[0].prompt, str) else json.dumps(spec.requests[0].prompt)
        ),
        "target_hash": hash_string(str(spec.target)),
    }
    return Grade(metrics, audit, reducer_inputs=reducer_inputs, filtered_resps=primary_filtered)
