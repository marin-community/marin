"""Prompt construction and response parsing for named judge protocols."""

import ast
import json
import random
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, JudgePlan, ModelRequest, RequestKind
from eval.graders.answer_equivalence import JUDGE_PROMPT, JudgeLabel
from eval.graders.simpleqa import SIMPLEQA_GRADER_TEMPLATE, _parse_simpleqa_judgment
from eval.contracts.failures import FailureCategory, MalformedJudgeResponse


@lru_cache
def template_text(path: str) -> str:
    return Path(path).read_text()


def _shorten_wildbench(text: str, max_words: int) -> str:
    if max_words > 0 and len(text.split(" ")) > max_words:
        return " ".join(text.split(" ")[:max_words]) + "... (truncated)"
    return text


def wildbench(spec: EvalSpec, responses: Sequence[Any]) -> Grade | JudgePlan:
    doc = spec.doc
    output = str(responses[-1])
    limit = spec.grader.reference["max_words"]
    history = ""
    for message in doc["conversation_input"][:-1]:
        if message["role"] in {"user", "assistant"}:
            history += message["role"].upper() + ": " + message["content"] + "\n\n"
    prompt = template_text(spec.grader.reference["template"])
    for key, value in {
        "history": _shorten_wildbench(history, limit),
        "user_query": _shorten_wildbench(doc["conversation_input"][-1]["content"], limit),
        "model_output": _shorten_wildbench(output, limit),
        "checklist": "".join(f"- {item}\n" for item in doc["checklist"]),
    }.items():
        prompt = prompt.replace("{$" + key + "}", value)
    parser = GraderSpec(
        "wildbench_score",
        {"output_length": len(output), "truncated": _shorten_wildbench(output, limit).endswith("... (truncated)")},
    )
    if not output.strip():
        return wildbench_score_with_reference(parser.reference, '{"score": "1"}')
    return JudgePlan((ModelRequest(RequestKind.GENERATE, prompt, {"temperature": 0, "max_gen_toks": 4096,
                                                               "response_format": {"type": "json_object"}}),), parser,
                     GraderSpec("judge_failed"))


def wildbench_score_with_reference(reference: dict[str, Any], raw: str) -> Grade:
    score = float(json.loads(raw)["score"])
    if not 1 <= score <= 10:
        raise ValueError("WildBench judge rating must be between 1 and 10")
    return Grade({"score": score, "adjusted_score": (score - 5) * 2, "avg_output_length": reference["output_length"],
                  "judge_failed": 0.0},
                 {"judge_raw": raw}, {"avg_output_length": float(not reference["truncated"] and reference["output_length"] > 0)})


def wildbench_score(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    return wildbench_score_with_reference(spec.grader.reference, str(responses[-1]))


def mixeval(spec: EvalSpec, responses: Sequence[Any]) -> JudgePlan:
    from eval.chat_benchmarks.MixEval.mix_eval.prompts.judge_prompts import (
        gpt_judge_for_closeended_freeform, gpt_judge_for_closeended_multiplechoice,
    )

    doc = spec.doc
    kind = spec.grader.reference["kind"]
    if kind == "freeform":
        gold = "; ".join(f"<answer {index + 1}> {answer}" for index, answer in enumerate(doc["target"]))
        prompt = gpt_judge_for_closeended_freeform(doc["prompt"], gold, str(responses[-1]))
    else:
        choices = "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(doc["options"]))
        prompt = gpt_judge_for_closeended_multiplechoice(doc["prompt"], choices, str(responses[-1]))
    reference = {"kind": kind, "target": doc["target"],
                 "num_options": len(doc["options"]) if kind == "multichoice" else 0}
    return JudgePlan((ModelRequest(RequestKind.GENERATE, prompt, {"max_gen_toks": 999}),),
                     GraderSpec("mixeval_score", reference), max_attempts=10,
                     on_exhausted=GraderSpec("mixeval_fallback", reference))


def mixeval_score(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    raw = str(responses[-1])
    reference = spec.grader.reference
    if reference["kind"] == "multichoice":
        match = re.search(r"\[\[([A-Z])\]\]", raw) or re.search(r"\[([A-Z])\]", raw)
        if match is None:
            raise MalformedJudgeResponse("MixEval judge returned no choice")
        score = float(match.group(1) == chr(65 + reference["target"][0]))
    else:
        match = re.search(r"\[\[(\d+\.?\d*)\]\]", raw) or re.search(r"\[(\d+\.?\d*)\]", raw)
        if match is None:
            raise MalformedJudgeResponse("MixEval judge returned no correctness score")
        score = float(ast.literal_eval(match.group(1)))
        if not 0 <= score <= 1:
            raise MalformedJudgeResponse("MixEval judge returned an out-of-range correctness score")
    return Grade({"accuracy": score}, {"judge_raw": raw})


def mixeval_fallback(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    # Preserve native fallback distributions without sharing mutable RNG state
    # with asynchronous transport retries or changing a grade on checkpoint replay.
    seed = json.dumps([spec.task, spec.namespace, spec.source_id, spec.trial], sort_keys=True)
    rng = random.Random(seed)
    reference = spec.grader.reference
    if reference["kind"] == "multichoice":
        choice = rng.choice([chr(65 + index) for index in range(reference["num_options"])])
        score = float(choice == chr(65 + reference["target"][0]))
    else:
        score = round(rng.random(), 1)
    return Grade({"accuracy": score}, {"judge_raw": str(responses[-1]),
                 "judge_fallback": "malformed_response_exhaustion", "judge_fallback_rng": "stable_item_v1"})


def simpleqa(spec: EvalSpec, responses: Sequence[Any]) -> JudgePlan:

    prompt = SIMPLEQA_GRADER_TEMPLATE.format(
        question=spec.doc["question"], target=spec.target, predicted_answer=str(responses[-1]),
    )
    return JudgePlan(
        (ModelRequest(RequestKind.GENERATE, prompt, {"temperature": 0, "max_gen_toks": 2048}),),
        GraderSpec("simpleqa_label"),
        GraderSpec("judge_failed"),
    )

def simpleqa_label(spec: EvalSpec, responses: Sequence[Any]) -> Grade:

    label = _parse_simpleqa_judgment(str(responses[-1])).label.value
    return Grade({"accuracy": float(label == "correct"), "incorrect": float(label == "incorrect"),
                  "not_attempted": float(label == "not_attempted"), "judge_failed": 0.0},
                 {"judge_label": label, "judge_raw": str(responses[-1])})

def equivalence_judge(spec: EvalSpec, responses: Sequence[Any]) -> JudgePlan:
    prompt = JUDGE_PROMPT.format(
        question=spec.grader.reference["question"],
        reference_answers=json.dumps(spec.grader.reference["answers"], ensure_ascii=False),
        candidate_answer=str(responses[-1]),
    )
    return JudgePlan(
        (ModelRequest(RequestKind.GENERATE, prompt, {"temperature": 0, "max_gen_toks": 2048}),),
        GraderSpec("equivalence_label", {"answer": str(responses[-1])}),
        GraderSpec("judge_failed"),
    )

def equivalence_label(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    raw = str(responses[-1])
    label = JudgeLabel(raw.strip().lower())
    return Grade(
        {
            "accuracy": float(label is JudgeLabel.CORRECT),
            "incorrect": float(label is JudgeLabel.INCORRECT),
            "not_attempted": float(label is JudgeLabel.NOT_ATTEMPTED),
            "judge_failed": 0.0,
        },
        {
            "method": "llm_judge",
            "judge_label": label.value,
            "judge_raw": raw,
        },
        filtered_resps=[spec.grader.reference["answer"]],
    )


def judge_failed(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    """Record an unavailable or invalid judge response without assigning a score."""
    metrics = {"judge_failed": 1.0}
    if "accuracy" in spec.grader.reference:
        metrics["accuracy"] = spec.grader.reference["accuracy"]
    return Grade(metrics, {
        "judge_label": None, "judge_raw": None,
        "failure_category": FailureCategory.GRADER_INFRASTRUCTURE.value,
        "judge_error": spec.grader.reference["judge_error"],
    })
