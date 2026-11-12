"""PDBThink's deterministic scorer, retaining provider termination signals."""

from collections.abc import Sequence
from typing import Any

from eval.completion_response import CompletionText
from eval.contracts.eval_spec import EvalSpec, Grade
from eval.runtime.checkpoints import input_digest


def pdbthink(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    # PDBThink is an optional benchmark dependency.
    from pdbthink.scoring import score_response

    response = responses[-1]
    truncated = refusal = tool_call = False
    if isinstance(response, CompletionText):
        choice = response.response.raw_choice
        message = choice.get("message", {})
        truncated = response.response.finish_reason == "length"
        refusal = bool(message.get("refusal")) or response.response.finish_reason == "content_filter"
        tool_call = bool(message.get("tool_calls") or message.get("function_call"))
    outcome = score_response(
        str(response),
        spec.grader.reference["answer_schema"],
        dict(spec.target),
        parameters=dict(spec.grader.reference["parameters"]),
        truncated=truncated,
        provider_refusal=refusal,
    )
    score = 0.0 if tool_call else float(outcome["score"]["score"])
    row = {
        **spec.doc["report_metadata"],
        "prompt_sha256": input_digest(spec.requests[0].prompt),
        "completion_index": spec.trial or 0,
        "score": score,
        "correct": bool(outcome["score"].get("correct")) and not tool_call,
        "detail": outcome["score"],
        "parsed_answer": outcome["parsed"],
        "format_error": outcome["format_error"],
        "refusal": outcome["refusal"],
        "truncated": outcome["truncated"],
        "api_error": False,
        "tool_call": tool_call,
    }
    return Grade(
        {"macro_score": score, "micro_score": score},
        audit={"pdbthink": outcome, "tool_call": tool_call},
        reducer_inputs={"pdbthink_row": row},
        filtered_resps=[outcome["parsed"]],
    )
