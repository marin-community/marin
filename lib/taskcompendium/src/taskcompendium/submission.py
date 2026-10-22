# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer-format instructions, compatibility checks, and chat requests.

An answer is the task's semantic result, identified by ``TaskSpec.answer_type``. A submission is
the typed value the task's answer format extracts from a completed attempt for grading. Plain
text, ``{"answer": "12"}``, and ``submit_answer(answer="12")`` all yield ``TextSubmission("12")``.
"""

import json
from dataclasses import dataclass
from typing import Any

from verifyit.spec import ExactSpec, JudgeSpec, PredictedActionSpec, Spec, StructuredExactSpec, UnsolvableSpec

from taskcompendium.direct_chat import unsupported_direct_chat_features
from taskcompendium.models import (
    ANSWER_CALL_NAME,
    ANSWER_FIELD,
    ActionSubmission,
    AnswerCall,
    AssistantToolCalls,
    BaseAnswerFormat,
    Boxed,
    ConversationEvent,
    FinalAction,
    FunctionDefinition,
    GradingAttempt,
    JsonAnswer,
    JsonSubmission,
    JsonValueAnswer,
    PlainText,
    StateSubmission,
    Submission,
    SubmissionFailure,
    TaskSpec,
    TextMessage,
    TextSubmission,
    VerifyitGrader,
    format_conversation,
    verifyit_spec,
)

EJECT_CALL_NAME = "eject_button"
EJECT_REASON_FIELD = "reason"


def eject_button() -> FunctionDefinition:
    """Return the optional terminal tool, independent of task solvability."""
    return FunctionDefinition(
        name=EJECT_CALL_NAME,
        description=(
            "End the interaction if the task cannot be solved or completing it would break guidelines. "
            "Provide a nonempty string reason."
        ),
        parameters={
            "type": "object",
            "properties": {EJECT_REASON_FIELD: {"type": "string"}},
            "required": [EJECT_REASON_FIELD],
            "additionalProperties": False,
        },
    )


def is_ejection(submission: Submission) -> bool:
    """Whether a submitted action is the sole terminal ejection call."""
    return (
        isinstance(submission, ActionSubmission)
        and isinstance(submission.message, AssistantToolCalls)
        and len(submission.message.calls) == 1
        and submission.message.calls[0].name == EJECT_CALL_NAME
    )


def has_ejection(event: ConversationEvent) -> bool:
    """Whether an assistant turn contains the reserved terminal action, including a mixed batch."""
    return isinstance(event, AssistantToolCalls) and any(call.name == EJECT_CALL_NAME for call in event.calls)


def extract_submission(answer_format: BaseAnswerFormat, attempt: GradingAttempt, *, context_length: int) -> Submission:
    """Dispatch terminal ejection before normal extraction, without private grading data."""
    # Completed calls in source history are input, not the current submission.
    for event in attempt.conversation.events[context_length:-1]:
        if has_ejection(event):
            raise SubmissionFailure("The interaction continued after eject_button")
    final = attempt.conversation.events[-1]
    if not has_ejection(final):
        return answer_format.extract(attempt)
    assert isinstance(final, AssistantToolCalls)
    if len(final.calls) != 1:
        raise SubmissionFailure("Ejection requires a single terminal call")
    arguments = final.calls[0].object_arguments()
    if (
        set(arguments) != {EJECT_REASON_FIELD}
        or not isinstance(arguments[EJECT_REASON_FIELD], str)
        or not arguments[EJECT_REASON_FIELD].strip()
    ):
        raise SubmissionFailure("Ejection requires a nonempty string reason")
    return ActionSubmission(final)


def visible_format(answer_format: BaseAnswerFormat) -> BaseAnswerFormat:
    """Return the normal presentation independently of private solvability labels."""
    if isinstance(answer_format, FinalAction) and answer_format.normal_submission is not None:
        return answer_format.normal_submission
    return answer_format


def answer_call_tool() -> dict[str, object]:
    """Return the function definition advertised by the answer-call format."""
    return {
        "type": "function",
        "function": {
            "name": ANSWER_CALL_NAME,
            "description": "Submit the final answer to the task.",
            "parameters": {
                "type": "object",
                "properties": {ANSWER_FIELD: {"type": "string"}},
                "required": [ANSWER_FIELD],
                "additionalProperties": False,
            },
        },
    }


def accepted_submission_types(verifier: Spec) -> tuple[type[Submission], ...]:
    """Declare the submissions a verifyit mode can grade."""
    if isinstance(verifier, StructuredExactSpec):
        return (JsonSubmission, StateSubmission)
    if isinstance(verifier, PredictedActionSpec | UnsolvableSpec):
        return (ActionSubmission,)
    if isinstance(verifier, JudgeSpec):
        return (TextSubmission, ActionSubmission)
    if isinstance(verifier, ExactSpec):
        return (TextSubmission, JsonSubmission, StateSubmission)
    return (TextSubmission,)


def native_action_text(message: TextMessage | AssistantToolCalls) -> str:
    """Serialize a typed final action for a private semantic judge, omitting transport call IDs."""
    if isinstance(message, TextMessage):
        value: dict[str, Any] = {"type": "message", "content": message.content}
    elif len(message.calls) == 1:
        call = message.calls[0]
        value = {"type": "function_call", "name": call.name, "arguments": call.arguments}
    else:
        value = {
            "type": "function_call_batch",
            "calls": [{"name": call.name, "arguments": call.arguments} for call in message.calls],
        }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class SubmissionCompatibility:
    """Whether the answer format preserves the task's result contract, with reasons when it does not."""

    reasons: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.reasons


def submission_compatibility(task: TaskSpec) -> SubmissionCompatibility:
    """Explain which parts of the task its answer format cannot carry."""
    answer_format = task.answer_format
    if not answer_format.supports(task.answer_type):
        return SubmissionCompatibility((f"{type(answer_format).__name__} cannot carry {task.answer_type.value}",))
    if isinstance(task.grader, VerifyitGrader):
        verifier = verifyit_spec(task.grader)
        if isinstance(verifier, UnsolvableSpec) and (
            not isinstance(answer_format, FinalAction)
            or not any(tool.name == EJECT_CALL_NAME for tool in task.final_tools)
        ):
            return SubmissionCompatibility(("Unsolvable tasks require final_action with ejection enabled",))
        accepted = accepted_submission_types(verifier)
        if not any(produced in accepted for produced in answer_format.submission_types):
            return SubmissionCompatibility(("Submission envelope is not accepted by the selected verifier",))
    if isinstance(answer_format, FinalAction) and not task.final_tools:
        return SubmissionCompatibility(("final action requires at least one final tool",))
    if isinstance(visible_format(answer_format), AnswerCall) and any(
        function.name == ANSWER_CALL_NAME for function in task.final_tools
    ):
        return SubmissionCompatibility(("final tool name collides with submit_answer",))
    return SubmissionCompatibility(())


def require_submission_compatibility(task: TaskSpec) -> None:
    """Raise ``ValueError`` naming the parts of the task its answer format cannot carry."""
    compatibility = submission_compatibility(task)
    if not compatibility.compatible:
        raise ValueError(
            f"Answer format {task.answer_format.kind!r} is incompatible: {'; '.join(compatibility.reasons)}"
        )


def submission_instruction(answer_format: BaseAnswerFormat) -> str:
    """Return the instruction added after a conversation prefix."""
    match visible_format(answer_format):
        case PlainText():
            return "Give your answer as plain text."
        case Boxed():
            return "Put your final answer within \\boxed{}."
        case JsonAnswer():
            return f'Give your answer as a JSON object with an "{ANSWER_FIELD}" field.'
        case JsonValueAnswer():
            return "Give your final answer as one JSON value, without Markdown fences."
        case AnswerCall():
            return f'Call {ANSWER_CALL_NAME} with your final answer as the "{ANSWER_FIELD}" string.'
        case FinalAction():
            return ""
        case _:
            raise ValueError(f"Unsupported answer format: {type(answer_format).__name__}")


def render_instruction(task: TaskSpec) -> str:
    """Return readable instruction text for the task's answer format."""
    require_submission_compatibility(task)
    if isinstance(visible_format(task.answer_format), FinalAction):
        return format_conversation(task.context.events)
    return f"{format_conversation(task.context.events)}\n\n{submission_instruction(task.answer_format)}\n"


def conversation_messages(events: tuple[ConversationEvent, ...]) -> list[dict[str, Any]]:
    """Convert conversation events to OpenAI-compatible chat messages."""
    messages: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, TextMessage):
            messages.append({"role": event.role, "content": event.content})
        elif isinstance(event, AssistantToolCalls):
            messages.append(
                {
                    "role": "assistant",
                    "content": event.content,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, separators=(",", ":"), ensure_ascii=False),
                            },
                        }
                        for call in event.calls
                    ],
                }
            )
        else:
            messages.append({"role": "tool", "tool_call_id": event.call_id, "content": event.content})
    return messages


def chat_request(task: TaskSpec) -> dict[str, Any]:
    """Prepare the conversation and tools for the task's answer format."""
    unsupported = unsupported_direct_chat_features(task)
    if unsupported:
        raise NotImplementedError(f"Direct chat cannot satisfy requirements: {', '.join(unsupported)}")
    require_submission_compatibility(task)
    answer_format = visible_format(task.answer_format)
    messages = conversation_messages(task.context.events)
    instruction = submission_instruction(answer_format)
    if instruction:
        messages.append({"role": "user", "content": instruction})
    request: dict[str, Any] = {"messages": messages}
    tools: list[dict[str, object]] = [
        {"type": "function", "function": function.model_dump(exclude_none=True)} for function in task.final_tools
    ]
    if isinstance(answer_format, AnswerCall):
        tools.append(answer_call_tool())
        request.update(tool_choice="required", parallel_tool_calls=False)
    if tools:
        request["tools"] = tools
    if isinstance(answer_format, FinalAction):
        if answer_format.require_call:
            request["tool_choice"] = "required"
        if answer_format.max_calls == 1:
            request["parallel_tool_calls"] = False
    return request
