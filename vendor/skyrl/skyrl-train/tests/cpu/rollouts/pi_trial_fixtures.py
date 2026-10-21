"""Independent attributed two-turn Pi evidence for shared scenario tests."""

from copy import deepcopy

USER = {"role": "user", "content": "Make a file"}
ACTION = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"echo 1"}'}},
    ],
}
OBS = {"tool_call_id": "call-1", "content": "one", "authored_by": "student"}


def sampled(request_id, messages, assistant, *, model="student", response_id=777):
    return {
        "request_id": request_id,
        "model": model,
        "messages": messages,
        "assistant": assistant,
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "chat_template_kwargs": {},
        "prompt_token_ids": [20, 21],
        "response_token_ids": [response_id],
        "logprobs": [-0.25],
    }


def trial(*, dagger=False, teacher_suffix=False):
    first = sampled("student-0", [USER], ACTION)
    label = sampled("label-0", [USER], ACTION, model="teacher", response_id=888)
    visited = [USER, ACTION, {"role": "tool", "tool_call_id": "call-1", "content": "one"}]
    second = sampled(
        "teacher-1" if teacher_suffix else "student-1",
        visited,
        {"role": "assistant", "content": "Done"},
        model="teacher" if teacher_suffix else "student",
        response_id=999,
    )
    label_second = (
        deepcopy(second)
        if teacher_suffix
        else sampled("label-1", visited, {"role": "assistant", "content": "Done"}, model="teacher", response_id=888)
    )
    return deepcopy(
        {
            "schema_version": 1,
            "trial_id": "trial-1",
            "policy_step": 3,
            "turns": [
                {
                    "index": 0,
                    "actor": "student",
                    "executed": first,
                    "executed_assistant": first["assistant"],
                    "observations": [OBS],
                    "teacher_label": label if dagger else None,
                },
                {
                    "index": 1,
                    "actor": "teacher" if teacher_suffix else "student",
                    "executed": second,
                    "executed_assistant": second["assistant"],
                    "observations": [],
                    "teacher_label": label_second if dagger else None,
                },
            ],
            "planned_executors": ["student", "teacher" if teacher_suffix else "student"] if dagger else None,
            "verifier_status": "graded",
            "verifier_reward": 0.0,
            "verifier_passed": False,
        }
    )
