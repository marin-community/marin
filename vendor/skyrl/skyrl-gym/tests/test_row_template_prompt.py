import json

from rolloutengine.models import Prompt

from skyrl_gym.task_sessions import row_template_prompt

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


def test_row_settings_override_the_prompt_options():
    prompt = Prompt(({"role": "user", "content": "hi"},), {"tools": [{"type": "function"}]})
    extras = {"chat_template_kwargs": {"enable_thinking": False}, "tools": TOOLS, "teacher_route": "tool_nonthinking"}
    result = row_template_prompt(prompt, extras)
    assert result.options == {"chat_template_kwargs": {"enable_thinking": False}, "tools": TOOLS}
    assert result.messages == prompt.messages


def test_json_string_settings_are_decoded():
    extras = {"chat_template_kwargs": json.dumps({"enable_thinking": True}), "tools": json.dumps(TOOLS)}
    result = row_template_prompt(Prompt(({"role": "user", "content": "hi"},), {}), extras)
    assert result.options == {"chat_template_kwargs": {"enable_thinking": True}, "tools": TOOLS}


def test_rows_without_settings_keep_the_prompt_unchanged():
    prompt = Prompt(({"role": "user", "content": "hi"},), {"tools": TOOLS})
    for extras in ({}, {"tools": ""}, {"tools": None, "chat_template_kwargs": {}}):
        assert row_template_prompt(prompt, extras).options == {"tools": TOOLS}
