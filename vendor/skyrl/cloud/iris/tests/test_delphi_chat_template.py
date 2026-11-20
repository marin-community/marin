from pathlib import Path

from jinja2 import Environment, StrictUndefined


_TEMPLATE_PATH = Path(__file__).parents[3] / "chat_templates" / "delphi_v1.jinja2"
_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _render(messages: list[dict[str, object]], *, enable_thinking: bool, add_generation_prompt: bool = True) -> str:
    template = Environment(undefined=StrictUndefined).from_string(_TEMPLATE_PATH.read_text())
    return template.render(
        bos_token="<|begin_of_text|>",
        messages=messages,
        tools=[],
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def test_delphi_v1_disabled_thinking_closes_the_canonical_reasoning_region() -> None:
    rendered = _render([{"role": "user", "content": "Solve this."}], enable_thinking=False)

    assert rendered.endswith(_ASSISTANT_HEADER + "<|start_think|>\n\n<|end_think|>\n\n")


def test_delphi_v1_enabled_thinking_does_not_prefill_a_reasoning_region() -> None:
    rendered = _render([{"role": "user", "content": "Solve this."}], enable_thinking=True)

    assert rendered.endswith(_ASSISTANT_HEADER)


def test_delphi_v1_uses_canonical_tokens_for_historical_reasoning() -> None:
    rendered = _render(
        [
            {"role": "user", "content": "Solve this."},
            {"role": "assistant", "content": "Four.", "reasoning_content": "Two plus two.", "tool_calls": []},
        ],
        enable_thinking=True,
        add_generation_prompt=False,
    )

    assert "<|start_think|>\nTwo plus two.\n<|end_think|>\n\nFour.<|eot_id|>" in rendered
