"""Record what the served chat template applied, so a reasoning check can tell a respected setting from an ignored one."""

from dataclasses import dataclass, field

import pytest
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from taskcompendium.models import ReasoningApplied

HONORS_FLAG = "{% if enable_thinking %}<think>{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"
IGNORES_FLAG = "{% for m in messages %}{{ m.content }}{% endfor %}"
TOKENS = {"<think>": 7, "</think>": 8, "42": 4}


@dataclass
class Tokenizer:
    chat_template: str
    eos_token_id: int = 99

    def decode(self, token_ids, skip_special_tokens):
        words = {token: word for word, token in TOKENS.items()}
        return "".join(words[token] for token in token_ids)


@dataclass
class Engine:
    """Render the template's thinking switch into the prompt tokens only when the template reads it."""

    tokenizer: Tokenizer
    response_ids: list[int]
    model_name: str = "fixture"
    bodies: list[dict] = field(default_factory=list)

    async def tokenize(self, request):
        thinking = request["json"].get("chat_template_kwargs", {}).get("enable_thinking", True)
        honors = "enable_thinking" in self.tokenizer.chat_template
        return {"tokens": [1, 2, *([TOKENS["<think>"]] if thinking and honors else [])]}

    async def chat_completion(self, request):
        self.bodies.append(request["json"])
        text = self.tokenizer.decode(self.response_ids, skip_special_tokens=False)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "token_ids": self.response_ids,
                    "finish_reason": "stop",
                }
            ]
        }


async def evidence(template: str, response_ids: list[int], options: dict):
    client = DirectModelClient(Engine(Tokenizer(template), response_ids))
    output = await client.generate(
        {
            "prompts": [[{"role": "user", "content": "Reply 42."}]],
            "chat_completion_params": [options],
            "sampling_params": {},
        }
    )
    [turn] = output["turn_evidence"]
    return turn


OFF = {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template,options,applied",
    [
        pytest.param(HONORS_FLAG, OFF, ReasoningApplied.OFF, id="honored-off"),
        pytest.param(HONORS_FLAG, {"chat_template_kwargs": {"enable_thinking": True}}, ReasoningApplied.ON, id="on"),
        pytest.param(IGNORES_FLAG, OFF, ReasoningApplied.UNVERIFIED, id="ignored"),
        pytest.param(HONORS_FLAG, {}, ReasoningApplied.UNVERIFIED, id="not-requested"),
    ],
)
async def test_reasoning_is_verified_only_when_the_template_renders_the_setting(template, options, applied):
    turn = await evidence(template, [4], options)
    assert turn.reasoning_applied == applied
    assert turn.request_settings.get("chat_template_kwargs") == options.get("chat_template_kwargs")


@pytest.mark.asyncio
async def test_turn_evidence_keeps_reasoning_delimiters_that_parsing_removes():
    turn = await evidence(HONORS_FLAG + "</think>", [7, 4, 8, 4], OFF)
    assert turn.raw_response == "<think>42</think>42"
    assert turn.reasoning_markers == ("<think>", "</think>")
