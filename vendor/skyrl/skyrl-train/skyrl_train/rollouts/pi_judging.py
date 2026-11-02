"""Independent post-trial Pi judgment; verifier outcomes never enter the judge request."""

import json
import re
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.secrets import resolve_secret_spec

from skyrl_train.rollouts.pi_scenario_etl import HydratedPiTrial, JudgeEvidence


class PiJudgeSettings(BaseModel):
    """An explicitly pinned, authenticated external judge, separate from the task grader."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str
    model: str
    revision: str
    auth: str
    principle: str
    prompt_revision: Literal["rlaif_judge_v1"]
    chat_template_kwargs: dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check(self) -> "PiJudgeSettings":
        endpoint = urlsplit(self.endpoint)
        if (
            not self.model.strip()
            or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None
            or not self.principle.strip()
            or not (
                (self.auth.startswith("env:") and self.auth[4:])
                or (self.auth.startswith("file:/") and self.auth[5:])
            )
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or set(self.chat_template_kwargs) - {"enable_thinking"}
            or self.chat_template_kwargs.get("enable_thinking") is True
            or not (
                endpoint.scheme == "https"
                or (endpoint.scheme == "http" and endpoint.hostname in ("localhost", "127.0.0.1"))
            )
        ):
            raise ValueError("Pi judge requires an HTTPS endpoint (or loopback), pinned model and secret reference")
        return self


class PiJudge(Protocol):
    async def score(self, trial: HydratedPiTrial) -> JudgeEvidence: ...

    async def close(self) -> None: ...


class ExternalPiJudge:
    """Score a validated physical trial once, using strict binary OpenAI chat output."""

    def __init__(self, settings: PiJudgeSettings, *, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.authorization = f"Bearer {resolve_secret_spec(settings.auth).value}"
        self.client = client if client is not None else httpx.AsyncClient(timeout=60)

    async def score(self, trial: HydratedPiTrial) -> JudgeEvidence:
        """Send only executed student messages and tool observations, never task grades."""
        if trial.judge is not None or trial.planned_executors is not None or trial.critique is not None or any(
            turn.actor != "student" or turn.teacher_label is not None for turn in trial.turns
        ):
            raise ValueError("Pi judge requires an unjudged student-only physical trial")
        last = trial.turns[-1]
        transcript = [
            *last.executed.messages,
            last.executed_assistant,
            *(
                {"role": "tool", "tool_call_id": item.tool_call_id, "content": item.content}
                for item in last.observations
            ),
        ]
        serialized = json.dumps(transcript, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > 65536:
            raise ValueError("Pi judge transcript exceeds the configured 64 KiB limit")
        try:
            response = await self.client.post(
                f"{self.settings.endpoint.rstrip('/')}/chat/completions",
                headers={"Authorization": self.authorization},
                json={
                    "model": self.settings.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Assess the student's executed task trajectory against this principle: "
                                f"{self.settings.principle}\nReturn exactly 1 if it satisfies the principle, "
                                "or exactly 0 otherwise. Return no other text."
                            ),
                        },
                        {"role": "user", "content": serialized},
                    ],
                    "temperature": 0,
                    "chat_template_kwargs": self.settings.chat_template_kwargs,
                    "max_tokens": 4,
                    "stream": False,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError:
            # The endpoint can be an Iris capability URL; httpx includes it in exception text.
            raise RuntimeError("Pi judge HTTP request failed") from None
        result = response.json()
        if result.get("model") != self.settings.model or not isinstance(result.get("id"), str) or not result["id"]:
            raise ValueError("Pi judge response lacks the pinned model or request identity")
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Pi judge must return exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        if (
            choice.get("finish_reason") != "stop"
            or not isinstance(message, dict)
            or message.get("role") != "assistant"
            or message.get("tool_calls")
            or message.get("content") not in ("0", "1")
        ):
            raise ValueError("Pi judge must finish with an exact binary score")
        return JudgeEvidence(
            request_id=result["id"],
            trial_id=trial.trial_id,
            student_request_ids=tuple(turn.executed.request_id for turn in trial.turns),
            model=self.settings.model,
            principle=self.settings.principle,
            prompt_revision=self.settings.prompt_revision,
            score=int(message["content"]),
        )

    async def close(self) -> None:
        """Release only the worker-owned HTTP transport, never the judge server."""
        await self.client.aclose()
