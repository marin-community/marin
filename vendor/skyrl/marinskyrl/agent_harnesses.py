"""Bind SkyRL's configured agents to RolloutEngine harnesses."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt
from rolloutengine.agents.model_relay import PromptTokenizer
from rolloutengine.agents.pi.agent import PI_HARNESS, PiOptions, PiThinking, PiThinkingFormat, pi_harness
from rolloutengine.agents.terminus.agent import TERMINUS_HARNESS, TerminusOptions, TerminusParser, terminus_harness
from rolloutengine.harness.core import Harness

class PiSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    thinking: PiThinking
    thinking_format: PiThinkingFormat
    setup_timeout: PositiveFloat


class TerminusSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_main_turns: PositiveInt
    setup_timeout: PositiveFloat


class AgentHarnessSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pi: PiSettings | None = None
    terminus: TerminusSettings | None = Field(default=None, alias=TERMINUS_HARNESS)


@dataclass(frozen=True)
class TaskAgentHarnesses:
    settings: AgentHarnessSettings
    model: str
    tokenize: PromptTokenizer

    def bind(self, context_window: int, max_output_tokens: int) -> dict[str, Harness]:
        """Bind the launch settings to the model budget for this batch."""
        harnesses: dict[str, Harness] = {}
        if pi := self.settings.pi:
            harnesses[PI_HARNESS] = pi_harness(
                PiOptions(
                    model=self.model,
                    thinking_format=pi.thinking_format,
                    max_input_tokens=context_window,
                    max_output_tokens=max_output_tokens,
                    version=pi.version,
                    thinking=pi.thinking,
                    setup_timeout=pi.setup_timeout,
                ),
                self.tokenize,
            )
        if terminus := self.settings.terminus:
            harnesses[TERMINUS_HARNESS] = terminus_harness(
                TerminusOptions(
                    parser=TerminusParser.JSON,
                    max_input_tokens=context_window,
                    max_output_tokens=max_output_tokens,
                    served_max_model_len=context_window,
                    max_main_turns=terminus.max_main_turns,
                    setup_timeout=terminus.setup_timeout,
                )
            )
        return harnesses
