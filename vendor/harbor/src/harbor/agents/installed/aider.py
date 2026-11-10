from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName

_UNSUPPORTED_REASON = (
    "Aider is unsupported: it requires blocked dependency LiteLLM and has no "
    "supported guest authentication path; use mini-swe-agent or terminus-2"
)


class Aider(BaseInstalledAgent):
    """Aider metadata; installation and execution are unsupported."""

    CLI_FLAGS = [
        CliFlag(
            "reasoning_effort",
            cli="--reasoning-effort",
            type="str",
        ),
        CliFlag(
            "thinking_tokens",
            cli="--thinking-tokens",
            type="int",
        ),
        CliFlag(
            "cache_prompts",
            cli="--cache-prompts",
            type="bool",
        ),
        CliFlag(
            "auto_lint",
            cli="--auto-lint",
            type="bool",
        ),
        CliFlag(
            "auto_test",
            cli="--auto-test",
            type="bool",
        ),
        CliFlag(
            "test_cmd",
            cli="--test-cmd",
            type="str",
        ),
        CliFlag(
            "stream",
            cli="--stream",
            type="bool",
        ),
        CliFlag(
            "map_tokens",
            cli="--map-tokens",
            type="int",
        ),
    ]

    @staticmethod
    def name() -> str:
        return AgentName.AIDER.value

    def get_version_command(self) -> str | None:
        return ". $HOME/.local/bin/env; aider --version"

    def parse_version(self, stdout: str) -> str:
        text = stdout.strip()
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line.removeprefix("aider").strip()
        return text

    async def install(self, environment: BaseEnvironment) -> None:
        raise RuntimeError(_UNSUPPORTED_REASON)

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        raise RuntimeError(_UNSUPPORTED_REASON)
