from pathlib import Path

from upath import UPath

from harbor.agents.base import BaseAgent
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.installed.opencode import OpenCode
from harbor.agents.installed.pi import Pi
from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_kira import TerminusKira
from harbor.models.agent.name import AgentName
from harbor.models.trial.config import AgentConfig
from harbor.utils.env import resolve_env_vars


class AgentFactory:
    _AGENT_MAP: dict[AgentName, type[BaseAgent]] = {
        AgentName(agent.name()): agent
        for agent in (Pi, MiniSweAgent, Codex, OpenCode, Terminus2, TerminusKira)
    }

    @classmethod
    def _unsupported(cls, name: str) -> ValueError:
        return ValueError(
            f"Agent {name!r} is not supported by Marin's vendored Harbor. "
            f"Supported agents: {', '.join(cls._AGENT_MAP)}"
        )

    @classmethod
    def get_agent_class(cls, name: AgentName) -> type[BaseAgent]:
        try:
            return cls._AGENT_MAP[name]
        except KeyError as exc:
            raise cls._unsupported(name) from exc

    @classmethod
    def create_agent_from_name(
        cls,
        name: AgentName,
        logs_dir: Path | UPath,
        model_name: str | None = None,
        model_alias: str | None = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent from an agent name.

        Args:
            name (AgentName): The name of the agent.

        Returns:
            BaseAgent: The created agent.

        Raises:
            ValueError: If the agent name is invalid.
        """
        agent_class = cls.get_agent_class(name)

        return agent_class(
            logs_dir=logs_dir,
            model_name=model_name,
            model_alias=model_alias,
            **kwargs,
        )

    @classmethod
    def create_agent_from_import_path(
        cls,
        import_path: str,
        logs_dir: Path | UPath,
        model_name: str | None = None,
        model_alias: str | None = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent from an import path.

        Args:
            import_path (str): The import path of the agent. In the format
                'module.path:ClassName'.

        Returns:
            BaseAgent: The created agent.

        Raises:
            ValueError: If the import path is invalid.
        """
        allowed = {
            f"{agent.__module__}:{agent.__name__}": agent
            for agent in cls._AGENT_MAP.values()
        }
        if import_path not in allowed:
            raise cls._unsupported(import_path)
        Agent = allowed[import_path]

        return Agent(
            logs_dir=logs_dir,
            model_name=model_name,
            model_alias=model_alias,
            **kwargs,
        )

    @classmethod
    def create_agent_from_config(
        cls,
        config: AgentConfig,
        logs_dir: Path | UPath,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent from an agent configuration.

        Args:
            config (AgentConfig): The configuration of the agent.

        Returns:
            BaseAgent: The created agent.

        Raises:
            ValueError: If the configuration is invalid.
        """
        extra_env = resolve_env_vars(config.env)
        if config.mode != "container":
            raise cls._unsupported(f"{config.name} (mode={config.mode})")

        if config.name is not None and config.name in AgentName.values():
            name = AgentName(config.name)
            agent_kwargs = {**config.kwargs, **kwargs}
            if config.override_setup_timeout_sec is not None:
                agent_kwargs["override_setup_timeout_sec"] = (
                    config.override_setup_timeout_sec
                )
            return cls.create_agent_from_name(
                name,
                logs_dir=logs_dir,
                model_name=config.model_name,
                model_alias=config.model_alias,
                extra_env=extra_env,
                **agent_kwargs,
            )
        elif config.import_path is not None:
            return cls.create_agent_from_import_path(
                config.import_path,
                logs_dir=logs_dir,
                model_name=config.model_name,
                model_alias=config.model_alias,
                extra_env=extra_env,
                **config.kwargs,
                **kwargs,
            )
        else:
            raise cls._unsupported(config.name or "<unspecified>")
