import logging
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from uuid import UUID

from upath import UPath

from harbor_config.model_name import split_provider_model_name

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.bridge import BridgeKind
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.paths import EnvironmentPaths
from harbor.models.trial.result import AgentInfo, ModelInfo
from harbor.utils.logger import logger as global_logger


class MalformedModelResponseError(Exception):
    """The model returned output that cannot be used safely by the agent."""


def encode_model_output(text: str) -> bytes:
    """Encode model-influenced text as UTF-8.

    Text a model produced or shaped (commands, terminal output, context
    excerpts) can contain an unpaired Unicode surrogate, which UTF-8 cannot
    encode. Raises :class:`MalformedModelResponseError` for such input
    instead of ``UnicodeEncodeError``.
    """
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise MalformedModelResponseError(
            "Agent output contains text that cannot be encoded as UTF-8"
        ) from error


class TurnCapExhaustedError(Exception):
    """The agent's turn budget (``max_turns``) ran out before it confirmed
    task completion.

    An agent loop that stops on its turn cap raises this rather than
    returning, so the trial records a classified stop instead of an
    unremarkable completion. Scored-outcome treatment is defined in the
    ``harbor_config.errors`` taxonomy.

    ``pending_completion`` is True when the agent declared task complete on
    its final turn and the cap cut off the confirmation exchange.
    """

    def __init__(self, message: str, *, pending_completion: bool = False) -> None:
        super().__init__(message)
        self.pending_completion = pending_completion


class BaseAgent(ABC):
    # ``logs_dir`` may be a remote ``UPath`` (e.g. gs://, s3://) when the trial
    # writes directly to object storage. ``UPath`` is NOT a ``pathlib.Path``
    # subclass, so the annotation must admit both — coercing a remote UPath to a
    # stdlib ``Path`` mangles the scheme (``gs://`` -> ``gs:/``) and silently
    # redirects writes (e.g. trajectory.json) to the local filesystem.
    logs_dir: Path | UPath
    environment_logs_dir: PurePosixPath
    model_name: str | None
    model_alias: str | None
    logger: logging.Logger

    # Whether agent supports Harbor's trajectory format (ATIF)
    # Subclasses should override this class variable to indicate ATIF support
    SUPPORTS_ATIF: bool = False

    # Whether agent populates literal token-level trace data
    # (step.metrics.{prompt_token_ids, completion_token_ids, logprobs}).
    # Agents that own their inference loop (e.g. terminus-2) or receive
    # token events from their harness can set this True so that
    # ``harbor traces export --literal`` emits token columns. Installed
    # agents that only recover decoded text from CLI logs must leave this
    # False — the literal exporter no-ops cleanly for them.
    SUPPORTS_LITERAL_TRACES: bool = False

    # Whether agent supports Windows container tasks.
    # Agents that only use Linux tools (bash, apt-get, tmux, etc.) in setup()
    # should keep the default False.  The trial runner checks this flag before
    # calling setup() and fails fast with a clear error on mismatch.
    SUPPORTS_WINDOWS: bool = False
    SUPPORTED_BRIDGES: frozenset[BridgeKind] = frozenset()

    session_id: str | None = None
    context_id: UUID | None = None

    def __init__(
        self,
        logs_dir: Path | UPath,
        model_name: str | None = None,
        model_alias: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig]
        | None = None,  # MCP servers from task config; see setup()/run() for usage
        skills_dir: str | None = None,  # Skills directory path in the environment
        *args,
        extra_env: dict[str, str] | None = None,
        environment_logs_dir: PurePosixPath | None = None,
        **kwargs,
    ):
        self.logs_dir = logs_dir
        self.environment_logs_dir = environment_logs_dir or EnvironmentPaths.agent_dir
        self.model_name = model_name
        self.model_alias = model_name if model_alias is None else model_alias
        self.logger = (logger or global_logger).getChild(__name__)
        self.mcp_servers = mcp_servers or []
        self.skills_dir = skills_dir
        if not hasattr(self, "_extra_env"):
            self._extra_env = dict(extra_env) if extra_env else {}

        self._init_model_info()

    @property
    def extra_env(self) -> dict[str, str]:
        return dict(self._extra_env)

    def _init_model_info(self):
        self._model_identity_provider = None
        self._model_identity_name = None

        if self.model_name is None:
            return

        self._model_identity_provider, self._model_identity_name = (
            split_provider_model_name(self.model_name)
        )

    def to_agent_info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name(),
            version=self.version() or "unknown",
            model_info=(
                ModelInfo(
                    name=self._model_identity_name,
                    provider=self._model_identity_provider,
                )
                if self._model_identity_name
                else None
            ),
        )

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the agent."""

    @abstractmethod
    def version(self) -> str | None:
        """The version of the agent."""

    @classmethod
    def import_path(cls) -> str:
        """
        The import path of the agent. Formatted as 'some.import.path:AgentClass'.
        """
        return f"{cls.__module__}:{cls.__name__}"

    @abstractmethod
    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Run commands to setup the agent & its tools.

        This is a good place to register the MCP servers in ``self.mcp_servers`` with
        the agent (e.g. by creating a .mcp.json file) and copy skills from
        ``self.skills_dir`` to the agent's expected location.

        The user the agent should run as is set on ``environment.default_user`` by
        the orchestrator before this method is called.

        Args:
            environment: The environment in which to setup the agent.
        """

    @abstractmethod
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Runs the agent in the environment. Be sure to populate the context with the
        results of the agent execution. Ideally, populate the context as the agent
        executes in case of a timeout or other error.

        If necessary, register the MCP servers in ``self.mcp_servers`` with
        the agent (e.g. by creating a .mcp.json file) and copy skills from
        ``self.skills_dir`` to the agent's expected location.

        The user the agent should run as is set on ``environment.default_user`` by
        the orchestrator before this method is called.  Any call to
        ``environment.exec`` that does not pass an explicit ``user`` argument will
        automatically run as that user.

        Args:
            instruction: The task instruction.
            environment: The environment in which to complete the task.
            context: The context to populate with the results of the agent execution.
        """

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Optionally backfill context after ``run()`` completes.

        Agents that write logs or trajectories during execution can override this
        to parse those outputs after the trial syncs logs back to the host.
        """
        pass
