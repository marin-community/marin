import asyncio
import copy
import json
import math
import os
import shlex
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from upath import UPath

from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from harbor_config.stop_reason import AgentStopReason

from harbor.agents.base import (
    BaseAgent,
    TurnCapExhaustedError,
    encode_model_output,
)
from harbor.agents.terminus_2.terminus_json_plain_parser import (
    TerminusJSONPlainParser,
)
from harbor.agents.terminus_2.terminus_xml_plain_parser import (
    TerminusXMLPlainParser,
)
from harbor.agents.terminus_2.tmux_session import (
    DEFAULT_BATCH_EXEC_TIMEOUT_MARGIN_SEC,
    TmuxKeystrokeBatch,
    TmuxSession,
)
from harbor.context_management import (
    ContextManagedResponse,
    ContextWindowManager,
    MESSAGES_PER_TURN,
    trim_latest_turn,
)
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import (
    BaseLLM,
    ContextBudgetExceededError,
    ContextLengthExceededError,
    ContextManagementInfrastructureError,
    LLMBackend,
    LLMResponse,
    LLMRequestTimeoutError,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat, assistant_message_from_response
from harbor.llms.lite_llm import LiteLLM
from harbor.llms.vllm_tokenization import shared_async_tokenize_client, tokenize_url
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.agent.rollout_detail import RolloutDetail
from harbor.models.agent.trajectory_config import TrajectoryConfig
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from harbor.utils.artifact_writer import get_artifact_writer
from harbor.utils.trajectory_utils import format_trajectory_json


# Shared, process-wide httpx client for the per-turn vLLM `/tokenize` probe in
# `_count_total_tokens`. A live py-spy on the CoreWeave RL path (v0g) caught the
# coordinator event loop repeatedly inside `httpx.Client(...)` -> `ssl.py
# create_default_context` — i.e. every proactive-summarization check built a
# BRAND-NEW client and a fresh SSL context per call, on the loop thread. The
# SSL-context construction is the expensive part; reusing one pooled client
# (httpx.Client is safe to share across threads for issuing requests) removes
# that per-call cost and keeps connections warm. Created lazily under a lock.
import threading as _threading

_TOKENIZE_HTTP_CLIENT: "httpx.Client | None" = None  # noqa: F821 (httpx imported lazily)
_TOKENIZE_HTTP_CLIENT_LOCK = _threading.Lock()


def _get_shared_tokenize_client(timeout: float = 10.0) -> "httpx.Client":  # noqa: F821
    """Return the process-wide pooled httpx client used for vLLM `/tokenize`.

    Lazily constructs a single ``httpx.Client`` (one SSL context, kept-alive
    connection pool) reused across all trials/turns instead of building a fresh
    client + SSL context per call. Thread-safe: the coordinator may drive this
    from worker threads via ``asyncio.to_thread``.

    Kept for the SYNC ``_count_total_tokens`` callers (the summarize / context
    -fallback ``_unwind_messages_to_free_tokens`` path). The per-turn RL hot path
    uses the shared async client in ``harbor.llms.vllm_tokenization``.
    """
    global _TOKENIZE_HTTP_CLIENT
    if _TOKENIZE_HTTP_CLIENT is None:
        import httpx

        with _TOKENIZE_HTTP_CLIENT_LOCK:
            if _TOKENIZE_HTTP_CLIENT is None:
                _TOKENIZE_HTTP_CLIENT = httpx.Client(timeout=timeout)
    return _TOKENIZE_HTTP_CLIENT


# Output-token budget reserved for context-summarization subagent calls.
#
# Proactive summarization fires when the context is already nearly full
# (free_tokens < proactive_summarization_threshold), so without an explicit
# output reservation the summary completion runs out of room, returns
# finish_reason=="length", and lite_llm raises OutputLengthExceededError —
# which previously degraded the agent to a no-LLM "instruction + last 1000
# chars" fallback (measured: 59% of summarizing trials, reward 0.19 vs 0.82).
#
# Before any summary call we unwind the input so context_limit - input_tokens >=
# RESERVED_SUMMARY_OUTPUT and pass that reserve as an explicit max_tokens. Clean
# summaries are ~573 completion tokens, so 4096 is generous. A thinking-enabled
# model can spend the whole reserve on reasoning tokens, though, so on truncation
# the retry grows the reserve up to the model's output limit. If the model also
# exhausts that legal maximum, the recovery boundary creates a bounded handoff
# from the partial completion and a conversation excerpt instead of failing the
# trial.
# See agent_logs/2026-06-18_919285_summarization_timing_divedive.md and
# agent_logs/2026-07-31_escalation-harbor-terminus2-context-budget-failure.md.
RESERVED_SUMMARY_OUTPUT = 4096
_MAX_RESERVED_SUMMARY_OUTPUT = 16384
_OVERLONG_SUMMARY_MAX_BYTES = 32768
_OVERLONG_SUMMARY_MIN_BYTES = 1024
_RECOVERY_STATE_MAX_BYTES = 8192
_OVERLONG_SUMMARY_MARKER = (
    "OVERLONG MODEL SUMMARIZATION WAS TRUNCATED. PLEASE RESUME YOUR WORK FROM "
    "THE PARTIAL HANDOFF AND CONVERSATION EXCERPT ABOVE."
)
_SUMMARY_FAILURE_MARKER = (
    "MODEL SUMMARIZATION FAILED. PLEASE RESUME FROM THE BOUNDED TASK STATE ABOVE."
)


@dataclass(frozen=True)
class SubagentRunResult:
    response: LLMResponse
    trajectory_ref: SubagentTrajectoryRef
    accepted_history: list[dict[str, Any]]


@dataclass
class SummaryProgress:
    recovery_state: str
    completed_summary: str = ""


@dataclass(frozen=True)
class EpisodeLoggingPaths:
    debug: Path | UPath | None
    prompt: Path | UPath | None
    response: Path | UPath | None


@dataclass(frozen=True)
class LLMRequestContext:
    logging_paths: EpisodeLoggingPaths
    original_instruction: str
    session: TmuxSession | None


def _trajectory_step_count(message_history_length: int) -> int:
    """Map system-plus-turn chat history to its parent trajectory slice."""
    return 1 + (message_history_length - 1) // MESSAGES_PER_TURN


def _accepted_trajectory_step_count(
    *, accepted_history_length: int, parent_history_length: int
) -> int:
    """Map accepted parent chat and one-step-per-message suffixes to steps."""
    accepted_parent_length = min(accepted_history_length, parent_history_length)
    accepted_suffix_length = max(0, accepted_history_length - parent_history_length)
    return _trajectory_step_count(accepted_parent_length) + accepted_suffix_length


def _decode_extra_body(extra_body: str) -> dict[str, Any]:
    """Decode an ``extra_body`` given as a JSON object string into a dict.

    Marin's model catalog types agent-kwargs values as strings, so
    ``extra_body`` can arrive as a JSON object string rather than a dict.
    Anything that does not decode to a JSON object is rejected with an error
    naming the field.
    """
    try:
        decoded = json.loads(extra_body)
    except json.JSONDecodeError as e:
        raise ValueError(
            "extra_body must be a dict or a JSON object string, "
            f"got undecodable string: {extra_body!r}"
        ) from e
    if not isinstance(decoded, dict):
        raise ValueError(
            "extra_body must be a dict or a JSON object string, "
            f"decoded {extra_body!r} to {type(decoded).__name__}"
        )
    return decoded


class TrajectoryDumpCadence(StrEnum):
    """How often the in-progress trajectory is persisted during the agent loop.

    ``FINAL`` (default): persist once when ``run()`` finishes (plus the
    mandatory continuation-boundary dumps in linear-history mode, whose
    segments would otherwise be lost to message truncation).
    ``PER_TURN``: additionally persist after every episode, trading extra
    storage writes for crash-granularity snapshots.
    """

    FINAL = "final"
    PER_TURN = "per_turn"


TRAJECTORY_DUMP_CADENCE_ENV = "HARBOR_TRAJECTORY_DUMP_CADENCE"


def _resolve_trajectory_dump_cadence(
    cadence: str | None,
) -> TrajectoryDumpCadence:
    """Explicit constructor arg wins; else the env override; else FINAL."""
    resolved = cadence or os.environ.get(TRAJECTORY_DUMP_CADENCE_ENV) or "final"
    return TrajectoryDumpCadence(resolved)


def _enqueue_episode_text(path: Path | UPath, text: str) -> None:
    """Persist a per-episode debug text file (prompt/response) off the loop.

    Only reached with ``enable_episode_logging=True``; the write (a blocking
    object-store upload for remote logs_dir) goes through the background
    artifact writer instead of the caller thread.
    """
    get_artifact_writer().submit_optional(
        path, lambda: text, description=f"terminus-2 episode log {path.name}"
    )


@dataclass
class Command:
    keystrokes: str
    duration_sec: float


@dataclass(frozen=True)
class LLMInteractionResult:
    commands: list[Command]
    is_task_complete: bool
    feedback: str
    analysis: str
    plan: str
    response: LLMResponse
    native_tool_calls: list[dict[str, Any]] | None = None
    custom_action: Any | None = None


class ParsedCommand(Protocol):
    @property
    def keystrokes(self) -> str: ...

    @property
    def duration(self) -> float: ...


class TerminusParseResult(Protocol):
    @property
    def commands(self) -> Sequence[ParsedCommand]: ...

    @property
    def is_task_complete(self) -> bool: ...

    @property
    def error(self) -> str: ...

    @property
    def warning(self) -> str: ...

    @property
    def analysis(self) -> str: ...

    @property
    def plan(self) -> str: ...


class TerminusResponseParser(Protocol):
    def parse_response(self, response: str) -> TerminusParseResult: ...

    def salvage_truncated_response(self, truncated_response: str) -> str | None:
        """Return text that remains valid after truncation, when available."""
        ...


@dataclass
class SubagentMetrics:
    """Metrics for subagent operations (summarization, etc.)."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0


class Terminus2(BaseAgent):
    SUPPORTS_ATIF: bool = True
    SUPPORTS_LITERAL_TRACES: bool = True

    def _init_llm(
        self,
        llm_backend: LLMBackend | str,
        model_name: str,
        temperature: float | None,
        collect_rollout_details: bool,
        llm_kwargs: dict | None,
        # LiteLLM-specific args
        api_base: str | None,
        session_id: str | None,
        max_thinking_tokens: int | None,
        reasoning_effort: str | None,
        model_info: dict | None,
        use_responses_api: bool,
    ) -> BaseLLM:
        """Initialize the LLM backend based on llm_backend parameter.

        Args:
            llm_backend: The LLM backend to use.
            model_name: Name of the model.
            temperature: Sampling temperature, if explicitly configured.
            collect_rollout_details: Whether to collect token IDs and logprobs.
            llm_kwargs: Additional kwargs passed to the LLM constructor.
            api_base: Base URL for LiteLLM API endpoint.
            session_id: Session ID for LiteLLM.
            max_thinking_tokens: Max thinking tokens for LiteLLM extended thinking.
            reasoning_effort: Reasoning effort level for LiteLLM.
            model_info: Model info dict for LiteLLM custom models.
            use_responses_api: Whether to use the Responses API.

        Returns:
            An initialized LLM instance.

        Raises:
            ValueError: If llm_backend is not a recognized backend.
        """
        # Normalize enum to string value for matching
        backend_value = (
            llm_backend.value if isinstance(llm_backend, LLMBackend) else llm_backend
        )
        constructor_kwargs = dict(llm_kwargs or {})
        if temperature is not None:
            constructor_kwargs["temperature"] = temperature

        match backend_value:
            case LLMBackend.LITELLM.value:
                return LiteLLM(
                    model_name=model_name,
                    api_base=api_base,
                    collect_rollout_details=collect_rollout_details,
                    session_id=session_id,
                    max_thinking_tokens=max_thinking_tokens,
                    reasoning_effort=reasoning_effort,
                    model_info=model_info,
                    use_responses_api=use_responses_api,
                    **constructor_kwargs,
                )
            case LLMBackend.TINKER.value:
                from harbor.llms.tinker import TinkerLLM

                return TinkerLLM(
                    model_name=model_name,
                    collect_rollout_details=collect_rollout_details,
                    **constructor_kwargs,
                )
            case _:
                raise ValueError(
                    f"Unknown llm_backend: {llm_backend!r}. "
                    f"Supported backends: {[b.value for b in LLMBackend]}"
                )

    def __init__(
        self,
        logs_dir: Path | UPath,
        model_name: str | None = None,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float | None = None,
        reasoning_effort: Literal[
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "default"
        ]
        | None = None,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        max_thinking_tokens: int | None = None,
        model_info: dict | None = None,
        trajectory_config: TrajectoryConfig | None = None,
        trajectory_dump_cadence: str | None = None,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        store_all_messages: bool = False,
        enable_episode_logging: bool = False,
        interleaved_thinking: bool = False,
        suppress_max_turns_warning: bool = False,
        use_responses_api: bool = False,
        llm_backend: LLMBackend | str = LLMBackend.LITELLM,
        llm_kwargs: dict | None = None,
        llm_call_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | str | None = None,
        extra_env: dict[str, str] | None = None,
        turn_callback: Callable | None = None,
        batch_turn_execs: bool = True,
        batch_exec_timeout_margin_sec: float = DEFAULT_BATCH_EXEC_TIMEOUT_MARGIN_SEC,
        *args,
        **kwargs,
    ):
        """Initialize Terminus 2 agent.

        Args:
            logs_dir: Directory to store logs
            model_name: Name of the model to use
            max_episodes: Maximum number of episodes (default: 1000000)
            parser_name: Parser to use - "json" or "xml" (default: "json")
            api_base: Base URL for the API endpoint
            temperature: Optional sampling temperature. If unset, no temperature is
                passed to the LLM backend. (default: None)
            reasoning_effort: Qualitative or quantitative measure of effort (default: None)
            collect_rollout_details: Whether to collect detailed rollout data including token IDs.
                NOTE: Rollout details will be incomplete if context summarization occurs.
                See class docstring for details. (default: False)
            session_id: Session ID for the agent (default: None)
            enable_summarize: Whether to enable context summarization (default: True)
            proactive_summarization_threshold: Number of free tokens below which to trigger
                proactive summarization. Set to 0 to disable proactive summarization. (default: 8000)
            max_thinking_tokens: Maximum thinking tokens for Anthropic extended thinking mode.
                Minimum value is 1024. (default: None)
            model_info: Optional dict containing model information for custom models.
                Used to register the model with litellm. Common fields include:
                - max_input_tokens: Maximum input tokens (context length)
                - max_output_tokens: Maximum output tokens
                - input_cost_per_token: Cost per input token (optional)
                - output_cost_per_token: Cost per output token (optional)
                (default: None)
            trajectory_config: Optional TrajectoryConfig containing trajectory-related configurations.
                Available options:
                - raw_content (bool): If True, dump raw LLM responses into trajectory without
                  parsing into tool_calls. Useful for SFT data export. (default: False)
                - linear_history (bool): If True, split trajectory into separate files when context
                  summarization occurs, ensuring each trajectory represents a continuous linear
                  history sent to the LLM. When False, keep all steps from the main agent
                  in a single trajectory file despite chat history resets. (default: False)
                (default: None)
            trajectory_dump_cadence: When to persist the in-progress trajectory:
                "final" (once at the end of ``run()``) or "per_turn" (also after
                every episode). Continuation-boundary dumps in linear-history
                mode always happen regardless of cadence — those segments are
                truncated from memory right after the dump. Overridable via the
                ``HARBOR_TRAJECTORY_DUMP_CADENCE`` env var when unset here.
                (default: "final")
            tmux_pane_width: Starting tmux pane width (maps to `tmux -x`, default: 160)
            tmux_pane_height: Starting tmux pane height (maps to `tmux -y`, default: 40)
            enable_episode_logging: Whether to write per-turn ``episode-N/{debug,prompt,
                response}`` files under ``logs_dir``. ``debug.json`` is the FULL raw
                litellm request+response (incl. the whole message history, MBs and
                growing) written by the synchronous litellm ``logger_fn``. When
                ``logs_dir`` is a remote path (gs://, s3://), each per-turn write is a
                blocking object-store upload that stalls the shared rollout event loop.
                DEFAULT False (off): RL training does NOT read these — logprobs/token_ids
                come from ``rollout_details`` and the trajectory from ``trajectory.json``,
                both written independently. Set True only for local debugging. (default: False)
            interleaved_thinking: Whether to include reasoning content in chat history
                and send to litellm in next round's conversation (default: False)
            suppress_max_turns_warning: Whether to suppress the warning about artificially
                limiting max_turns (default: False)
            llm_backend: LLM backend to use. Use LLMBackend.LITELLM or "litellm".
                (default: LLMBackend.LITELLM)
            llm_kwargs: Additional kwargs to pass to the LLM constructor.
                (default: None)
            llm_call_kwargs: Extra kwargs to forward to LLM calls (e.g., extra_body).
            extra_body: Optional provider-specific request-body params to forward to
                every LLM chat-completion call (folded into llm_call_kwargs.extra_body).
                May be a dict or a JSON object string (decoded at construction);
                string-typed agent-kwargs sources such as Marin's model catalog
                deliver the latter. Used for vLLM features like chat_template_kwargs.
                For Qwen3.5 thinking
                models this is how the rollout enables the proper thinking generation
                prompt (mirrors the eval path):
                ``{"chat_template_kwargs": {"enable_thinking": True}}``. Without it the
                qwen3.5 chat template renders an empty ``<think></think>`` block (the
                ``enable_thinking is false`` branch), which diverges from the served
                completion ids at TIS re-tokenization. (default: None)
            batch_turn_execs: Execute each turn's liveness probes, keystrokes, waits,
                and terminal captures through one environment call. (default: True)
            batch_exec_timeout_margin_sec: Additional time allowed beyond the sum of
                a turn's declared command waits. (default:
                DEFAULT_BATCH_EXEC_TIMEOUT_MARGIN_SEC)
            **kwargs: Additional arguments
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self._extra_env = extra_env

        if model_name is None:
            raise ValueError("model_name is required for Terminus 2")

        if self.model_alias is None:
            raise ValueError("model_alias could not be resolved for Terminus 2")

        self._last_response_model_name: str | None = None
        self._parser_name = parser_name
        self._collect_rollout_details = collect_rollout_details
        self._reasoning_effort = reasoning_effort
        resolved_model_info = self._resolve_model_info(self.model_alias, model_info)

        self._llm = self._init_llm(
            llm_backend=llm_backend,
            model_name=self.model_alias,
            temperature=temperature,
            collect_rollout_details=collect_rollout_details,
            llm_kwargs=llm_kwargs,
            api_base=api_base,
            session_id=session_id,
            max_thinking_tokens=max_thinking_tokens,
            reasoning_effort=reasoning_effort,
            model_info=resolved_model_info,
            use_responses_api=use_responses_api,
        )
        self._parser = self._get_parser()
        self._prompt_template = self._get_prompt_template_path().read_text()
        self._timeout_template = self._get_timeout_template_path().read_text()
        self._temperature = temperature
        self._context_window_manager = ContextWindowManager(self._llm)

        # Handle deprecated 'episodes' kwarg
        episodes_from_kwargs = kwargs.get("episodes")
        if episodes_from_kwargs is not None:
            self.logger.warning(
                "The 'episodes' parameter is deprecated and will be removed in a future version. "
                "Please use 'max_turns' instead."
            )

        # Determine the final max episodes value with proper precedence:
        # 1. max_turns (new parameter)
        # 2. max_episodes (deprecated but still supported)
        # 3. episodes from kwargs (deprecated)
        # 4. Default value of 1000000
        final_max_episodes = None
        if max_turns is not None:
            final_max_episodes = max_turns
        elif kwargs.get("max_episodes") is not None:
            final_max_episodes = kwargs.get("max_episodes")
        elif episodes_from_kwargs is not None:
            final_max_episodes = episodes_from_kwargs

        if final_max_episodes is not None:
            self._max_episodes = final_max_episodes
        else:
            self._max_episodes = 1000000
        self._chat: Chat | None = None
        self._context: AgentContext | None = None
        self._pending_completion = False
        self._session: TmuxSession | None = None
        self._api_request_times: list[float] = []
        self._n_episodes: int = 0
        self._user_provided_session_id: str | None = session_id
        self._session_id = session_id if session_id else str(uuid.uuid4())
        self._trajectory_steps: list[Step] = []
        self._enable_episode_logging = enable_episode_logging
        if (
            not math.isfinite(batch_exec_timeout_margin_sec)
            or batch_exec_timeout_margin_sec <= 0
        ):
            raise ValueError(
                "batch_exec_timeout_margin_sec must be positive and finite"
            )
        self._batch_turn_execs = batch_turn_execs
        self._batch_exec_timeout_margin_sec = batch_exec_timeout_margin_sec
        self._session_alive_hint: bool | None = None
        self._llm_call_kwargs = dict(llm_call_kwargs) if llm_call_kwargs else {}
        # Fold an explicit `extra_body` into llm_call_kwargs so it is forwarded to
        # every LLM call's request body (e.g. chat_template_kwargs.enable_thinking for
        # Qwen3.5 thinking models — see __init__ docstring). Merge into any extra_body
        # already supplied via llm_call_kwargs rather than clobber it.
        if isinstance(extra_body, str):
            extra_body = _decode_extra_body(extra_body)
        if extra_body:
            merged_extra_body = dict(self._llm_call_kwargs.get("extra_body") or {})
            merged_extra_body.update(extra_body)
            self._llm_call_kwargs["extra_body"] = merged_extra_body

        self._summarization_count: int = (
            0  # Track number of summarization subagents created
        )
        self._continuation_segments_persisted: int = (
            0  # Completed trajectory boundary dumps (canonical index of the
            # current in-memory segment); diverges from _summarization_count
            # when summarization is attempted but the boundary dump is skipped.
        )
        self._pending_subagent_refs: list[SubagentTrajectoryRef] | None = (
            None  # Track subagent refs to include in next step
        )
        self._pending_handoff_prompt: str | None = (
            None  # Track handoff prompt to include as user step
        )
        self._subagent_metrics = SubagentMetrics()  # Track subagent metrics separately
        self._subagent_rollout_details: list[
            RolloutDetail
        ] = []  # Track rollout details for each subagent
        self._enable_summarize = (
            enable_summarize  # Toggle for proactive and context limit summarization
        )
        self._proactive_summarization_threshold = proactive_summarization_threshold
        self._tmux_pane_width = tmux_pane_width
        self._tmux_pane_height = tmux_pane_height

        # Trajectory configuration
        self._trajectory_config = trajectory_config or {}
        self._save_raw_content_in_trajectory = self._trajectory_config.get(
            "raw_content", False
        )
        self._linear_history = self._trajectory_config.get("linear_history", False)
        self._trajectory_dump_cadence = _resolve_trajectory_dump_cadence(
            trajectory_dump_cadence
        )
        # Optional: include full chat messages in TrialResult metadata (can be large)
        self._store_all_messages = store_all_messages
        self._interleaved_thinking = interleaved_thinking
        self._llm_kwargs = llm_kwargs
        # Per-turn callback invoked after every episode. Used by PRMs (e.g.
        # teacher_hint) to inject hints into the next observation OR to
        # request early termination. Wired in via agent_kwargs from
        # SkyRL's harbor_config (turn_callback = prm.as_turn_callback()).
        # See also: _early_termination_reason below + the call site in run().
        self._turn_callback = turn_callback
        self._early_termination_reason: str | None = None
        # Why the agent loop ended. Published in AgentContext.metadata as
        # "stop_reason" so downstream consumers can monitor termination
        # outcomes even when the trial still scores.
        self._stop_reason: AgentStopReason | None = None

    def _resolve_model_info(
        self, model_name: str | None, provided_model_info: dict | None
    ) -> dict | None:
        if provided_model_info:
            return provided_model_info
        if model_name and "hosted_vllm" in model_name:
            self.logger.warning(
                "Model info is required when using hosted_vllm models. "
                "Please set `model_info` in your Terminus 2 configuration with "
                "`max_input_tokens`, `max_output_tokens`, and cost fields. "
                "Falling back to LiteLLM defaults, which may cause context or pricing issues."
            )
        return None

    @staticmethod
    def name() -> str:
        return AgentName.TERMINUS_2.value

    def version(self) -> str | None:
        return "2.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        self._session = TmuxSession(
            session_name=self.name(),
            environment=environment,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
            batch_exec_timeout_margin_sec=self._batch_exec_timeout_margin_sec,
        )
        await self._session.start()

    def _get_parser(self) -> TerminusResponseParser:
        """Return the appropriate parser instance for this format."""
        if self._parser_name == "json":
            return TerminusJSONPlainParser()
        elif self._parser_name == "xml":
            return TerminusXMLPlainParser()
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _get_prompt_template_path(self) -> Path:
        """Return the path to the prompt template for this format."""
        if self._parser_name == "json":
            return Path(__file__).parent / "templates" / "terminus-json-plain.txt"
        elif self._parser_name == "xml":
            return Path(__file__).parent / "templates" / "terminus-xml-plain.txt"
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _get_timeout_template_path(self) -> Path:
        """Return the path to the timeout template for this format."""
        return Path(__file__).parent / "templates" / "timeout.txt"

    @staticmethod
    def _parse_skill_frontmatter(content: str) -> dict[str, str] | None:
        """Parse YAML frontmatter from SKILL.md content, returning name and description."""
        import re

        import yaml

        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return None
        try:
            fm = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
        if not isinstance(fm, dict) or "name" not in fm or "description" not in fm:
            return None
        return {"name": fm["name"], "description": fm["description"]}

    async def _build_skills_section(self, environment: BaseEnvironment) -> str | None:
        """Discover Agent Skills in skills_dir and return an <available_skills> XML block.

        Follows the Agent Skills spec: scans for subdirectories containing SKILL.md
        inside the remote environment, parses YAML frontmatter for name/description,
        and provides the absolute path so the model can ``cat`` the file to activate
        a skill.
        """
        if not self.skills_dir:
            return None

        if not await environment.is_dir(self.skills_dir):
            return None

        # List subdirectories containing SKILL.md in the remote environment
        result = await environment.exec(
            f"find {shlex.quote(self.skills_dir)} -mindepth 2 -maxdepth 2"
            " -name SKILL.md -type f | sort",
            timeout_sec=10,
        )

        if result.return_code != 0 or not result.stdout or not result.stdout.strip():
            return None

        skill_md_paths = result.stdout.strip().splitlines()
        entries: list[tuple[str, str, str]] = []  # (name, description, location)

        for skill_md_path in skill_md_paths:
            cat_result = await environment.exec(
                f"cat {shlex.quote(skill_md_path)}", timeout_sec=10
            )
            if cat_result.return_code != 0 or not cat_result.stdout:
                continue
            fm = self._parse_skill_frontmatter(cat_result.stdout)
            if fm is None:
                continue
            entries.append((fm["name"], fm["description"], skill_md_path))

        if not entries:
            return None

        from xml.etree.ElementTree import Element, SubElement, indent, tostring

        root = Element("available_skills")
        for name, description, location in entries:
            skill = SubElement(root, "skill")
            SubElement(skill, "name").text = name
            SubElement(skill, "description").text = description
            SubElement(skill, "location").text = location

        indent(root, space="  ")
        return "\n" + tostring(root, encoding="unicode")

    def _get_error_response_type(self) -> str:
        """Return the response type name for error messages.

        Examples: 'JSON response', 'response'
        """
        if self._parser_name == "json":
            return "JSON response"
        elif self._parser_name == "xml":
            return "response"
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Return the format-specific task completion confirmation message."""
        if self._parser_name == "json":
            return (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                'make any further corrections. If so, include "task_complete": true '
                "in your JSON response again."
            )
        elif self._parser_name == "xml":
            return (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                "make any further corrections. If so, include "
                "<task_complete>true</task_complete> again."
            )
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _setup_episode_logging(
        self, logging_dir: Path | UPath | None, episode: int
    ) -> EpisodeLoggingPaths:
        # Default OFF: no per-turn episode-N/{debug,prompt,response} writes. This
        # returns all-None so `_query_llm` skips prompt/response writes AND the
        # per-call raw request+response `debug.json` is never written (LiteLLM.call
        # only writes the episode log when logging_path is not None) — eliminating
        # the synchronous, multi-MB, object-store `debug.json` write that otherwise
        # blocks the shared rollout event loop on every LLM call. RL training is
        # unaffected (it reads rollout_details + trajectory.json, written
        # independently).
        if not self._enable_episode_logging:
            return EpisodeLoggingPaths(None, None, None)
        if logging_dir is None:
            return EpisodeLoggingPaths(None, None, None)

        episode_logging_dir = logging_dir / f"episode-{episode}"
        # No mkdir here: the artifact-writer worker creates local parent dirs
        # on write, and object stores need no directory at all — a remote mkdir
        # would be one more blocking round-trip on the event loop.
        return EpisodeLoggingPaths(
            episode_logging_dir / "debug.json",
            episode_logging_dir / "prompt.txt",
            episode_logging_dir / "response.txt",
        )

    def _count_total_tokens(
        self,
        chat: Chat,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count total tokens across all messages in the chat.

        Prefers vLLM's `/tokenize` endpoint so the count matches what
        vLLM applies at serve time (same BPE, same chat template). Falls
        back to `litellm.utils.token_counter` when no vLLM api_base is
        available — e.g. an OpenAI-hosted backend.

        ``extra_messages`` (e.g. a summary prompt appended to the retained
        history only for the request) are included in the count so budget
        accounting matches the real request size.

        Why: litellm's counter disagrees with vLLM by exactly +1 token
        on long, near-budget chat-template renders for unknown HF repos
        (e.g. laion finetunes), manifesting as `VLLMValidationError:
        32769 input tokens` rejections that churn Ray ObjectRefs and
        eventually trigger the SkyRL driver refcount-race SIGABRT.
        """
        # No messages yet (e.g. the proactive-summarization check on the very
        # first episode, before the initial prompt has been appended to the
        # chat): the token count is trivially 0. Short-circuit BEFORE probing
        # vLLM — handing `/tokenize` an empty `messages: []` crashes it. HF's
        # apply_chat_template indexes `conversation[0]` (tokenization_utils_base)
        # before its own empty-guard runs, which vLLM re-raises as a ValueError
        # and fails the request. Never dispatch an empty-messages request.
        if not chat.messages and not extra_messages:
            return 0

        prep = self._prep_tokenize_request(chat, extra_messages)
        if prep is not None:
            url, payload = prep
            try:
                # Reuse the process-wide pooled client (see
                # _get_shared_tokenize_client) so we don't rebuild an SSL
                # context per turn on the coordinator loop.
                client = _get_shared_tokenize_client()
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    return int(resp.json()["count"])
            except Exception:
                pass

        return self._token_counter_fallback(chat, extra_messages)

    def _prep_tokenize_request(
        self,
        chat: Chat,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> "tuple[str, dict[str, Any]] | None":
        """Build the ``(url, json_payload)`` for the vLLM `/tokenize` probe.

        Returns ``None`` when the model isn't served by a vLLM ``api_base``
        (e.g. an OpenAI-hosted backend), in which case the caller falls back to
        the litellm counter. Shared verbatim by the sync (``_count_total_tokens``)
        and truly-async (``_acount_total_tokens``) paths so the request shape /
        `/v1`-stripping stays identical.

        ``extra_messages`` are appended to ``chat.messages`` in the payload so a
        caller can count the exact request size — e.g. the summary prompt that is
        sent alongside the retained history but is NOT part of ``chat.messages``.
        """
        api_base = getattr(self._llm, "_api_base", None)
        if not api_base:
            return None
        model_for_request = self.model_alias
        for prefix in ("hosted_vllm/", "openai/"):
            if model_for_request.startswith(prefix):
                model_for_request = model_for_request[len(prefix) :]
                break
        messages = [
            m.model_dump() if hasattr(m, "model_dump") else m for m in chat.messages
        ]
        if extra_messages:
            messages.extend(extra_messages)
        return (
            tokenize_url(api_base),
            {"model": model_for_request, "messages": messages},
        )

    def _token_counter_fallback(
        self,
        chat: Chat,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        """litellm's local (CPU-bound) token counter — the `/tokenize` fallback."""
        from litellm.utils import token_counter

        messages = list(chat.messages)
        if extra_messages:
            messages.extend(extra_messages)
        return token_counter(model=self.model_alias, messages=messages)

    async def _acount_total_tokens(
        self,
        chat: Chat,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count total tokens via a TRULY-ASYNC vLLM `/tokenize` probe on the loop.

        This is the per-turn RL hot path: reached EVERY turn (~169 concurrent
        trials per coordinator) via ``_check_proactive_summarization``, which
        always runs (even with ``enable_summarize=False``) to stop an oversized
        request before it reaches the inference engine.

        Round-2 pooled the client + SSL context; round-3 moved the blocking POST
        off the loop via ``asyncio.to_thread``. But ``to_thread`` routes through
        the DEFAULT ``ThreadPoolExecutor`` (bounded ~min(32, cpu+4) threads): at
        169-concurrency the 169 per-turn probes SERIALIZE ~32-wide AND contend
        the same default executor litellm's ``acompletion`` uses for its
        ``run_in_executor(None, ...)`` request preamble — so a live py-spy on the
        round-3 image (v0i) still showed the loop saturation-bursty (~1/3 TDP).

        The concurrency math is why this MUST be async, not ``to_thread``: an
        ``httpx.AsyncClient`` POST ``await``-ed on the loop lets all 169 probes
        be in-flight at once (each yields during the network wait), instead of
        queueing behind 32 worker threads. One pooled client per event loop (see
        ``_get_shared_async_tokenize_client``) — reused for the whole run.

        The sync ``_count_total_tokens`` is kept intact for the remaining
        NON-async callers (``_unwind_messages_to_free_tokens`` on the summarize /
        context-fallback path, which is OFF in RL).
        """
        if not chat.messages and not extra_messages:
            return 0

        prep = self._prep_tokenize_request(chat, extra_messages)
        if prep is not None:
            url, payload = prep
            try:
                client = shared_async_tokenize_client()
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    return int(resp.json()["count"])
            except Exception:
                pass

        # Fallback path only (probe failed / no vLLM api_base). litellm's
        # token_counter is local CPU (GIL-bound) — keep it off the loop so the
        # degraded path can't stall dispatch either. This fallback is rare, so
        # its use of the default pool is not a concurrency concern.
        return await asyncio.to_thread(
            self._token_counter_fallback, chat, extra_messages
        )

    def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:
        """
        Limit output to specified byte length, keeping first and last portions.

        Args:
            output: The terminal output to potentially truncate
            max_bytes: Maximum allowed bytes (default 10000)

        Returns:
            str: Original output if under limit, or truncated with middle omitted
        """
        output_bytes = encode_model_output(output)
        if len(output_bytes) <= max_bytes:
            return output

        # Calculate portions (half each for first and last)
        portion_size = max_bytes // 2

        # Get first portion
        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")

        # Get last portion
        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")

        # Calculate omitted bytes
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode("utf-8"))
            - len(last_portion.encode("utf-8"))
        )

        return (
            f"{first_portion}\n[... output limited to {max_bytes} bytes; "
            f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
        )

    def _unwind_messages_to_free_tokens(
        self,
        chat: Chat,
        target_free_tokens: int,
        extra_request_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Remove recent messages until we have enough free tokens.

        ``extra_request_messages`` are content appended to the history only for
        the real request (e.g. a summary prompt) and are counted toward the
        budget but never unwound away — they live outside ``chat.messages``.
        """
        context_limit = self._llm.get_model_context_limit()

        while True:
            current_tokens = self._count_total_tokens(chat, extra_request_messages)
            free_tokens = context_limit - current_tokens

            if free_tokens >= target_free_tokens:
                break

            trimmed_messages = trim_latest_turn(chat.messages)
            if trimmed_messages is None:
                break
            chat._messages = trimmed_messages

        chat.reset_response_chain()
        free_tokens = context_limit - self._count_total_tokens(
            chat, extra_request_messages
        )
        self.logger.debug(
            f"Unwound messages. Remaining messages: {len(chat.messages)}, "
            f"Free tokens: approximately {free_tokens}"
        )

    def _collect_subagent_rollout_detail(self, response: LLMResponse) -> None:
        """Collect rollout details from a subagent LLM response.

        Args:
            response: The LLM response containing token IDs and logprobs
        """
        if not self._collect_rollout_details:
            return

        rollout_detail: RolloutDetail = {}
        if response.prompt_token_ids is not None:
            rollout_detail["prompt_token_ids"] = [response.prompt_token_ids]
        if response.completion_token_ids is not None:
            rollout_detail["completion_token_ids"] = [response.completion_token_ids]
        if response.logprobs is not None:
            rollout_detail["logprobs"] = [response.logprobs]
        if response.extra is not None:
            rollout_detail["extra"] = {
                key: [value] for key, value in response.extra.items()
            }

        if rollout_detail:
            self._subagent_rollout_details.append(rollout_detail)

    def _update_subagent_metrics(self, usage_info) -> None:
        """Update subagent metrics with usage information from an LLM response.

        Args:
            usage_info: Usage information from LLM response containing token counts and cost
        """
        if not usage_info:
            return

        self._subagent_metrics.total_prompt_tokens += usage_info.prompt_tokens
        self._subagent_metrics.total_completion_tokens += usage_info.completion_tokens
        self._subagent_metrics.total_cached_tokens += usage_info.cache_tokens
        self._subagent_metrics.total_cost_usd += usage_info.cost_usd

    def _extract_usage_metrics(self, usage_info) -> tuple[int, int, int, float]:
        """Extract and normalize metrics from usage info.

        Args:
            usage_info: Usage information from LLM response

        Returns:
            Tuple of (prompt_tokens, completion_tokens, cached_tokens, cost_usd)
        """
        if not usage_info:
            return 0, 0, 0, 0

        return (
            usage_info.prompt_tokens,
            usage_info.completion_tokens,
            usage_info.cache_tokens,
            usage_info.cost_usd if usage_info.cost_usd > 0 else 0,
        )

    def _track_api_request_time(self, start_time: float) -> None:
        """Track API request time from start timestamp.

        Args:
            start_time: Start time from time.time()
        """
        end_time = time.time()
        request_time_ms = (end_time - start_time) * 1000
        self._api_request_times.append(request_time_ms)

    async def _dispatch_subagent_request(
        self,
        *,
        prompt: str,
        message_history: list[dict[str, Any]],
        max_tokens: int,
    ) -> ContextManagedResponse:
        """Dispatch one context-managed subagent request and record its duration."""
        call_kwargs = {
            key: value
            for key, value in self._llm_call_kwargs.items()
            if key != "max_tokens"
        }
        start_time = time.time()
        managed_response = await self._context_window_manager.call(
            prompt=prompt,
            message_history=message_history,
            max_tokens=max_tokens,
            call_kwargs=call_kwargs,
        )
        self._track_api_request_time(start_time)
        return managed_response

    async def _run_subagent(
        self,
        prompt: str,
        message_history: list[dict[str, Any]],
        steps: list[Step],
        session_id: str,
        agent_name: str,
        filename_suffix: str,
        summary_text: str,
        max_tokens: int,
        parent_history_length: int | None = None,
        subagent_name_for_logging: str = "subagent",
    ) -> SubagentRunResult:
        """Run a subagent and return its response and trajectory.

        ``steps`` is truncated when necessary so its copied context matches the
        history the provider accepted.

        Args:
            prompt: The prompt to send to the LLM
            message_history: The message history to provide context
            steps: The trajectory steps (will be modified to include prompt and response steps)
            session_id: Session ID for the subagent
            agent_name: Name of the subagent (e.g., "terminus-2-summarization-summary")
            filename_suffix: Suffix for trajectory filename (e.g., "summary", "questions", "answers")
            summary_text: Human-readable summary for the trajectory ref
            subagent_name_for_logging: Name used in logging messages
            max_tokens: Output-token budget for the LLM call. This guarantees the
                summary completion cannot be silently truncated by running out of
                output room.
            parent_history_length: Length of the parent chat prefix. Messages after
                this prefix correspond one-to-one with copied trajectory steps.
        Returns:
            The response, saved trajectory reference, and provider-accepted history.
        """
        managed_response = await self._dispatch_subagent_request(
            prompt=prompt,
            message_history=message_history,
            max_tokens=max_tokens,
        )
        response = managed_response.response
        if parent_history_length is None:
            parent_history_length = len(message_history)
        accepted_steps = _accepted_trajectory_step_count(
            accepted_history_length=len(managed_response.accepted_history),
            parent_history_length=parent_history_length,
        )
        del steps[accepted_steps:]

        prompt_step_id = len(steps) + 1
        steps.append(
            Step(
                step_id=prompt_step_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=prompt,
            )
        )
        response_step_id = prompt_step_id + 1

        usage_info = response.usage
        self._update_subagent_metrics(usage_info)

        self._append_subagent_response_step(
            steps,
            response_step_id,
            response,
            usage_info,
            subagent_name_for_logging,
        )

        self._collect_subagent_rollout_detail(response)

        trajectory_ref = self._save_subagent_trajectory(
            session_id=session_id,
            agent_name=agent_name,
            steps=steps,
            usage_info=usage_info,
            filename_suffix=filename_suffix,
            summary_text=summary_text,
        )

        return SubagentRunResult(
            response=response,
            trajectory_ref=trajectory_ref,
            accepted_history=managed_response.accepted_history,
        )

    async def _summarize_with_retry(
        self,
        chat: Chat,
        original_instruction: str,
        session: TmuxSession,
        pending_prompt: str = "",
    ) -> tuple[str, list[SubagentTrajectoryRef] | None]:
        """Build a handoff through full, short, then deterministic recovery."""
        original_messages = list(chat.messages)
        progress = SummaryProgress(
            recovery_state=await self._capture_recovery_state(
                original_instruction=original_instruction,
                pending_prompt=pending_prompt,
                session=session,
            )
        )
        failed_summary = ""
        summary_output_exhausted = False
        try:
            output_limit = self._llm.get_model_output_limit()
            max_reserve = (
                _MAX_RESERVED_SUMMARY_OUTPUT if output_limit is None else output_limit
            )
            if max_reserve < 1:
                raise ContextManagementInfrastructureError(
                    "Context recovery requires a positive output budget."
                )
            reserve = min(RESERVED_SUMMARY_OUTPUT, max_reserve)
            while True:
                chat._messages = list(original_messages)
                try:
                    handoff, refs = await self._summarize(
                        chat,
                        original_instruction,
                        session,
                        reserved_output=reserve,
                        progress=progress,
                    )
                    if self._handoff_has_required_gap(chat, handoff):
                        return handoff, refs
                    raise ContextLengthExceededError(
                        "The complete generated handoff does not leave enough "
                        "context for the next model response."
                    )
                except OutputLengthExceededError as error:
                    next_reserve = min(reserve * 2, max_reserve)
                    if next_reserve == reserve:
                        failed_summary = (
                            error.partial_response.content
                            or error.partial_response.reasoning_content
                            or ""
                        )
                        summary_output_exhausted = True
                        raise
                    reserve = next_reserve
        except Exception as error:
            chat._messages = original_messages
            chat.reset_response_chain()
            self.logger.warning(
                "Full context summary failed; trying compact recovery: %s: %s",
                type(error).__name__,
                error,
            )

        # This ladder is a required resilience boundary. Models can fail during
        # compaction in arbitrary ways, so keep both the independent short-summary
        # attempt and the deterministic handoff when changing summary internals.
        short_prompt = (
            "Briefly summarize how another agent should continue this task in 2-3 "
            "sentences. Preserve concrete results and the immediate next action.\n\n"
            f"{progress.recovery_state}"
        )
        short_ref = None
        try:
            output_limit = self._llm.get_model_output_limit()
            short_output_tokens = min(
                _MAX_RESERVED_SUMMARY_OUTPUT if output_limit is None else output_limit,
                _MAX_RESERVED_SUMMARY_OUTPUT,
            )
            if short_output_tokens < 1:
                raise ContextManagementInfrastructureError(
                    "Short context recovery requires a positive output budget."
                )
            short_result = await self._run_subagent(
                prompt=short_prompt,
                message_history=[],
                steps=[],
                session_id=(
                    f"{self._session_id}-summarization-"
                    f"{self._summarization_count}-short"
                ),
                agent_name="terminus-2-summarization-short",
                filename_suffix="short",
                summary_text="Short context recovery from bounded task state",
                max_tokens=short_output_tokens,
                subagent_name_for_logging="short summary subagent",
            )
            short_ref = short_result.trajectory_ref
            source = (
                f"{progress.recovery_state}\n\nSHORT HANDOFF SUMMARY:\n"
                f"{short_result.response.content}"
            )
        except Exception as error:
            self.logger.warning(
                "Short context summary failed; using deterministic recovery: %s: %s",
                type(error).__name__,
                error,
            )
            source = progress.recovery_state

        if progress.completed_summary:
            source += f"\n\nCOMPLETED SUMMARY STAGE:\n{progress.completed_summary}"
        if failed_summary:
            source += f"\n\nPARTIAL SUMMARY:\n{failed_summary}"

        source += "\n\nCONVERSATION EXCERPT:\n" + self._conversation_excerpt_text(
            original_messages
        )
        marker = (
            _OVERLONG_SUMMARY_MARKER
            if summary_output_exhausted
            else _SUMMARY_FAILURE_MARKER
        )
        handoff = self._bounded_recovery_handoff(
            chat=chat,
            original_messages=original_messages,
            source=source,
            marker=marker,
        )
        return handoff, [short_ref] if short_ref is not None else None

    async def _capture_recovery_state(
        self,
        *,
        original_instruction: str,
        pending_prompt: str,
        session: TmuxSession,
    ) -> str:
        """Capture state that must survive every lossy summary fallback."""
        try:
            terminal_screen = await session.capture_pane(capture_entire=False)
        except Exception as error:
            self.logger.warning(
                "Could not capture terminal state for context recovery: %s: %s",
                type(error).__name__,
                error,
            )
            terminal_screen = ""

        sections = [
            (
                "ORIGINAL TASK",
                self._limit_output_length(
                    original_instruction, max_bytes=_RECOVERY_STATE_MAX_BYTES
                ),
            ),
            (
                "PENDING OBSERVATION",
                self._limit_output_length(
                    pending_prompt, max_bytes=_RECOVERY_STATE_MAX_BYTES
                ),
            ),
            (
                "CURRENT TERMINAL SCREEN",
                self._limit_output_length(
                    terminal_screen, max_bytes=_RECOVERY_STATE_MAX_BYTES
                ),
            ),
        ]
        return "\n\n".join(
            f"{heading}:\n{content}" for heading, content in sections if content
        )

    def _bounded_recovery_handoff(
        self,
        *,
        chat: Chat,
        original_messages: list[dict[str, Any]],
        source: str,
        marker: str,
    ) -> str:
        """Replace live history with a bounded handoff and output headroom."""
        chat._messages = original_messages[:1]
        chat.reset_response_chain()

        context_limit = self._llm.get_model_context_limit()
        required_gap = min(
            max(RESERVED_SUMMARY_OUTPUT, self._proactive_summarization_threshold),
            max(1, context_limit // 2),
        )
        max_bytes = min(
            _OVERLONG_SUMMARY_MAX_BYTES,
            max(_OVERLONG_SUMMARY_MIN_BYTES, context_limit * 2),
        )
        while max_bytes >= _OVERLONG_SUMMARY_MIN_BYTES:
            excerpt = self._limit_output_length(source, max_bytes=max_bytes)
            handoff = f"{excerpt}\n\n{marker}"
            used_tokens = self._count_total_tokens(
                chat, extra_messages=[{"role": "user", "content": handoff}]
            )
            if context_limit - used_tokens >= required_gap:
                self.logger.warning(
                    "Continuing with a %s-byte bounded context-recovery handoff",
                    len(encode_model_output(handoff)),
                )
                return handoff
            max_bytes //= 2

        raise ContextManagementInfrastructureError(
            "The bounded summary fallback could not leave enough context to resume."
        )

    def _handoff_has_required_gap(self, chat: Chat, handoff: str) -> bool:
        context_limit = self._llm.get_model_context_limit()
        required_gap = min(
            max(RESERVED_SUMMARY_OUTPUT, self._proactive_summarization_threshold),
            max(1, context_limit // 2),
        )
        used_tokens = self._count_total_tokens(
            chat, extra_messages=[{"role": "user", "content": handoff}]
        )
        return context_limit - used_tokens >= required_gap

    def _conversation_excerpt_text(self, messages: list[dict[str, Any]]) -> str:
        rendered_messages = []
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            rendered_messages.append(f"{role}: {content}")
        return self._limit_output_length(
            "\n\n".join(rendered_messages),
            max_bytes=_OVERLONG_SUMMARY_MAX_BYTES,
        )

    async def _summarize(
        self,
        chat: Chat,
        original_instruction: str,
        session: TmuxSession,
        reserved_output: int = RESERVED_SUMMARY_OUTPUT,
        progress: SummaryProgress | None = None,
    ) -> tuple[str, list[SubagentTrajectoryRef] | None]:
        """Create a summary of the agent's work to pass to a new agent instance.

        This method implements a three-step context summarization process using separate
        subagents to compress conversation history while preserving critical information:

        **Step 1: Summary Generation (SUBAGENT 1)**
        - Message history: Unwound chat.messages (limited context after token freeing)
        - Prompt: Generate comprehensive summary of all work completed so far
        - Response: summary_response.content (detailed narrative of progress)

        **Step 2: Question Asking (SUBAGENT 2)**
        - Message history: [] (fresh start, no conversation context)
        - Prompt: Given the original task, summary from Step 1, and current terminal screen,
        generate questions about information missing from the summary.
        - Response: model_questions (list of clarifying questions)

        **Step 3: Answer Providing (SUBAGENT 3)**
        - Message history: chat.messages + [summary_prompt, summary_response] (extended context)
        - Prompt: Given the questions from Step 2, answer the questions based on full conversation history.
        - Response: answers_response.content (detailed answers)

        **Final handoff:**
        - Chat history replaced with: [system, question_prompt (which includes summary from step 1), model_questions]
        - Result: Compressed context that preserves critical task-specific information, allowing
        the main agent to continue working on the task without losing context.

        **Why three steps?** This question-answer approach ensures the summarization doesn't
        lose important details - the questions subagent identifies gaps, and the answers
        subagent (with full context) fills them in.

        Args:
            chat: Chat object containing conversation history (may be unwound)
            original_instruction: The original task instruction from user
            session: TmuxSession for capturing current terminal state

        Returns:
            tuple: (handoff_prompt, subagent_trajectory_refs)
                - handoff_prompt: The prompt to continue with (includes answers + instructions)
                - subagent_trajectory_refs: List of 3 SubagentTrajectoryRef objects, or None if summarization failed
        """
        if len(chat.messages) == 0:
            return original_instruction, None

        # ===== SUBAGENT 1 prompt (built before reserving) =====
        # The summary prompt is sent alongside the retained history but is NOT
        # part of chat.messages, so it must be counted when reserving output
        # headroom — otherwise a near-full chat plus the prompt overflows the
        # window by exactly the prompt size. Build it first so the unwind
        # targets the real request size.
        summary_prompt = f"""You are about to hand off your work to another AI agent.
            Please provide a comprehensive summary of what you have
            accomplished so far on this task:

Original Task: {original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed
            and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file
            locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you
            encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.


Be comprehensive and detailed. The next agent needs to understand everything
            that has happened so far in order to continue."""
        if progress is not None and progress.recovery_state:
            summary_prompt += (
                "\n\nThe following state was captured before any history was "
                f"trimmed:\n{progress.recovery_state}"
            )
        summary_overhead_messages = [{"role": "user", "content": summary_prompt}]

        # Guarantee output headroom BEFORE computing the trajectory slice: unwind
        # the input so the summary completion has at least ``reserved_output`` free
        # tokens counting the retained history PLUS the summary prompt, then pass
        # that same reserve as an explicit max_tokens on every summary
        # subagent call below. Provider rejection remains the final admission
        # authority and can trim the copied history further.
        self._unwind_messages_to_free_tokens(
            chat,
            target_free_tokens=reserved_output,
            extra_request_messages=summary_overhead_messages,
        )

        # Increment summarization count
        self._summarization_count += 1
        subagent_trajectory_refs = []

        # ===== SUBAGENT 1: Summary Generation =====
        summary_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-summary"
        )

        # Trajectory needs to reflect what is sent to LLM: essentially the previous chat
        # history, minus the messages that were removed to free up tokens.
        # Calculate how many trajectory steps to include based on remaining chat messages:
        # - Chat has: [system, agent1, user1, agent2, user2, ...]
        # - Trajectory has: [step1_system, step2_agent, step3_agent, ...]
        steps_to_include = _trajectory_step_count(len(chat.messages))
        summary_steps, step_id_counter = self._prepare_copied_trajectory_steps(
            steps_to_include
        )

        summary_result = await self._run_subagent(
            prompt=summary_prompt,
            message_history=chat.messages,
            steps=summary_steps,
            session_id=summary_session_id,
            agent_name="terminus-2-summarization-summary",
            filename_suffix="summary",
            summary_text=f"Context summarization {self._summarization_count}: Step 1 - Summary generation",
            subagent_name_for_logging="summary generation LLM call",
            max_tokens=reserved_output,
        )
        chat._messages = summary_result.accepted_history
        summary_response = summary_result.response
        if progress is not None:
            progress.completed_summary = summary_response.content
        subagent_trajectory_refs.append(summary_result.trajectory_ref)

        # ===== SUBAGENT 2: Question Asking =====
        current_screen = await session.capture_pane(capture_entire=False)
        questions_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-questions"
        )
        questions_steps = []

        question_prompt = f"""You are picking up work from a previous AI agent on this task:

**Original Task:** {original_instruction}

**Summary from Previous Agent:**
{summary_response.content}

**Current Terminal Screen:**
{current_screen}

Please begin by asking several questions (at least five, more if necessary)
about the current state of the solution that are not answered in the summary
from the prior agent. After you ask these questions you will be on your own,
so ask everything you need to know."""

        questions_result = await self._run_subagent(
            prompt=question_prompt,
            message_history=[],
            steps=questions_steps,
            session_id=questions_session_id,
            agent_name="terminus-2-summarization-questions",
            filename_suffix="questions",
            summary_text=f"Context summarization {self._summarization_count}: Step 2 - Question asking",
            subagent_name_for_logging="questions subagent",
            max_tokens=reserved_output,
        )
        questions_response = questions_result.response
        model_questions = questions_response.content
        subagent_trajectory_refs.append(questions_result.trajectory_ref)

        # ===== SUBAGENT 3: Answer Providing =====
        answers_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-answers"
        )

        # Reuse the exact trajectory accepted for the summary request. The
        # provider may have trimmed more history than the local estimate.
        steps_to_include = _trajectory_step_count(len(chat.messages))
        answers_steps, step_id_counter = self._prepare_copied_trajectory_steps(
            steps_to_include
        )

        # Add the summary prompt and response steps that are part of the message history
        # Mark these as copied context since they were already part of the summary subagent trajectory
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=summary_prompt,
                is_copied_context=True,
            )
        )
        step_id_counter += 1

        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=summary_response.model_name or self.model_alias,
                message=summary_response.content,
                reasoning_content=summary_response.reasoning_content,
                is_copied_context=True,
                extra={
                    "note": "Copied from summary subagent - metrics already recorded there"
                },
            )
        )
        step_id_counter += 1

        answer_request_prompt = (
            "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n"
            + model_questions
        )

        # The answer subagent should see: unwound chat history + summary prompt + summary response
        answers_message_history = chat.messages + [
            {"role": "user", "content": summary_prompt},
            assistant_message_from_response(
                summary_response, self._interleaved_thinking
            ),
        ]

        answers_result = await self._run_subagent(
            prompt=answer_request_prompt,
            message_history=answers_message_history,
            steps=answers_steps,
            session_id=answers_session_id,
            agent_name="terminus-2-summarization-answers",
            filename_suffix="answers",
            summary_text=f"Context summarization {self._summarization_count}: Step 3 - Answer providing",
            subagent_name_for_logging="answers subagent",
            max_tokens=reserved_output,
            parent_history_length=len(chat.messages),
        )
        answers_response = answers_result.response
        subagent_trajectory_refs.append(answers_result.trajectory_ref)

        # Update chat history with the handoff context
        # Note: We only include questions, not answers. The answers will be provided
        # via the handoff_prompt to the next agent iteration.
        chat._messages = [
            chat.messages[0],
            {"role": "user", "content": question_prompt},
            assistant_message_from_response(
                questions_response, self._interleaved_thinking
            ),
        ]
        chat.reset_response_chain()

        handoff_prompt = (
            "Here are the answers the other agent provided.\n\n"
            + answers_response.content
            + "\n\n"
            + "Continue working on this task from where the previous agent left off."
            " You can no longer ask questions. Please follow the spec to interact with "
            "the terminal."
        )

        return handoff_prompt, subagent_trajectory_refs

    async def _check_proactive_summarization(
        self,
        chat: Chat,
        original_instruction: str,
        session: TmuxSession | None,
        pending_prompt: str = "",
    ) -> tuple[str, list[SubagentTrajectoryRef] | None] | None:
        """Check context length and either summarize or proactively raise.

        This method ALWAYS runs regardless of the ``enable_summarize`` setting so
        that we never dispatch a generation request that the inference engine will
        reject for exceeding the context window. The estimate includes
        ``pending_prompt`` because it will be part of the next request.

        - When ``enable_summarize=False`` (e.g. RL rollouts that keep raw traces):
          if the prompt is within ``minimal_threshold`` of the context limit, raise
          ``ContextLengthExceededError`` BEFORE the oversized request is sent to the
          inference engine. This is the same exception type that ``_query_llm`` would
          re-raise on the post-overflow path, so the downstream rollback/mask handling
          is unchanged; we simply avoid the wasted overflow request (and the
          inference-engine validation error + litellm async-logging churn it triggers,
          which leaks file descriptors at high concurrency).
        - When ``enable_summarize=True``: behaves as before, summarizing once the
          ``_proactive_summarization_threshold`` is crossed.

        Ported from penfever/temp-override `_check_context_length`
        (commit 61df5270) — proactive raise instead of inert no-op when
        summarization is disabled.

        Returns:
            tuple: (summary_prompt, subagent_trajectory_ref) if summarization occurred
            None: if context length is within limits or summarization is disabled

        Raises:
            ContextLengthExceededError: if the prompt is within ``minimal_threshold``
                of the context limit and summarization is disabled.
        """
        context_limit = self._llm.get_model_context_limit()
        pending_messages = (
            [{"role": "user", "content": pending_prompt}] if pending_prompt else None
        )
        exact_prompt_token_ids = None
        if pending_prompt:
            exact_prompt_token_ids = await chat.prepare_prompt_token_ids(
                pending_prompt, **self._llm_call_kwargs
            )
        if exact_prompt_token_ids is not None:
            current_tokens = len(exact_prompt_token_ids)
        else:
            current_tokens = await self._acount_total_tokens(chat, pending_messages)
        free_tokens = context_limit - current_tokens

        # When summarization is disabled, raise only once we've actually reached the
        # limit (within minimal_threshold) — not at the larger proactive threshold —
        # so RL training still uses (nearly) the full context window. Raising here
        # stops the oversized request before it reaches the inference engine.
        if not self._enable_summarize:
            minimal_threshold = 2048
            if free_tokens < minimal_threshold:
                raise ContextLengthExceededError(
                    f"Context length limit reached: {current_tokens} tokens used, "
                    f"only {free_tokens} free tokens remaining (limit: {context_limit}). "
                    f"Summarization is disabled (enable_summarize=False)."
                )
            return None

        if free_tokens < self._proactive_summarization_threshold:
            self.logger.debug(
                f"Proactively summarizing. Free tokens: approximately {free_tokens}"
            )
            if session is None:
                raise RuntimeError("Cannot summarize without session")
            return await self._summarize_with_retry(
                chat,
                original_instruction,
                session,
                pending_prompt=pending_prompt,
            )

        return None

    @retry(
        stop=stop_after_attempt(3),
        # Context-capacity and exhausted-recovery outcomes need a new conversation
        # state or a new trial; repeating the identical call cannot resolve them.
        retry=(
            retry_if_not_exception_type(
                (
                    ContextLengthExceededError,
                    ContextBudgetExceededError,
                    ContextManagementInfrastructureError,
                    LLMRequestTimeoutError,
                )
            )
            # To avoid asyncio.CancelledError retries which inherits from BaseException
            # rather than Exception
            & retry_if_exception_type(Exception)
        ),
        reraise=True,
    )
    async def _query_llm(
        self,
        chat: Chat,
        prompt: str,
        request_context: LLMRequestContext,
    ) -> LLMResponse:
        logging_paths = request_context.logging_paths
        logging_path = logging_paths.debug
        prompt_path = logging_paths.prompt
        response_path = logging_paths.response

        if prompt_path is not None:
            _enqueue_episode_text(prompt_path, prompt)

        try:
            return await self._dispatch_llm_request(
                chat, prompt, logging_path, response_path
            )

        except ContextLengthExceededError as error:
            return await self._recover_context_capacity(
                chat=chat,
                error=error,
                pending_prompt=prompt,
                request_context=request_context,
            )

        except OutputLengthExceededError as e:
            return await self._recover_output_overflow(
                chat=chat,
                prompt=prompt,
                error=e,
                request_context=request_context,
            )

    async def _dispatch_llm_request(
        self,
        chat: Chat,
        prompt: str,
        logging_path: Path | UPath | None,
        response_path: Path | UPath | None,
    ) -> LLMResponse:
        start_time = time.time()
        response = await chat.chat(
            prompt,
            logging_path=logging_path,
            **self._llm_call_kwargs,
        )
        self._track_api_request_time(start_time)
        if response_path is not None:
            _enqueue_episode_text(response_path, response.content)
        return response

    async def _recover_context_capacity(
        self,
        *,
        chat: Chat,
        error: ContextLengthExceededError,
        pending_prompt: str,
        request_context: LLMRequestContext,
    ) -> LLMResponse:
        if not self._enable_summarize:
            self.logger.debug("Context capacity exceeded and summarization is OFF.")
            raise error
        if request_context.session is None:
            raise RuntimeError("Cannot handle context length error without session")

        self.logger.debug("Context capacity exceeded. Recovering context.")
        summary_prompt, subagent_refs = await self._summarize_with_retry(
            chat,
            request_context.original_instruction,
            request_context.session,
            pending_prompt=pending_prompt,
        )
        self._pending_subagent_refs = subagent_refs
        self._pending_handoff_prompt = summary_prompt
        if request_context.logging_paths.prompt is not None:
            _enqueue_episode_text(request_context.logging_paths.prompt, summary_prompt)

        try:
            return await self._dispatch_llm_request(
                chat,
                summary_prompt,
                request_context.logging_paths.debug,
                request_context.logging_paths.response,
            )
        except ContextLengthExceededError:
            # Provider-side templates can exceed local token estimates. Degrade
            # once more to system prompt plus a bounded handoff before giving up.
            original_messages = list(chat.messages)
            recovery_state = await self._capture_recovery_state(
                original_instruction=request_context.original_instruction,
                pending_prompt=summary_prompt,
                session=request_context.session,
            )
            compact_prompt = self._bounded_recovery_handoff(
                chat=chat,
                original_messages=original_messages,
                source=recovery_state,
                marker=_SUMMARY_FAILURE_MARKER,
            )
            self._pending_handoff_prompt = compact_prompt
            if request_context.logging_paths.prompt is not None:
                _enqueue_episode_text(
                    request_context.logging_paths.prompt, compact_prompt
                )
            try:
                return await self._dispatch_llm_request(
                    chat,
                    compact_prompt,
                    request_context.logging_paths.debug,
                    request_context.logging_paths.response,
                )
            except ContextLengthExceededError as error:
                raise ContextManagementInfrastructureError(
                    "The bounded context-recovery handoff still exceeded the "
                    "provider context limit."
                ) from error

    async def _recover_output_overflow(
        self,
        *,
        chat: Chat,
        prompt: str,
        error: OutputLengthExceededError,
        request_context: LLMRequestContext,
    ) -> LLMResponse:
        self.logger.debug(f"Output length exceeded: {error}")
        truncated_response = (
            error.truncated_response or "[TRUNCATED RESPONSE NOT AVAILABLE]"
        )
        response_path = request_context.logging_paths.response

        salvaged_response = self._parser.salvage_truncated_response(truncated_response)
        if salvaged_response:
            self.logger.debug(
                "Output exceeded length but found valid response. "
                "Using truncated version."
            )
            if response_path is not None:
                _enqueue_episode_text(response_path, salvaged_response)
            return salvaged_response

        warnings_text = ""
        parse_result = self._parser.parse_response(truncated_response)
        if parse_result.warning:
            warnings_text = (
                f"\n\nParser warnings from your truncated response:\n"
                f"{parse_result.warning}"
            )

        output_limit = self._llm.get_model_output_limit()
        limit_str = (
            f"{output_limit} tokens"
            if output_limit is not None
            else "the maximum output length"
        )
        error_msg = (
            "ERROR!! NONE of the actions you just requested were performed "
            f"because you exceeded {limit_str}. "
            f"Your outputs must be less than {limit_str}. Re-issue this request, "
            f"breaking it into chunks each of which is less than {limit_str}."
            f"{warnings_text}"
        )
        chat.append_response(prompt, error.partial_response)
        # The retained partial response is still contiguous with the served
        # prompt, so only the provider-side response chain is invalid here.
        chat.reset_provider_response_chain()
        if response_path is not None:
            _enqueue_episode_text(response_path, error_msg)

        return await self._query_llm(
            chat=chat,
            prompt=error_msg,
            request_context=request_context,
        )

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: EpisodeLoggingPaths,
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> LLMInteractionResult:
        request_context = LLMRequestContext(
            logging_paths=logging_paths,
            original_instruction=original_instruction,
            session=session,
        )
        llm_response = await self._query_llm(chat, prompt, request_context)

        result = self._parser.parse_response(llm_response.content)

        feedback = ""
        if result.error:
            feedback += f"ERROR: {result.error}"
            if result.warning:
                feedback += f"\nWARNINGS: {result.warning}"
        elif result.warning:
            feedback += f"WARNINGS: {result.warning}"

        if result.warning:
            self.logger.debug(f"Parser warnings: {result.warning}")

        commands = []
        for parsed_cmd in result.commands:
            commands.append(
                Command(
                    keystrokes=parsed_cmd.keystrokes,
                    duration_sec=min(parsed_cmd.duration, 60),
                )
            )

        return LLMInteractionResult(
            commands=commands,
            is_task_complete=result.is_task_complete,
            feedback=feedback,
            analysis=result.analysis,
            plan=result.plan,
            response=llm_response,
        )

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        """Execute a batch of commands in the terminal.

        Args:
            commands: List of commands to execute
            session: TmuxSession instance

        Returns:
            Tuple of (timeout_occurred, terminal_output)
        """
        if self._batch_turn_execs:
            terminal_output, session_alive = await session.send_keys_and_capture(
                [
                    TmuxKeystrokeBatch(command.keystrokes, command.duration_sec)
                    for command in commands
                ]
            )
            self._session_alive_hint = session_alive
            return False, self._limit_output_length(terminal_output)

        for command in commands:
            try:
                await session.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=command.duration_sec,
                )
            except TimeoutError:
                return True, self._timeout_template.format(
                    timeout_sec=command.duration_sec,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(
                        await session.get_incremental_output()
                    ),
                )

        return False, self._limit_output_length(await session.get_incremental_output())

    async def _execute_interaction(
        self,
        interaction: LLMInteractionResult,
        session: TmuxSession,
    ) -> str:
        """Execute one parsed interaction and return the next observation."""
        _, terminal_output = await self._execute_commands(interaction.commands, session)
        return terminal_output

    def _append_native_tool_results(
        self,
        chat: Chat,
        interaction: LLMInteractionResult,
        observation: str,
    ) -> None:
        """Complete a native-tool conversation, when the agent uses one."""

    def _trajectory_tool_calls(
        self,
        interaction: LLMInteractionResult,
        episode: int,
    ) -> list[ToolCall] | None:
        """Represent the interaction's actions in the ATIF trajectory."""
        tool_calls: list[ToolCall] = []
        for index, command in enumerate(interaction.commands, start=1):
            tool_calls.append(
                ToolCall(
                    tool_call_id=f"call_{episode}_{index}",
                    function_name="bash_command",
                    arguments={
                        "keystrokes": command.keystrokes,
                        "duration": command.duration_sec,
                    },
                )
            )
        if interaction.is_task_complete:
            tool_calls.append(
                ToolCall(
                    tool_call_id=f"call_{episode}_task_complete",
                    function_name="mark_task_complete",
                    arguments={},
                )
            )
        return tool_calls or None

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        logging_dir: Path | UPath | None = None,
        original_instruction: str = "",
    ) -> None:
        if self._context is None:
            raise RuntimeError("Agent context is not set. This should never happen.")

        prompt = initial_prompt

        self._context.n_input_tokens = 0
        self._context.n_output_tokens = 0
        self._context.n_cache_tokens = 0
        self._context.cost_usd = None

        if self._session is None:
            raise RuntimeError("Session is not set. This should never happen.")

        # Step ID offset accounts for initial steps in the trajectory:
        # Step 1: system message (includes task instruction)
        # Steps 2+: agent episodes (starting from episode 0)

        for episode in range(self._max_episodes):
            self._n_episodes = episode + 1
            session_alive = self._session_alive_hint
            self._session_alive_hint = None
            if session_alive is None:
                session_alive = await self._session.is_session_alive()
            if not session_alive:
                self.logger.debug("Session has ended, breaking out of agent loop")
                self._stop_reason = AgentStopReason.SESSION_ENDED
                return

            # Always check context length: summarize if enabled, or proactively
            # raise ContextLengthExceededError if disabled — before dispatching an
            # oversized request to the inference engine.
            if original_instruction:
                proactive_summary_result = await self._check_proactive_summarization(
                    chat,
                    original_instruction,
                    self._session,
                    pending_prompt=prompt,
                )
                if proactive_summary_result:
                    prompt, subagent_refs = proactive_summary_result
                    # Store subagent_refs to add a system step later
                    self._pending_subagent_refs = subagent_refs
                    # Also store the handoff prompt to add as a user step
                    self._pending_handoff_prompt = prompt

            logging_paths = self._setup_episode_logging(logging_dir, episode)

            # Track token counts and cost before this step
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            interaction = await self._handle_llm_interaction(
                chat, prompt, logging_paths, original_instruction, self._session
            )
            is_task_complete = interaction.is_task_complete
            feedback = interaction.feedback
            analysis = interaction.analysis
            plan = interaction.plan
            llm_response = interaction.response

            # If we have pending subagent refs, add a system step to record the delegation
            # This must happen before we build the agent step
            # We use len(self._trajectory_steps) + 1 as the step_id to ensure it's sequential
            if self._pending_subagent_refs:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context summarization and handoff to continue task.",
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    subagent_trajectory_ref=self._pending_subagent_refs
                                )
                            ]
                        ),
                    )
                )
                self._pending_subagent_refs = None

            # Handle handoff prompt based on linear_history mode
            if self._pending_handoff_prompt:
                # If linear_history mode is enabled, split trajectory immediately WITHOUT adding handoff step
                # The handoff step will be added to the continuation trajectory during the split
                if self._linear_history:
                    self._split_trajectory_on_summarization(
                        self._pending_handoff_prompt
                    )
                else:
                    # For non-linear mode, add the handoff prompt as a user step
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=self._pending_handoff_prompt,
                        )
                    )
                self._pending_handoff_prompt = None

            # Create message content from analysis and plan, or use raw response if raw_content is enabled
            if self._save_raw_content_in_trajectory:
                # Use the raw LLM response content for SFT data export
                message_content = llm_response.content
            else:
                # Parse into structured format (analysis + plan)
                message_parts = []
                if analysis:
                    message_parts.append(f"Analysis: {analysis}")
                if plan:
                    message_parts.append(f"Plan: {plan}")
                message_content = "\n".join(message_parts) if message_parts else ""

            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                self._append_native_tool_results(chat, interaction, prompt)
                # For error cases, we still want to record the step
                # Use the raw response as the message since parsing failed
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before

                if llm_response.model_name:
                    self._last_response_model_name = llm_response.model_name
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=llm_response.model_name or self.model_alias,
                        message=llm_response.content,
                        reasoning_content=llm_response.reasoning_content,
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    content=prompt,
                                )
                            ]
                        ),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens
                            - tokens_before_output,
                            cached_tokens=cache_tokens_used
                            if cache_tokens_used > 0
                            else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                continue

            terminal_output = await self._execute_interaction(
                interaction, self._session
            )

            # Capture the pending completion state before potentially modifying it
            was_pending_completion = self._pending_completion

            # Construct the observation (what gets sent back to the LLM)
            if is_task_complete:
                if self._pending_completion:
                    observation = terminal_output
                else:
                    self._pending_completion = True
                    observation = self._get_completion_confirmation_message(
                        terminal_output
                    )
            else:
                self._pending_completion = False
                if feedback and "WARNINGS:" in feedback:
                    observation = (
                        f"Previous response had warnings:\n{feedback}\n\n"
                        f"{self._limit_output_length(terminal_output)}"
                    )
                else:
                    observation = self._limit_output_length(terminal_output)

            self._append_native_tool_results(chat, interaction, observation)

            # Record the step in trajectory
            cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
            step_cost = chat.total_cost - cost_before

            # Create tool_calls array from commands and task completion
            # Note: Although Terminus 2 doesn't offer native tool calling (it uses text-based command parsing),
            # we represent parsed commands as tool calls for better trajectory analysis and compatibility with tooling.
            # However, when raw_content mode is enabled, we skip tool_calls generation to preserve the raw LLM response.
            observation_results: list[ObservationResult] = []

            if not self._save_raw_content_in_trajectory:
                tool_calls = self._trajectory_tool_calls(interaction, episode)
                observation_results.append(ObservationResult(content=observation))
            else:
                # In raw_content mode, just add observation without tool_calls
                tool_calls = None
                observation_results.append(
                    ObservationResult(
                        content=observation,
                    )
                )

            # Build the step object using Pydantic models
            if llm_response.model_name:
                self._last_response_model_name = llm_response.model_name
            self._trajectory_steps.append(
                Step(
                    step_id=len(self._trajectory_steps) + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=llm_response.model_name or self.model_alias,
                    message=message_content,
                    reasoning_content=llm_response.reasoning_content,
                    tool_calls=tool_calls,
                    observation=Observation(results=observation_results),
                    metrics=Metrics(
                        prompt_tokens=chat.total_input_tokens - tokens_before_input,
                        completion_tokens=chat.total_output_tokens
                        - tokens_before_output,
                        cached_tokens=cache_tokens_used
                        if cache_tokens_used > 0
                        else None,
                        cost_usd=step_cost if step_cost > 0 else None,
                        prompt_token_ids=llm_response.prompt_token_ids,
                        completion_token_ids=llm_response.completion_token_ids,
                        logprobs=llm_response.logprobs,
                    ),
                )
            )

            # Optional per-episode trajectory snapshot (cadence-gated, default
            # off): the final dump in run()'s finally block plus the mandatory
            # continuation-boundary dumps cover the durable record.
            self._maybe_persist_trajectory_per_turn()

            # Per-turn callback (e.g., PRM hint injection or early termination).
            # Ported from penfever/temp-override 2026-05-23: the OT-Agent
            # teacher_hint PRM relied on this hook to inject "[HINT FROM TEACHER]:"
            # markers, but the call site was missing on penfever/otagent-latest,
            # silently turning every _hint ablation into a no-op base run.
            # See OT-Agent prm/teacher_hint.py:as_turn_callback() and SkyRL
            # examples/terminal_bench/harbor_config.py:813-815 for how the
            # callable gets wired into agent_kwargs.
            if self._turn_callback is not None:
                cb_result = self._turn_callback(
                    turn=episode,
                    trajectory_steps=self._trajectory_steps,
                    messages=chat.messages,
                )
                if isinstance(cb_result, str) and cb_result:
                    # Hint injection: prepend hint to observation so the
                    # student sees it in the next prompt.
                    # Guard: check if injecting the hint would exceed context limit.
                    context_limit = self._llm.get_model_context_limit()
                    # Offloaded off the coordinator event loop (blocking /tokenize POST).
                    current_tokens = await self._acount_total_tokens(chat)
                    hint_tokens = (
                        len(cb_result) // 3
                    )  # conservative char-to-token estimate
                    if current_tokens + hint_tokens > context_limit - 512:
                        raise ContextLengthExceededError(
                            f"Cannot inject teacher hint: would exceed context limit. "
                            f"Current tokens: {current_tokens}, estimated hint tokens: "
                            f"{hint_tokens}, limit: {context_limit}."
                        )
                    self.logger.info(
                        f"Turn callback injected hint at episode {episode}"
                    )
                    observation = cb_result + observation
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="system",
                            message=cb_result,
                        )
                    )
                    self._maybe_persist_trajectory_per_turn()
                elif cb_result is True:
                    # Early termination requested by callback (e.g., PRM
                    # detected agent collapse or stuck loop).
                    self.logger.info(
                        f"Turn callback requested early termination at episode {episode}"
                    )
                    self._early_termination_reason = (
                        f"turn_callback requested stop at episode {episode}"
                    )
                    self._stop_reason = AgentStopReason.EARLY_TERMINATION
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="system",
                            message=(
                                f"[Early termination] Turn callback stopped the agent "
                                f"at episode {episode}."
                            ),
                        )
                    )
                    self._maybe_persist_trajectory_per_turn()
                    break

            if is_task_complete:
                if was_pending_completion:
                    # Task is confirmed complete (this is the second time task_complete was True), return
                    self._stop_reason = AgentStopReason.TASK_COMPLETE
                    return
                else:
                    # First completion attempt - ask for confirmation and continue
                    prompt = observation
                    continue

            prompt = observation

        # A turn-callback early termination leaves the loop via `break` and is
        # a deliberate external stop, not budget exhaustion — it keeps its own
        # recorded outcome and must not be reclassified below. Keyed on
        # `_stop_reason` (cleared per run) rather than
        # `_early_termination_reason`, which is write-only and survives
        # `_reset_per_run_state` across multi-step trials.
        if self._stop_reason is AgentStopReason.EARLY_TERMINATION:
            return

        # The episode budget ran out without a confirmed task completion. A
        # capped rollout must NOT finalize like a delivered one: record the
        # boundary in the trajectory, then raise the classified error so the
        # trial (and downstream consumers via the harbor_config taxonomy) can
        # see and count the outcome while the verifier still scores the work.
        self._stop_reason = AgentStopReason.TURN_CAP_EXHAUSTED
        if self._pending_completion:
            detail = (
                "The agent declared task complete on its final turn, but the "
                "cap cut off the confirmation exchange."
            )
        else:
            detail = "The agent never declared the task complete."
        self._trajectory_steps.append(
            Step(
                step_id=len(self._trajectory_steps) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="system",
                message=(
                    f"[Turn cap exhausted] The agent ran out of turns "
                    f"(max_turns={self._max_episodes}) without confirming "
                    f"task completion. {detail}"
                ),
            )
        )
        self._maybe_persist_trajectory_per_turn()
        raise TurnCapExhaustedError(
            f"Agent exhausted its turn cap of {self._max_episodes} episodes "
            f"without confirming task completion. {detail}",
            pending_completion=self._pending_completion,
        )

    def _reset_per_run_state(self) -> None:
        """Reset all per-run state. The same Terminus2 instance is reused
        across multiple `run()` invocations in multi-step trials, so any
        accumulator that should be scoped to a single step must be reset here.
        """
        self._trajectory_steps = []
        self._api_request_times = []
        self._n_episodes = 0
        self._summarization_count = 0
        self._continuation_segments_persisted = 0
        self._subagent_metrics = SubagentMetrics()
        self._subagent_rollout_details = []
        self._pending_completion = False
        self._pending_subagent_refs = None
        self._pending_handoff_prompt = None
        self._stop_reason = None
        self._session_alive_hint = None
        self._session_id = self._user_provided_session_id or str(uuid.uuid4())

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._reset_per_run_state()
        self._chat = Chat(self._llm, interleaved_thinking=self._interleaved_thinking)
        self._context = context

        if self._session is None:
            raise RuntimeError("Session is not set")

        # Get the terminal state for the initial prompt
        terminal_state = self._limit_output_length(
            await self._session.get_incremental_output()
        )

        augmented_instruction = instruction
        if self.mcp_servers:
            mcp_info = "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n"
            for s in self.mcp_servers:
                if s.transport == "stdio":
                    args_str = " ".join(s.args)
                    mcp_info += f"- {s.name}: stdio transport, command: {s.command} {args_str}\n"
                else:
                    mcp_info += f"- {s.name}: {s.transport} transport, url: {s.url}\n"
            augmented_instruction = instruction + mcp_info

        skills_section = await self._build_skills_section(environment)
        if skills_section:
            augmented_instruction += skills_section

        initial_prompt = self._prompt_template.format(
            instruction=augmented_instruction,
            terminal_state=terminal_state,
        )

        self._trajectory_steps.append(
            Step(
                step_id=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=initial_prompt,
            )
        )

        try:
            await self._run_agent_loop(
                initial_prompt=initial_prompt,
                chat=self._chat,
                logging_dir=self.logs_dir,
                original_instruction=instruction,
            )
        except asyncio.CancelledError:
            # The trial timed out: `asyncio.wait_for(agent.run(...))` in
            # trial._execute_agent cancelled us. Historically this branch reaped
            # LiteLLM's detached async_success_handler logging tasks (which held
            # the trial's Ray ObjectRefs via their closures → reference_count.cc
            # SIGABRT, Perlmutter 53509359). The completion transport is now
            # openai.AsyncOpenAI (no litellm LoggingWorker / detached callback
            # tasks), so there are no orphan background tasks to reap — the
            # cancellation simply propagates as AgentTimeoutError with an empty
            # rollout, semantics unchanged.
            raise
        finally:
            # Always capture metrics and rollout details even if the agent times out or errors
            # Combine main chat rollout details with subagent rollout details
            context.rollout_details = (
                self._chat.rollout_details + self._subagent_rollout_details
            )

            # Include subagent metrics in context totals
            context.n_input_tokens = (
                self._chat.total_input_tokens
                + self._subagent_metrics.total_prompt_tokens
            )
            context.n_output_tokens = (
                self._chat.total_output_tokens
                + self._subagent_metrics.total_completion_tokens
            )
            context.n_cache_tokens = (
                self._chat.total_cache_tokens
                + self._subagent_metrics.total_cached_tokens
            )
            total_cost = self._chat.total_cost + self._subagent_metrics.total_cost_usd
            context.cost_usd = total_cost if total_cost > 0 else None
            context.metadata = {
                "n_episodes": self._n_episodes,
                "api_request_times_msec": self._api_request_times,
                "summarization_count": self._summarization_count,
                "stop_reason": self._stop_reason,
            }
            if self._store_all_messages:
                context.metadata["all_messages"] = self._chat.messages

            # Attempt a final trajectory dump on every exit path (success,
            # timeout/cancellation, error). Optional admission never delays the
            # result; a saturated diagnostic lane records and logs the drop.
            self._persist_final_trajectory()

    # ===== Trajectory-related helper functions =====

    @staticmethod
    def _remove_metrics_from_copied_steps(steps: list[Step]) -> None:
        """Remove metrics from copied trajectory steps and mark as copied context.

        Args:
            steps: List of trajectory steps to modify in-place
        """
        for step in steps:
            # Mark all copied steps with is_copied_context=True
            step.is_copied_context = True

            # Remove metrics to avoid duplication
            if step.metrics:
                step.metrics = None
                if step.extra is None:
                    step.extra = {}
                step.extra["note"] = (
                    "Metrics omitted to avoid duplication - already recorded in parent trajectory"
                )

    def _prepare_copied_trajectory_steps(
        self, steps_to_include: int
    ) -> tuple[list[Step], int]:
        """Prepare trajectory steps for subagent by copying and removing metrics.

        Args:
            steps_to_include: Number of steps to include from main trajectory

        Returns:
            Tuple of (copied_steps, next_step_id)
        """
        copied_steps = copy.deepcopy(self._trajectory_steps[:steps_to_include])
        self._remove_metrics_from_copied_steps(copied_steps)
        next_step_id = len(copied_steps) + 1
        return copied_steps, next_step_id

    def _append_subagent_response_step(
        self,
        steps: list[Step],
        step_id: int,
        response: LLMResponse,
        usage_info,
        subagent_name: str,
    ) -> None:
        """Append a response step with conditional metrics to trajectory steps.

        Args:
            steps: List of steps to append to
            step_id: ID for the new step
            response: LLM response
            usage_info: Usage info (may be None)
            subagent_name: Name of subagent for warning message
        """
        if usage_info:
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=response.model_name or self.model_alias,
                    message=response.content,
                    reasoning_content=response.reasoning_content,
                    metrics=Metrics(
                        prompt_tokens=usage_info.prompt_tokens,
                        completion_tokens=usage_info.completion_tokens,
                        cached_tokens=usage_info.cache_tokens,
                        cost_usd=usage_info.cost_usd
                        if usage_info.cost_usd > 0
                        else None,
                        prompt_token_ids=response.prompt_token_ids,
                        completion_token_ids=response.completion_token_ids,
                        logprobs=response.logprobs,
                    ),
                )
            )
        else:
            self.logger.debug(f"Failed to get token usage for {subagent_name}")
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=response.model_name or self.model_alias,
                    message=response.content,
                    reasoning_content=response.reasoning_content,
                )
            )

    def _save_subagent_trajectory(
        self,
        session_id: str,
        agent_name: str,
        steps: list[Step],
        usage_info,
        filename_suffix: str,
        summary_text: str,
    ) -> SubagentTrajectoryRef:
        """Save a subagent trajectory to disk and return its reference.

        Args:
            session_id: Session ID for the subagent
            agent_name: Name of the subagent (e.g., "terminus-2-summarization-summary")
            steps: List of trajectory steps
            usage_info: Usage information from LLM response
            filename_suffix: Suffix for trajectory filename (e.g., "summary", "questions", "answers")
            summary_text: Human-readable summary for the trajectory ref extra field

        Returns:
            SubagentTrajectoryRef for inclusion in parent trajectory
        """
        total_prompt, total_completion, total_cached, total_cost = (
            self._extract_usage_metrics(usage_info)
        )

        trajectory = Trajectory(
            session_id=session_id,
            agent=Agent(
                name=agent_name,
                version=self.version() or "unknown",
                model_name=self.model_alias,
                extra={
                    "parent_session_id": self._session_id,
                    "summarization_index": self._summarization_count,
                },
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_cached_tokens=total_cached,
                total_cost_usd=total_cost if total_cost > 0 else None,
            ),
        )

        trajectory_path = (
            self.logs_dir
            / f"trajectory.summarization-{self._summarization_count}-{filename_suffix}.json"
        )
        # Runs on the event-loop thread at summarization boundaries; enqueue the
        # serialization + (possibly remote fsspec) write on the artifact writer.
        # `trajectory` is local to this call, so late serialization is safe.
        get_artifact_writer().submit_optional(
            trajectory_path,
            lambda: format_trajectory_json(trajectory.to_json_dict()),
            description=f"terminus-2 {filename_suffix} subagent trajectory",
        )

        return SubagentTrajectoryRef(
            session_id=session_id,
            trajectory_path=str(trajectory_path.name),
            extra={"summary": summary_text},
        )

    def _convert_chat_messages_to_steps(
        self,
        chat_messages: list[dict],
        additional_user_message: str | None = None,
        mark_as_copied: bool = False,
    ) -> list[Step]:
        """Convert chat messages to trajectory steps.

        Args:
            chat_messages: List of chat messages with 'role' and 'content' fields
            additional_user_message: Optional additional user message to append as final step
            mark_as_copied: If True, mark all steps with is_copied_context=True (for continuation trajectories)

        Returns:
            List of Step objects representing the chat history
        """
        steps = []
        step_id = 1

        for msg in chat_messages:
            role = msg["role"]
            content = msg["content"]

            # Map chat role to trajectory source
            if role == "assistant":
                source = "agent"
                step = Step(
                    step_id=step_id,
                    source=source,
                    message=content,
                    model_name=self._last_response_model_name or self.model_alias,
                )
            else:
                source = role
                step = Step(
                    step_id=step_id,
                    source=source,
                    message=content,
                )

            # Mark as copied context if this is for a continuation trajectory
            if mark_as_copied:
                step.is_copied_context = True

            steps.append(step)
            step_id += 1

        # Add the additional user message if provided
        # Note: The additional user message is NOT marked as copied since it's the new handoff prompt
        if additional_user_message:
            steps.append(
                Step(
                    step_id=step_id,
                    source="user",
                    message=additional_user_message,
                )
            )

        return steps

    def _split_trajectory_on_summarization(self, handoff_prompt: str) -> None:
        """Split trajectory on summarization when linear_history is enabled.

        Saves current trajectory segment and resets for continuation with full linear history.

        Args:
            handoff_prompt: The handoff prompt containing answers
        """
        # Save current trajectory segment before creating a continuation.
        # This boundary dump is MANDATORY regardless of trajectory_dump_cadence:
        # the steps below are truncated from memory right after, so this segment
        # exists nowhere else. _persist_trajectory snapshots the step list on
        # this thread before the reset, then serializes + writes in the
        # background artifact writer. The continuation index is the number of
        # boundary dumps already completed (_continuation_segments_persisted),
        # advanced after this dump so a later exit writes the next segment.
        self._persist_trajectory(self._continuation_segments_persisted, continued=True)
        self._continuation_segments_persisted += 1

        # Create new session_id for continuation
        self._session_id = (
            f"{self._session_id.split('-cont-')[0]}-cont-{self._summarization_count}"
        )

        # After dumping the trajectory till the summarization stage, reset trajectory by
        # converting from chat messages (excluding the last response which will be added
        # by the normal agent loop flow). Mark all these steps as copied context since they
        # were already present in the previous trajectory segment.
        if self._chat:
            self._trajectory_steps = self._convert_chat_messages_to_steps(
                self._chat.messages[:-1], mark_as_copied=True
            )

    def _build_trajectory(
        self, continuation_index: int, *, continued: bool = False
    ) -> tuple[Trajectory, Path | UPath] | None:
        """Build the trajectory model + destination path for a dump.

        Cheap enough to run on the event-loop thread. The step list is
        snapshotted (shallow copy) so the built trajectory is immune to later
        appends/resets; steps themselves are never mutated after being appended.
        The expensive serialization (``to_json_dict`` + ``format_trajectory_json``)
        happens later on an artifact-writer worker thread.

        Args:
            continuation_index: The continuation index to use for filename and metadata.
                               For the initial trajectory, use 0.
                               For the first continuation, use 1, etc.
            continued: Whether a later segment follows this one (boundary dump
                       during a completed split). The final exit-path dump and
                       per-turn snapshots pass False: no continuation follows.
        """
        if not self._context:
            self.logger.debug("No context available, skipping trajectory dump")
            return None

        # Construct the trajectory using Pydantic models for validation
        final_metrics = FinalMetrics(
            total_prompt_tokens=self._context.n_input_tokens,
            total_completion_tokens=self._context.n_output_tokens,
            total_cached_tokens=self._context.n_cache_tokens or 0,
            total_cost_usd=self._context.cost_usd if self._context.cost_usd else None,
        )

        agent_extra: dict[str, Any] = {
            "parser": self._parser_name,
        }
        if self._temperature is not None:
            agent_extra["temperature"] = self._temperature
        if self._llm_kwargs:
            agent_extra["llm_kwargs"] = self._llm_kwargs
        if self._linear_history and continuation_index > 0:
            agent_extra["continuation_index"] = continuation_index

        # A continuation ref is only correct when this dump is the boundary
        # segment that a completed split is carving a follow-up out of. The
        # final exit-path dump is the last segment; an interrupted summarization
        # has no follow-up even though _summarization_count was incremented.
        continued_trajectory_ref = None
        if self._linear_history and continued:
            next_continuation_index = continuation_index + 1
            continued_trajectory_ref = f"trajectory.cont-{next_continuation_index}.json"

        trajectory = Trajectory(
            session_id=self._session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_alias,
                extra=agent_extra,
            ),
            steps=list(self._trajectory_steps),
            final_metrics=final_metrics,
            continued_trajectory_ref=continued_trajectory_ref,
        )

        # Determine trajectory filename based on continuation index
        if self._linear_history and continuation_index > 0:
            trajectory_path = (
                self.logs_dir / f"trajectory.cont-{continuation_index}.json"
            )
        else:
            trajectory_path = self.logs_dir / "trajectory.json"
        return trajectory, trajectory_path

    def _persist_trajectory(
        self, continuation_index: int, *, continued: bool = False
    ) -> None:
        """Enqueue a trajectory dump on the background artifact writer.

        Non-blocking for the caller: the writer's own small worker pool runs
        both the CPU-bound ``format_trajectory_json`` pass and the (possibly
        remote fsspec/s3fs) write. Neither may run on the shared event loop
        nor on asyncio's default executor — under concurrent trials the
        default executor fills with blocked uploads and starves every other
        ``to_thread`` user (live py-spy caught 12-30 blocked upload threads
        per coordinator). Write errors and optional-lane drops are logged by
        the writer and never raise here.
        """
        built = self._build_trajectory(continuation_index, continued=continued)
        if built is None:
            return
        trajectory, trajectory_path = built
        get_artifact_writer().submit_optional(
            trajectory_path,
            lambda: format_trajectory_json(trajectory.to_json_dict()),
            description=f"terminus-2 {trajectory_path.name}",
        )

    def _persist_final_trajectory(self) -> None:
        """Persist the final in-memory trajectory segment on every run() exit.

        Uses the completed boundary-dump count as the continuation index: when
        no split completed (including a summarization cancelled mid-flight) the
        full conversation lands in the canonical ``trajectory.json``; after a
        completed split it lands in the next ``trajectory.cont-N.json``.
        """
        self._persist_trajectory(self._continuation_segments_persisted)

    def _maybe_persist_trajectory_per_turn(self) -> None:
        """Per-episode dump hook: no-op unless cadence is PER_TURN.

        The default FINAL cadence relies on the run()-finally dump (always
        written) plus the mandatory continuation-boundary dumps, eliminating
        the per-turn cross-region write that collapsed rollout throughput.
        """
        if self._trajectory_dump_cadence is TrajectoryDumpCadence.PER_TURN:
            self._persist_trajectory(self._continuation_segments_persisted)
