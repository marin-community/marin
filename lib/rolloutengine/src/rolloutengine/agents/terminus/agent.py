# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Terminus-2 agent loop, and its ``AgentHarness``.

Adapted from Harbor's ``Terminus2`` (Apache-2.0); see ``rolloutengine.agents.terminus``. Terminus sends one
user message per turn: the task prompt first, then the terminal output that the model's keystrokes produced.
The model answers in JSON or XML. A parse error asks the model to fix its response, and a completion claim
asks it to confirm. Near the context limit, or when the model rejects a request as too long, Terminus asks
subagents to summarize its work and continues from their handoff.

The harness records every request, including the summarization subagents', so each counts toward
``RolloutLimits.max_turns``. The rollout limits end the loop; the agent has no turn cap of its own.
"""

import asyncio
import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from taskcompendium.models import TextMessage

from rolloutengine.agents.environment import MachineEnvironment
from rolloutengine.agents.terminus.llm import (
    Chat,
    ContextLengthExceededError,
    ContextManagedResponse,
    ContextManagementInfrastructureError,
    ContextWindowManager,
    LLMResponse,
    OutputLengthExceededError,
    TerminusLLM,
    assistant_message_from_response,
    trim_latest_turn,
)
from rolloutengine.agents.terminus.terminus_json_plain_parser import TerminusJSONPlainParser
from rolloutengine.agents.terminus.terminus_xml_plain_parser import TerminusXMLPlainParser
from rolloutengine.agents.terminus.tmux_session import (
    MalformedModelResponseError,
    ModelInitiatedTmuxFailure,
    TmuxKeystrokeBatch,
    TmuxSession,
    encode_model_output,
)
from rolloutengine.harness.core import HarnessContext
from rolloutengine.harness.external import AgentCompletion, AgentHarness, LengthStopPolicy, ReplyMatch
from rolloutengine.models import RolloutEnded

logger = logging.getLogger(__name__)

TEMPLATES = Path(__file__).parent / "templates"
SESSION_NAME = "terminus-2"
VLLM_CONTEXT_SAFETY_BUFFER = 8
"""Tokens that Harbor keeps free below the server's ``max_model_len``, against off-by-one prompt renderings."""
MAX_COMMAND_DURATION = 60
OUTPUT_LIMIT_BYTES = 10000

RESERVED_SUMMARY_OUTPUT = 4096
_MAX_RESERVED_SUMMARY_OUTPUT = 16384
_OVERLONG_SUMMARY_MAX_BYTES = 32768
_OVERLONG_SUMMARY_MIN_BYTES = 1024
_RECOVERY_STATE_MAX_BYTES = 8192
_OVERLONG_SUMMARY_MARKER = (
    "OVERLONG MODEL SUMMARIZATION WAS TRUNCATED. PLEASE RESUME YOUR WORK FROM "
    "THE PARTIAL HANDOFF AND CONVERSATION EXCERPT ABOVE."
)
_SUMMARY_FAILURE_MARKER = "MODEL SUMMARIZATION FAILED. PLEASE RESUME FROM THE BOUNDED TASK STATE ABOVE."


class TerminusParser(StrEnum):
    """The response format that the prompt asks for and the parser reads."""

    JSON = "json"
    XML = "xml"


@dataclass(frozen=True)
class TerminusOptions:
    """Terminus-2's settings, with Harbor's defaults.

    ``max_input_tokens`` and ``max_output_tokens`` are Harbor's ``model_info`` limits. Each main request asks
    for ``max_output_tokens``, and summaries reserve up to that many. The context limit is ``max_input_tokens``,
    or the served ``max_model_len`` less 8 tokens when that is smaller; ``served_max_model_len`` of ``None``
    applies ``max_input_tokens`` alone. ``extra_body`` goes into every request, for example chat template
    arguments. With ``interleaved_thinking``, the history keeps each reply's reasoning. ``enable_summarize``
    compacts the history near the context limit, when fewer than ``proactive_summarization_threshold`` tokens
    are free, and after a context rejection; without it, the agent fails there. ``batch_turn_execs`` runs each
    turn's keystrokes, waits and terminal captures in one machine command.
    """

    parser: TerminusParser
    max_input_tokens: int
    max_output_tokens: int
    served_max_model_len: int | None
    temperature: float | None = None
    reasoning_effort: str | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    interleaved_thinking: bool = False
    enable_summarize: bool = True
    proactive_summarization_threshold: int = 8000
    tmux_pane_width: int = 160
    tmux_pane_height: int = 40
    batch_turn_execs: bool = True

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("Terminus token limits must be positive")

    @property
    def context_limit(self) -> int:
        if self.served_max_model_len is None:
            return self.max_input_tokens
        return min(self.max_input_tokens, max(1, self.served_max_model_len - VLLM_CONTEXT_SAFETY_BUFFER))


def terminus_harness(options: TerminusOptions) -> AgentHarness:
    """An ``AgentHarness`` that runs Terminus-2 with ``options`` on the task machine.

    After a reply cut by the output limit, Terminus asks the model again in a new conversation, as Harbor does.
    """

    async def run(context: HarnessContext, complete: AgentCompletion) -> None:
        await run_terminus(options, context, complete)

    return AgentHarness(run, length_stop=LengthStopPolicy.NEW_CONVERSATION, reply_match=ReplyMatch.EXACT)


async def run_terminus(options: TerminusOptions, context: HarnessContext, complete: AgentCompletion) -> None:
    """Run Terminus-2 on the task's instruction until it confirms completion, runs out of context, or the attempt ends.

    Raises:
        ValueError: The task is not one user instruction, or it has no machine.
    """
    events = context.task.context.events
    if len(events) != 1 or not isinstance(events[0], TextMessage) or events[0].role != "user":
        raise ValueError("Terminus requires one user instruction as the task context")
    if context.machine is None:
        raise ValueError("Terminus requires a task machine")
    session = TmuxSession(
        session_name=SESSION_NAME,
        environment=MachineEnvironment(context.machine, context.run_id),
        pane_width=options.tmux_pane_width,
        pane_height=options.tmux_pane_height,
    )
    await session.start()
    # Ending the session kills the jobs the agent started in it, such as a server that the tests check, so it
    # ends after grading, as in Harbor.
    context.cleanup.push_after_grading("terminus_tmux", session.close)
    try:
        await Terminus2(options, complete, session).run(events[0].content)
    except ContextLengthExceededError as error:
        # As in Harbor, an agent out of context ends its attempt, and the engine grades the work it left.
        logger.warning("Terminus ran out of context in attempt %s; grading its work: %s", context.run_id, error)
    except OutputLengthExceededError as error:
        # Terminus recovers a cut reply in its turn loop, so one escapes only from context recovery: the agent
        # could not continue after its context ran out. Harbor leaves such a trial unscored; the engine grades
        # its work, as it does when the context runs out.
        logger.warning(
            "Terminus reached the output limit while recovering its context in attempt %s; grading its work: %s",
            context.run_id,
            error,
        )
    except ModelInitiatedTmuxFailure as error:
        # As in Harbor, a model that ends its own terminal ends its attempt, and the engine grades its work.
        logger.warning("The model ended its terminal in attempt %s; grading its work: %s", context.run_id, error)
    except MalformedModelResponseError as error:
        # As in Harbor, keystrokes that cannot be sent to the terminal end the attempt, and the engine grades its work.
        logger.warning(
            "The model sent unencodable keystrokes in attempt %s; grading its work: %s", context.run_id, error
        )


@dataclass
class SummaryProgress:
    recovery_state: str
    completed_summary: str = ""


@dataclass(frozen=True)
class Command:
    keystrokes: str
    duration_sec: float


@dataclass(frozen=True)
class LLMInteractionResult:
    commands: list[Command]
    is_task_complete: bool
    feedback: str
    response: LLMResponse


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


class TerminusResponseParser(Protocol):
    def parse_response(self, response: str) -> TerminusParseResult: ...

    def salvage_truncated_response(self, truncated_response: str, /) -> str | None:
        """Return text that remains valid after truncation, when available."""
        ...


class Terminus2:
    """One attempt of Terminus-2: its chat, its terminal and its context recovery."""

    def __init__(self, options: TerminusOptions, complete: AgentCompletion, session: TmuxSession):
        self._parser_name = options.parser
        self._llm = TerminusLLM(
            complete,
            context_limit=options.context_limit,
            output_limit=options.max_output_tokens,
            temperature=options.temperature,
            reasoning_effort=options.reasoning_effort,
        )
        self._parser = self._get_parser()
        self._prompt_template = self._get_prompt_template_path().read_text()
        self._timeout_template = (TEMPLATES / "timeout.txt").read_text()
        self._context_window_manager = ContextWindowManager(self._llm)
        self._session = session
        self._pending_completion = False
        self._batch_turn_execs = options.batch_turn_execs
        self._session_alive_hint: bool | None = None
        self._llm_call_kwargs: dict[str, Any] = {"max_tokens": options.max_output_tokens}
        if options.extra_body:
            self._llm_call_kwargs["extra_body"] = dict(options.extra_body)
        self._enable_summarize = options.enable_summarize
        self._proactive_summarization_threshold = options.proactive_summarization_threshold
        self._interleaved_thinking = options.interleaved_thinking
        self._chat = Chat(self._llm, interleaved_thinking=self._interleaved_thinking)

    def _get_parser(self) -> TerminusResponseParser:
        """Return the appropriate parser instance for this format."""
        match self._parser_name:
            case TerminusParser.JSON:
                return TerminusJSONPlainParser()
            case TerminusParser.XML:
                return TerminusXMLPlainParser()

    def _get_prompt_template_path(self) -> Path:
        """Return the path to the prompt template for this format."""
        match self._parser_name:
            case TerminusParser.JSON:
                return TEMPLATES / "terminus-json-plain.txt"
            case TerminusParser.XML:
                return TEMPLATES / "terminus-xml-plain.txt"

    def _get_error_response_type(self) -> str:
        """Return the response type name for error messages.

        Examples: 'JSON response', 'response'
        """
        match self._parser_name:
            case TerminusParser.JSON:
                return "JSON response"
            case TerminusParser.XML:
                return "response"

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Return the format-specific task completion confirmation message."""
        match self._parser_name:
            case TerminusParser.JSON:
                return (
                    f"Current terminal state:\n{terminal_output}\n\n"
                    "Are you sure you want to mark the task as complete? "
                    "This will trigger your solution to be graded and you won't be able to "
                    'make any further corrections. If so, include "task_complete": true '
                    "in your JSON response again."
                )
            case TerminusParser.XML:
                return (
                    f"Current terminal state:\n{terminal_output}\n\n"
                    "Are you sure you want to mark the task as complete? "
                    "This will trigger your solution to be graded and you won't be able to "
                    "make any further corrections. If so, include "
                    "<task_complete>true</task_complete> again."
                )

    def _count_total_tokens(self, chat: Chat, extra_messages: list[dict[str, Any]] | None = None) -> int:
        """Count total tokens across all messages in the chat, followed by ``extra_messages``.

        ``extra_messages`` (e.g. a summary prompt appended to the retained
        history only for the request) are included in the count so budget
        accounting matches the real request size.
        """
        if not chat.messages and not extra_messages:
            return 0
        return chat.count_tokens(extra_messages)

    def _limit_output_length(self, output: str, max_bytes: int = OUTPUT_LIMIT_BYTES) -> str:
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

        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")

        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")

        omitted_bytes = len(output_bytes) - len(first_portion.encode("utf-8")) - len(last_portion.encode("utf-8"))

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

        free_tokens = context_limit - self._count_total_tokens(chat, extra_request_messages)
        logger.debug(
            f"Unwound messages. Remaining messages: {len(chat.messages)}, Free tokens: approximately {free_tokens}"
        )

    async def _run_subagent(
        self,
        prompt: str,
        message_history: list[dict[str, Any]],
        max_tokens: int,
    ) -> ContextManagedResponse:
        """Send one context-managed subagent request without the main request's ``max_tokens``.

        ``max_tokens`` is the output-token budget for the call. This guarantees the
        summary completion cannot be silently truncated by running out of output room.
        """
        call_kwargs = {key: value for key, value in self._llm_call_kwargs.items() if key != "max_tokens"}
        return await self._context_window_manager.call(
            prompt=prompt,
            message_history=message_history,
            max_tokens=max_tokens,
            call_kwargs=call_kwargs,
        )

    async def _summarize_with_retry(
        self,
        chat: Chat,
        original_instruction: str,
        session: TmuxSession,
        pending_prompt: str = "",
    ) -> str:
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
            max_reserve = output_limit
            if max_reserve < 1:
                raise ContextManagementInfrastructureError("Context recovery requires a positive output budget.")
            reserve = min(RESERVED_SUMMARY_OUTPUT, max_reserve)
            while True:
                chat._messages = list(original_messages)
                try:
                    handoff = await self._summarize(
                        chat,
                        original_instruction,
                        session,
                        reserved_output=reserve,
                        progress=progress,
                    )
                    if self._handoff_has_required_gap(chat, handoff):
                        return handoff
                    raise ContextLengthExceededError(
                        "The complete generated handoff does not leave enough context for the next model response."
                    )
                except OutputLengthExceededError as error:
                    next_reserve = min(reserve * 2, max_reserve)
                    if next_reserve == reserve:
                        failed_summary = error.partial_response.content or error.partial_response.reasoning_content or ""
                        summary_output_exhausted = True
                        raise
                    reserve = next_reserve
        # An ended rollout refuses every later request, so no fallback can recover it.
        except RolloutEnded:
            raise
        except Exception as error:
            chat._messages = original_messages
            logger.warning(
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
        try:
            output_limit = self._llm.get_model_output_limit()
            short_output_tokens = min(output_limit, _MAX_RESERVED_SUMMARY_OUTPUT)
            if short_output_tokens < 1:
                raise ContextManagementInfrastructureError("Short context recovery requires a positive output budget.")
            short_result = await self._run_subagent(
                prompt=short_prompt,
                message_history=[],
                max_tokens=short_output_tokens,
            )
            source = f"{progress.recovery_state}\n\nSHORT HANDOFF SUMMARY:\n{short_result.response.content}"
        except RolloutEnded:
            raise
        except Exception as error:
            logger.warning(
                "Short context summary failed; using deterministic recovery: %s: %s",
                type(error).__name__,
                error,
            )
            source = progress.recovery_state

        if progress.completed_summary:
            source += f"\n\nCOMPLETED SUMMARY STAGE:\n{progress.completed_summary}"
        if failed_summary:
            source += f"\n\nPARTIAL SUMMARY:\n{failed_summary}"

        source += "\n\nCONVERSATION EXCERPT:\n" + self._conversation_excerpt_text(original_messages)
        marker = _OVERLONG_SUMMARY_MARKER if summary_output_exhausted else _SUMMARY_FAILURE_MARKER
        return self._bounded_recovery_handoff(
            chat=chat,
            original_messages=original_messages,
            source=source,
            marker=marker,
        )

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
            logger.warning(
                "Could not capture terminal state for context recovery: %s: %s",
                type(error).__name__,
                error,
            )
            terminal_screen = ""

        sections = [
            (
                "ORIGINAL TASK",
                self._limit_output_length(original_instruction, max_bytes=_RECOVERY_STATE_MAX_BYTES),
            ),
            (
                "PENDING OBSERVATION",
                self._limit_output_length(pending_prompt, max_bytes=_RECOVERY_STATE_MAX_BYTES),
            ),
            (
                "CURRENT TERMINAL SCREEN",
                self._limit_output_length(terminal_screen, max_bytes=_RECOVERY_STATE_MAX_BYTES),
            ),
        ]
        return "\n\n".join(f"{heading}:\n{content}" for heading, content in sections if content)

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
            used_tokens = self._count_total_tokens(chat, extra_messages=[{"role": "user", "content": handoff}])
            if context_limit - used_tokens >= required_gap:
                logger.warning(
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
        used_tokens = self._count_total_tokens(chat, extra_messages=[{"role": "user", "content": handoff}])
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
    ) -> str:
        """Create a summary of the agent's work to pass to a new agent instance.

        Three subagent requests compress the history:

        1. Summary: the unwound history and a request for a full summary.
        2. Questions: a fresh conversation with the original task, that summary and the current screen, asking
           for questions that the summary does not answer.
        3. Answers: the unwound history, the summary exchange and the questions, asking for the answers.

        The chat becomes the first prompt, the questions prompt and the questions, and the returned handoff
        prompt holds the answers.
        """
        if len(chat.messages) == 0:
            return original_instruction

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
                f"\n\nThe following state was captured before any history was trimmed:\n{progress.recovery_state}"
            )
        summary_overhead_messages = [{"role": "user", "content": summary_prompt}]

        # Guarantee output headroom: unwind the input so the summary completion
        # has at least ``reserved_output`` free tokens counting the retained
        # history PLUS the summary prompt, then pass that same reserve as an
        # explicit max_tokens on every summary subagent call below. Provider
        # rejection remains the final admission authority and can trim the
        # copied history further.
        self._unwind_messages_to_free_tokens(
            chat,
            target_free_tokens=reserved_output,
            extra_request_messages=summary_overhead_messages,
        )

        # ===== SUBAGENT 1: Summary Generation =====
        summary_result = await self._run_subagent(
            prompt=summary_prompt,
            message_history=chat.messages,
            max_tokens=reserved_output,
        )
        chat._messages = summary_result.accepted_history
        summary_response = summary_result.response
        if progress is not None:
            progress.completed_summary = summary_response.content

        # ===== SUBAGENT 2: Question Asking =====
        current_screen = await session.capture_pane(capture_entire=False)

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
            max_tokens=reserved_output,
        )
        questions_response = questions_result.response
        model_questions = questions_response.content

        # ===== SUBAGENT 3: Answer Providing =====
        answer_request_prompt = (
            "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n"
            + model_questions
        )

        # The answer subagent should see: unwound chat history + summary prompt + summary response
        answers_message_history = [
            *chat.messages,
            {"role": "user", "content": summary_prompt},
            assistant_message_from_response(summary_response, self._interleaved_thinking),
        ]

        answers_result = await self._run_subagent(
            prompt=answer_request_prompt,
            message_history=answers_message_history,
            max_tokens=reserved_output,
        )
        answers_response = answers_result.response

        # Update chat history with the handoff context
        # Note: We only include questions, not answers. The answers will be provided
        # via the handoff_prompt to the next agent iteration.
        chat._messages = [
            chat.messages[0],
            {"role": "user", "content": question_prompt},
            assistant_message_from_response(questions_response, self._interleaved_thinking),
        ]

        handoff_prompt = (
            "Here are the answers the other agent provided.\n\n"
            + answers_response.content
            + "\n\n"
            + "Continue working on this task from where the previous agent left off."
            " You can no longer ask questions. Please follow the spec to interact with "
            "the terminal."
        )

        return handoff_prompt

    async def _check_proactive_summarization(
        self,
        chat: Chat,
        original_instruction: str,
        session: TmuxSession,
        pending_prompt: str = "",
    ) -> str | None:
        """Check context length and either summarize or proactively raise.

        This runs regardless of ``enable_summarize`` so that an oversized request
        is not sent. The estimate includes ``pending_prompt`` because it will be
        part of the next request.

        - Without summarization, raise ``ContextLengthExceededError`` once the
          prompt is within ``minimal_threshold`` of the context limit.
        - With summarization, summarize once fewer than
          ``proactive_summarization_threshold`` tokens are free.

        Returns:
            The handoff prompt if summarization occurred, else None.
        """
        context_limit = self._llm.get_model_context_limit()
        pending_messages = [{"role": "user", "content": pending_prompt}] if pending_prompt else None
        current_tokens = self._count_total_tokens(chat, pending_messages)
        free_tokens = context_limit - current_tokens

        # When summarization is disabled, raise only once we've actually reached the
        # limit (within minimal_threshold) — not at the larger proactive threshold —
        # so the agent still uses (nearly) the full context window.
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
            logger.debug(f"Proactively summarizing. Free tokens: approximately {free_tokens}")
            return await self._summarize_with_retry(
                chat,
                original_instruction,
                session,
                pending_prompt=pending_prompt,
            )

        return None

    async def _query_llm(self, chat: Chat, prompt: str, original_instruction: str) -> LLMResponse:
        try:
            return await chat.chat(prompt, **self._llm_call_kwargs)

        except ContextLengthExceededError as error:
            return await self._recover_context_capacity(
                chat=chat,
                error=error,
                pending_prompt=prompt,
                original_instruction=original_instruction,
            )

        except OutputLengthExceededError as e:
            return await self._recover_output_overflow(
                chat=chat,
                prompt=prompt,
                error=e,
                original_instruction=original_instruction,
            )

    async def _recover_context_capacity(
        self,
        *,
        chat: Chat,
        error: ContextLengthExceededError,
        pending_prompt: str,
        original_instruction: str,
    ) -> LLMResponse:
        if not self._enable_summarize:
            logger.debug("Context capacity exceeded and summarization is OFF.")
            raise error

        logger.debug("Context capacity exceeded. Recovering context.")
        summary_prompt = await self._summarize_with_retry(
            chat,
            original_instruction,
            self._session,
            pending_prompt=pending_prompt,
        )

        try:
            return await chat.chat(summary_prompt, **self._llm_call_kwargs)
        except ContextLengthExceededError:
            # Provider-side templates can exceed local token estimates. Degrade
            # once more to system prompt plus a bounded handoff before giving up.
            original_messages = list(chat.messages)
            recovery_state = await self._capture_recovery_state(
                original_instruction=original_instruction,
                pending_prompt=summary_prompt,
                session=self._session,
            )
            compact_prompt = self._bounded_recovery_handoff(
                chat=chat,
                original_messages=original_messages,
                source=recovery_state,
                marker=_SUMMARY_FAILURE_MARKER,
            )
            try:
                return await chat.chat(compact_prompt, **self._llm_call_kwargs)
            except ContextLengthExceededError as error:
                raise ContextManagementInfrastructureError(
                    "The bounded context-recovery handoff still exceeded the provider context limit."
                ) from error

    async def _recover_output_overflow(
        self,
        *,
        chat: Chat,
        prompt: str,
        error: OutputLengthExceededError,
        original_instruction: str,
    ) -> LLMResponse:
        logger.debug(f"Output length exceeded: {error}")
        truncated_response = error.truncated_response or "[TRUNCATED RESPONSE NOT AVAILABLE]"

        salvaged_response = self._parser.salvage_truncated_response(truncated_response)
        if salvaged_response:
            logger.debug("Output exceeded length but found valid response. Using truncated version.")
            # Harbor returns the salvaged text itself, which fails where its caller reads the response's
            # content. The XML parser alone salvages; the response is not added to the chat, as in Harbor.
            return LLMResponse(
                content=salvaged_response,
                reasoning_content=None,
                tool_calls=None,
                usage=error.partial_response.usage,
                prompt_tokens=error.partial_response.prompt_tokens,
            )

        warnings_text = ""
        parse_result = self._parser.parse_response(truncated_response)
        if parse_result.warning:
            warnings_text = f"\n\nParser warnings from your truncated response:\n{parse_result.warning}"

        output_limit = self._llm.get_model_output_limit()
        limit_str = f"{output_limit} tokens"
        error_msg = (
            "ERROR!! NONE of the actions you just requested were performed "
            f"because you exceeded {limit_str}. "
            f"Your outputs must be less than {limit_str}. Re-issue this request, "
            f"breaking it into chunks each of which is less than {limit_str}."
            f"{warnings_text}"
        )
        chat.append_response(prompt, error.partial_response)

        return await self._query_llm(chat=chat, prompt=error_msg, original_instruction=original_instruction)

    async def _handle_llm_interaction(
        self, chat: Chat, prompt: str, original_instruction: str = ""
    ) -> LLMInteractionResult:
        llm_response = await self._query_llm(chat, prompt, original_instruction)

        result = self._parser.parse_response(llm_response.content)

        feedback = ""
        if result.error:
            feedback += f"ERROR: {result.error}"
            if result.warning:
                feedback += f"\nWARNINGS: {result.warning}"
        elif result.warning:
            feedback += f"WARNINGS: {result.warning}"

        if result.warning:
            logger.debug(f"Parser warnings: {result.warning}")

        commands = []
        for parsed_cmd in result.commands:
            commands.append(
                Command(
                    keystrokes=parsed_cmd.keystrokes,
                    duration_sec=min(parsed_cmd.duration, MAX_COMMAND_DURATION),
                )
            )

        return LLMInteractionResult(
            commands=commands,
            is_task_complete=result.is_task_complete,
            feedback=feedback,
            response=llm_response,
        )

    async def _execute_commands(self, commands: list[Command], session: TmuxSession) -> str:
        """Execute a batch of commands in the terminal and return the terminal output."""
        if self._batch_turn_execs:
            terminal_output, session_alive = await session.send_keys_and_capture(
                [TmuxKeystrokeBatch(command.keystrokes, command.duration_sec) for command in commands]
            )
            self._session_alive_hint = session_alive
            return self._limit_output_length(terminal_output)

        for command in commands:
            try:
                await session.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=command.duration_sec,
                )
            except TimeoutError:
                return self._timeout_template.format(
                    timeout_sec=command.duration_sec,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(await session.get_incremental_output()),
                )

        return self._limit_output_length(await session.get_incremental_output())

    async def _run_agent_loop(self, initial_prompt: str, chat: Chat, original_instruction: str) -> None:
        prompt = initial_prompt

        for _episode in itertools.count():
            session_alive = self._session_alive_hint
            self._session_alive_hint = None
            if session_alive is None:
                session_alive = await self._session.is_session_alive()
            if not session_alive:
                logger.debug("Session has ended, breaking out of agent loop")
                return

            # Always check context length: summarize if enabled, or proactively
            # raise ContextLengthExceededError if disabled — before dispatching an
            # oversized request.
            if original_instruction:
                handoff_prompt = await self._check_proactive_summarization(
                    chat,
                    original_instruction,
                    self._session,
                    pending_prompt=prompt,
                )
                if handoff_prompt:
                    prompt = handoff_prompt

            interaction = await self._handle_llm_interaction(chat, prompt, original_instruction)
            is_task_complete = interaction.is_task_complete
            feedback = interaction.feedback

            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                continue

            terminal_output = await self._execute_commands(interaction.commands, self._session)

            # Capture the pending completion state before potentially modifying it
            was_pending_completion = self._pending_completion

            # Construct the observation (what gets sent back to the LLM)
            if is_task_complete:
                if self._pending_completion:
                    observation = terminal_output
                else:
                    self._pending_completion = True
                    observation = self._get_completion_confirmation_message(terminal_output)
            else:
                self._pending_completion = False
                if feedback and "WARNINGS:" in feedback:
                    observation = (
                        f"Previous response had warnings:\n{feedback}\n\n{self._limit_output_length(terminal_output)}"
                    )
                else:
                    observation = self._limit_output_length(terminal_output)

            if is_task_complete and was_pending_completion:
                # Task is confirmed complete (this is the second time task_complete was True)
                return

            prompt = observation

    async def run(self, instruction: str) -> None:
        terminal_state = self._limit_output_length(await self._session.get_incremental_output())

        initial_prompt = self._prompt_template.format(
            instruction=instruction,
            terminal_state=terminal_state,
        )

        try:
            await self._run_agent_loop(
                initial_prompt=initial_prompt,
                chat=self._chat,
                original_instruction=instruction,
            )
        except asyncio.CancelledError:
            await self._session.interrupt()
            raise
