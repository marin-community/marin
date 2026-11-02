import copy
import json
import os
import re
import shlex
import uuid
from datetime import datetime, timezone
from typing import Any

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    ModelAuthenticationError,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError
from harbor.literal.proxy import CORRELATION_ID_HEADER, LITERAL_LOG_FILENAME
from harbor.literal.rollout_build import (
    build_rollout_details_from_pairs,
    read_literal_pairs,
)
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.agent.rollout_detail import RolloutDetail
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.trial.errors import AgentTimeoutError
from harbor.utils.artifact_writer import get_artifact_writer
from harbor.utils.trajectory_utils import format_trajectory_json


class OpenCode(BaseInstalledAgent):
    """
    The OpenCode agent uses the opencode-ai tool to solve tasks.

    Parses the JSON lines emitted by ``opencode run --format=json`` (captured
    to ``opencode.txt``) into an ATIF trajectory.

    Literal token-level traces: OpenCode's JSON stream carries token *counts*
    but not token IDs / logprobs. When run behind a
    :class:`~harbor.literal.proxy.RecordProxy` (which records the
    ``literal.jsonl`` log into the agent logs dir), the proxy pairs are merged
    into trajectory steps by turn index in ``populate_context_post_run``.

    Stdout JSON line types:
        user          - optional user prompt emitted outside agent turns
        text          - agent text output (part.type == "text")
        reasoning     - optional explicit reasoning text (part.type == "reasoning")
        tool_use      - tool call with input/output (part.type == "tool")
        step_start    - marks the beginning of an agent turn
        step_finish   - marks the end of a turn, carries cost & token data
        error         - error event
    """

    SUPPORTS_ATIF: bool = True
    SUPPORTS_LITERAL_TRACES: bool = True

    _OUTPUT_FILENAME = "opencode.txt"
    _NVM_SOURCE_COMMAND = "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh"
    CLI_FLAGS = [
        CliFlag("variant", cli="--variant", type="str"),
    ]

    # Base config written to opencode.json before each run.
    # Extend per-job via ``opencode_config`` in agents[].kwargs, e.g.:
    #
    #   opencode_config:
    #     experimental:
    #       continue_loop_on_deny: true
    _DEFAULT_CONFIG: dict[str, Any] = {}

    # Fallback token budgets used to bound the opencode per-model ``limit`` when
    # the caller does not plumb ``model_info``. opencode defaults a custom
    # provider's ``max_tokens`` to ~32000 (opencode #1735); against a
    # 32768-token served window that leaves only ~768 tokens for the prompt, so
    # any real prompt overflows with a 400 ("maximum context length ... total
    # ... tokens") and every trial fails. A bounded ``limit.output`` caps the
    # per-request ``max_tokens``; ``limit.context`` sizes opencode's
    # compaction window. Both are chosen so ``prompt + max_tokens`` can never
    # exceed the served window.
    _DEFAULT_MODEL_WINDOW: int = 32768
    _DEFAULT_MAX_OUTPUT: int = 4096

    # Server-side hard timeout for the opencode run exec. opencode has no native
    # turn cap and harbor runs the ENTIRE agent loop as ONE detached (run_async)
    # sandbox command, so a runaway ``opencode`` (looping forever) is otherwise
    # only bounded by the trial-level ``asyncio.wait_for`` — which CANCELS the
    # local poll but never kills the detached process, orphaning the sandbox
    # (auto_stop=0) long past its deadline (observed: 51-82 min sandboxes vs a
    # 30-min timeout, 0 completions). run() wraps the invocation in a shell
    # ``timeout`` so the SANDBOX reaps opencode at the deadline regardless of
    # client-side polling. The server-side TERM fires ``_SERVER_TIMEOUT_GRACE_SEC``
    # before the app-level wait_for and the SIGKILL follows
    # ``_SERVER_TIMEOUT_KILL_AFTER_SEC`` later — both strictly before the app
    # deadline — so opencode is always reaped server-side FIRST, closing the
    # un-killed-cancel race.
    _SERVER_TIMEOUT_GRACE_SEC: int = 15
    _SERVER_TIMEOUT_KILL_AFTER_SEC: int = 5
    # GNU ``timeout`` returns 124 when its deadline expires without
    # ``--preserve-status``. Signal exits such as 137 or 143 may instead come
    # from opencode or a sandbox lifecycle event and must retain their original
    # AgentKilledBySignalError classification.
    _TIMEOUT_EXIT_CODE: int = 124

    def __init__(
        self,
        *args,
        opencode_config: dict[str, Any] | None = None,
        collect_rollout_details: bool = False,
        max_turns: int | None = None,
        **kwargs,
    ):
        # Capture the served-model api_base the RL generator passes as a kwarg, BEFORE
        # super() (BaseInstalledAgent keeps only CLI_FLAG/ENV_VAR descriptor kwargs;
        # BaseAgent drops the rest). It is the PRIMARY baseURL source in
        # _build_register_config_command: unlike the env vars it travels WITH the trial
        # dispatch as data, so it is process-boundary-safe. The RL trial's opencode
        # config-build runs in a context that does NOT inherit the run_rl process's
        # late-set HARBOR_MODEL_ENDPOINT/OPENAI_BASE_URL, which is why the env-only path
        # produced a `"undefined/chat/completions"` baseURL (fullgate1-3).
        self._api_base: str | None = kwargs.pop("api_base", None)
        # Served-model token budgets, popped BEFORE super() for the same
        # process-boundary reason as api_base (BaseAgent drops non-descriptor
        # kwargs). Used to bound the per-model ``limit`` written into
        # opencode.json so opencode never requests a ``max_tokens`` that
        # overflows the served context window. When the caller does not plumb
        # it (``model_info`` absent), _resolve_model_limit falls back to safe
        # defaults — so opencode is ALWAYS bounded, on both the RL-generator and
        # the eval construction paths. Mirrors openhands.py / kimi_cli.py.
        self._model_info: dict[str, Any] | None = kwargs.pop("model_info", None)
        super().__init__(*args, **kwargs)
        self._opencode_config: dict[str, Any] = opencode_config or {}
        # OpenCode's JSON stream may omit the prompt, so retain the rendered
        # instruction until the post-run trajectory conversion.
        self._instruction: str | None = None
        # Graceful per-run agentic-iteration cap. opencode exposes NO ``run`` CLI
        # flag for this, but its config supports a per-agent ``steps`` field
        # (legacy alias ``maxSteps``) that forces the agent to stop iterating and
        # respond with text once the cap is hit. Written into opencode.json for
        # the default ``build`` agent in _build_register_config_command. None →
        # unbounded (field omitted) → byte-identical to before. This is the
        # GRACEFUL companion to the hard server-side ``timeout`` applied in run():
        # ``steps`` lets opencode wind down cleanly; the ``timeout`` is the hard
        # backstop that forcibly reaps a run that ignores or outlives the cap.
        self._max_turns: int | None = max_turns
        # When True, surface the RecordProxy-captured literal token_ids/logprobs
        # into ``context.rollout_details`` (what the RL rollout loop consumes),
        # in addition to the existing merge into ``trajectory.json``. OpenCode is
        # a CLI agent that bypasses harbor ``Chat``, so unlike terminus-2 its
        # ``rollout_details`` is otherwise never populated. Off by default →
        # behavior is byte-identical to before (rollout_details stays None).
        self._collect_rollout_details: bool = collect_rollout_details
        # Per-trial correlation id, minted only when collecting rollout details.
        # Injected into every opencode->vLLM request (provider options.headers) so a
        # shared controller RecordProxy can attribute each interleaved log entry to
        # THIS trial (see CORRELATION_ID_HEADER). Also surfaced on
        # ``context.metadata`` so the RL-loop glue knows which id to correlate.
        # None when the flag is off → no header, no metadata (byte-identical).
        self.rollout_correlation_id: str | None = (
            uuid.uuid4().hex if collect_rollout_details else None
        )

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge *override* into *base* in place, recursing into nested dicts."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenCode._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def _user_event_text(event: dict[str, Any]) -> str | None:
        parts = event.get("parts")
        if not isinstance(parts, list):
            return None
        texts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(text for text in texts if text) or None

    def _prepend_user_step(
        self,
        steps: list[Step],
        user_message: str | None,
        user_timestamp: int | None,
    ) -> None:
        user_text = user_message or self._instruction
        if not user_text:
            return

        steps.insert(
            0,
            Step(
                step_id=1,
                timestamp=self._millis_to_iso(user_timestamp),
                source="user",
                message=user_text,
            ),
        )
        for index, step in enumerate(steps, start=1):
            step.step_id = index

    @staticmethod
    def name() -> str:
        return AgentName.OPENCODE.value

    def get_version_command(self) -> str | None:
        # ~/.nvm is absent on musl images, where Node comes from apk instead.
        return f"{self._NVM_SOURCE_COMMAND}; opencode --version"

    async def install(self, environment: BaseEnvironment) -> None:
        # Daytona can run task sandboxes with all runtime egress disabled. Allow
        # those tasks to bake the pinned CLI into their image during snapshot
        # construction, then avoid touching apt, curl, or npm inside the trial.
        # Only accept an exact pinned version: an unpinned agent still means
        # "install latest", and a mismatched image must not silently change the
        # executable selected by the job configuration.
        if self._version is not None:
            installed = await environment.exec(command=self.get_version_command())
            if (
                installed.return_code == 0
                and self.parse_version(installed.stdout) == self._version
            ):
                return

        await self.exec_as_root(
            environment,
            command=(
                "if [ -f /etc/alpine-release ]; then "
                "apk add --no-cache curl bash coreutils nodejs npm; "
                "else apt-get update && apt-get install -y curl bash coreutils; fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        # RL literal bridge: the per-trial correlation-id header (x-ot-trial-id in
        # provider options.headers) is confirmed to forward through
        # @ai-sdk/openai-compatible on opencode-ai 1.18.2. Header forwarding is
        # version-sensitive (older 1.0.x dropped provider options — opencode #5674),
        # so RL runs that rely on the literal bridge MUST pin ``version: "1.18.2"``
        # in the agent config (self._version) rather than drifting on ``@latest``.
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then "
                "node --version && npm --version; "
                "else "
                "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
                'export NVM_DIR="$HOME/.nvm" && '
                '\\. "$NVM_DIR/nvm.sh" || true && '
                "command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
                "nvm install 22 && npm -v; fi && "
                f"npm i -g opencode-ai{version_spec} && "
                "opencode --version"
            ),
        )

    @staticmethod
    def _millis_to_iso(timestamp_ms: int | float | None) -> str | None:
        """Convert a millisecond Unix timestamp to ISO 8601 string."""
        if timestamp_ms is None:
            return None
        try:
            return datetime.fromtimestamp(
                timestamp_ms / 1000, tz=timezone.utc
            ).isoformat()
        except (OSError, ValueError, OverflowError):
            return None

    @staticmethod
    def _parse_event_text(output: str) -> list[dict[str, Any]]:
        """Parse OpenCode's JSON-lines event stream."""
        events: list[dict[str, Any]] = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def _parse_stdout(self) -> list[dict[str, Any]]:
        """Read and parse JSON lines from the opencode stdout file."""
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if not output_path.exists():
            return []
        return self._parse_event_text(output_path.read_text())

    def _error_messages(self) -> list[str]:
        """Return messages from OpenCode error events in stdout."""
        messages: list[str] = []
        for event in self._parse_stdout():
            if event.get("type") != "error":
                continue
            error = event.get("error")
            if isinstance(error, dict):
                data = error.get("data")
                message = data.get("message") if isinstance(data, dict) else None
                messages.append(str(message or error.get("name") or error))
            else:
                messages.append(str(error))
        return messages

    # Substrings that identify a served-context-window overflow in opencode's
    # JSON error stream. When a prompt exceeds the served window the upstream
    # vLLM / OpenAI endpoint returns HTTP 400 ("maximum context length is N
    # tokens ... total ...") and opencode surfaces it as a ``ContextOverflowError``
    # error event (``error.name``) whose ``data.message`` / ``data.responseBody``
    # carry the upstream text. Matched case-insensitively.
    _CONTEXT_OVERFLOW_MARKERS = (
        "contextoverflow",  # opencode error.name == "ContextOverflowError"
        "maximum context length",  # upstream 400 BadRequestError message
        "context_length_exceeded",  # OpenAI-style error code
        "reduce the length of the messages",  # message tail (defensive)
    )

    _AUTHENTICATION_MARKERS = (
        "authenticationerror",  # OpenAI client's typed 401 error
        "authentication error",  # Provider text without a typed error name
    )

    def _matching_error_message(
        self, markers: tuple[str, ...], *, extra_output: str | None = None
    ) -> str | None:
        """Return detail from the first OpenCode error event matching *markers*."""
        events = self._parse_stdout()
        if extra_output:
            events.extend(self._parse_event_text(extra_output))
        for event in events:
            if event.get("type") != "error":
                continue
            error = event.get("error")
            name = ""
            blob = ""
            if isinstance(error, dict):
                name = str(error.get("name") or "")
                data = error.get("data")
                if isinstance(data, dict):
                    blob = (
                        f"{data.get('message') or ''} {data.get('responseBody') or ''}"
                    )
            else:
                blob = str(error)
            haystack = f"{name} {blob}".lower()
            if any(marker in haystack for marker in markers):
                detail = f"{name}: {blob}".strip(": ").strip()
                return (detail or "model endpoint error")[:500]
        return None

    def _context_overflow_message(
        self, *, extra_output: str | None = None
    ) -> str | None:
        """Return the overflow text if opencode signalled a served-context
        overflow, else ``None``.

        A served-window overflow is a *clean, classified* terminal condition,
        not a generic agent crash: with summarization/compaction disabled the
        agent cannot recover, so the trial should end gracefully (the verifier
        still scores whatever was produced) via
        :class:`~harbor.llms.base.ContextLengthExceededError` rather than a
        ``NonZeroAgentExitCodeError``. (When summarization is *enabled* opencode
        compacts and continues, so no error event is emitted and this returns
        ``None``.)
        """
        return self._matching_error_message(
            self._CONTEXT_OVERFLOW_MARKERS, extra_output=extra_output
        )

    def _authentication_failure_message(self) -> str | None:
        """Return model-endpoint authentication detail from OpenCode stdout."""
        return self._matching_error_message(self._AUTHENTICATION_MARKERS)

    def _convert_events_to_trajectory(
        self, events: list[dict[str, Any]]
    ) -> Trajectory | None:
        """Convert opencode stdout JSON events into an ATIF trajectory.

        Events are grouped into agent steps by ``step_start`` / ``step_finish``
        boundaries. Each completed group becomes one ATIF Step with
        source="agent". A final partial group is also retained when OpenCode is
        interrupted before emitting ``step_finish``. The leading user step comes
        from OpenCode's user event when available, or from the instruction passed
        to ``run()`` when the event stream omits it.
        """
        if not events:
            return None

        session_id: str | None = None
        for event in events:
            sid = event.get("sessionID")
            if sid:
                session_id = sid
                break

        # Group events into turns delimited by step_start / step_finish
        turns: list[dict[str, Any]] = []
        current_turn: dict[str, Any] | None = None
        user_message: str | None = None
        user_timestamp: int | None = None

        for event in events:
            etype = event.get("type")

            if etype == "user":
                if user_message is None:
                    user_message = self._user_event_text(event)
                    user_timestamp = event.get("timestamp")
                continue

            if etype == "step_start":
                if current_turn is not None and current_turn["parts"]:
                    turns.append(current_turn)
                current_turn = {
                    "parts": [],
                    "finish": None,
                    "timestamp": event.get("timestamp"),
                }
                continue

            if etype == "step_finish":
                if current_turn is not None:
                    current_turn["finish"] = event.get("part", {})
                    turns.append(current_turn)
                    current_turn = None
                continue

            if current_turn is not None and etype in ("text", "reasoning", "tool_use"):
                current_turn["parts"].append(event.get("part", {}))

        if current_turn is not None and current_turn["parts"]:
            turns.append(current_turn)

        steps: list[Step] = []
        step_id = 1
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read = 0

        for turn in turns:
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls_list: list[ToolCall] = []
            observation_results: list[ObservationResult] = []
            timestamp = self._millis_to_iso(turn.get("timestamp"))

            for part in turn["parts"]:
                ptype = part.get("type")

                if ptype == "text":
                    text = part.get("text", "")
                    if text:
                        text_parts.append(text)

                elif ptype == "reasoning":
                    reasoning = part.get("text", "")
                    if reasoning:
                        reasoning_parts.append(reasoning)

                elif ptype == "tool":
                    state = part.get("state", {})
                    tool_name = part.get("tool", "")
                    tool_input = state.get("input", {})
                    tool_output = state.get("output")
                    call_id = part.get("callID", part.get("id", ""))

                    if not isinstance(tool_input, dict):
                        tool_input = {"value": tool_input} if tool_input else {}

                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=tool_name,
                            arguments=tool_input,
                        )
                    )

                    if tool_output is not None:
                        observation_results.append(
                            ObservationResult(
                                source_call_id=call_id or None,
                                content=str(tool_output),
                            )
                        )

            # Extract metrics from step_finish
            finish = turn.get("finish") or {}
            tokens = finish.get("tokens", {})
            cost = finish.get("cost", 0) or 0
            input_tok = tokens.get("input", 0) or 0
            output_tok = tokens.get("output", 0) or 0
            reasoning_tok = tokens.get("reasoning", 0) or 0
            cache = tokens.get("cache", {})
            cache_read = cache.get("read", 0) or 0
            cache_write = cache.get("write", 0) or 0

            # Defensive literal-token capture: OpenCode's JSON stream may
            # (now or in future) carry token IDs / logprobs directly on the
            # step_finish part. Read them if present; they are merged with any
            # proxy-captured literal data in populate_context_post_run.
            stream_prompt_ids = finish.get("prompt_token_ids")
            stream_completion_ids = finish.get("completion_token_ids")
            stream_logprobs = finish.get("logprobs")

            total_cost += cost
            total_input_tokens += input_tok + cache_read
            total_output_tokens += output_tok
            total_cache_read += cache_read

            metrics: Metrics | None = None
            has_token_usage = bool(input_tok or output_tok or cache_read)
            has_literal = (
                stream_prompt_ids is not None
                or stream_completion_ids is not None
                or stream_logprobs is not None
            )
            if has_token_usage or has_literal:
                metrics = Metrics(
                    prompt_tokens=(input_tok + cache_read) or None,
                    completion_tokens=output_tok or None,
                    cached_tokens=cache_read if cache_read else None,
                    cost_usd=cost if cost else None,
                    prompt_token_ids=stream_prompt_ids,
                    completion_token_ids=stream_completion_ids,
                    logprobs=stream_logprobs,
                    extra={
                        k: v
                        for k, v in {
                            "reasoning_tokens": reasoning_tok,
                            "cache_write_tokens": cache_write,
                        }.items()
                        if v
                    }
                    or None,
                )

            message_text = "\n".join(text_parts) if text_parts else ""
            observation = (
                Observation(results=observation_results)
                if observation_results
                else None
            )

            step_kwargs: dict[str, Any] = {
                "step_id": step_id,
                "timestamp": timestamp,
                "source": "agent",
                "message": message_text or "(tool use)",
                "model_name": self.model_alias,
            }
            if reasoning_parts:
                step_kwargs["reasoning_content"] = "\n\n".join(reasoning_parts)
            if tool_calls_list:
                step_kwargs["tool_calls"] = tool_calls_list
            if observation:
                step_kwargs["observation"] = observation
            if metrics:
                step_kwargs["metrics"] = metrics

            steps.append(Step(**step_kwargs))
            step_id += 1

        if not steps:
            return None

        self._prepend_user_step(steps, user_message, user_timestamp)

        final_metrics = FinalMetrics(
            total_prompt_tokens=total_input_tokens or None,
            total_completion_tokens=total_output_tokens or None,
            total_cached_tokens=total_cache_read or None,
            total_cost_usd=total_cost if total_cost else None,
            total_steps=len(steps),
        )

        return Trajectory(
            schema_version="ATIF-v1.6",
            session_id=session_id or "unknown",
            agent=Agent(
                name="opencode",
                version=self.version() or "unknown",
                model_name=self.model_alias,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

    def _parse_literal_proxy_log(self) -> list[dict[str, Any]] | None:
        """Read literal token pairs captured by a RecordProxy.

        When OpenCode is run behind a :class:`~harbor.literal.proxy.RecordProxy`,
        each forwarded completion is appended (one JSON object per line) to
        ``literal.jsonl`` in the agent logs dir. Each line's ``literal`` field
        holds ``{prompt_token_ids, completion_token_ids, logprobs}``.

        Returns the list of literal dicts (one per assistant turn, in call
        order), or ``None`` if no proxy log is present.
        """
        pairs = read_literal_pairs(self.logs_dir / LITERAL_LOG_FILENAME)
        return pairs if pairs else None

    def _merge_literal_into_trajectory(
        self,
        trajectory: Trajectory,
        literal_pairs: list[dict[str, Any]],
    ) -> None:
        """Merge proxy-captured literal token data into trajectory steps in place.

        The proxy log and the trajectory share turn ordering (one entry per
        assistant turn), so pairs are merged by index. Agent steps without a
        matching pair are left untouched. A count mismatch logs a warning but
        still merges as many pairs as possible (defensive against partial logs
        from interrupted runs).
        """
        agent_steps = [s for s in trajectory.steps if s.source == "agent"]
        if len(literal_pairs) != len(agent_steps):
            self.logger.debug(
                f"Literal proxy log has {len(literal_pairs)} pairs for "
                f"{len(agent_steps)} agent steps; merging first "
                f"{min(len(literal_pairs), len(agent_steps))}."
            )
        for step, pair in zip(agent_steps, literal_pairs, strict=False):
            prompt_ids = pair.get("prompt_token_ids")
            completion_ids = pair.get("completion_token_ids")
            logprobs = pair.get("logprobs")
            if prompt_ids is None and completion_ids is None and logprobs is None:
                continue
            if step.metrics is None:
                step.metrics = Metrics()
            # Prefer proxy-captured (ground-truth) values over any stream values.
            if prompt_ids is not None:
                step.metrics.prompt_token_ids = prompt_ids
            if completion_ids is not None:
                step.metrics.completion_token_ids = completion_ids
            if logprobs is not None:
                step.metrics.logprobs = logprobs

    def _build_rollout_details_from_literal(
        self,
        literal_pairs: list[dict[str, Any]],
    ) -> list[RolloutDetail]:
        """Build a single :class:`RolloutDetail` from proxy-captured literal pairs.

        Thin wrapper over
        :func:`harbor.literal.rollout_build.build_rollout_details_from_pairs`
        (shared with the RL-loop per-trial correlation glue). Mirrors
        ``Chat.rollout_details`` with the ``8737426c`` per-turn length-parity guard
        (a mismatched/absent-logprob turn records ``[]`` so TIS flags it rather
        than mis-pairing). See that function for the full contract.
        """
        return build_rollout_details_from_pairs(literal_pairs)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Parse opencode stdout and convert to ATIF trajectory.

        If a ``literal.jsonl`` proxy log is present in the logs dir, its
        per-turn token IDs / logprobs are merged into the trajectory's agent
        steps (by index) before the trajectory is written to disk. When
        ``collect_rollout_details`` is set, the same per-turn data is also
        surfaced into ``context.rollout_details`` for the RL rollout loop.
        """
        # Surface the per-trial correlation id BEFORE any early return so the
        # RL-loop glue can correlate the shared proxy log to this trial even if the
        # trajectory failed to parse (an errored trial still emitted requests).
        if self.rollout_correlation_id:
            context.metadata = {
                **(context.metadata or {}),
                "rollout_correlation_id": self.rollout_correlation_id,
            }

        events = self._parse_stdout()
        if not events:
            return

        try:
            trajectory = self._convert_events_to_trajectory(events)
        except Exception:
            self.logger.exception("Failed to convert opencode events to trajectory")
            return

        if not trajectory:
            return

        # Merge ground-truth literal token data captured by the recording proxy.
        literal_pairs = self._parse_literal_proxy_log()
        if literal_pairs:
            self._merge_literal_into_trajectory(trajectory, literal_pairs)
            # RL bridge: also surface the literal per-turn token_ids/logprobs into
            # context.rollout_details (what MarinSkyRL consumes). Flag-gated so
            # non-RL / datagen paths stay byte-identical (rollout_details None).
            if self._collect_rollout_details:
                rollout_details = self._build_rollout_details_from_literal(
                    literal_pairs
                )
                if rollout_details:
                    context.rollout_details = rollout_details

        trajectory_path = self.logs_dir / "trajectory.json"
        # Serialization + (possibly remote fsspec) write run on the background
        # artifact writer, off the shared event loop; failures are logged there.
        get_artifact_writer().submit_optional(
            trajectory_path,
            lambda: format_trajectory_json(trajectory.to_json_dict()),
            description="opencode trajectory.json",
        )

        if trajectory.final_metrics:
            fm = trajectory.final_metrics
            context.cost_usd = fm.total_cost_usd
            context.n_input_tokens = fm.total_prompt_tokens or 0
            context.n_output_tokens = fm.total_completion_tokens or 0
            context.n_cache_tokens = fm.total_cached_tokens or 0

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills to OpenCode's skills directory."""
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p ~/.config/opencode/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            f"~/.config/opencode/skills/ 2>/dev/null || true"
        )

    def _resolve_model_limit(self) -> dict[str, int]:
        """Return a bounded ``{"context", "output"}`` for the opencode model config.

        opencode writes the per-model ``max_tokens`` from the model's ``limit.output``
        and defaults it to ~32000 for a custom (openai-compatible) provider when unset
        (opencode #1735). Against a served window of N tokens that leaves only
        ``N - 32000`` for the prompt, so every real prompt overflows the server with a
        400 and the whole run fails. This computes an ``output`` (the per-request
        ``max_tokens`` cap) that is RESERVED inside the window, so
        ``prompt + output <= window`` always holds:

            output  = model_info.max_output_tokens or _DEFAULT_MAX_OUTPUT
            window  = model_info.max_input_tokens  or _DEFAULT_MODEL_WINDOW
            context = window - output            # opencode keeps the prompt <= context

        Values come from ``model_info`` (the served model's ``max_input_tokens`` /
        ``max_output_tokens``, plumbed as the ctor kwarg the RL generator and the eval
        path pass) when present, else the safe class defaults — so the returned limit is
        ALWAYS bounded, model-agnostic, and never over-requests, even when the caller
        plumbs nothing. Non-numeric / non-positive inputs fall back to the defaults.
        """
        info = self._model_info or {}

        def _as_positive_int(value: Any) -> int | None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed > 0 else None

        output = (
            _as_positive_int(info.get("max_output_tokens")) or self._DEFAULT_MAX_OUTPUT
        )
        window = (
            _as_positive_int(info.get("max_input_tokens")) or self._DEFAULT_MODEL_WINDOW
        )
        # Reserve the output budget inside the window so that the server-side check
        # ``prompt_tokens + max_tokens <= max_model_len`` can never be violated. Clamp
        # output below the window first so ``context`` stays >= 1 for tiny windows.
        output = min(output, max(1, window - 1))
        # Reserve a small SAFETY MARGIN so ``context + output`` stays strictly BELOW the
        # window. opencode compacts the prompt to ~``limit.context``, but the rendered
        # prompt vLLM actually counts can exceed ``context`` by chat-template / tokenizer
        # overhead — with ``context = window - output`` (zero slack) even a 1-token
        # overshoot 400s ("maximum context length ... total 32769" at prompt 16385 vs
        # context 16384, observed on the 32768/16384 eval profile). The margin absorbs
        # that overhead so ``prompt + output <= window`` holds in practice.
        margin = min(1024, max(0, window - output - 1))
        context = max(1, window - output - margin)
        return {"context": context, "output": output}

    def _build_register_config_command(self) -> str | None:
        """Return a shell command that writes the opencode config to ~/.config/opencode/opencode.json.

        The config may include MCP server definitions and/or a provider model
        registration so opencode recognises models not in its built-in registry.
        """
        config: dict[str, Any] = {}

        if self.mcp_servers:
            mcp: dict[str, dict[str, Any]] = {}
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    cmd_list = [server.command] + server.args if server.command else []
                    mcp[server.name] = {"type": "local", "command": cmd_list}
                else:  # sse or streamable-http
                    # Harbor cannot supply OAuth credentials for an MCP server.
                    # Disable OpenCode's browser-based OAuth auto-detection for
                    # remote servers, which benchmark runs cannot complete.
                    mcp[server.name] = {
                        "type": "remote",
                        "url": server.url,
                        "oauth": False,
                    }
            config["mcp"] = mcp

        if self.model_alias and "/" in self.model_alias:
            provider, model_id = self.model_alias.split("/", 1)
            # Bound the model's token ``limit`` so opencode never requests a
            # ``max_tokens`` that overflows the served window (opencode #1735
            # defaults it to ~32000 for a custom provider → 100% 400-overflow at
            # every real prompt). Applies to BOTH the RL and eval paths; a
            # job-level ``opencode_config`` may still override via deep-merge below.
            provider_config: dict[str, Any] = {
                "models": {model_id: {"limit": self._resolve_model_limit()}}
            }
            # The self-hosted vLLM baseURL comes from HARBOR_MODEL_ENDPOINT — the
            # harbor-specific var the RL runner publishes for the co-located served
            # model. OPENAI_BASE_URL is a back-compat FALLBACK ONLY: it is reserved for
            # genuine OpenAI traffic (the LLM-judge verifiers on the worker read it), so
            # the RL runner must NOT overload it with a vLLM endpoint — that would
            # silently misroute every judge call to vLLM. Prefer the harbor var here so
            # the runner never has to touch OPENAI_BASE_URL.
            # PRIMARY source is self._api_base (the ctor kwarg the RL generator passes) —
            # it travels with the trial dispatch as data, so it works where the process's
            # env does not carry HARBOR_MODEL_ENDPOINT (the fullgate1-3 undefined-URL bug).
            # The env vars stay as fallbacks for the datagen/non-RL paths that set them.
            if self._api_base:
                base_url = self._api_base
            elif provider == "anthropic":
                base_url = os.environ.get("ANTHROPIC_BASE_URL")
            elif provider == "google":
                base_url = os.environ.get("GOOGLE_BASE_URL")
            else:
                base_url = (
                    self._api_base
                    or os.environ.get("HARBOR_MODEL_ENDPOINT")
                    or os.environ.get("OPENAI_BASE_URL")
                )
            # Prefer the inert OPENCODE_DUMMY_KEY placeholder so the real host
            # OPENAI_API_KEY (needed by the LLM-judge verifiers on the worker) never
            # enters the sandbox. The capability token rides in the URL path, so the
            # header value is inert and any non-empty string works.
            api_key = os.environ.get("OPENCODE_DUMMY_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
            if provider in ("vllm", "hosted_vllm"):
                # Route a self-hosted vLLM through ai-sdk's openai-compatible
                # provider, NOT the built-in ``openai`` provider. Both the bare
                # ``vllm`` id (datagen) and the ``hosted_vllm`` id the RL rollout
                # generator forces (terminal_bench_generator: model_name=
                # ``hosted_vllm/<alias>``) name the same self-hosted endpoint, so
                # both take this branch — otherwise the RL literal bridge's
                # baseURL/apiKey/x-ot-trial-id header never gets written. The ``openai``
                # provider id makes opencode call the Responses API
                # (POST /v1/responses), which vLLM 404s ("AI_APICallError: Not
                # Found"). A non-``openai`` id backed by "@ai-sdk/openai-compatible"
                # routes to POST /v1/chat/completions, which vLLM serves.
                # opencode reads baseURL/apiKey from provider.options.
                provider_config["npm"] = "@ai-sdk/openai-compatible"
                options = provider_config.setdefault("options", {})
                if base_url:
                    options["baseURL"] = base_url
                if api_key:
                    options["apiKey"] = api_key
                # RL literal bridge: stamp the per-trial correlation id into every
                # request so the shared RecordProxy can attribute each interleaved
                # log entry to this trial. @ai-sdk/openai-compatible forwards
                # provider ``options.headers`` on every /v1/chat/completions call
                # (confirmed opencode-ai 1.18.2; provider-ROOT headers are NOT
                # forwarded). No id (flag off) → no header (byte-identical).
                if self.rollout_correlation_id:
                    headers = options.setdefault("headers", {})
                    headers[CORRELATION_ID_HEADER] = self.rollout_correlation_id
            elif base_url and provider in {"anthropic", "google", "openai"}:
                # opencode reads baseURL from provider.options, not the provider root.
                # See: https://github.com/anomalyco/opencode config.ts ProviderConfig schema.
                provider_config.setdefault("options", {})["baseURL"] = base_url
            config["provider"] = {provider: provider_config}

        # Graceful iteration cap on the default ``build`` agent — the agent
        # opencode uses for ``opencode run`` when no --agent is given. ``steps``
        # bounds the agentic loop server-agnostically (opencode has no ``run``
        # CLI flag for it). Layered into ``config`` (NOT the final dict) so a
        # job-level ``opencode_config`` may still override agent.build.steps via
        # the deep-merge below. Omitted when unset → byte-identical to before.
        if self._max_turns is not None:
            build_agent = config.setdefault("agent", {}).setdefault("build", {})
            build_agent["steps"] = self._max_turns

        # Layer: defaults → auto-generated → job-level overrides.
        # Deep-merge preserves sibling keys within nested dicts (e.g. provider, experimental).
        config = self._deep_merge(copy.deepcopy(self._DEFAULT_CONFIG), config)
        config = self._deep_merge(config, self._opencode_config)

        if not config:
            return None

        config_json = json.dumps(config, indent=2)
        escaped = shlex.quote(config_json)
        return f"mkdir -p ~/.config/opencode && echo {escaped} > ~/.config/opencode/opencode.json"

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._instruction = instruction
        escaped_instruction = shlex.quote(instruction)

        if not self.model_alias or "/" not in self.model_alias:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, _ = self.model_alias.split("/", 1)

        env = {}
        keys = []

        # Get provider environment variables
        if provider == "amazon-bedrock":
            keys.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"])
        elif provider == "anthropic":
            keys.extend(["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"])
        elif provider == "azure":
            keys.extend(["AZURE_RESOURCE_NAME", "AZURE_API_KEY"])
        elif provider == "deepseek":
            keys.append("DEEPSEEK_API_KEY")
        elif provider == "github-copilot":
            keys.append("GITHUB_TOKEN")
        elif provider == "google":
            keys.extend(
                [
                    "GEMINI_API_KEY",
                    "GOOGLE_GENERATIVE_AI_API_KEY",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_CLOUD_PROJECT",
                    "GOOGLE_CLOUD_LOCATION",
                    "GOOGLE_GENAI_USE_VERTEXAI",
                    "GOOGLE_API_KEY",
                    "GOOGLE_BASE_URL",
                ]
            )
        elif provider == "groq":
            keys.append("GROQ_API_KEY")
        elif provider == "huggingface":
            keys.append("HF_TOKEN")
        elif provider == "llama":
            keys.append("LLAMA_API_KEY")
        elif provider == "mistral":
            keys.append("MISTRAL_API_KEY")
        elif provider in ("openai", "vllm", "hosted_vllm"):
            # ``vllm`` / ``hosted_vllm`` are self-hosted OpenAI-compatible endpoints
            # registered via the "@ai-sdk/openai-compatible" provider (see
            # _build_register_config_command); they read the same env as ``openai``.
            keys.append("OPENAI_API_KEY")
            keys.append("OPENAI_BASE_URL")
            # Symmetry with the baseURL source in _build_register_config_command
            # (HARBOR_MODEL_ENDPOINT preferred over OPENAI_BASE_URL). Non-load-bearing:
            # the baseURL is already baked into opencode.json, so the sandbox does not
            # strictly need the env var — forwarded only so both vars are visible there.
            keys.append("HARBOR_MODEL_ENDPOINT")
        elif provider == "opencode":
            keys.append("OPENCODE_API_KEY")
        elif provider == "xai":
            keys.append("XAI_API_KEY")
        elif provider == "openrouter":
            keys.append("OPENROUTER_API_KEY")

        for key in keys:
            if key in os.environ:
                env[key] = os.environ[key]

        # The OpenAI-compatible sandbox api key must be the inert OPENCODE_DUMMY_KEY
        # placeholder, NOT the real host OPENAI_API_KEY — that stays on the worker for
        # the LLM-judge verifiers, and must not leak into the sandbox. The capability
        # token is in the URL path, so the header value is inert.
        if provider in ("openai", "vllm", "hosted_vllm"):
            dummy_key = os.environ.get("OPENCODE_DUMMY_KEY")
            if dummy_key:
                env["OPENAI_API_KEY"] = dummy_key

        # Enable fake VCS for OpenCode
        env["OPENCODE_FAKE_VCS"] = "git"
        # Disable the file-finding feature that triggers redundant harness reads.
        env["OPENCODE_DISABLE_FFF"] = "1"

        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command, env=env)

        mcp_command = self._build_register_config_command()
        if mcp_command:
            await self.exec_as_agent(environment, command=mcp_command, env=env)

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        timeout_prefix = self._server_timeout_prefix(environment)
        try:
            await self.exec_as_agent(
                environment,
                # Note that the --thinking flag just means thinking blocks will be included in the json formatted output
                command=(
                    f"{self._NVM_SOURCE_COMMAND}; "
                    f"{timeout_prefix}opencode --model={self.model_alias} run --format=json {cli_flags_arg}--thinking --auto -- {escaped_instruction} "
                    f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt"
                ),
                env=env,
            )
        except NonZeroAgentExitCodeError as exc:
            if timeout_prefix and self._is_timeout_exit(exc):
                raise AgentTimeoutError(
                    "opencode exceeded its agent-phase deadline and was terminated server-side"
                ) from exc
            # A served-context overflow surfaces as opencode exit-1 with a
            # ContextOverflowError event already teed into opencode.txt before
            # the CLI exits. Reclassify it as the clean, passthrough-scored
            # ContextLengthExceededError so the trial ends gracefully instead of
            # tanking with a generic non-zero agent exit. Any other non-zero
            # exit (incl. a signal kill -> AgentKilledBySignalError) propagates
            # unchanged.
            overflow = self._context_overflow_message(
                extra_output=exc.result.stdout if exc.result is not None else None
            )
            if overflow is not None:
                raise ContextLengthExceededError(
                    f"OpenCode exceeded the served context window: {overflow}"
                ) from exc
            authentication_failure = self._authentication_failure_message()
            if authentication_failure is not None:
                raise ModelAuthenticationError(
                    "OpenCode could not authenticate to its model endpoint "
                    f"(possibly an expired capability URL): {authentication_failure}"
                ) from exc
            raise

        if messages := self._error_messages():
            # opencode exited 0 but emitted an error event. A context-window
            # overflow is still the clean classified error; anything else keeps
            # the generic non-zero-exit signal.
            overflow = self._context_overflow_message()
            if overflow is not None:
                raise ContextLengthExceededError(
                    f"OpenCode exceeded the served context window: {overflow}"
                )
            authentication_failure = self._authentication_failure_message()
            if authentication_failure is not None:
                raise ModelAuthenticationError(
                    "OpenCode could not authenticate to its model endpoint "
                    f"(possibly an expired capability URL): {authentication_failure}"
                )
            raise NonZeroAgentExitCodeError(
                "OpenCode emitted error event(s): " + "; ".join(messages[:3])
            )

    def _server_timeout_prefix(self, environment: BaseEnvironment) -> str:
        """Build the shell ``timeout`` prefix for the opencode run, or ``""``.

        Reads the CURRENT agent-phase deadline the trial records on the
        environment (``agent_timeout_sec`` — per-step in multi-step trials). The
        server-side TERM fires ``_SERVER_TIMEOUT_GRACE_SEC`` before that deadline
        and the SIGKILL ``_SERVER_TIMEOUT_KILL_AFTER_SEC`` after the TERM — both
        strictly before the trial-level wait_for — so opencode is reaped
        server-side first. Returns ``""`` when no deadline is set (unbounded, as
        before), keeping the emitted command byte-identical.
        """
        agent_timeout = getattr(environment, "agent_timeout_sec", None)
        # Only a real, positive number is a deadline. The trial always sets a
        # ``float | None``; the isinstance guard also keeps this a no-op for a
        # test double / any env that doesn't carry a numeric deadline.
        if not isinstance(agent_timeout, (int, float)) or agent_timeout <= 0:
            return ""
        term_sec = max(1, int(agent_timeout) - self._SERVER_TIMEOUT_GRACE_SEC)
        return f"timeout -k {self._SERVER_TIMEOUT_KILL_AFTER_SEC} -s TERM {term_sec} "

    @staticmethod
    def _is_timeout_exit(exc: NonZeroAgentExitCodeError) -> bool:
        """True if the agent-exec failure was a shell ``timeout`` kill.

        ``BaseInstalledAgent._exec`` formats non-zero failures as
        ``Command failed (exit <code>): ...`` and raises rather than returning
        the ExecResult, so the exit code is recovered from the message.
        """
        match = re.search(r"exit (\d+)", str(exc))
        if not match:
            return False
        return int(match.group(1)) == OpenCode._TIMEOUT_EXIT_CODE
