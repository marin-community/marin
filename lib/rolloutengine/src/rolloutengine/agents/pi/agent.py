# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Pi agent and its ``AgentHarness``.

Adapted from Harbor's ``Pi`` and ``BaseInstalledAgent`` (Apache-2.0); see ``rolloutengine.agents.pi``. Pi runs
inside the task machine, from the task image, in print mode with JSON output and without a session file. It
reaches the policy model through the model relay at ``http://127.0.0.1:8788/v1``, which serves each request
with the harness's ``complete``, so the rollout records every request, including Pi's compaction requests.
Its provider configuration, compaction settings and context-budget extension are Harbor's: before each request
the extension counts the prompt's tokens with the relay's ``/tokenize`` route, shortens the largest tool outputs
to fit the context window and lowers the request's output limit to the room that is left.

Pi's output streams to ``/logs/agent/pi.txt`` on the task machine. The rollout, not the log, records the turns.
"""

import asyncio
import json
import logging
import shlex
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from taskcompendium.models import ProviderRequirement, TaskSpec, TextMessage

from rolloutengine.agents.environment import ExecResult, MachineEnvironment
from rolloutengine.agents.model_relay import ModelRelay, PromptTokenizer, serve_model_relay, tokenize_url
from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.harness.external import AgentCompletion, AgentHarness, LengthStopPolicy, ReplyMatch
from rolloutengine.models import ContextRecovery, RolloutLimits

logger = logging.getLogger(__name__)

PI_HARNESS = "pi"
PROVIDER = "hosted_vllm"
"""Pi's provider name for the policy model, Harbor's name for a hosted vLLM endpoint."""
RELAY_PORT = 8788
"""The task machine's loopback port of the model relay: Harbor's candidate-model relay port."""
API_KEY = "EMPTY"
"""The credential that Pi sends. The relay ignores it; the policy model's credentials stay on the host."""
AGENT_LOGS_DIR = "/logs/agent"
INSTALLED_AGENT_DIR = "/installed-agent"
OUTPUT_FILENAME = "pi.txt"
API_BASE_ENV = "HARBOR_PI_API_BASE"
API_KEY_ENV = "HARBOR_PI_API_KEY"
STAGING_DIR_TEMPLATE = "/tmp/harbor-pi.XXXXXXXX"
ENDPOINT_ENV_FILENAME = "harbor-pi-endpoint.env"
CONTEXT_BUDGET_EXTENSION_FILENAME = "harbor-pi-context-budget.js"
MCP_INTERFACE = "mcp:v1"
"""The action interface of a task's MCP server, which Pi's MCP extension connects to."""
MCP_TRANSPORTS = frozenset({"sse", "streamable-http"})
MCP_EXTENSION_PATH = "/opt/harbor-pi/node_modules/pi-mcp-extension/src/index.ts"
MCP_CONFIGURE_SCRIPT = Path(__file__).with_name("pi_configure_mcp.cjs")
MCP_CONFIGURE_PATH = f"{INSTALLED_AGENT_DIR}/pi_configure_mcp.cjs"
PROVIDER_CONFIG_RELATIVE_PATH = ".pi/agent/models.json"
SETTINGS_CONFIG_RELATIVE_PATH = ".pi/agent/settings.json"
AUTHENTICATION_FAILURE_MARKER = "HARBOR_PI_MODEL_AUTHENTICATION_ERROR"
CONTEXT_LENGTH_FAILURE_MARKER = "HARBOR_PI_CONTEXT_LENGTH_EXCEEDED"
OUTPUT_PREVIEW_CHARACTERS = 1000
VALIDATE_OUTPUT_SCRIPT = (
    """
const readline = require("readline");
let lastAssistantStopReason;
let lastAssistantErrorMessage;
const lines = readline.createInterface({ input: process.stdin });
lines.on("line", (line) => {
  process.stdout.write(line + "\\n");
  try {
    const event = JSON.parse(line);
    const message = event.type === "message_end" ? event.message : undefined;
    if (message?.role === "assistant") {
      lastAssistantStopReason = message.stopReason;
      lastAssistantErrorMessage = message.errorMessage;
    }
  } catch {
    // Pi can mix diagnostics into its JSON stream. Preserve them without
    // treating unstructured text as a successful or failed assistant turn.
  }
});
lines.on("close", () => {
  if (lastAssistantStopReason !== "error") return;
  if (/authentication required|authenticationerror/i.test(lastAssistantErrorMessage ?? "")) {
    process.stderr.write("__AUTHENTICATION_FAILURE_MARKER__\\n");
  } else if (/maximum context length|context_length_exceeded/i.test(lastAssistantErrorMessage ?? "")) {
    process.stderr.write("__CONTEXT_LENGTH_FAILURE_MARKER__\\n");
  }
  process.exitCode = 1;
});
""".replace(
        "__AUTHENTICATION_FAILURE_MARKER__", AUTHENTICATION_FAILURE_MARKER
    )
    .replace("__CONTEXT_LENGTH_FAILURE_MARKER__", CONTEXT_LENGTH_FAILURE_MARKER)
    .strip()
)
REGISTER_PROVIDER_SCRIPT = """
const fs = require("fs");
const path = require("path");
const config = JSON.parse(process.argv[1]);
const provider = process.argv[2];
const baseUrlEnv = process.argv[3];
const configRelativePath = process.argv[4];
const baseUrl = process.env[baseUrlEnv];
if (!baseUrl) throw new Error(`${baseUrlEnv} is required`);
config.providers[provider].baseUrl = baseUrl;
const configPath = path.join(process.env.HOME, configRelativePath);
fs.mkdirSync(path.dirname(configPath), { recursive: true });
fs.writeFileSync(configPath, JSON.stringify(config));
""".strip()
REGISTER_COMPACTION_SCRIPT = """
const fs = require("fs");
const path = require("path");
const compaction = JSON.parse(process.argv[1]);
const configPath = path.join(process.env.HOME, process.argv[2]);
let settings = {};
if (fs.existsSync(configPath)) settings = JSON.parse(fs.readFileSync(configPath, "utf8"));
settings.compaction = { ...(settings.compaction ?? {}), ...compaction };
fs.mkdirSync(path.dirname(configPath), { recursive: true });
fs.writeFileSync(configPath, JSON.stringify(settings));
""".strip()
CONTEXT_BUDGET_EXTENSION = """
const tokenizeUrl = __TOKENIZE_URL__;
const contextWindow = __CONTEXT_WINDOW__;

const tokenizationKeys = [
  "chat_template",
  "chat_template_kwargs",
  "documents",
  "mm_processor_kwargs",
  "tools",
];

function tokenizationPayload(payload) {
  const request = {
    model: payload.model,
    messages: payload.messages,
    add_generation_prompt: true,
  };
  for (const key of tokenizationKeys) {
    if (payload[key] !== undefined) request[key] = payload[key];
  }
  return request;
}

async function countPromptTokens(payload) {
  const response = await fetch(tokenizeUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(tokenizationPayload(payload)),
  });
  if (!response.ok) {
    throw new Error(`vLLM /tokenize returned HTTP ${response.status}: ${await response.text()}`);
  }
  const body = await response.json();
  if (!Array.isArray(body.tokens) || !body.tokens.every(Number.isInteger)) {
    throw new Error("vLLM /tokenize did not return an integer token list");
  }
  return body.tokens.length;
}

function toolTextSlots(messages) {
  const slots = [];
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
    const message = messages[messageIndex];
    if (message?.role !== "tool") continue;
    if (typeof message.content === "string") {
      slots.push({ messageIndex, blockIndex: undefined, text: message.content });
      continue;
    }
    if (!Array.isArray(message.content)) continue;
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex];
      if (block?.type === "text" && typeof block.text === "string") {
        slots.push({ messageIndex, blockIndex, text: block.text });
      }
    }
  }
  return slots.sort((left, right) => right.text.length - left.text.length);
}

function truncateToolText(text, retainedCharacters) {
  const characters = Array.from(text);
  if (retainedCharacters >= characters.length) return text;
  const prefixLength = Math.ceil(retainedCharacters / 2);
  const suffixLength = Math.floor(retainedCharacters / 2);
  const omitted = characters.length - retainedCharacters;
  const marker = `\n...[tool output truncated to fit model context; ${omitted} characters omitted]...\n`;
  return characters.slice(0, prefixLength).join("") + marker +
    characters.slice(characters.length - suffixLength).join("");
}

function replaceToolText(messages, slot, text) {
  const updated = structuredClone(messages);
  if (slot.blockIndex === undefined) {
    updated[slot.messageIndex].content = text;
  } else {
    updated[slot.messageIndex].content[slot.blockIndex].text = text;
  }
  return updated;
}

async function fitToolOutputs(payload, promptBudget, promptTokens) {
  let adjusted = payload;
  let adjustedTokens = promptTokens;
  for (const slot of toolTextSlots(payload.messages)) {
    if (adjustedTokens <= promptBudget) break;
    const characterCount = Array.from(slot.text).length;
    let low = 0;
    let high = characterCount;
    let bestMessages = replaceToolText(
      adjusted.messages,
      slot,
      truncateToolText(slot.text, 0),
    );
    let bestTokens = await countPromptTokens({ ...adjusted, messages: bestMessages });
    if (bestTokens > promptBudget) {
      adjusted = { ...adjusted, messages: bestMessages };
      adjustedTokens = bestTokens;
      continue;
    }
    while (low <= high) {
      const retained = Math.floor((low + high) / 2);
      const messages = replaceToolText(
        adjusted.messages,
        slot,
        truncateToolText(slot.text, retained),
      );
      const tokens = await countPromptTokens({ ...adjusted, messages });
      if (tokens <= promptBudget) {
        bestMessages = messages;
        bestTokens = tokens;
        low = retained + 1;
      } else {
        high = retained - 1;
      }
    }
    adjusted = { ...adjusted, messages: bestMessages };
    adjustedTokens = bestTokens;
  }
  return { payload: adjusted, promptTokens: adjustedTokens };
}

export default function (pi) {
  pi.on("before_provider_request", async (event) => {
    const payload = event.payload;
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      return payload;
    }

    const maxTokensField = Object.hasOwn(payload, "max_tokens")
      ? "max_tokens"
      : Object.hasOwn(payload, "max_completion_tokens")
        ? "max_completion_tokens"
        : undefined;
    if (!maxTokensField || !Array.isArray(payload.messages) || typeof payload.model !== "string") {
      return payload;
    }

    const configuredMaxTokens = payload[maxTokensField];
    if (!Number.isSafeInteger(configuredMaxTokens) || configuredMaxTokens <= 0) {
      throw new Error(`${maxTokensField} must be a positive integer`);
    }
    let promptTokens = await countPromptTokens(payload);
    const promptBudget = Math.max(1, contextWindow - configuredMaxTokens);
    let adjustedPayload = payload;
    if (promptTokens > promptBudget) {
      const fitted = await fitToolOutputs(payload, promptBudget, promptTokens);
      adjustedPayload = fitted.payload;
      promptTokens = fitted.promptTokens;
    }
    const availableOutputTokens = Math.max(1, contextWindow - promptTokens);
    return {
      ...adjustedPayload,
      [maxTokensField]: Math.min(configuredMaxTokens, availableOutputTokens),
    };
  });
}
""".strip()


class PiThinkingFormat(StrEnum):
    """How Pi turns thinking on and off: the ``enable_thinking`` chat-template argument of the served model."""

    CHAT_TEMPLATE = "chat-template"
    QWEN_CHAT_TEMPLATE = "qwen-chat-template"


class PiThinking(StrEnum):
    """Pi's ``--thinking`` level."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass(frozen=True)
class PiOptions:
    """Pi's settings, as Harbor's agent arguments name them.

    ``model`` is the served model name that Pi reports; the relay serves every request with the policy model.
    ``max_input_tokens`` is the context window and ``max_output_tokens`` each request's output limit, Harbor's
    ``model_info`` limits. Pi compacts its history when fewer than ``compaction_reserve_tokens`` tokens of the
    window are free, by default ``max_output_tokens``, and keeps half of the rest. ``version``, when set, must
    equal the version of the image's Pi. ``thinking`` sets Pi's ``--thinking`` level, and ``mcp_excluded_tools``
    hides tools of the task's MCP servers. ``setup_timeout`` bounds the version check, Harbor's agent setup.
    ``relay_port`` is the model relay's loopback port in the task machine.
    """

    model: str
    thinking_format: PiThinkingFormat
    max_input_tokens: int
    max_output_tokens: int
    compaction_reserve_tokens: int | None = None
    version: str | None = None
    thinking: PiThinking | None = None
    mcp_excluded_tools: tuple[str, ...] = ()
    setup_timeout: float | None = None
    relay_port: int = RELAY_PORT

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("Pi requires the served model name")
        if min(self.max_input_tokens, self.max_output_tokens, self.reserve_tokens) < 1:
            raise ValueError("Pi token limits must be positive")
        if self.max_output_tokens > self.max_input_tokens:
            raise ValueError("max_output_tokens must not exceed max_input_tokens")
        if self.reserve_tokens < self.max_output_tokens:
            raise ValueError("compaction_reserve_tokens must be at least max_output_tokens")
        if self.reserve_tokens >= self.max_input_tokens:
            raise ValueError("compaction_reserve_tokens must be less than max_input_tokens")

    @property
    def reserve_tokens(self) -> int:
        return self.max_output_tokens if self.compaction_reserve_tokens is None else self.compaction_reserve_tokens

    @property
    def keep_recent_tokens(self) -> int:
        return (self.max_input_tokens - self.reserve_tokens) // 2

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.relay_port}/v1"


class NonZeroAgentExitCodeError(RuntimeError):
    """A Pi command exited with a non-zero exit code."""

    def __init__(self, message: str, *, result: ExecResult):
        super().__init__(message)
        self.result = result


class ModelAuthenticationError(NonZeroAgentExitCodeError):
    """Pi could not authenticate to its model endpoint."""


class AgentKilledBySignalError(NonZeroAgentExitCodeError):
    """A Pi command exited with the shell's status for a signal, for example after an out-of-memory kill."""

    def __init__(self, message: str, *, signal_number: int, result: ExecResult):
        super().__init__(message, result=result)
        self.signal_number = signal_number


class ContextLengthExceededError(Exception):
    """Pi ended on a request that exceeded the model's context window."""


@dataclass(frozen=True)
class PiHarness:
    """A harness that runs Pi on the task machine, with the task's MCP servers.

    ``check`` accepts the task's ``mcp:v1`` tool providers, which Pi's MCP extension serves, and otherwise
    applies ``AgentHarness.check``. The task context must be one user instruction.
    """

    harness: AgentHarness

    @property
    def context_recovery(self) -> ContextRecovery:
        return self.harness.context_recovery

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        task_instruction(task)
        requirements = task.environment_requirements
        servers = mcp_servers(requirements.tool_providers)
        others = {name: provider for name, provider in requirements.tool_providers.items() if name not in servers}
        self.harness.check(
            task.model_copy(
                update={"environment_requirements": requirements.model_copy(update={"tool_providers": others})}
            ),
            limits,
        )

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        await self.harness.run(context, session)


def pi_harness(options: PiOptions, tokenize: PromptTokenizer) -> PiHarness:
    """A harness that runs Pi with ``options`` and counts its prompt tokens with ``tokenize``.

    Pi decides itself what to do after a reply cut by its output limit, so the harness never ends the attempt
    there: such a request starts a new conversation.
    """

    async def run(context: HarnessContext, complete: AgentCompletion) -> None:
        await run_pi(options, tokenize, context, complete)

    return PiHarness(AgentHarness(run, length_stop=LengthStopPolicy.NEW_CONVERSATION, reply_match=ReplyMatch.EXACT))


def task_instruction(task: TaskSpec) -> str:
    events = task.context.events
    if len(events) != 1 or not isinstance(events[0], TextMessage) or events[0].role != "user":
        raise ValueError("Pi requires one user instruction as the task context")
    return events[0].content


def mcp_servers(providers: Mapping[str, ProviderRequirement]) -> dict[str, dict[str, Any]]:
    """Pi's MCP extension configuration of each ``mcp:v1`` provider, by name.

    Raises:
        ValueError: A provider's initial state is not an HTTP transport and URL.
    """
    servers: dict[str, dict[str, Any]] = {}
    for name, provider in providers.items():
        if provider.action_interface != MCP_INTERFACE:
            continue
        state = provider.initial_state
        if (
            not isinstance(state, dict)
            or state.keys() != {"transport", "url"}
            or state["transport"] not in MCP_TRANSPORTS
            or not isinstance(state["url"], str)
        ):
            raise ValueError(f"MCP server {name!r} must name an SSE or streamable HTTP transport and its URL")
        servers[name] = {"transport": state["transport"], "lifecycle": "eager", "url": state["url"]}
    return servers


async def run_pi(
    options: PiOptions, tokenize: PromptTokenizer, context: HarnessContext, complete: AgentCompletion
) -> None:
    """Check the image's Pi, then run it on the task's instruction with the model relay.

    When setup has its own deadline, the turn budget starts as soon as setup completes.
    Pi's non-zero exit and its context-length failure end the agent normally, so the engine grades Pi's work, as
    Harbor runs its verifier after them. A signal kill or an authentication failure fails the attempt, as does an
    exit after the relay could not reach the policy model's tokenizer, with the tokenizer's error.

    Raises:
        ValueError: The task is not one user instruction, or it has no machine.
        TimeoutError: The setup outlived ``options.setup_timeout``.
    """
    instruction = task_instruction(context.task)
    if context.machine is None:
        raise ValueError("Pi requires a task machine")
    agent = PiAgent(
        options,
        MachineEnvironment(context.machine, context.rollout_id),
        mcp_servers(context.task.environment_requirements.tool_providers),
    )
    relay = ModelRelay(complete, tokenize)
    async with serve_model_relay(context.machine, relay, port=options.relay_port):
        try:
            async with asyncio.timeout(options.setup_timeout):
                await agent.setup()
        except TimeoutError as error:
            raise TimeoutError(f"Pi setup timed out after {options.setup_timeout} seconds") from error
        if options.setup_timeout is not None:
            context.begin_turn_budget()
        try:
            await agent.run(instruction)
        except (AgentKilledBySignalError, ModelAuthenticationError):
            raise
        except (NonZeroAgentExitCodeError, ContextLengthExceededError) as error:
            if relay.upstream_failure is not None:
                # Pi stopped because the relay could not reach the policy model, so the attempt fails with that error.
                raise relay.upstream_failure from error
            logger.warning("Pi failed in attempt %s; grading its work: %s", context.rollout_id, error)


class PiAgent:
    """Harbor's Pi agent over the task machine: its setup, which checks the image's Pi, and its run."""

    def __init__(self, options: PiOptions, environment: MachineEnvironment, mcp_servers: dict[str, dict[str, Any]]):
        self.options = options
        self.environment = environment
        self.mcp_servers = mcp_servers
        self.version: str | None = options.version
        # Harbor's launcher gives an installed agent's commands the guest endpoint of its model relay.
        self.extra_env = {
            "OPENAI_BASE_URL": options.api_base,
            "OPENAI_API_BASE": options.api_base,
            "OPENAI_API_KEY": API_KEY,
        }

    async def setup(self) -> None:
        """Create the agent directories, and check that the image's Pi has the pinned version."""
        await self.environment.exec(f"mkdir -p {AGENT_LOGS_DIR} && chmod 777 {AGENT_LOGS_DIR}", user="root")
        await self.environment.exec(command=f"mkdir -p {INSTALLED_AGENT_DIR}", user="root")
        await self.check_version()

    async def check_version(self) -> None:
        """Check that the environment image's Pi has the pinned version, and record that version."""
        result = await self.exec_as_agent(command="pi --version")
        installed_version = result.stdout.strip().splitlines()[-1].strip()
        if self.version is not None and installed_version != self.version:
            raise RuntimeError(f"Image provides Pi {installed_version}, but the agent requires {self.version}")
        self.version = installed_version

    def _truncate_output(self, text: str | None) -> str:
        if not text:
            return "None"
        text = text.replace(API_KEY, "<redacted>")
        if len(text) > OUTPUT_PREVIEW_CHARACTERS:
            return text[:OUTPUT_PREVIEW_CHARACTERS] + " ... [truncated]"
        return text

    async def _exec(self, command: str, user: str | None = None, env: dict[str, str] | None = None) -> ExecResult:
        """Run ``command`` with the agent environment; raise on a non-zero exit."""
        result = await self.environment.exec(
            command=f"set -o pipefail; {command}", user=user, env={**(env or {}), **self.extra_env}
        )
        if result.return_code == 0:
            return result
        message = (
            f"Command failed (exit {result.return_code}): {command}\n"
            f"stdout: {self._truncate_output(result.stdout)}\n"
            f"stderr: {self._truncate_output(result.stderr)}"
        )
        # A return code of 128 + N (N in 1..64) is the shell's convention for "terminated by signal N".
        if 128 < result.return_code <= 128 + 64:
            raise AgentKilledBySignalError(message, signal_number=result.return_code - 128, result=result)
        raise NonZeroAgentExitCodeError(message, result=result)

    async def exec_as_root(self, command: str) -> ExecResult:
        return await self._exec(command, user="root")

    async def exec_as_agent(self, command: str, env: dict[str, str] | None = None) -> ExecResult:
        return await self._exec(command, env=env)

    async def configure_mcp_servers(self) -> None:
        """Configure the image-provided pi-mcp-extension before Pi starts."""
        if not self.mcp_servers and not self.options.mcp_excluded_tools:
            return
        await self.environment.upload_file(MCP_CONFIGURE_SCRIPT, MCP_CONFIGURE_PATH)
        await self.exec_as_agent(
            command=(
                f"node {shlex.quote(MCP_CONFIGURE_PATH)} "
                f"{shlex.quote(MCP_EXTENSION_PATH)} "
                f"{shlex.quote(json.dumps({'mcpServers': self.mcp_servers}))} "
                f"{shlex.quote(json.dumps(list(self.options.mcp_excluded_tools)))}"
            ),
        )

    def register_provider_command(self) -> str:
        """A shell command that registers the relay as Pi's OpenAI-compatible provider of the model."""
        provider_document = {
            "providers": {
                PROVIDER: {
                    "baseUrl": "",
                    "api": "openai-completions",
                    "apiKey": API_KEY_ENV,
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                        "requiresReasoningContentOnAssistantMessages": True,
                        "thinkingFormat": self.options.thinking_format.value,
                        "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}},
                    },
                    "models": [
                        {
                            "id": self.options.model,
                            "name": self.options.model,
                            "reasoning": True,
                            "input": ["text"],
                            "contextWindow": self.options.max_input_tokens,
                            "maxTokens": self.options.max_output_tokens,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        }
                    ],
                }
            }
        }
        serialized = json.dumps(provider_document, separators=(",", ":"))
        return (
            f"node -e {shlex.quote(REGISTER_PROVIDER_SCRIPT)} "
            f"{shlex.quote(serialized)} {shlex.quote(PROVIDER)} "
            f"{shlex.quote(API_BASE_ENV)} "
            f"{shlex.quote(PROVIDER_CONFIG_RELATIVE_PATH)}"
        )

    def register_compaction_command(self) -> str:
        compaction = {
            "enabled": True,
            "reserveTokens": self.options.reserve_tokens,
            "keepRecentTokens": self.options.keep_recent_tokens,
        }
        serialized = json.dumps(compaction, separators=(",", ":"))
        return (
            f"node -e {shlex.quote(REGISTER_COMPACTION_SCRIPT)} "
            f"{shlex.quote(serialized)} "
            f"{shlex.quote(SETTINGS_CONFIG_RELATIVE_PATH)}"
        )

    async def _create_staging_dir(self) -> PurePosixPath:
        # Create as the agent so it can unlink provider-owned uploads. Keep
        # capability URLs outside collected logs even when cleanup is killed.
        result = await self.exec_as_agent(command=f"mktemp -d {shlex.quote(STAGING_DIR_TEMPLATE)}")
        return PurePosixPath(result.stdout.strip())

    async def _stage_endpoint_env(self, target_path: PurePosixPath) -> None:
        """Upload the endpoint variables without placing their values in a command."""
        contents = (
            f"export {API_BASE_ENV}={shlex.quote(self.options.api_base)}\n"
            f"export {API_KEY_ENV}={shlex.quote(API_KEY)}\n"
        )
        await self._upload_staged_text(contents, "endpoint.env", target_path)

    async def _stage_context_budget_extension(self, target_path: PurePosixPath) -> None:
        """Stage exact vLLM request budgeting for the hosted Pi provider."""
        extension = (
            CONTEXT_BUDGET_EXTENSION.replace(
                "__TOKENIZE_URL__", json.dumps(tokenize_url(self.options.api_base))
            ).replace("__CONTEXT_WINDOW__", str(self.options.max_input_tokens))
            + "\n"
        )
        await self._upload_staged_text(extension, CONTEXT_BUDGET_EXTENSION_FILENAME, target_path)

    async def _upload_staged_text(self, contents: str, filename: str, target_path: PurePosixPath) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / filename
            local_path.write_text(contents)
            await self.environment.upload_file(local_path, str(target_path))
        # The parent was created by the task user; uploads can be root-owned.
        quoted_path = shlex.quote(str(target_path))
        await self.exec_as_root(
            command=(
                f"chown --reference={shlex.quote(str(target_path.parent))} -- {quoted_path} && "
                f"chmod 600 -- {quoted_path}"
            ),
        )

    async def run(self, instruction: str) -> None:
        """Run Pi on ``instruction`` until it exits.

        Raises:
            ModelAuthenticationError: Pi could not authenticate to the relay.
            ContextLengthExceededError: Pi's last request exceeded the context window.
            AgentKilledBySignalError: A signal ended a command.
            NonZeroAgentExitCodeError: A command, or Pi, failed otherwise.
        """
        escaped_instruction = shlex.quote(instruction)
        await self.configure_mcp_servers()
        model_args = f"--provider {PROVIDER} --model {self.options.model} "
        cli_flags = "" if self.options.thinking is None else f"--thinking {self.options.thinking.value} "

        runtime_prelude = f"set -e -o pipefail; {self.register_compaction_command()}; "
        staging_dir = await self._create_staging_dir()
        endpoint_file = staging_dir / ENDPOINT_ENV_FILENAME
        extension_file = staging_dir / CONTEXT_BUDGET_EXTENSION_FILENAME
        await self._stage_endpoint_env(endpoint_file)
        await self._stage_context_budget_extension(extension_file)
        endpoint_path = shlex.quote(str(endpoint_file))
        extension_path = shlex.quote(str(extension_file))
        staging_path = shlex.quote(str(staging_dir))
        provider_path = f'"$HOME/{PROVIDER_CONFIG_RELATIVE_PATH}"'
        runtime_prelude = (
            "set -e; "
            f"trap 'rm -rf {staging_path} {provider_path}' EXIT; "
            f"{runtime_prelude}"
            f". {endpoint_path}; {self.register_provider_command()}; rm -f {endpoint_path}; "
        )
        cli_flags += f"--extension {extension_path} "
        try:
            await self.exec_as_agent(
                command=(
                    f"{runtime_prelude}"
                    f"pi --print --mode json --no-session "
                    f"{model_args}"
                    f"{cli_flags}"
                    f"{escaped_instruction} "
                    "2>&1 </dev/null | "
                    f"node -e {shlex.quote(VALIDATE_OUTPUT_SCRIPT)} | "
                    f"stdbuf -oL tee {AGENT_LOGS_DIR}/{OUTPUT_FILENAME}"
                ),
                env={},
            )
        except NonZeroAgentExitCodeError as error:
            error_lines = (error.result.stderr or "").lower().splitlines()
            if AUTHENTICATION_FAILURE_MARKER.lower() in error_lines:
                raise ModelAuthenticationError(
                    "Pi could not authenticate to its model endpoint", result=error.result
                ) from error
            if CONTEXT_LENGTH_FAILURE_MARKER.lower() in error_lines:
                raise ContextLengthExceededError("Pi exceeded the model's served context window") from error
            raise
