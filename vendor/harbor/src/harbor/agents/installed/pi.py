import base64
import binascii
import json
import logging
import os
import shlex
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version
from upath import UPath

from harbor.agents.installed.pi_literal_recorder import RECORDER_SCRIPT
from harbor.agents.installed.pi_in_memory_capture import EXTENSION_SCRIPT
from harbor_config.models.trial.pi import PiAgentConfig
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    ModelAuthenticationError,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError
from harbor.literal.proxy import CORRELATION_ID_HEADER
from harbor.literal.rollout_build import (
    build_pi_trial_evidence,
)
from harbor.models.agent.context import AgentContext
from harbor.models.agent.pi_trajectory import PiCollatedTrajectory
from harbor.llms.vllm_tokenization import tokenize_url
from harbor.models.agent.name import AgentName


_logger = logging.getLogger(__name__)
_BASE64_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)

_CURRENT_PI_PACKAGE = "@earendil-works/pi-coding-agent"
_NODE_VERSION = "22"
_LEGACY_PI_PACKAGE = "@mariozechner/pi-coding-agent"
_PI_PACKAGE_RENAME_VERSION = Version("0.74.0")
_REMOTE_LOG_READ_BLOCK_SIZE = 1024 * 1024
# pi-ai 1.3.0 subtracts this reserve inside buildBaseOptions before it sends
# the request. Advertise a virtual window to cancel that internal reserve;
# the before_provider_request hook separately enforces the real served window.
_PI_SDK_CONTEXT_SAFETY_TOKENS = 4096


@dataclass(frozen=True)
class _ModelLimits:
    # Harbor defines model_info.max_input_tokens as the total context window;
    # model_info.max_output_tokens is a cap within it, not additional capacity.
    context_window: int
    max_tokens: int
    compaction_reserve_tokens: int

    @property
    def compaction_keep_recent_tokens(self) -> int:
        return (self.context_window - self.compaction_reserve_tokens) // 2


def _decode_pi_evidence(
    stdout: str, *, dump_oversized_frame: bool = False
) -> dict[str, Any]:
    """Reassemble one in-memory Pi frame from bounded, numbered stdout writes."""
    prefix = "HARBOR_PI_EVIDENCE_V2:"
    chunks: dict[int, str] = {}
    expected_total: int | None = None
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            # Capture-mode stdout belongs exclusively to the evidence channel.
            # Never silently discard an unexpected Pi event or diagnostic here:
            # it could mask a transport regression or a private side channel.
            raise ValueError("Pi evidence stdout contained non-frame output")
        header, separator, fragment = line[len(prefix) :].partition(":")
        index_text, slash, total_text = header.partition("/")
        # Report only fixed shape labels: the stdout line may contain private
        # task text, model output, or credentials and must not enter diagnostics.
        if not separator:
            raise ValueError(
                "Pi returned malformed in-memory evidence chunks: missing separator"
            )
        if not slash or not index_text.isdecimal() or not total_text.isdecimal():
            raise ValueError(
                "Pi returned malformed in-memory evidence chunks: invalid header"
            )
        if not fragment:
            raise ValueError(
                "Pi returned malformed in-memory evidence chunks: empty fragment"
            )
        if len(fragment) > 1024:
            # Never log the line or a fragment of it: stdout can contain private
            # prompts, model responses, and credentials. Shape counters can
            # distinguish concatenated frames from unrelated interleaved text.
            _logger.warning(
                "Pi evidence oversized fragment: chars=%d embedded_frame_markers=%d "
                "contains_non_base64=%s chunks_before_error=%d",
                len(fragment),
                min(fragment.count(prefix), 2),
                any(char not in _BASE64_CHARS for char in fragment),
                len(chunks),
            )
            # One-off controlled diagnostics only. This line contains encoded
            # trajectory data and may include tool output or credentials; never
            # enable this in ordinary evaluation/training jobs.
            if dump_oversized_frame:
                _logger.warning("Pi rejected raw evidence line: %s", line)
            raise ValueError(
                "Pi returned malformed in-memory evidence chunks: oversized fragment"
            )
        index, total = int(index_text), int(total_text)
        if (
            total <= 0
            or total > 65536
            or index >= total
            or (expected_total is not None and total != expected_total)
            or index in chunks
        ):
            raise ValueError("Pi returned conflicting in-memory evidence chunks")
        expected_total = total
        chunks[index] = fragment
    if expected_total is None or len(chunks) != expected_total:
        raise ValueError(
            "Pi did not return exactly one in-memory evidence frame (missing chunks)"
        )
    try:
        payload = json.loads(
            base64.b64decode(
                "".join(chunks[index] for index in range(expected_total)), validate=True
            )
        )
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Pi returned a corrupted in-memory evidence frame") from error
    if not isinstance(payload, dict):
        raise ValueError("Pi in-memory evidence must be an object")
    return payload


class Pi(BaseInstalledAgent):
    _OUTPUT_FILENAME = "pi.txt"
    _DEFAULT_CONTEXT_WINDOW = 32768
    _DEFAULT_MAX_TOKENS = 16384
    _SUPPORTED_THINKING_FORMATS = frozenset({"chat-template", "qwen-chat-template"})
    _API_BASE_ENV = "HARBOR_PI_API_BASE"
    _API_KEY_ENV = "HARBOR_PI_API_KEY"
    _ENDPOINT_ENV_PATH = "/tmp/harbor-pi-endpoint.env"
    _CONTEXT_BUDGET_EXTENSION_PATH = "/tmp/harbor-pi-context-budget.js"
    _LITERAL_RECORDER_PATH = "/tmp/harbor-pi-literal-recorder.mjs"
    _PROVIDER_CONFIG_RELATIVE_PATH = ".pi/agent/models.json"
    _SETTINGS_CONFIG_RELATIVE_PATH = ".pi/agent/settings.json"
    _AUTHENTICATION_FAILURE_MARKER = "HARBOR_PI_MODEL_AUTHENTICATION_ERROR"
    _CONTEXT_LENGTH_FAILURE_MARKER = "HARBOR_PI_CONTEXT_LENGTH_EXCEEDED"
    _VALIDATE_OUTPUT_SCRIPT = (
        """
const readline = require("readline");
const captureOnly = process.env.HARBOR_PI_CAPTURE_ONLY === "1";
let lastAssistantStopReason;
let lastAssistantErrorMessage;
const lines = readline.createInterface({ input: process.stdin });
lines.on("line", (line) => {
  if (!captureOnly) process.stdout.write(line + "\\n");
  try {
    const event = JSON.parse(line);
    const message = event.type === "message_end" ? event.message : undefined;
    if (message?.role === "assistant") {
      lastAssistantStopReason = message.stopReason;
      lastAssistantErrorMessage = message.errorMessage;
    }
  } catch {
    // Pi can mix diagnostics into its JSON stream. In capture mode, the
    // dedicated evidence FD carries the only stdout; never echo JSON events
    // into the evidence channel.
  }
});
lines.on("close", () => {
  if (lastAssistantStopReason === undefined) {
    process.stderr.write("HARBOR_PI_MISSING_ASSISTANT_RESULT\\n");
    process.exitCode = 1;
    return;
  }
  if (lastAssistantStopReason !== "error") return;
  if (/authentication required|authenticationerror/i.test(lastAssistantErrorMessage ?? "")) {
    process.stderr.write("__AUTHENTICATION_FAILURE_MARKER__\\n");
  } else if (/maximum context length|context_length_exceeded/i.test(lastAssistantErrorMessage ?? "")) {
    process.stderr.write("__CONTEXT_LENGTH_FAILURE_MARKER__\\n");
  } else {
    // Do not put an untrusted model error message on the evidence channel.
    process.stderr.write("HARBOR_PI_ASSISTANT_ERROR\\n");
  }
  process.exitCode = 1;
});
""".replace("__AUTHENTICATION_FAILURE_MARKER__", _AUTHENTICATION_FAILURE_MARKER)
        .replace("__CONTEXT_LENGTH_FAILURE_MARKER__", _CONTEXT_LENGTH_FAILURE_MARKER)
        .strip()
    )
    _REGISTER_PROVIDER_SCRIPT = """
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
    _REGISTER_COMPACTION_SCRIPT = """
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
    _CONTEXT_BUDGET_EXTENSION = """
import fs from 'node:fs';

const tokenizeUrl = __TOKENIZE_URL__;
const contextWindow = __CONTEXT_WINDOW__;
const diagnostics = process.env.HARBOR_PI_SANDBOX_DIAGNOSTICS === '1';
let budgetAttempt = 0;

function recordBudget(phase, fields) {
  if (!diagnostics) return;
  fs.appendFileSync(process.env.HARBOR_PI_SANDBOX_DIAGNOSTICS_PATH,
    JSON.stringify({ source: 'budget', phase, ...fields }) + '\\n', { mode: 0o600 });
}

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
    recordBudget('tokenize_failed', { http_status: response.status });
    throw new Error(`vLLM /tokenize returned HTTP ${response.status}`);
  }
  const body = await response.json();
  if (!Array.isArray(body.tokens) || !body.tokens.every(Number.isInteger)) {
    recordBudget('tokenize_invalid', {});
    throw new Error("vLLM /tokenize did not return an integer token list");
  }
  recordBudget('tokenize_completed', { prompt_tokens: body.tokens.length });
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
  recordBudget('registered', { context_window: contextWindow });
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
    const attempt = ++budgetAttempt;
    recordBudget('request', { attempt, context_window: contextWindow,
      configured_output_tokens: configuredMaxTokens, message_count: payload.messages.length,
      tool_count: Array.isArray(payload.tools) ? payload.tools.length : 0 });
    let promptTokens = await countPromptTokens(payload);
    const originalPromptTokens = promptTokens;
    const promptBudget = Math.max(1, contextWindow - configuredMaxTokens);
    let adjustedPayload = payload;
    if (promptTokens > promptBudget) {
      const fitted = await fitToolOutputs(payload, promptBudget, promptTokens);
      adjustedPayload = fitted.payload;
      promptTokens = fitted.promptTokens;
    }
    const availableOutputTokens = Math.max(1, contextWindow - promptTokens);
    const outputTokens = Math.min(configuredMaxTokens, availableOutputTokens);
    recordBudget('decision', { attempt, original_prompt_tokens: originalPromptTokens,
      fitted_prompt_tokens: promptTokens, prompt_budget: promptBudget,
      available_output_tokens: availableOutputTokens, output_tokens: outputTokens,
      tool_outputs_fitted: adjustedPayload !== payload });
    return { ...adjustedPayload, [maxTokensField]: outputTokens };
  });
}
""".strip()

    CLI_FLAGS = [
        CliFlag(
            "thinking",
            cli="--thinking",
            type="enum",
            choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        ),
    ]

    def __init__(
        self,
        *args,
        api_base: str | None = None,
        api_key: str | None = None,
        collect_rollout_details: bool = False,
        verbose_sandbox_diagnostics: bool = False,
        dump_oversized_frame: bool = False,
        pi_config: PiAgentConfig | None = None,
        teacher_api_key: str | None = None,
        compaction_reserve_tokens: int | None = None,
        model_info: dict[str, Any] | None = None,
        thinking_format: str | None = None,
        **kwargs,
    ) -> None:
        if api_base is not None and (not isinstance(api_base, str) or not api_base):
            raise ValueError("api_base must be a non-empty string")
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise ValueError("api_key must be a non-empty string")
        if (
            thinking_format is not None
            and thinking_format not in self._SUPPORTED_THINKING_FORMATS
        ):
            supported = ", ".join(sorted(self._SUPPORTED_THINKING_FORMATS))
            raise ValueError(
                f"Unsupported thinking_format '{thinking_format}'. "
                f"Valid values: {supported}"
            )
        if api_base is not None and thinking_format is None:
            raise ValueError("thinking_format is required when api_base is configured")
        if (collect_rollout_details or pi_config is not None) and api_base is None:
            raise ValueError(
                "collect_rollout_details requires api_base so Pi can stamp every "
                "model request with a trial correlation ID"
            )

        if type(verbose_sandbox_diagnostics) is not bool:
            raise ValueError("verbose_sandbox_diagnostics must be a boolean")
        if verbose_sandbox_diagnostics and (
            api_base is None or not (collect_rollout_details or pi_config is not None)
        ):
            raise ValueError(
                "Verbose Pi sandbox diagnostics require hosted in-memory capture"
            )
        if type(dump_oversized_frame) is not bool:
            raise ValueError("dump_oversized_frame must be a boolean")
        if dump_oversized_frame and not verbose_sandbox_diagnostics:
            raise ValueError(
                "Raw Pi frame diagnostics require verbose sandbox diagnostics"
            )
        self._verbose_sandbox_diagnostics = verbose_sandbox_diagnostics
        self._dump_oversized_frame = dump_oversized_frame
        self._api_base = api_base
        self._api_key = api_key or "not-needed"
        if pi_config is not None and not isinstance(pi_config, PiAgentConfig):
            raise ValueError("Pi capability must be a validated PiAgentConfig")
        self._dagger_plan = pi_config.plan if pi_config is not None else None
        self._critique_plan = pi_config.critique if pi_config is not None else None
        self._capture_actor = (
            "teacher"
            if pi_config is not None and pi_config.mode == "teacher_trace"
            else None
        )
        collect_rollout_details = collect_rollout_details or pi_config is not None
        self._teacher_api_key = teacher_api_key or "not-needed"
        self._dagger_turns: list[dict[str, Any]] | None = None
        self._critique_evidence: dict[str, Any] | None = None
        self._collect_rollout_details = collect_rollout_details
        self._in_memory_pi_output: str | None = None
        self._in_memory_entries: list[dict[str, Any]] | None = None
        self._in_memory_trajectory: dict[str, Any] | None = None
        self._in_memory_capture_failed = False
        self.rollout_correlation_id = (
            uuid.uuid4().hex if collect_rollout_details else None
        )
        self._trial_identity = (
            pi_config.trial_id
            if pi_config is not None and pi_config.trial_id is not None
            else self._dagger_plan.trial_id
            if self._dagger_plan is not None
            else self._critique_plan.trial_id
            if self._critique_plan is not None
            else self.rollout_correlation_id
        )
        self._policy_step = (
            pi_config.policy_step
            if pi_config is not None and pi_config.policy_step is not None
            else self._dagger_plan.policy_step
            if self._dagger_plan is not None
            else 0
        )
        self._thinking_format = thinking_format
        model_info = model_info or {}
        self._configure_compaction = (
            api_base is not None
            or compaction_reserve_tokens is not None
            or any(
                key in model_info for key in ("max_input_tokens", "max_output_tokens")
            )
        )
        self._model_limits = self._resolve_model_limits(
            model_info, compaction_reserve_tokens
        )
        if self._model_limits.max_tokens > self._model_limits.context_window:
            raise ValueError("max_output_tokens must not exceed max_input_tokens")
        if self._model_limits.compaction_reserve_tokens < self._model_limits.max_tokens:
            raise ValueError(
                "compaction_reserve_tokens must be at least max_output_tokens"
            )
        if (
            self._model_limits.compaction_reserve_tokens
            >= self._model_limits.context_window
        ):
            raise ValueError(
                "compaction_reserve_tokens must be less than max_input_tokens"
            )

        super().__init__(*args, **kwargs)

    @classmethod
    def _resolve_model_limit(
        cls, model_info: dict[str, Any], key: str, default: int
    ) -> int:
        value = model_info.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"model_info.{key} must be a positive integer")
        return value

    @classmethod
    def _resolve_model_limits(
        cls,
        model_info: dict[str, Any],
        compaction_reserve_tokens: int | None,
    ) -> _ModelLimits:
        max_tokens = cls._resolve_model_limit(
            model_info, "max_output_tokens", cls._DEFAULT_MAX_TOKENS
        )
        if compaction_reserve_tokens is None:
            compaction_reserve_tokens = max_tokens
        elif (
            isinstance(compaction_reserve_tokens, bool)
            or not isinstance(compaction_reserve_tokens, int)
            or compaction_reserve_tokens <= 0
        ):
            raise ValueError("compaction_reserve_tokens must be a positive integer")
        return _ModelLimits(
            context_window=cls._resolve_model_limit(
                model_info, "max_input_tokens", cls._DEFAULT_CONTEXT_WINDOW
            ),
            max_tokens=max_tokens,
            compaction_reserve_tokens=compaction_reserve_tokens,
        )

    def _truncate_output(self, text: str | None, max_len: int = 1000) -> str:
        if text is None:
            return super()._truncate_output(text, max_len)
        if self._api_key:
            text = text.replace(self._api_key, "<redacted>")
        return super()._truncate_output(text, max_len)

    @property
    def verbose_sandbox_diagnostics(self) -> bool:
        """Whether bounded sandbox-side diagnostic counters are enabled for this trial."""
        return self._verbose_sandbox_diagnostics

    @staticmethod
    def name() -> str:
        return AgentName.PI.value

    def get_version_command(self) -> str | None:
        return f"{self.nvm_shell_command()} && pi --version"

    @staticmethod
    def nvm_shell_command() -> str:
        """Return the shell prelude that exposes the NVM-managed Pi executable."""
        return (
            'export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh" && '
            f"nvm use {_NODE_VERSION} >/dev/null"
        )

    def hosted_endpoint_env(self) -> dict[str, str]:
        """Return the secret environment required by a registered hosted provider."""
        if self._api_base is None:
            return {}
        return {self._API_KEY_ENV: self._api_key}

    async def configure_hosted_provider(self, environment: BaseEnvironment) -> None:
        """Register this agent's dynamic OpenAI-compatible endpoint with Pi."""
        if self._api_base is None:
            return
        if not self.model_alias or "/" not in self.model_alias:
            raise ValueError("Pi model alias must be in the format provider/model_name")

        provider, model_id = self.model_alias.split("/", 1)
        provider_command = self._build_register_provider_command(provider, model_id)
        if provider_command is None:
            raise RuntimeError("Pi provider registration command was not created")

        await self._stage_endpoint_env(environment)
        endpoint_path = shlex.quote(self._ENDPOINT_ENV_PATH)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"trap 'rm -f {endpoint_path}' EXIT; "
                f"{self.nvm_shell_command()}; "
                f". {endpoint_path}; {provider_command}; rm -f {endpoint_path}"
            ),
        )

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1].strip()

    def _package_name(self) -> str:
        if self._version:
            try:
                if Version(self._version) < _PI_PACKAGE_RENAME_VERSION:
                    return _LEGACY_PI_PACKAGE
            except InvalidVersion as exc:
                raise ValueError(
                    f"Pi version {self._version!r} is not a valid package version"
                ) from exc
        return _CURRENT_PI_PACKAGE

    async def install(self, environment: BaseEnvironment) -> None:
        package_name = self._package_name()
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
                'export NVM_DIR="$HOME/.nvm" && '
                '\\. "$NVM_DIR/nvm.sh" || true && '
                "command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
                f"nvm install {_NODE_VERSION} && npm -v && "
                f"npm install -g --ignore-scripts {package_name}{version_spec} && "
                "pi --version"
            ),
        )

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills to Pi's skills directory."""
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p $HOME/.agents/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            f"$HOME/.agents/skills/ 2>/dev/null || true"
        )

    def _build_register_provider_command(
        self, provider: str, model_id: str
    ) -> str | None:
        if self._api_base is None:
            return None

        provider_document = {
            "providers": {
                provider: {
                    "baseUrl": "",
                    "api": "openai-completions",
                    "apiKey": self._API_KEY_ENV,
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                        "requiresReasoningContentOnAssistantMessages": True,
                        "thinkingFormat": self._thinking_format,
                        "chatTemplateKwargs": {
                            "enable_thinking": {"$var": "thinking.enabled"}
                        },
                    },
                    "models": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "reasoning": True,
                            "input": ["text"],
                            "contextWindow": self._model_limits.context_window
                            + _PI_SDK_CONTEXT_SAFETY_TOKENS,
                            "maxTokens": self._model_limits.max_tokens,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            }
        }
        if self.rollout_correlation_id:
            # Pi applies provider-level models.json headers to every request made
            # with this model, including tool-loop turns and context compaction.
            provider_document["providers"][provider]["headers"] = {
                CORRELATION_ID_HEADER: self.rollout_correlation_id
            }
        serialized = json.dumps(provider_document, separators=(",", ":"))
        return (
            f"node -e {shlex.quote(self._REGISTER_PROVIDER_SCRIPT)} "
            f"{shlex.quote(serialized)} {shlex.quote(provider)} "
            f"{shlex.quote(self._API_BASE_ENV)} "
            f"{shlex.quote(self._PROVIDER_CONFIG_RELATIVE_PATH)}"
        )

    def _build_register_compaction_command(self) -> str | None:
        if not self._configure_compaction:
            return None
        compaction = {
            "enabled": self._dagger_plan is None,
            "reserveTokens": self._model_limits.compaction_reserve_tokens
            + (_PI_SDK_CONTEXT_SAFETY_TOKENS if self._api_base is not None else 0),
            "keepRecentTokens": self._model_limits.compaction_keep_recent_tokens,
        }
        serialized = json.dumps(compaction, separators=(",", ":"))
        return (
            f"node -e {shlex.quote(self._REGISTER_COMPACTION_SCRIPT)} "
            f"{shlex.quote(serialized)} "
            f"{shlex.quote(self._SETTINGS_CONFIG_RELATIVE_PATH)}"
        )

    async def _stage_endpoint_env(self, environment: BaseEnvironment) -> None:
        """Upload the endpoint variables without placing their values in a command."""
        if self._api_base is None:
            raise ValueError("api_base is required to stage the endpoint environment")
        contents = (
            f"export {self._API_BASE_ENV}={shlex.quote(self._api_base)}\n"
            f"export {self._API_KEY_ENV}={shlex.quote(self._api_key)}\n"
        )
        if self._dagger_plan is not None:
            contents += f"export HARBOR_PI_TEACHER_API_KEY={shlex.quote(self._teacher_api_key)}\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "endpoint.env"
            local_path.write_text(contents)
            await environment.upload_file(local_path, self._ENDPOINT_ENV_PATH)

    async def _stage_context_budget_extension(
        self, environment: BaseEnvironment
    ) -> None:
        """Stage exact vLLM request budgeting for the hosted Pi provider."""
        if self._api_base is None:
            raise ValueError(
                "api_base is required to stage the context budget extension"
            )
        extension = (
            self._CONTEXT_BUDGET_EXTENSION.replace(
                "__TOKENIZE_URL__", json.dumps(tokenize_url(self._api_base))
            ).replace("__CONTEXT_WINDOW__", str(self._model_limits.context_window))
            + "\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "harbor-pi-context-budget.js"
            local_path.write_text(extension)
            await environment.upload_file(
                local_path, self._CONTEXT_BUDGET_EXTENSION_PATH
            )

    async def _stage_capture_extension(self, environment: BaseEnvironment) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "harbor-pi-in-memory-capture.mjs"
            local_path.write_text(EXTENSION_SCRIPT)
            await environment.upload_file(
                local_path, "/tmp/harbor-pi-in-memory-capture.mjs"
            )

    async def _stage_literal_recorder(self, environment: BaseEnvironment) -> None:
        """Install the trial-local vLLM recorder in the task sandbox."""
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "harbor-pi-literal-recorder.mjs"
            local_path.write_text(RECORDER_SCRIPT)
            await environment.upload_file(local_path, self._LITERAL_RECORDER_PATH)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._in_memory_pi_output = None
        self._in_memory_entries = None
        self._in_memory_trajectory = None
        self._dagger_turns = None
        self._critique_evidence = None
        self._in_memory_capture_failed = False
        escaped_instruction = shlex.quote(instruction)

        if not self.model_alias or "/" not in self.model_alias:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, model_id = self.model_alias.split("/", 1)
        if not provider or not model_id:
            raise ValueError("Model name must be in the format provider/model_name")

        env: dict[str, str] = {}
        if self._verbose_sandbox_diagnostics:
            env["HARBOR_PI_SANDBOX_DIAGNOSTICS"] = "1"
            env["HARBOR_PI_SANDBOX_DIAGNOSTICS_PATH"] = str(
                self.environment_logs_dir / "pi-sandbox-diagnostics.jsonl"
            )
        if self._collect_rollout_details:
            if not self._trial_identity:
                raise ValueError("Pi trial has no bound identity")
            env["HARBOR_PI_TRIAL_ID"] = self._trial_identity
            env["HARBOR_PI_POLICY_STEP"] = str(self._policy_step)
        if self._dagger_plan is not None:
            env["HARBOR_PI_DAGGER_PLAN"] = json.dumps(asdict(self._dagger_plan))
        if self._capture_actor is not None:
            env["HARBOR_PI_CAPTURE_ACTOR"] = self._capture_actor
        if self._critique_plan is not None:
            env["HARBOR_PI_CRITIQUE_PLAN"] = json.dumps(asdict(self._critique_plan))
        keys: list[str] = []

        if self._api_base is None:
            if provider == "amazon-bedrock":
                keys.extend(
                    ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]
                )
            elif provider == "anthropic":
                keys.extend(
                    ["ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_BASE_URL"]
                )
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
                    ]
                )
            elif provider == "groq":
                keys.append("GROQ_API_KEY")
            elif provider == "huggingface":
                keys.append("HF_TOKEN")
            elif provider == "mistral":
                keys.append("MISTRAL_API_KEY")
            elif provider == "openai":
                keys.extend(["OPENAI_API_KEY", "OPENAI_BASE_URL"])
            elif provider == "openrouter":
                keys.append("OPENROUTER_API_KEY")
            elif provider == "xai":
                keys.append("XAI_API_KEY")

        for key in keys:
            val = os.environ.get(key)
            if val:
                env[key] = val

        model_args = f"--provider {provider} --model {model_id} "

        cli_flags = self.build_cli_flags()
        if cli_flags:
            cli_flags += " "

        provider_command = self._build_register_provider_command(provider, model_id)
        compaction_command = self._build_register_compaction_command()
        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command)

        runtime_prelude = f"set -e -o pipefail; {self.nvm_shell_command()}; "
        if compaction_command is not None:
            runtime_prelude += f"{compaction_command}; "
        if self._api_base is not None:
            if provider_command is None:
                raise ValueError(
                    "Provider registration command is required with api_base"
                )
            await self._stage_endpoint_env(environment)
            await self._stage_context_budget_extension(environment)
            endpoint_path = shlex.quote(self._ENDPOINT_ENV_PATH)
            extension_path = shlex.quote(self._CONTEXT_BUDGET_EXTENSION_PATH)
            provider_path = f'"$HOME/{self._PROVIDER_CONFIG_RELATIVE_PATH}"'
            cleanup = f"rm -f {endpoint_path} {extension_path} {provider_path}"
            recorder_setup = ""
            if self._collect_rollout_details:
                await self._stage_literal_recorder(environment)
                await self._stage_capture_extension(environment)
                cleanup += " /tmp/harbor-pi-in-memory-capture.mjs"
                recorder_setup = (
                    f'export HARBOR_PI_UPSTREAM_BASE="${self._API_BASE_ENV}"; '
                    'export HARBOR_PI_PACKAGE_ROOT="$(npm root -g)/@earendil-works/pi-coding-agent"; '
                    f"export HARBOR_PI_PROVIDER={shlex.quote(provider)}; "
                )
            runtime_prelude = (
                "set -e; "
                f"trap {shlex.quote(cleanup)} EXIT; "
                f"{runtime_prelude}"
                f". {endpoint_path}; {recorder_setup}"
                f"{provider_command}; rm -f {endpoint_path}; "
            )
            cli_flags += f"--extension {extension_path} "
            if self._collect_rollout_details:
                cli_flags += "--extension /tmp/harbor-pi-in-memory-capture.mjs "
        if self._collect_rollout_details:
            # Reserve stdout as an evidence-only channel before Pi's own JSON
            # stdout enters the validator pipe. Pi's extension writes to FD 3.
            runtime_prelude += "exec 3>&1; export HARBOR_PI_CAPTURE_ONLY=1; "
        sink = (
            ""
            if self._collect_rollout_details
            else f" | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
        )
        try:
            result = await self.exec_as_agent(
                environment,
                command=(
                    f"{runtime_prelude}"
                    f"pi --print --mode json --no-session "
                    f"{model_args}"
                    f"{cli_flags}"
                    f"{escaped_instruction} "
                    "2>&1 </dev/null | "
                    f"node -e {shlex.quote(self._VALIDATE_OUTPUT_SCRIPT)}{sink}"
                ),
                env=env,
            )
            if self._collect_rollout_details:
                self._in_memory_pi_output = result.stdout
                payload = _decode_pi_evidence(
                    result.stdout or "", dump_oversized_frame=self._dump_oversized_frame
                )
                if self._critique_plan is not None:
                    if not isinstance(payload, dict) or set(payload) != {
                        "entries",
                        "critique",
                        "failure",
                        "trajectory",
                    }:
                        raise ValueError(
                            "Pi critique did not return its complete evidence frame"
                        )
                    if payload["failure"] is not None:
                        raise ValueError(f"Pi critique failed: {payload['failure']}")
                    self._critique_evidence = payload["critique"]
                    entries = payload["entries"]
                elif self._dagger_plan is not None:
                    if not isinstance(payload, dict) or set(payload) != {
                        "entries",
                        "dagger_turns",
                        "trajectory",
                    }:
                        raise ValueError(
                            "Pi DAgger did not return executed turns and teacher labels"
                        )
                    self._dagger_turns = payload["dagger_turns"]
                    entries = payload["entries"]
                else:
                    if not isinstance(payload, dict) or set(payload) != {
                        "entries",
                        "trajectory",
                    }:
                        raise ValueError(
                            "Pi evidence frame must contain one collated trajectory"
                        )
                    entries = payload["entries"]
                if not isinstance(entries, list) or not isinstance(
                    payload["trajectory"], dict
                ):
                    raise ValueError(
                        "Pi evidence frame must contain model calls and one collated trajectory"
                    )
                self._in_memory_entries = entries
                self._in_memory_trajectory = payload["trajectory"]
        except NonZeroAgentExitCodeError as error:
            self._in_memory_capture_failed = self._collect_rollout_details
            if self._collect_rollout_details:
                self.logger.warning(
                    "Pi in-memory capture failed without a validated trajectory"
                )
            detail = (
                (error.result.stderr or "").lower() if error.result is not None else ""
            )
            if self._AUTHENTICATION_FAILURE_MARKER.lower() in detail:
                raise ModelAuthenticationError(
                    "Pi could not authenticate to its model endpoint "
                    "(possibly an expired or invalid capability URL)",
                    result=error.result,
                ) from error
            if self._CONTEXT_LENGTH_FAILURE_MARKER.lower() in detail:
                raise ContextLengthExceededError(
                    "Pi exceeded the model's served context window"
                ) from error
            raise

    def _validate_dagger_turns(self, entries: list[dict[str, Any]]) -> None:
        plan = self._dagger_plan
        turns = self._dagger_turns
        if (
            plan is None
            or not entries
            or not isinstance(turns, list)
            or len(turns) != len(entries)
        ):
            raise ValueError("Pi DAgger must label exactly every executed turn")
        if len(turns) > len(plan.executors):
            raise ValueError("Pi DAgger exceeded its pre-action plan")
        for index, (entry, turn) in enumerate(zip(entries, turns, strict=True)):
            if not isinstance(turn, dict) or set(turn) != {
                "turn",
                "actor",
                "teacher_label",
                "executed_request_id",
            }:
                raise ValueError("Pi DAgger has a malformed attributed turn")
            actor = plan.executors[index]
            label = turn["teacher_label"]
            if (
                turn["turn"] != index
                or turn["actor"] != actor
                or entry.get("actor") != actor
                or turn["executed_request_id"] != entry.get("request_id")
                or not isinstance(label, dict)
                or label.get("request_id") is None
            ):
                raise ValueError(
                    "Pi DAgger pre-action actor attribution does not match execution"
                )
            build_pi_trial_evidence([*entries[:index], {**label, "actor": "teacher"}])
            label_request = label["request"]
            executed_request = entry["request"]
            if {
                key: value for key, value in label_request.items() if key != "model"
            } != {
                key: value for key, value in executed_request.items() if key != "model"
            } or label_request.get("model") != plan.teacher_model:
                raise ValueError(
                    "Pi DAgger teacher label and executed action have different pre-action states"
                )
            if actor == "teacher" and label != {
                key: value for key, value in entry.items() if key != "actor"
            }:
                raise ValueError(
                    "Pi DAgger teacher-executed action must be the sampled teacher label"
                )
            if actor == "student" and label["request_id"] == entry["request_id"]:
                raise ValueError(
                    "Pi DAgger private teacher label was reused as a student action"
                )

    def _validate_critique(self, entries: list[dict[str, Any]]) -> None:
        plan = self._critique_plan
        critique = self._critique_evidence
        if (
            plan is None
            or len(entries) != 2
            or not isinstance(critique, dict)
            or set(critique)
            != {
                "trial_id",
                "teacher_model",
                "student_request_id",
                "teacher_request_id",
                "teacher_request",
                "teacher_response",
                "feedback",
            }
        ):
            raise ValueError(
                "Pi critique needs one student attempt, teacher feedback, and one revision"
            )
        first, revision = entries
        feedback = critique["feedback"]
        teacher_request = critique["teacher_request"]
        if (
            first.get("actor") != "student"
            or revision.get("actor") != "revision"
            or critique["trial_id"] != plan.trial_id
            or critique["teacher_model"] != plan.teacher_model
            or critique["student_request_id"] != first.get("request_id")
            or not isinstance(critique["teacher_request_id"], str)
            or not critique["teacher_request_id"]
            or not isinstance(feedback, str)
            or not feedback.strip()
            or not isinstance(teacher_request, dict)
            or teacher_request.get("model") != plan.teacher_model
            or teacher_request.get("stream") is not False
            or not isinstance(critique["teacher_response"], dict)
            or not isinstance(critique["teacher_response"].get("content"), str)
            or not critique["teacher_response"]["content"].strip()
            or critique["teacher_response"]["content"].strip() not in feedback
        ):
            raise ValueError(
                "Pi critique actor or teacher request/feedback attribution is invalid"
            )
        student_messages = first["request"]["messages"]
        teacher_messages = teacher_request.get("messages")
        revision_messages = revision["request"]["messages"]
        if (
            not isinstance(teacher_messages, list)
            or not isinstance(revision_messages, list)
            or len(teacher_messages) < len(student_messages) + 2
            or teacher_messages[: len(student_messages)] != student_messages
            or teacher_messages[len(student_messages)]
            != first["literal"]["assistant_message"]
            or len(revision_messages) < len(student_messages) + 2
            or revision_messages[: len(student_messages)] != student_messages
            or revision_messages[-1]
            != {"role": "user", "content": [{"type": "text", "text": feedback}]}
            or teacher_messages[-1] in revision_messages
            or revision["literal"]["assistant_message"].get("tool_calls")
        ):
            raise ValueError(
                "Pi critique private instruction leaked, feedback missing, or revision used tools"
            )
        calls = first["literal"]["assistant_message"].get("tool_calls", [])
        observations = teacher_messages[len(student_messages) + 1 : -1]
        executed_observations = revision_messages[len(student_messages) + 1 : -1]
        if (
            observations != executed_observations
            or len(observations) != len(calls)
            or any(
                observation.get("role") != "tool"
                or observation.get("tool_call_id") != call["id"]
                for call, observation in zip(calls, observations, strict=True)
            )
        ):
            raise ValueError(
                "Pi critique did not receive the executed tool observations"
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        if self.rollout_correlation_id:
            context.metadata = {
                **(context.metadata or {}),
                "rollout_correlation_id": self.rollout_correlation_id,
            }

        if self._in_memory_capture_failed:
            return
        if self._collect_rollout_details:
            if self._in_memory_entries is None:
                raise ValueError("Pi trial missing its in-memory model evidence")
            rollout_details, messages = build_pi_trial_evidence(self._in_memory_entries)
            request_ids = [entry.get("request_id") for entry in self._in_memory_entries]
            if any(
                not isinstance(value, str) or not value for value in request_ids
            ) or len(set(request_ids)) != len(request_ids):
                raise ValueError("Pi trial model requests need unique nonempty IDs")
            context.rollout_details = rollout_details
            trajectory = PiCollatedTrajectory.model_validate(self._in_memory_trajectory)
            if (
                trajectory.trial_id != self._trial_identity
                or trajectory.policy_step != self._policy_step
            ):
                raise ValueError(
                    "Pi trajectory identity does not match the bound trial"
                )
            if len(trajectory.turns) != len(self._in_memory_entries):
                raise ValueError(
                    "Pi trajectory must contain exactly the executed model turns"
                )
            for turn, entry in zip(
                trajectory.turns, self._in_memory_entries, strict=True
            ):
                if turn.executed.request_id != entry[
                    "request_id"
                ] or turn.executed.response_token_ids != tuple(
                    entry["literal"]["completion_token_ids"]
                ):
                    raise ValueError(
                        "Pi trajectory differs from its sampled model evidence"
                    )
            if self._dagger_plan is not None:
                self._validate_dagger_turns(self._in_memory_entries)
            if self._critique_plan is not None:
                self._validate_critique(self._in_memory_entries)
            if self._capture_actor is not None:
                if not self._in_memory_entries or any(
                    entry.get("actor") != self._capture_actor
                    for entry in self._in_memory_entries
                ):
                    raise ValueError(
                        "Pi teacher trace requires every executed model turn to be teacher-authored"
                    )
            context.metadata = {
                **(context.metadata or {}),
                "all_messages": messages,
                "summarization_count": 0,
                "pi_literal_capture": True,
                "pi_hydrated_trajectory": trajectory.model_dump(mode="json"),
                "pi_literal_source": "in_memory",
                "pi_literal_request_ids": request_ids,
                **(
                    {"pi_executed_actors": ["teacher"] * len(request_ids)}
                    if self._capture_actor
                    else {
                        "pi_executed_actors": ["student", "revision"],
                        "pi_critique": self._critique_evidence,
                    }
                    if self._critique_plan is not None
                    else {}
                ),
                **(
                    {
                        "pi_dagger_turns": self._dagger_turns,
                        "pi_dagger_plan": asdict(self._dagger_plan),
                        "pi_dagger_executed_messages": [
                            *self._in_memory_entries[-1]["request"]["messages"],
                            self._in_memory_entries[-1]["literal"]["assistant_message"],
                        ],
                    }
                    if self._dagger_plan is not None
                    else {}
                ),
            }

        output_file = self.logs_dir / self._OUTPUT_FILENAME
        if not self._collect_rollout_details and not output_file.exists():
            return

        assistant_messages = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_write_tokens = 0
        total_cost = 0.0

        skipped_records = 0
        if self._collect_rollout_details:
            if self._in_memory_entries is None:
                raise ValueError("Pi trial missing its in-memory assistant evidence")
            # Pi's JSON events are deliberately not echoed into the evidence
            # channel. Use the already validated, attributed model calls for
            # these informational counts, never a second stdout transcript.
            assistant_messages = len(self._in_memory_entries)
            total_input_tokens = sum(
                len(entry["literal"]["prompt_token_ids"])
                for entry in self._in_memory_entries
            )
            total_output_tokens = sum(
                len(entry["literal"]["completion_token_ids"])
                for entry in self._in_memory_entries
            )
        else:
            if isinstance(output_file, UPath):
                lines_context = output_file.open(
                    "rb", block_size=_REMOTE_LOG_READ_BLOCK_SIZE
                )
            else:
                lines_context = output_file.open("rb")
            with lines_context as lines:
                for raw_line in lines:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("type") == "message_end":
                            message = event.get("message") or {}
                            if message.get("role") == "assistant":
                                assistant_messages += 1
                                usage = message.get("usage") or {}
                                total_input_tokens += usage.get("input", 0)
                                total_output_tokens += usage.get("output", 0)
                                total_cache_read_tokens += usage.get("cacheRead", 0)
                                total_cache_write_tokens += usage.get("cacheWrite", 0)
                                cost = usage.get("cost") or {}
                                total_cost += cost.get("total", 0.0)
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        skipped_records += 1

        self._in_memory_pi_output = None
        self._in_memory_entries = None
        if self._collect_rollout_details and context.rollout_details is not None:
            captured_turns = len(context.rollout_details[0]["completion_token_ids"])
            if assistant_messages != captured_turns:
                raise ValueError(
                    f"Pi trial has {assistant_messages} assistant turns but {captured_turns} literal model calls"
                )

        if skipped_records:
            self.logger.debug(
                "Skipped %d malformed Pi output record(s) in %s",
                skipped_records,
                output_file,
            )

        context.n_input_tokens = total_input_tokens + total_cache_read_tokens
        context.n_output_tokens = total_output_tokens
        context.n_cache_tokens = total_cache_read_tokens
        context.cost_usd = total_cost if total_cost > 0 else None
