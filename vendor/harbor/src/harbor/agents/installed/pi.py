import json
import os
import shlex
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version
from upath import UPath

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
from harbor.llms.vllm_tokenization import tokenize_url
from harbor.models.agent.name import AgentName


_CURRENT_PI_PACKAGE = "@earendil-works/pi-coding-agent"
_NODE_VERSION = "22"
_LEGACY_PI_PACKAGE = "@mariozechner/pi-coding-agent"
_PI_PACKAGE_RENAME_VERSION = Version("0.74.0")
_REMOTE_LOG_READ_BLOCK_SIZE = 1024 * 1024


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


class Pi(BaseInstalledAgent):
    _OUTPUT_FILENAME = "pi.txt"
    _DEFAULT_CONTEXT_WINDOW = 32768
    _DEFAULT_MAX_TOKENS = 16384
    _SUPPORTED_THINKING_FORMATS = frozenset({"chat-template", "qwen-chat-template"})
    _API_BASE_ENV = "HARBOR_PI_API_BASE"
    _API_KEY_ENV = "HARBOR_PI_API_KEY"
    _ENDPOINT_ENV_PATH = "/tmp/harbor-pi-endpoint.env"
    _CONTEXT_BUDGET_EXTENSION_PATH = "/tmp/harbor-pi-context-budget.js"
    _PROVIDER_CONFIG_RELATIVE_PATH = ".pi/agent/models.json"
    _SETTINGS_CONFIG_RELATIVE_PATH = ".pi/agent/settings.json"
    _AUTHENTICATION_FAILURE_MARKER = "HARBOR_PI_MODEL_AUTHENTICATION_ERROR"
    _CONTEXT_LENGTH_FAILURE_MARKER = "HARBOR_PI_CONTEXT_LENGTH_EXCEEDED"
    _VALIDATE_OUTPUT_SCRIPT = (
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
        if collect_rollout_details and api_base is None:
            raise ValueError(
                "collect_rollout_details requires api_base so Pi can stamp every "
                "model request with a trial correlation ID"
            )

        self._api_base = api_base
        self._api_key = api_key or "not-needed"
        self._collect_rollout_details = collect_rollout_details
        self.rollout_correlation_id = (
            uuid.uuid4().hex if collect_rollout_details else None
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
                            "contextWindow": self._model_limits.context_window,
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
            "enabled": True,
            "reserveTokens": self._model_limits.compaction_reserve_tokens,
            "keepRecentTokens": (self._model_limits.compaction_keep_recent_tokens),
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

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_alias or "/" not in self.model_alias:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, model_id = self.model_alias.split("/", 1)
        if not provider or not model_id:
            raise ValueError("Model name must be in the format provider/model_name")

        env: dict[str, str] = {}
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
            runtime_prelude = (
                "set -e; "
                f"trap 'rm -f {endpoint_path} {extension_path} {provider_path}' EXIT; "
                f"{runtime_prelude}"
                f". {endpoint_path}; {provider_command}; rm -f {endpoint_path}; "
            )
            cli_flags += f"--extension {extension_path} "
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    f"{runtime_prelude}"
                    f"pi --print --mode json --no-session "
                    f"{model_args}"
                    f"{cli_flags}"
                    f"{escaped_instruction} "
                    "2>&1 </dev/null | "
                    f"node -e {shlex.quote(self._VALIDATE_OUTPUT_SCRIPT)} | "
                    f"stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
                ),
                env=env,
            )
        except NonZeroAgentExitCodeError as error:
            error_lines = (
                (error.result.stderr or "").splitlines()
                if error.result is not None
                else []
            )
            if self._AUTHENTICATION_FAILURE_MARKER in error_lines:
                raise ModelAuthenticationError(
                    "Pi could not authenticate to its model endpoint "
                    "(possibly an expired or invalid capability URL)",
                    result=error.result,
                ) from error
            if self._CONTEXT_LENGTH_FAILURE_MARKER in error_lines:
                raise ContextLengthExceededError(
                    "Pi exceeded the model's served context window"
                ) from error
            raise

    def populate_context_post_run(self, context: AgentContext) -> None:
        if self.rollout_correlation_id:
            context.metadata = {
                **(context.metadata or {}),
                "rollout_correlation_id": self.rollout_correlation_id,
            }

        if self._collect_rollout_details:
            rollout_details = build_rollout_details_from_pairs(
                read_literal_pairs(self.logs_dir / LITERAL_LOG_FILENAME)
            )
            if rollout_details:
                context.rollout_details = rollout_details

        output_file = self.logs_dir / self._OUTPUT_FILENAME
        if not output_file.exists():
            return

        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_write_tokens = 0
        total_cost = 0.0

        skipped_records = 0
        if isinstance(output_file, UPath):
            lines_context = output_file.open(
                "rb", block_size=_REMOTE_LOG_READ_BLOCK_SIZE
            )
        else:
            lines_context = output_file.open("rb")

        with lines_context as lines:
            for raw_line in lines:
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    skipped_records += 1
                    continue
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") == "message_end":
                        message = event.get("message") or {}
                        if message.get("role") == "assistant":
                            usage = message.get("usage") or {}
                            total_input_tokens += usage.get("input", 0)
                            total_output_tokens += usage.get("output", 0)
                            total_cache_read_tokens += usage.get("cacheRead", 0)
                            total_cache_write_tokens += usage.get("cacheWrite", 0)
                            cost = usage.get("cost") or {}
                            total_cost += cost.get("total", 0.0)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    skipped_records += 1

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
