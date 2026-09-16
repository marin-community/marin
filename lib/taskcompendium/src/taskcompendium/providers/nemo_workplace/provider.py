# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor provider adapter for the pinned NeMo Workplace Assistant source."""

import hashlib
import json
from pathlib import Path
from typing import Any

import msgspec
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from taskcompendium.models import ActionInterface, GradingResult, Outcome
from taskcompendium.providers.nemo_workplace.tools import equivalent_state, get_tools, source_state

SEED_SHA256 = "abcfd3d4727c66b6dfc145b59f720b819ac9de1b65df285cd30bc80bc10b3b8b"
ADAPTER = "nemo_workplace_v1"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _seed_digest() -> str:
    root = Path(__file__).parent / "vendor/csv_data"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.csv")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


class NemoWorkplaceEnvironment(BaseEnvironment):
    """One isolated original Workplace Assistant tool environment per Harbor trial."""

    def __init__(self, *args, interface: dict[str, Any], seed_sha256: str, **kwargs: Any) -> None:
        self.interface = msgspec.convert(interface, type=ActionInterface)
        if self.interface.name != "workplace_assistant" or self.interface.version != "nemo-gym-v1":
            raise ValueError("Nemo Workplace adapter received an incompatible action interface")
        if seed_sha256 != self.interface.seed_sha256 or seed_sha256 != SEED_SHA256 or _seed_digest() != SEED_SHA256:
            raise ValueError("Nemo Workplace seed does not match its pinned digest")
        self.tool_env = get_tools()
        self.trace: list[dict[str, str]] = []
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "taskcompendium-nemo-workplace"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        # Workplace state is provided by the pinned package, so its lowering
        # has no filesystem payload and may omit an empty environment directory.
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool) -> None:
        pass

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        if command == "pwd":
            return ExecResult(stdout="/app\n", stderr="", return_code=0)
        raise ValueError("Workplace provider does not expose shell execution")

    async def empty_dirs(self, dirs, *, chmod: bool = True) -> None:
        if not set(map(str, dirs)).issubset({"/logs/agent", "/logs/verifier", "/logs/artifacts", "/tests"}):
            raise ValueError("Workplace provider does not expose filesystem operations")

    async def upload_file(self, source_path, target_path) -> None:
        raise ValueError("Workplace provider does not expose filesystem uploads")

    async def upload_dir(self, source_dir, target_dir) -> None:
        raise ValueError("Workplace provider does not expose filesystem uploads")

    async def download_file(self, source_path, target_path) -> None:
        raise ValueError("Workplace provider does not expose filesystem downloads")

    async def download_dir(self, source_dir, target_dir) -> None:
        if source_dir not in {"/logs/agent", "/logs/artifacts"}:
            raise ValueError("Workplace provider does not expose filesystem downloads")

    async def native_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description"),
                    "parameters": schema["parameters"],
                    "strict": schema.get("strict", False),
                },
            }
            for schema in self.tool_env["schemas"]
        ]

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str:
        try:
            payload = json.loads(arguments)
            if not isinstance(payload, dict):
                raise ValueError("Tool arguments must be a JSON object")
            output = self.tool_env["functions"][name](**payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            output = f"Error executing tool '{name}': {error}"
        encoded = json.dumps({"output": output}, default=_json_default, separators=(",", ":"))
        self.trace.append({"call_id": call_id, "name": name, "arguments": arguments, "output": encoded})
        return encoded

    async def authoritative_state(self) -> dict[str, Any]:
        """Return the mutable provider state; this never consumes the agent transcript."""
        return source_state(self.tool_env)

    async def grade_provider_state(self, adapter: str, parameters: dict[str, Any]) -> GradingResult:
        if adapter != ADAPTER:
            return GradingResult(Outcome.INVALID_TASK, None, {"error": "Unexpected provider-state adapter"})
        gold = parameters.get("ground_truth")
        if not isinstance(gold, list) or not all(
            isinstance(action, dict) and isinstance(action.get("name"), str) and isinstance(action.get("arguments"), str)
            for action in gold
        ):
            return GradingResult(Outcome.INVALID_TASK, None, {"error": "Invalid pinned Workplace ground truth"})
        try:
            expected = source_state(_execute_source_actions(gold))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return GradingResult(
                Outcome.INVALID_TASK,
                None,
                {"error": f"Invalid pinned Workplace ground truth: {error}"},
            )
        actual = await self.authoritative_state()
        matched = equivalent_state(actual, expected)
        return GradingResult(Outcome.GRADED, float(matched), {"source_adapter": ADAPTER})


def _execute_source_actions(actions: list[dict[str, str]]) -> dict[str, Any]:
    """Recreate the upstream verifier's fresh-state action replay without using agent output."""
    environment = get_tools()
    for action in actions:
        arguments = json.loads(action["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("Workplace verifier arguments must decode to an object")
        environment["functions"][action["name"]](**arguments)
    return environment
