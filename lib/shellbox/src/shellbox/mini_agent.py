# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side native mini-swe-agent controller with an offline QEMU guest."""

import asyncio
import errno
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from harbor.agents.base import BaseAgent
from harbor.agents.installed.mini_swe_agent import convert_mini_swe_agent_to_atif
from harbor.agents.utils import get_api_key_var_names_from_model_name
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from shellbox.backends.qemu.environment import QemuEnvironment
from shellbox.machine import HARBOR_EXEC_OUTPUT_LIMIT_BYTES, Command, ExitReason

MINI_VERSION = "2.1.0"
CONTROLLER_STOP_TIMEOUT = 5
BRIDGE_REQUEST_LIMIT = 1024 * 1024


def _native_exception_output(error: Exception, output: str = "") -> dict:
    return {
        "output": output,
        "returncode": -1,
        "exception_info": f"An error occurred while executing the command: {error}",
        "extra": {"exception_type": type(error).__name__, "exception": str(error)},
    }


class NativeMiniAgent(BaseAgent):
    """Run one native controller per Harbor trial."""

    SUPPORTS_ATIF = True

    def __init__(self, *args, config_specs: list[str] | None = None, api_base: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_specs = config_specs
        self.api_base = api_base

    @staticmethod
    def name() -> str:
        return "marin-native-mini-qemu"

    def version(self) -> str:
        return MINI_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        if importlib.metadata.version("mini-swe-agent") != MINI_VERSION:
            raise ValueError(f"NativeMiniAgent requires mini-swe-agent {MINI_VERSION}")
        if not isinstance(environment, QemuEnvironment) or environment.machine is None:
            raise TypeError("NativeMiniAgent requires a running QemuEnvironment")
        if self.mcp_servers or self.skills_dir:
            raise ValueError("NativeMiniAgent does not support MCP servers or skills")
        if environment.default_user not in (None, "root", 0):
            raise ValueError("NativeMiniAgent supports only the adapted root user")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self.setup(environment)
        assert isinstance(environment, QemuEnvironment)
        assert environment.machine is not None
        machine = environment.machine
        model = self.model_alias
        if not model or "/" not in model:
            raise ValueError("NativeMiniAgent requires model_alias in provider/model format")
        provider_env = {"MSWEA_CONFIGURED": "true", "MSWEA_COST_TRACKING": "ignore_errors"}
        if "MSWEA_API_KEY" not in os.environ:
            if model.startswith("hosted_vllm/"):
                provider_env["MSWEA_API_KEY"] = "EMPTY"
            else:
                for key in get_api_key_var_names_from_model_name(model):
                    if key not in os.environ:
                        raise ValueError(f"Unset API variable {key}; set MSWEA_API_KEY as a fallback")
        if self.api_base is not None:
            provider_env["OPENAI_API_BASE"] = self.api_base
            if model.startswith("hosted_vllm/"):
                provider_env["HOSTED_VLLM_API_BASE"] = self.api_base
        guest_env = machine.image_env | machine.spec.env | (environment._merge_env(None) or {})
        environment_result = await machine.run(Command(("env", "-0"), env=guest_env))
        if environment_result.exit_code != 0:
            raise RuntimeError("Cannot read guest environment metadata")
        guest_env = dict(
            item.decode(errors="replace").split("=", 1) for item in environment_result.stdout.split(b"\0") if item
        )
        platform_result = await machine.run(
            Command(
                (
                    "/harbor/busybox",
                    "sh",
                    "-c",
                    'for flag in -s -n -r -v -m; do /harbor/busybox uname "$flag" || exit; printf "\\0"; done',
                )
            )
        )
        if platform_result.exit_code != 0:
            raise RuntimeError("Cannot read guest platform metadata")
        platform = dict(
            zip(
                ("system", "node", "release", "version", "machine"),
                (item.decode(errors="replace").rstrip("\n") for item in platform_result.stdout.split(b"\0")[:-1]),
                strict=True,
            )
        )

        handlers: set[asyncio.Task] = set()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            handlers.add(task)
            try:
                request = json.loads(await reader.readline())
                env = guest_env | request["env"]
                if request["operation"] == "template":
                    response = platform | env
                elif request["operation"] == "execute":
                    command = request["command"]
                    timeout = request["timeout"]
                    base_cwd = machine.spec.workdir or machine.image_workdir
                    reported_cwd = request["cwd"] or base_cwd
                    cwd = str(PurePosixPath(base_cwd) / reported_cwd)
                    preflight = await machine.run(
                        Command(
                            (
                                "/harbor/busybox",
                                "sh",
                                "-c",
                                'if [ ! -e "$1" ]; then echo cwd_missing; '
                                'elif [ ! -d "$1" ]; then echo cwd_file; '
                                "elif [ ! -e /bin/sh ]; then echo shell_missing; "
                                "elif [ ! -x /bin/sh ]; then echo shell_permission; fi",
                                "native-mini-preflight",
                                cwd,
                            ),
                            cwd="/",
                        )
                    )
                    if preflight.exit_code != 0:
                        raise RuntimeError("QEMU native command preflight failed")
                    failure = preflight.stdout.strip()
                    if failure == b"cwd_missing":
                        response = _native_exception_output(
                            FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), reported_cwd)
                        )
                    elif failure == b"cwd_file":
                        response = _native_exception_output(
                            NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), reported_cwd)
                        )
                    elif failure == b"shell_missing":
                        response = _native_exception_output(
                            FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "/bin/sh")
                        )
                    elif failure == b"shell_permission":
                        response = _native_exception_output(
                            PermissionError(errno.EACCES, os.strerror(errno.EACCES), "/bin/sh")
                        )
                    elif failure:
                        raise RuntimeError("Unknown QEMU native command preflight result")
                    else:
                        result = await machine.run(
                            Command(
                                ("/bin/sh", "-c", "exec 2>&1\n" + command),
                                cwd=cwd,
                                env=env,
                                timeout=timeout,
                                output_limit_bytes=HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
                            )
                        )
                        if result.stdout_truncated or result.stderr_truncated:
                            raise RuntimeError("Native mini command output exceeds the transport limit")
                        output = (result.stdout + result.stderr).decode(errors="replace")
                        if result.reason is ExitReason.TIMED_OUT:
                            response = _native_exception_output(
                                subprocess.TimeoutExpired(command, timeout, output=output), output
                            )
                        else:
                            response = {"output": output, "returncode": result.exit_code, "exception_info": ""}
                else:
                    raise ValueError("Unknown native mini environment operation")
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
            except Exception as error:
                writer.write(json.dumps({"error": f"{type(error).__name__}: {error}"}).encode() + b"\n")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.remove(task)

        with tempfile.TemporaryDirectory(prefix="marin-native-mini-") as directory:
            root = Path(directory)
            socket_path = root / "guest.sock"
            trajectory = root / "mini-swe-agent.trajectory.json"
            task_config = root / "task.yaml"
            task_config.write_text(json.dumps({"run": {"task": instruction}}))
            arguments = [
                sys.executable,
                "-m",
                "minisweagent.run.mini",
                "--yolo",
                "--exit-immediately",
                "--cost-limit=0",
                f"--model={model}",
                f"--output={trajectory}",
                "--environment-class=shellbox.mini_environment.MiniQemuEnvironment",
            ]
            for spec in (
                self.config_specs
                if self.config_specs is not None
                else [os.getenv("MSWEA_MINI_CONFIG_PATH", "mini.yaml")]
            ):
                arguments.extend(("-c", spec))
            arguments.extend(("-c", str(task_config), "-c", f"environment.socket_path={socket_path}"))
            child_env = (
                os.environ
                | provider_env
                | {
                    "MSWEA_GLOBAL_CONFIG_DIR": str(root / "config"),
                }
            )
            process = None
            controller_log = root / "controller.log"
            async with await asyncio.start_unix_server(handle, socket_path, limit=BRIDGE_REQUEST_LIMIT) as server:
                try:
                    with controller_log.open("wb") as output:
                        process = await asyncio.create_subprocess_exec(
                            *arguments,
                            env=child_env,
                            stdin=asyncio.subprocess.DEVNULL,
                            stdout=output,
                            stderr=asyncio.subprocess.STDOUT,
                            start_new_session=True,
                        )
                        await process.wait()
                finally:
                    if process is not None and process.returncode is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(process.wait(), CONTROLLER_STOP_TIMEOUT)
                        except TimeoutError:
                            os.killpg(process.pid, signal.SIGKILL)
                            await process.wait()
                    server.close()
                    for handler in tuple(handlers):
                        handler.cancel()
                    await asyncio.gather(*handlers, return_exceptions=True)
                    self.logs_dir.mkdir(parents=True, exist_ok=True)
                    if controller_log.exists():
                        (self.logs_dir / "mini-swe-agent.txt").write_bytes(controller_log.read_bytes())
                    if trajectory.exists():
                        data = trajectory.read_bytes()
                        (self.logs_dir / trajectory.name).write_bytes(data)
            assert process is not None
            if process.returncode != 0:
                raise RuntimeError(f"Native mini controller failed with exit code {process.returncode}")
            native = json.loads(trajectory.read_bytes())
            atif = convert_mini_swe_agent_to_atif(native, environment.session_id)
            (self.logs_dir / "trajectory.json").write_text(json.dumps(atif.to_json_dict(), indent=2))
            context.cost_usd = native["info"]["model_stats"]["instance_cost"]
            context.metadata = (context.metadata or {}) | {
                "runtime": "adapted-host-native-mini-qemu",
                "exit_status": native["info"]["exit_status"],
            }
            # Only assistant messages have provider responses; provider usage is optional.
            usage = [m["extra"]["response"].get("usage") or {} for m in native["messages"] if m["role"] == "assistant"]
            context.n_input_tokens = sum(item.get("prompt_tokens", 0) for item in usage)
            context.n_output_tokens = sum(item.get("completion_tokens", 0) for item in usage)
            context.n_cache_tokens = sum(
                (item.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for item in usage
            )
