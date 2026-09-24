# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a complete local Harbor trial with a deterministic fake model."""

import argparse
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

from harbor.job import Job
from harbor.models.job.config import JobConfig


class ModelHandler(BaseHTTPRequestHandler):
    bash_commands: ClassVar[list[str]] = ["echo 'hello from qemu' > /workspace/answer.txt"]

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["tools"][0]["function"]["name"] == "Bash"
        completed = sum(message["role"] == "tool" for message in request["messages"])
        if completed >= len(self.bash_commands):
            message = {"role": "assistant", "content": "Done."}
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": json.dumps({"command": self.bash_commands[completed]}),
                        },
                    }
                ],
            }
        body = json.dumps(
            {"choices": [{"message": message}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        pass


async def main(
    jobs_dir: Path,
    task_path: Path,
    commands: list[str],
    environment_import_path: str,
    environment_kwargs: dict,
):
    ModelHandler.bash_commands = commands
    server = HTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = JobConfig.model_validate(
            {
                "job_name": "sandbox-smoke",
                "jobs_dir": str(jobs_dir),
                "n_concurrent_trials": 1,
                "tasks": [{"path": str(task_path)}],
                "environment": {
                    "import_path": environment_import_path,
                    "kwargs": environment_kwargs,
                },
                "agents": [
                    {
                        "import_path": "harbor_qemu.agent:BashAgent",
                        "model_name": "openai/fake",
                        "kwargs": {"base_url": f"http://127.0.0.1:{server.server_port}/v1"},
                    }
                ],
            }
        )
        result = await (await Job.create(config)).run()
        summary = result.model_dump(mode="json")["stats"]
        assert summary["n_errored_trials"] == 0, result.model_dump_json(indent=2)
        mean = next(iter(summary["evals"].values()))["metrics"][0]["mean"]
        assert mean == 1.0, result.model_dump_json(indent=2)
        print(f"Harbor smoke passed: reward={mean}")
    finally:
        server.shutdown()
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_or_source", help="guest bundle path, 'task-image', or 'shellsim'")
    parser.add_argument("jobs_dir", type=Path)
    parser.add_argument("--task-path", type=Path, default=Path(__file__).parent / "task")
    parser.add_argument("--command-file", type=Path)
    parser.add_argument("--commands-json", type=Path)
    parser.add_argument("--network-policy", choices=("require-offline-task", "deny"), default="require-offline-task")
    parser.add_argument("--acceleration", choices=("auto", "kvm", "tcg"), default="auto")
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--skopeo", type=Path)
    parser.add_argument("--qemu-assets-json", type=Path)
    args = parser.parse_args()
    if args.command_file is not None and args.commands_json is not None:
        parser.error("--command-file and --commands-json cannot be combined")
    if args.commands_json is not None:
        commands = json.loads(args.commands_json.read_text())
    elif args.command_file is not None:
        commands = [args.command_file.read_text()]
    else:
        commands = ModelHandler.bash_commands
    if args.bundle_or_source == "shellsim":
        environment_import_path = "harbor_qemu.backends.shellsim.environment:ShellSimEnvironment"
        environment_kwargs = {"network_policy": args.network_policy}
    elif args.bundle_or_source == "task-image":
        environment_import_path = "harbor_qemu.backends.qemu.environment:QemuEnvironment"
        if args.image_cache is None or args.skopeo is None or args.qemu_assets_json is None:
            parser.error("task-image requires --image-cache, --skopeo, and --qemu-assets-json")
        environment_kwargs = {
            "image_cache": str(args.image_cache),
            "skopeo": str(args.skopeo),
            "qemu_assets": json.loads(args.qemu_assets_json.read_text()),
            "network_policy": args.network_policy,
            "acceleration": args.acceleration,
        }
    else:
        environment_import_path = "harbor_qemu.backends.qemu.environment:QemuEnvironment"
        environment_kwargs = {
            "guest_bundle": args.bundle_or_source,
            "network_policy": args.network_policy,
            "acceleration": args.acceleration,
        }
    asyncio.run(main(args.jobs_dir, args.task_path, commands, environment_import_path, environment_kwargs))
