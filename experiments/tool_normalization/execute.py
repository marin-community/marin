# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute one candidate's native calls in a fresh, isolated Linux workspace."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy

from experiments.tool_normalization.native_tools import observations


@dataclass(frozen=True)
class ExecutionConfig:
    image: str
    workdir: str
    recorder_python: str
    cluster: str
    scheduling_timeout: int
    execution_timeout: int
    memory_mb: int
    cpus: int
    output_limit_bytes: int


async def execute(calls: list[dict], config: ExecutionConfig, directory: Path) -> list[dict]:
    """Return observed tool messages; retain receipts even when a replay fails."""
    if "@sha256:" not in config.image:
        raise ValueError("Execution image must be digest-pinned")
    directory.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).with_name("native_tools.py")
    call_file = directory / "calls.json"
    call_file.write_text(json.dumps(calls, ensure_ascii=False))
    manifest = {
        "config": asdict(config),
        "calls_sha256": hashlib.sha256(call_file.read_bytes()).hexdigest(),
        "executor_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
        "network": "deny",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    factory = IrisMachineFactory(
        cluster=config.cluster,
        scheduling_timeout=config.scheduling_timeout,
        job_ttl=config.scheduling_timeout + config.execution_timeout + 180,
    )
    machine = await factory.create(
        MachineSpec(
            source=RegistryImage(config.image),
            workdir=config.workdir,
            network=NetworkPolicy.DENY,
            cpus=config.cpus,
            memory_mb=config.memory_mb,
        )
    )
    try:
        await machine.upload(code, "/tmp/native_tools.py")
        await machine.upload(call_file, "/tmp/calls.json")
        result = await machine.run(
            Command(
                (
                    config.recorder_python,
                    "/tmp/native_tools.py",
                    "--binary",
                    "/usr/local/bin/opencode",
                    "--calls",
                    "/tmp/calls.json",
                    "--workspace",
                    config.workdir,
                    "--evidence",
                    "/tmp/native-evidence",
                    "--timeout",
                    str(config.execution_timeout),
                ),
                timeout=config.execution_timeout + 30,
                output_limit_bytes=config.output_limit_bytes,
            )
        )
        (directory / "stdout.json").write_bytes(result.stdout)
        (directory / "stderr.txt").write_bytes(result.stderr)
        receipt = {
            "exit_code": result.exit_code,
            "reason": result.reason,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
        }
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
        if result.exit_code != 0 or result.stdout_truncated or result.stderr_truncated:
            raise ValueError("Incomplete execution; inspect saved receipt")
        report = json.loads(result.stdout)
        if report["exit_code"] or report["dispatch_errors"]:
            raise ValueError("Native dispatch failed")
        messages = observations(calls, report["tool_events"])
        (directory / "observations.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2))
        return messages
    finally:
        await machine.close()
