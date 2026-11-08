# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise native Daytona environment startup and Pi installation on CPUs."""

import argparse
import asyncio
import logging
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from harbor.host_capabilities import DAYTONA_API_URL, DaytonaConnection, HostCapabilities
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from experiments.post_training.long_loop.eval_runtime.artifacts import publish
from experiments.post_training.long_loop.eval_runtime.environment_inventory import restored_environments
from experiments.post_training.long_loop.eval_runtime.profile import (
    DATASET,
    DESTINATION,
    MODEL,
    REVISION,
    SKIPPED_BENCHMARKS,
    counterpart_environment,
    pi_agent_config,
)


async def main(
    smoke: bool, credential_env: str, data_uri: str, benchmarks: list[str], report_name: str, output_prefix: str
):
    logging.basicConfig(level=logging.INFO)
    host_capabilities = HostCapabilities(
        daytona=DaytonaConnection(
            api_key=os.environ[credential_env],
            api_url=DAYTONA_API_URL,
            target=os.environ["DAYTONA_TARGET"],
        ),
        protected_values=frozenset((os.environ[credential_env], os.environ["TOGETHER_API_KEY"])),
    )
    receipts = []
    failures = []
    semaphore = asyncio.Semaphore(4)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = [row for row in restored_environments(root / "tasks") if row.benchmark not in SKIPPED_BENCHMARKS]
        unique = {row.image_source_key: row for row in reversed(rows)}
        selected = {row.benchmark: row.image_source_key for row in reversed(rows)}
        keys = list(selected.values()) if smoke else list(unique)
        if benchmarks:
            keys = [key for key in keys if unique[key].benchmark in benchmarks]

        async def probe(key):
            row = unique[key]
            async with semaphore:
                identity = "mini840-preflight-" + uuid4().hex[:16]
                config = EnvironmentConfig(
                    type="daytona",
                    override_storage_mb=row.storage_override_mb,
                )
                started = time.monotonic()
                try:
                    trial = await Trial.create(
                        TrialConfig(
                            task=TaskConfig(path=Path(row.context).parent),
                            trial_name=identity,
                            trials_dir=root / "preflight-trials",
                            environment=config,
                            agent=AgentConfig.model_validate(
                                {
                                    **pi_agent_config(),
                                    "model_name": "hosted_vllm/" + MODEL,
                                    "mcp_servers": row.agent_mcp_servers,
                                    "kwargs": {
                                        **pi_agent_config()["kwargs"],
                                        "api_base": "http://unused-during-setup.invalid/v1",
                                        "api_key": "EMPTY",
                                    },
                                }
                            ),
                        ),
                        host_capabilities=host_capabilities,
                        guest_environment={
                            **counterpart_environment(),
                            "OPENAI_API_KEY": "EMPTY",
                            "OPENAI_BASE_URL": "http://unused-during-setup.invalid/v1",
                        },
                    )
                    environment = trial.agent_environment
                    agent = trial.agent
                    async with asyncio.TaskGroup() as service_tasks:
                        try:
                            async with asyncio.timeout(1800):
                                await environment.start(force_build=False, service_tasks=service_tasks)
                            async with asyncio.timeout(360):
                                await agent.setup(environment)
                            result = await environment.exec(command="uname -m", timeout_sec=30)
                            if result.return_code != 0:
                                raise RuntimeError("Environment execution failed")
                            receipts.append(
                                {
                                    **asdict(row),
                                    "architecture": result.stdout.strip(),
                                    "elapsed": time.monotonic() - started,
                                }
                            )
                            logging.info("Validated %s (%s), %d/%d", row.benchmark, key, len(receipts), len(keys))
                        finally:
                            async with asyncio.timeout(60):
                                await environment.stop(delete=True)
                except Exception as error:
                    failures.append(
                        {
                            "image_source_key": key,
                            "benchmark": row.benchmark,
                            "error_type": type(error).__name__,
                            "message": str(error).replace(os.environ[credential_env], "[redacted]")[-16000:],
                        }
                    )
                    logging.error("Preflight failed: %s %s", row.benchmark, type(error).__name__)
                finally:
                    publish(
                        "native-environment-preflight-progress.json",
                        {"ready": receipts, "failures": failures},
                        output_prefix,
                    )

        await asyncio.gather(*(probe(key) for key in keys))
        report = {
            "dataset": DATASET,
            "revision": REVISION,
            "credential_env": credential_env,
            "data_uri": data_uri,
            "scope": "diagnostic" if benchmarks else ("smoke" if smoke else "full"),
            "provisioning": "native_harbor_daytona",
            "excluded_benchmarks": SKIPPED_BENCHMARKS,
            "named_snapshots_required": False,
            "required_source_keys": keys,
            "expected_source_keys": len(keys),
            "validated_source_keys": [r["image_source_key"] for r in receipts],
            "environment_overrides": {
                key: {"override_storage_mb": unique[key].storage_override_mb}
                for key in keys
                if unique[key].storage_override_mb is not None
            },
            "agent_mcp_servers": {key: unique[key].agent_mcp_servers for key in keys if unique[key].agent_mcp_servers},
            "verifier_env": {
                key: {
                    "OPENAI_API_KEY": "${JUDGE_API_KEY}",
                    "OPENAI_BASE_URL": "${JUDGE_BASE_URL}",
                    "MODEL_NAME": "MiniMaxAI/MiniMax-M3-MXFP8",
                }
                for key in keys
                if unique[key].benchmark == "bixbench-pi"
            },
            "receipts": receipts,
            "failures": failures,
        }
        publish(
            report_name
            or ("diagnostic" if benchmarks else ("smoke" if smoke else "full")) + "-native-environments.json",
            report,
            output_prefix,
        )
        if failures:
            raise RuntimeError(f"{len(failures)} environment preflights failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-name", default="")
    parser.add_argument("--output-prefix", default=DESTINATION)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--credential-env", required=True)
    parser.add_argument("--data-uri", required=True)
    args = parser.parse_args()
    asyncio.run(
        main(args.smoke, args.credential_env, args.data_uri, args.benchmark, args.report_name, args.output_prefix)
    )
