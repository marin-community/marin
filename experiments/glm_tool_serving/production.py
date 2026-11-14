# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the prepared conversion shards without a laptop or port-forward."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

DESTINATION = "s3://marin-us-east-02a/marin/users/benfeuer/tool-normalization/h100-production-20261103"


def main() -> None:
    configure_coreweave_s3()
    data = Path(__file__).resolve().parent / "production_data"
    root = Path("/tmp/tool-conversion-production")
    root.mkdir(exist_ok=False)
    configs = []
    for replica in (1, 2):
        shard = root / f"r{replica}"
        shard.mkdir()
        cfg = json.loads((data / f"config-{replica}.json").read_text())
        for key in ("serving_tokenizer", "sft_tokenizer"):
            cfg[key] = str(data / cfg[key])
        for evidence in cfg["adapter_evidence"]:
            evidence["path"] = str(data / evidence["path"])
        model = cfg["model"]
        for key, filename in (
            ("api_key_file", "api-key.txt"),
            ("capacity_file", "capacity.json"),
            ("ledger", "ledger.sqlite3"),
        ):
            model[key] = str(shard / filename)
        (shard / "api-key.txt").write_text("unused-local-cluster-endpoint")
        (shard / "capacity.json").write_text(
            json.dumps(
                {
                    "generation": model["generation"],
                    "conversion_slots": 16,
                    "science_slots": 0,
                    "expires_at_unix": time.time() + 36 * 3600,
                }
            )
        )
        (shard / "config.json").write_text(json.dumps(cfg, indent=2))
        rows = [json.loads(line) for line in (data / f"manifest-{replica}.json").read_text().splitlines()]
        for row in rows:
            row["source_file"] = str(data / row["source_file"])
        (shard / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        configs.append(cfg)

    # Both engines must be idle in consecutive observations before admitting new work.
    idle_samples = 0
    deadline = time.monotonic() + 3600
    while idle_samples < 2:
        if time.monotonic() > deadline:
            raise TimeoutError("Prior benchmark requests did not drain within one hour")
        loads = []
        for cfg in configs:
            endpoint = cfg["model"]["base_url"].removesuffix("/v1")
            with urllib.request.urlopen(endpoint + "/metrics", timeout=30) as response:
                metrics = response.read().decode()
            values = [
                float(line.rsplit(" ", 1)[1])
                for line in metrics.splitlines()
                if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
            ]
            if len(values) < 2:
                raise ValueError("Missing direct-engine admission metrics")
            loads.append(sum(values))
        print(json.dumps({"phase": "draining", "engine_requests": loads}), flush=True)
        idle_samples = idle_samples + 1 if all(value == 0 for value in loads) else 0
        if idle_samples < 2:
            time.sleep(10)

    # A storage preflight ensures completed outputs have a durable destination.
    with fsspec.open(DESTINATION + "/launch.json", "w") as stream:
        json.dump({"started_at_unix": time.time(), "workers": 32, "configs": configs}, stream)
    processes = []
    logs = []
    for replica in (1, 2):
        shard = root / f"r{replica}"
        log = (shard / "driver.log").open("w")
        logs.append(log)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.tool_normalization.driver",
                    "--config",
                    str(shard / "config.json"),
                    "--manifest",
                    str(shard / "manifest.jsonl"),
                    "--root",
                    str(shard / "run"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
    print("PRODUCTION_STARTED workers=32 replicas=2", flush=True)
    archive = Path("/tmp/tool-conversion-production.tar.gz")
    while True:
        codes = [process.poll() for process in processes]
        final = all(code is not None for code in codes)
        with tarfile.open(archive, "w:gz") as tar:
            # Only finished record directories are immutable during dispatch.
            # Avoid archiving live SQLite journals or partially written receipts.
            for shard in sorted(root.glob("r*")):
                for name in ("config.json", "manifest.jsonl", "driver.log"):
                    tar.add(shard / name, arcname=f"production/{shard.name}/{name}")
                for result in sorted((shard / "run/records").glob("*/result.json")):
                    tar.add(result.parent, arcname=f"production/{shard.name}/run/records/{result.parent.name}")
                if final:
                    for name in ("queue.sqlite3", "summary.json", "timing.json", "config.json", "sources.jsonl"):
                        path = shard / "run" / name
                        if path.exists():
                            tar.add(path, arcname=f"production/{shard.name}/run/{name}")
            tar.add(data, arcname="inputs")
        with archive.open("rb") as source, fsspec.open(DESTINATION + "/latest.tar.gz", "wb") as target:
            shutil.copyfileobj(source, target)
        print(
            json.dumps(
                {
                    "phase": "complete" if final else "running",
                    "exit_codes": codes,
                    "backup_bytes": archive.stat().st_size,
                    "destination": DESTINATION,
                }
            ),
            flush=True,
        )
        if final:
            for log in logs:
                log.close()
            output = Path(os.environ["IRIS_OUTPUT_DIR"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "result.json").write_text(json.dumps({"exit_codes": codes, "destination": DESTINATION}))
            if any(code != 0 for code in codes):
                raise RuntimeError(f"Conversion driver failed: {codes}")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
