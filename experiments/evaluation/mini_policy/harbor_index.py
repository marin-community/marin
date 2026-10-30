# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve canonical Harbor task identities without downloading task payloads."""

import asyncio
import json
import os
from pathlib import Path

import yaml
from harbor.models.job.config import DatasetConfig
from harbor.registry.client import RegistryClientFactory
from harbor.registry.client.package import PackageDatasetClient
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.identities import document_identity

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled"
DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/harbor-source-index-20261030"


async def main():
    cohort = json.loads(ROOT.joinpath("experiments/evaluation/mini_policy/cohort.json").read_text())
    benchmarks = sorted({row["benchmark"] for row in cohort["cells"]} - set(cohort["predictor_exclusions"]))
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    summary = []
    for benchmark in benchmarks:
        path = CAMPAIGN / "harbor-configs" / f"{benchmark}.yaml"
        if not path.exists():
            continue
        config = yaml.safe_load(path.read_text())
        resolved = []
        for raw in config["datasets"]:
            dataset = DatasetConfig.model_validate(raw)
            if dataset.is_package():
                client = PackageDatasetClient()
                reference = f'{dataset.name}@{dataset.ref or "latest"}'
            elif dataset.is_registry():
                client = RegistryClientFactory.create(
                    registry_url=dataset.registry_url, registry_path=dataset.registry_path
                )
                reference = f"{dataset.name}@{dataset.version}" if dataset.version else dataset.name
            else:
                raise ValueError(f"Unsupported discovery source: {benchmark}")
            metadata = await client.get_dataset_metadata(reference)
            rows = [
                {
                    "task_id": document_identity(benchmark, {}, task.get_name()),
                    "task_name": task.get_name(),
                    "identity": task.model_dump(mode="json"),
                }
                for task in metadata.task_ids
            ]
            resolved.append({"dataset": raw, "metadata": metadata.model_dump(mode="json"), "rows": rows})
        document = {
            "benchmark": benchmark,
            "config_path": str(path),
            "config_text": path.read_text(),
            "resolved": resolved,
        }
        data = json.dumps(document, indent=2).encode()
        (output / (benchmark + ".json")).write_bytes(data)
        fs, key = filesystem_for(DEST + "/" + benchmark + ".json")
        fs.pipe_file(key, data)
        summary.append({"benchmark": benchmark, "tasks": sum(len(part["rows"]) for part in resolved)})
        print(json.dumps(summary[-1]), flush=True)
    data = json.dumps(summary, indent=2).encode()
    (output / "summary.json").write_bytes(data)
    fs, key = filesystem_for(DEST + "/summary.json")
    fs.pipe_file(key, data)


if __name__ == "__main__":
    asyncio.run(main())
