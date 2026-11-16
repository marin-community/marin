# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Request, shard naming, and completion metadata for Hero exports."""

import re
from dataclasses import asdict
from datetime import UTC, datetime

from levanter.compat.hf_export import HFShardRecord, HFShardResume, write_export_json
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest

MANIFEST_FILENAME = "export-manifest.json"
REQUEST_FILENAME = "export-request.json"
INDEX_FILENAME = "model.safetensors.index.json"
EXPORT_VERSION = 3
EXECUTION_CONFIG_KEYS = frozenset(
    {
        "read_host_budget_bytes",
        "request_concurrency",
        "export_host_budget_bytes",
        "max_concurrent_shards",
        "expert_axis_size",
        "replica_axis_size",
    }
)
EXPERT_BANK = re.compile(r"^(.*\.mlp\.experts)\.(gate_proj|up_proj|down_proj)\.weight$")


def artifact_config(config: dict) -> dict:
    """Keep output-defining inputs in an encoded export configuration."""
    return {key: value for key, value in config.items() if key not in EXECUTION_CONFIG_KEYS}


def execution_config(config: dict) -> dict:
    """Extract resource and concurrency settings for completion metadata."""
    return {key: value for key, value in config.items() if key in EXECUTION_CONFIG_KEYS}


def bf16_hf_config(model: GrugModelConfig) -> dict:
    config = model.to_hf_config(model.vocab_size).to_dict()
    config["dtype"] = "bfloat16"
    return config


def split_expert_names(name: str, num_experts: int) -> tuple[str, ...]:
    match = EXPERT_BANK.fullmatch(name)
    if match is None:
        return (name,)
    return tuple(f"{match[1]}.{i}.{match[2]}.weight" for i in range(num_experts))


def prepare_export(destination: str, request: dict) -> HFShardResume:
    """Write or verify the artifact request, rejecting completed or unrelated output."""
    if re.fullmatch(r"[0-9a-f]{40}", request["config"]["source_revision"]) is None:
        raise ValueError("source_revision must be a 40-character lowercase hexadecimal Marin commit")
    root = StoragePath(destination)
    if (root / MANIFEST_FILENAME).exists():
        raise FileExistsError(f"Export already complete: {root}")
    if not (root / REQUEST_FILENAME).exists() and root.exists() and root.ls():
        raise FileExistsError(f"Export requires a fresh destination: {root}")
    write_export_json(root / REQUEST_FILENAME, request)
    return HFShardResume(root, digest(request))


def write_export_metadata(
    resume: HFShardResume, request: dict, records: list[HFShardRecord], payload_size: int, execution: dict
) -> None:
    """Commit the config and index, then publish the completion manifest last."""
    write_export_json(resume.root / "config.json", request["hf_config"])
    write_export_json(
        resume.root / INDEX_FILENAME,
        {
            "metadata": {"total_size": payload_size},
            "weight_map": {name: record.filename for record in records for name in record.tensor_names},
        },
    )
    write_export_json(
        resume.root / MANIFEST_FILENAME,
        {
            "export_id": resume.export_id,
            "created_at": datetime.now(UTC).isoformat(),
            "request": request,
            "shards": [asdict(record) for record in records],
            **execution,
        },
    )
