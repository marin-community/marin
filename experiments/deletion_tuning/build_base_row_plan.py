# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Micro210 base-row plan for the Snowball Deletion Tuning task.

The base row measures the unmodified Snowball 262K base under the same Micro210
contract as every variant row. The plan copies the round-1 SFT746 eval template
verbatim — judge cache, coordinator, evaluation execution, and the frozen
standard-evaluation digest — and swaps only the model binding (the staged
in-region HF copy of the base), the output prefix, and the version.

Usage (after the staging job writes `model-binding-inputs.json`):

    uv run python -m experiments.deletion_tuning.build_base_row_plan \
        --binding s3://.../base-hf/2026.11.18.1/model-binding-inputs.json \
        --output experiments/deletion_tuning/plans/base-row-full.json
"""

import argparse
import hashlib
import json

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

TEMPLATE_PATH = "experiments/deletion_tuning/plans/template-micro210-r1-sft746.json"
MODEL_KEY = "snowball-deletion-tuning-base"
VERSION = "2026.11.18.1"
OUTPUT_PREFIX = (
    "s3://marin-us-east-02a/tmp/ttl=30d/benfeuer/snowball-deletion-tuning/micro210-base-20261118/parent-base-full"
)
POLICY_REPLICAS = 16
"""The Exp3 .61 binding's serving geometry: 16 engines, 128 H100 (DP8/EP8/TP1)."""

# The binding packet this plan must match (verified SHA-256 e3078d7a...).
BINDING61_URI = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/evidence/micro210-binding61.json"
BINDING61_SHA256 = "e3078d7a11f3a11a2bbb4c42849037d29bdb51a00cb8e037580230067ad96109"


def build(binding: dict, binding61: dict) -> dict:
    template = json.loads(StoragePath(TEMPLATE_PATH).read_text())
    r1_model = template["model"]
    sidecar_names = sorted(binding["sidecar_sha256"])
    model = {
        "key": MODEL_KEY,
        "hf_uri": binding["hf_uri"],
        "tokenizer_revision": r1_model["tokenizer_revision"],
        "evidence": [
            {
                "uri": binding["hf_uri"] + "/model-binding-inputs.json",
                "sha256": binding["self_sha256"],
            }
        ],
        "sidecars": [
            {"uri": binding["hf_uri"] + "/" + name, "sha256": binding["sidecar_sha256"][name]} for name in sidecar_names
        ],
        "shards": [
            {
                "path": shard["path"],
                "bytes": shard["bytes"],
                "header_sha256": shard["header_sha256"],
            }
            for shard in binding["shards"]
        ],
        "provenance": {
            "repo": binding["repo"],
            "revision": binding["revision"],
            "source_receipt": binding["source_receipt"],
            "source_model_ref": binding["source_model_ref"],
            "copy_manifest": binding["copy_manifest"],
        },
    }
    plan = dict(template)
    plan["model"] = model
    plan["version"] = VERSION
    plan["output_prefix"] = OUTPUT_PREFIX
    plan["temporary_prefix"] = template["temporary_prefix"]
    # The Exp3 .61 binding's serving geometry: 16 engines over 128 H100.
    plan["policy_replicas"] = POLICY_REPLICAS
    plan["binding61_reference"] = {
        "uri": BINDING61_URI,
        "sha256": BINDING61_SHA256,
        "seed": binding61["seed"],
        "attempts": binding61["attempts"],
        "unique_tasks": binding61["unique_tasks"],
        "selection_uri": binding61["selection_uri"],
        "selection_sha256": binding61["selection_sha256"],
        "dataset": binding61["dataset"],
        "sampling_params": binding61["sampling_params"],
        "inference": binding61["inference"],
        "judge": binding61["judge"],
        "judge_gpu_count": binding61["judge_gpu_count"],
        "judge_gpu_type": binding61["judge_gpu_type"],
        "reusable_builder": binding61["reusable_builder"],
        "source_revision": binding61["source_revision"],
    }
    # Contract alignment checks against the frozen binding packet.
    expected = {
        "seed": 20261030,
        "attempts": 440,
        "unique_tasks": 200,
    }
    for field, value in expected.items():
        if binding61[field] != value:
            raise ValueError(f"binding61 {field}={binding61[field]} does not match the pinned contract {value}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, help="URI of the staged model-binding-inputs.json")
    parser.add_argument("--output", required=True, help="Where to write the plan JSON")
    args = parser.parse_args()

    configure_coreweave_s3()
    binding = json.loads(StoragePath(args.binding).read_bytes())
    binding61 = json.loads(StoragePath(BINDING61_URI).read_bytes())
    binding["self_sha256"] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    plan = build(binding, binding61)
    with open(args.output, "w") as handle:
        json.dump(plan, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(
        f"wrote {args.output}: model {plan['model']['hf_uri']}"
        f" shards={len(plan['model']['shards'])} policy_replicas={plan['policy_replicas']}"
    )


if __name__ == "__main__":
    main()
