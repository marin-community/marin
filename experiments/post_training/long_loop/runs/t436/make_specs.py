# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the task-436 spec set from Ben's specs: 10/05 native init, contributor data, interactive priority.

    python make_specs.py SPECS_DIR OUTPUT_DIR MARIN_REVISION

OUTPUT_DIR gets a copy of every SPECS_DIR yaml, the task-436 changes and fresh SHA-256 pins.
Launch from an untracked OUTPUT_DIR inside the worktree with MARIN_REVISION = HEAD.
"""

import hashlib
import shutil
import sys
from pathlib import Path

import yaml

# Measured 2026-11-12 after /karan/build-contrib-packed262k-t436 and /karan/copy-native-1005-t436 succeeded.
SCRATCH = "s3://marin-rno2a/tmp/ttl=14d/long-loop-436"
LEDGER_SHA256 = "232477641413cd7858417e7b575fe97e3b1d00aa6845cb4913b68a267b0be63a"
SUMMARY_SHA256 = "2d848c515a37d007832c3509c6e95631a72daab080f6ef0f67b0ba11b1a67209"
COPY_SHA256 = {
    "copy-receipt/receipt.json": "a68824fc4256521fddedd969a3ff5f984a58670104a71124f1ff00997757e8cf",
    "native/step-0/metadata.json": "c462604c60c17c30dba5e71c107a9b8e644b91f4026b0459b91ab3386db6f697",
    "native/step-0/manifest.json": "cbe1fb6b4f524d9f27544273c4981f3f286fff58063bc65a12811235cd90017d",
}

source, d, revision = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
d.mkdir(parents=True, exist_ok=True)
for path in source.glob("*.yaml"):
    shutil.copyfile(path, d / path.name)


def load(name: str):
    return yaml.safe_load((d / name).read_text())


def dump(name: str, value) -> None:
    (d / name).write_text(yaml.safe_dump(value, sort_keys=False))


# Recipe: shootout optimizer and precision at 262k packing, batch 2 x CP8 = 2 nodes.
sft = load("sft.yaml")
sft["optimizer"] = {
    "type": "adam",
    "learning_rate": 1.0e-6,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "warmup": 0,
    "lr_schedule": "constant",
    "min_lr_ratio": 1.0,
}
sft["stop_after_steps"] = 678
sft["eval"] = None
sft["resources"]["replicas"] = 2
t = sft["trainer"]
t["special_token_lr_multiplier"] = 1.0
t["reseed_special_token_rows"] = False
t["z_loss_weight"] = 0.0
t["trainer"]["mp"] = "compute=bfloat16,params=float32,output=bfloat16"
t["trainer"]["train_batch_size"] = 2
t["trainer"]["num_train_steps"] = 678
t["trainer"]["seed"] = 42
t["data_seed"] = 42
t["trainer"]["checkpointer"]["keep"] = []  # The final save is permanent; fp32 params + Adam are ~0.8 TB each.
t["gradient_accumulation_steps"] = 1
t["trainer"]["tracker"]["tags"] = ["long-loop", "t436", "contributor-sft"]
t["trainer"]["initialize_from"] = "UNBOUND"
dump("sft-contrib.yaml", sft)

# Data manifest: one packed prebuilt cache per epoch, one stage per epoch, in file order.
CACHE = "s3://marin-rno2a/tmp/ttl=14d/long-loop-436/contrib-v01-packed262k"
fmt = {"type": "prebuilt", "input_ids_key": "input_ids", "loss_weights_key": "loss_weights"}
names = [f"epoch-{e}" for e in range(3)]
data = {
    "tokenizer": "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18",
    "vocab_size": None,
    "cache_dir": None,
    "enforce_eos": False,
    "auto_build_caches": False,
    "chat_template": None,
    "shuffle": False,
    "block_cross_document_attention": True,
    "stop_strategy": "restart",
    "mixture_block_size": 2,
    "components": {
        n: {
            "type": "cached",
            "cache_dir": f"{CACHE}/{n}/train",
            "format": fmt,
            "pack": False,
            "split": "train",
            "flat_cache": True,
        }
        for n in names
    },
    "train_weights": [[226 * e, {m: float(m == n) for m in names}] for e, n in enumerate(names)],
}
dump(
    "sft-contrib-data.yaml",
    {
        "data": data,
        # The three ledgers are identical (row counts, not data); summary.json pins per-update supervised counts.
        "files": (
            [
                {"uri": f"{SCRATCH}/contrib-v01-packed262k/epoch-{e}/train/shard_ledger.json", "sha256": LEDGER_SHA256}
                for e in range(3)
            ]
            + [{"uri": f"{SCRATCH}/contrib-v01-packed262k/summary.json", "sha256": SUMMARY_SHA256}]
        ),
        "provenance": {
            "source": "s3://marin-us-east-02a/marin/experiments/sft-shootout/rows/t202-contrib-v01-20261022/plain",
            "manifest_sha256_prefix": "507e73f0",
            "packing": "8 consecutive 32,768-token rows per 262,144-token row; runs/t436/build_packed_cache.py",
            "rows_per_epoch": 3616,
            "packed_rows_per_epoch": 452,
            "updates_per_epoch": 226,
            "supervised_tokens_per_epoch": 16137263,
        },
        "sections": {},
    },
)

# Initializer: verbatim copy of the 10/05 native final step.
dump(
    "initializer-1005.yaml",
    {
        "uri": "s3://marin-rno2a/tmp/ttl=14d/long-loop-436/native-1005",
        "evidence": "native_copy",
        "source": (
            "gs://marin-us-central2/users/held/grug_sft/grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped/checkpoints/step-161750"
        ),
        "hf_release": {
            "repository": "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05",
            "revision": "ac60d9401820953943f5a56e06f36e13c86b12d6",
        },
        "files": [{"uri": f"{SCRATCH}/native-1005/{path}", "sha256": sha} for path, sha in COPY_SHA256.items()],
    },
)

# One shared priority: interactive first.
for name, path in (
    ("execution.yaml", ("coordinator",)),
    ("rl.yaml", ("execution",)),
    ("evaluation.yaml", ("execution",)),
    ("endpoint.yaml", ()),
):
    doc = load(name)
    node = doc
    for key in path:
        node = node[key]
    node["priority"] = "interactive"
    dump(name, doc)

# SFT and export read only these; the shared default also asks for the RL and eval sandbox and judge keys.
execution = load("execution.yaml")
execution["coordinator"]["required_env_vars"] = ["MARIN_PREFIX", "MARIN_TEMP_PREFIX", "HF_TOKEN"]
dump("execution.yaml", execution)

spec = load("round1.yaml")
spec["name"] = "long-loop/t436-contrib"
spec["marin_revision"] = revision
spec["datasets"]["sft"]["path"] = "sft-contrib-data.yaml"
spec["sft"]["path"] = "sft-contrib.yaml"
spec["initializer"]["path"] = "initializer-1005.yaml"
spec["smoke_overrides"] = {"sft_steps": 2, "rl_steps": 2, "eval_tasks_per_benchmark": 1, "eval_replicas": 1}
# Compute is on rno2a; intermediates stay in its TTL scratch.
spec["storage"] = {
    "output_prefix": "s3://marin-rno2a/tmp/ttl=14d/long-loop-t436",
    "temporary_prefix": "s3://marin-rno2a/tmp/ttl=3d/long-loop-t436",
}


def pin(section):
    section["sha256"] = hashlib.sha256((d / section["path"]).read_bytes()).hexdigest()


for section in spec["datasets"].values():
    pin(section)
for key in ("sft", "initializer", "tokenizer", "endpoint", "execution", "archive", "rl", "evaluation"):
    pin(spec[key])
dump("round1.yaml", spec)
for name in ("sft.yaml", "sft-data.yaml", "initializer.yaml"):
    (d / name).unlink()
print("ok", spec["name"])
