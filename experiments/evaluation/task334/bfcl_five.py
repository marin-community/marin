# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BFCL x5 for task-334 merge candidates, with the arms' model profile and Harbor config.

``model`` writes ``model-<tag>.yaml``: the M51 profile (serve, parser, tokenizer, agent) with the
candidate's name and location. ``command`` prints the ``bfcl_parity launch`` call at interactive
priority with fresh output namespaces. Neither submits.
"""

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path

import yaml
from rigging.filesystem.storage_path import StoragePath

HERE = Path(__file__).parent
HARBOR_CONFIG = HERE / "bfclparity-pi-arms.yaml"
HARBOR_SHA256 = "73bf0dc40d438b04fedaf1ac31bc4c35d3fc07ce851d815730c5079f168666e1"
SCREEN_ROOT = "s3://marin-rno2a/tmp/ttl=14d/snowball-identity/task-334/screens"
# The M51 BFCL model profile, with name and location left to each candidate.
PROFILE = {
    "revision": None,
    "apply_chat_template": True,
    "resource_hint": {"gpu": {"H100": 8}, "memory": "512g"},
    "serve": {
        "hf_hub_disable_xet": True,
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "max_model_len": 73728,
        "max_num_batched_tokens": 7168,
        "max_num_seqs": 32,
        "tool_call_parser": "hermes",
        "auto_overrides": False,
        "vllm_extra_args": ["--enable-expert-parallel", "--model-loader-extra-config", '{"distributed":true}'],
    },
    "agent": {"agent_kwargs": {"thinking_format": "chat-template"}},
    "tokenizer": "open-athena/Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12",
    "tokenizer_revision": "4eeb756b404db25bc04e4daf49dfb9817bceb741",
}


def checked_tag(tag: str) -> str:
    if re.fullmatch(r"[a-z0-9-]+", tag) is None:
        raise ValueError("Tags are lowercase, e.g. ties-t02")
    return tag


def model(tag: str, admission_uri: str) -> Path:
    report = json.loads(StoragePath(admission_uri).read_bytes())
    if report["admitted"] is not True:
        raise ValueError("Bind only an admitted merge")
    value = {"name": f"snowball-t334-bfcl-{checked_tag(tag)}-r1", "location": report["output"], **PROFILE}
    path = HERE / f"model-{tag}.yaml"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def command(tag: str) -> list[str]:
    if hashlib.sha256(HARBOR_CONFIG.read_bytes()).hexdigest() != HARBOR_SHA256:
        raise ValueError("The arms' BFCL Harbor config changed")
    path = HERE / f"model-{checked_tag(tag)}.yaml"
    root = f"{SCREEN_ROOT}/{tag}/bfcl-five-r1"
    return [
        "python", "-m", "experiments.evaluation.bfcl_parity", "launch",
        "--model-config", str(path.relative_to(HERE.parents[2])),
        "--harbor-config", str(HARBOR_CONFIG.relative_to(HERE.parents[2])),
        "--run-prefix", f"t334-{tag}-bfcl",
        "--runs", "5",
        "--federated-cluster", "cw-rno2a",
        "--records-prefix", root + "/records",
        "--accelerator", "H100x8",
        "--priority-band", "interactive",
    ]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    m = sub.add_parser("model")
    m.add_argument("--tag", required=True)
    m.add_argument("--admission", required=True)
    c = sub.add_parser("command")
    c.add_argument("--tag", required=True)
    args = parser.parse_args()
    print(model(args.tag, args.admission) if args.kind == "model" else shlex.join(command(args.tag)))


if __name__ == "__main__":
    main()
