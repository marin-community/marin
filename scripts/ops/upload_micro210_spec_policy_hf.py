# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the spec-complete Micro210 policy and private prepared table to pinned HF tags."""

import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, create_repo
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.micro210_policy import (
    SPEC_POLICY_REFERENCE,
    POLICY_REPO,
    PREPARED_REPO,
)
from experiments.evaluation.micro210_spec_grading import driver_sha256
from scripts.ops.upload_micro210_policy_hf import prepare_hf_token, publish_version, read_and_verify

POLICY_CARD = """\
---
license: apache-2.0
language: [en]
tags: [evaluation, reinforcement-learning]
---

# Micro210 evaluation policy

The `2026.11.23-together` version makes every script-graded task self-contained: each row
embeds a vendored grading library and a standalone driver, and the verifier package spec
drops the ambient evalchemy and lm_eval wheels. It keeps the Together judge and the frozen
task selection, permutes TruthfulQA choices with recorded per-task seeds, grades CRUXEval
with the canonical tuple-safe extraction (#11922), accepts Kyiv in the IFBench European
capitals checker (#11926), and applies a float-tolerance fix to the MBPP+ polar_rect row
(#12159). See [marin-community/marin#12060](https://github.com/marin-community/marin/issues/12060).
The policy JSON is public; prepared tasks are in a separate private dataset.
"""

PREPARED_CARD = """\
---
license: other
language: [en]
tags: [evaluation, reinforcement-learning]
---

# Micro210 prepared task inputs

The `2026.11.23-together` tag contains 440 prepared attempts derived from the pinned
Micro210 selection. Every script-graded row embeds its grading library, driver, and
provenance; TruthfulQA choices have recorded per-task seeds. The dataset remains private
because it re-hosts restricted benchmark items, including GPQA-Diamond. See
[marin-community/marin#12060](https://github.com/marin-community/marin/issues/12060).
"""


def main() -> None:
    prepare_hf_token()
    api = HfApi()
    identity = api.whoami()
    if not any(org.get("name") == "open-athena" for org in identity.get("orgs", [])):
        raise SystemExit("HF identity is not an open-athena member; refusing to publish")
    raw = read_and_verify(SPEC_POLICY_REFERENCE.uri, SPEC_POLICY_REFERENCE.sha256, SPEC_POLICY_REFERENCE.byte_size)
    policy = json.loads(raw)
    if policy["policy_version"] != SPEC_POLICY_REFERENCE.revision:
        raise ValueError("Staged policy version differs from its pin")
    if policy["grader_embed"]["driver_sha256"] != driver_sha256():
        raise ValueError("Staged policy driver digest differs from the committed driver")
    environment = read_and_verify(
        str(StoragePath(policy["prepared_inputs_uri"]) / "environment.json"), policy["task_environment_sha256"]
    )
    if len(json.loads(environment)["tasks"]) != 440:
        raise ValueError("Staged environment manifest does not carry 440 tasks")
    prepared_uri = str(StoragePath(policy["prepared_inputs_uri"]) / "rollouts.parquet")
    prepared = read_and_verify(prepared_uri, policy["prepared_rollouts_sha256"])
    with tempfile.TemporaryDirectory() as temporary:
        prepared_path = Path(temporary) / "rollouts.parquet"
        prepared_path.write_bytes(prepared)
        if pq.read_metadata(prepared_path).num_rows != 440:
            raise ValueError("Staged Micro210 table must contain 440 attempts")
        create_repo(POLICY_REPO, repo_type="dataset", private=False, exist_ok=True)
        create_repo(PREPARED_REPO, repo_type="dataset", private=True, exist_ok=True)
        policy_commit = publish_version(
            api,
            POLICY_REPO,
            SPEC_POLICY_REFERENCE.revision,
            f"Publish Micro210 policy {SPEC_POLICY_REFERENCE.revision}",
            POLICY_CARD,
            raw,
            "policy.json",
            SPEC_POLICY_REFERENCE.sha256,
            SPEC_POLICY_REFERENCE.byte_size,
        )
        prepared_commit = publish_version(
            api,
            PREPARED_REPO,
            SPEC_POLICY_REFERENCE.revision,
            f"Publish Micro210 prepared tasks {SPEC_POLICY_REFERENCE.revision}",
            PREPARED_CARD,
            str(prepared_path),
            "rollouts.parquet",
            policy["prepared_rollouts_sha256"],
            len(prepared),
        )
    read_and_verify(
        f"hf://datasets/{POLICY_REPO}@{policy_commit}/policy.json",
        SPEC_POLICY_REFERENCE.sha256,
        SPEC_POLICY_REFERENCE.byte_size,
    )
    read_and_verify(
        f"hf://datasets/{PREPARED_REPO}@{prepared_commit}/rollouts.parquet",
        policy["prepared_rollouts_sha256"],
        len(prepared),
    )
    public = requests.get(
        f"https://huggingface.co/datasets/{POLICY_REPO}/resolve/{policy_commit}/policy.json", timeout=120
    )
    private = requests.get(
        f"https://huggingface.co/datasets/{PREPARED_REPO}/resolve/{prepared_commit}/rollouts.parquet",
        timeout=120,
        allow_redirects=False,
    )
    if public.status_code != 200 or public.content != raw:
        raise ValueError("Published policy is not anonymously readable with pinned bytes")
    if private.status_code in (200, 301, 302, 307, 308):
        raise ValueError("Published prepared table is anonymously readable")
    print(f"SPEC_POLICY_COMMIT = {policy_commit}")
    print(f"SPEC_PREPARED_COMMIT = {prepared_commit}")


if __name__ == "__main__":
    main()
