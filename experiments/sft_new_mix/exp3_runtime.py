# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for the SFT candidate pipeline stages.

Every CLI stage imports from this module; it may not import a stage module,
and ``implementation_identity`` hashes only the files a stage names.
"""

import gzip
import hashlib
import json
import os
from collections import Counter
from functools import cache
from pathlib import Path

from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_filters import CompProgLabels, FilterConfig, Stage

# The build stages whose bytes determine candidate shard output. exp3_plan and
# exp3_build must pin exactly this list.
BUILD_IMPLEMENTATION_FILES = ("exp3_runtime.py", "exp3_build.py", "exp3_filters.py", "loss_modes.py")


def read_pinned(uri: str, sha256: str) -> bytes:
    """Read bytes and refuse any content that does not match its pin."""
    content = StoragePath(uri).read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"Digest changed for {uri}")
    return content


def configure_storage() -> None:
    configure_coreweave_s3()
    os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"] = s3_credentials(StoreType.COREWEAVE)
    os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = data_buckets()["marin-us-east-02a"].signing_region


@cache
def tokenizer_for(uri: str, sha256: str) -> Tokenizer:
    return Tokenizer.from_str(read_pinned(uri, sha256).decode())


@cache
def build_config(uri: str, expected_sha256: str) -> tuple[dict, FilterConfig]:
    value = json.loads(read_pinned(uri, expected_sha256))
    rescue_enabled = Stage.COMP_PROG_LABEL_RESCUE.value in value["enabled"]
    pinned_labels = value.get("comp_prog_labels")
    if rescue_enabled != (pinned_labels is not None):
        raise ValueError("comp_prog_label_rescue stage and its pinned label table must be set together")
    labels = None
    if pinned_labels is not None:
        table = gzip.decompress(read_pinned(pinned_labels["table_uri"], pinned_labels["table_sha256"])).decode()
        labels = CompProgLabels.from_records(json.loads(line) for line in table.splitlines())
    config = FilterConfig(
        enabled=frozenset(Stage(stage) for stage in value["enabled"]),
        think_limits=value["think_limits"],
        tools_by_source={source: frozenset(tools) for source, tools in value["tools_by_source"].items()},
        give_up_sources=frozenset(value["give_up_sources"]),
        comp_prog_labels=labels,
    )
    return value, config


def count_stage(
    stats: dict[str, Counter], stage: str, before: int, after: int, reason: str = "", *, changed: bool = False
) -> None:
    row = stats[stage]
    row["input_documents"] += 1
    row["input_tokens"] += before
    row["output_documents"] += not bool(reason)
    row["output_tokens"] += after
    row["dropped_documents"] += bool(reason)
    row["dropped_tokens"] += before if reason else 0
    row["changed_documents"] += not reason and changed
    if reason:
        row["drop_reason:" + reason] += 1


def implementation_identity(files: tuple[str, ...]) -> str:
    """Hash the named module files; every stage pins the list it depends on."""
    digest = hashlib.sha256()
    for filename in files:
        digest.update(filename.encode())
        digest.update(Path(__file__).with_name(filename).read_bytes())
    return digest.hexdigest()
