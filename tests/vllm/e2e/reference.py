# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accelerator-independent reference data and assertions for vLLM e2es."""

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
from rigging.filesystem import StoragePath

RUN_NAME = "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3"
CHECKPOINT_STEP = 42150
CHECKPOINT_NAME = f"step-{CHECKPOINT_STEP}"
EXPORT_TREE_SHA256 = "781bc3291c81ce282be6762520280ebd5ef5b85e88ba65129c2d0162d48ee632"
S3_MODEL_URI = (
    f"s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/{CHECKPOINT_NAME}/"
    f"hf-bf16-vllm/{EXPORT_TREE_SHA256[:16]}/"
)
GCS_MODEL_URI = (
    f"gs://marin-us-east5/models/grug/june-67b-a2b/{CHECKPOINT_NAME}/" f"hf-bf16-vllm/{EXPORT_TREE_SHA256[:16]}/"
)
GCS_MODEL_COMPLETION_URI = f"{GCS_MODEL_URI.rstrip('/')}.complete"
LOGPROBS_RESOURCE = Path(__file__).parent / "resources" / f"june_tpu_67b_a2b_step_{CHECKPOINT_STEP}_logprobs.json"
RETURNED_LOGPROBS = 50
# Clean checkpoint dev runs stayed below 0.0052 max probability error and 0.0078 L1
# across both GPU attention backends, leaving cross-accelerator margin within these bounds.
MAX_PROBABILITY_ERROR = 0.008
TOP_PROBABILITY_L1_ERROR = 0.012


@dataclass(frozen=True)
class TokenLogprobReference:
    logprob: float
    text: str
    token_id: int


@dataclass(frozen=True)
class InferenceReference:
    moe_implementation: str
    mp: str
    prompt: str
    prompt_token_ids: list[int]
    tokenizer: str
    top_logprobs: list[TokenLogprobReference]


@dataclass(frozen=True)
class InferenceMetrics:
    lane: int
    max_abs_logprob_error: float
    max_abs_probability_error: float
    top_probability_l1_error: float
    seconds: float | None = None


def read_inference_reference(path: Path = LOGPROBS_RESOURCE) -> InferenceReference:
    return draccus.decode(InferenceReference, json.loads(path.read_text()))


def _combined_tree_sha256(relative_paths: list[str], file_digests: list[bytes]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_digest in zip(relative_paths, file_digests, strict=True):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(file_digest)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    relative_paths = [path.relative_to(root).as_posix() for path in paths]
    file_digests = []
    for path in paths:
        with path.open("rb") as file:
            file_digests.append(hashlib.file_digest(file, "sha256").digest())
    return _combined_tree_sha256(relative_paths, file_digests)


def storage_tree_sha256(uri: str) -> str:
    root = StoragePath(uri)
    paths = sorted(
        (directory / filename for directory, _subdirectories, filenames in root.walk() for filename in filenames),
        key=str,
    )
    if not paths:
        raise FileNotFoundError(f"No files found under {uri}")

    def file_sha256(path: StoragePath) -> bytes:
        file_digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(8 * 1024 * 1024):
                file_digest.update(chunk)
        return file_digest.digest()

    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
        file_digests = list(executor.map(file_sha256, paths))

    relative_paths = [path.relative_to(root) for path in paths]
    return _combined_tree_sha256(relative_paths, file_digests)


def completion_request(expected: InferenceReference) -> dict[str, Any]:
    return {
        "prompt": expected.prompt,
        "add_special_tokens": False,
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": RETURNED_LOGPROBS,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }


def assert_completion_matches_reference(
    expected: InferenceReference,
    choice: dict[str, Any],
    *,
    lane: int,
) -> InferenceMetrics:
    expected_logprobs = {entry.token_id: entry.logprob for entry in expected.top_logprobs}
    assert choice["prompt_token_ids"] == expected.prompt_token_ids
    assert choice["token_ids"] == [expected.top_logprobs[0].token_id]
    actual_logprobs = {
        int(token.removeprefix("token_id:")): logprob for token, logprob in choice["logprobs"]["top_logprobs"][0].items()
    }
    missing_token_ids = expected_logprobs.keys() - actual_logprobs.keys()
    assert not missing_token_ids, sorted(missing_token_ids)
    max_logprob_error = max(
        abs(actual_logprobs[token_id] - expected_logprob) for token_id, expected_logprob in expected_logprobs.items()
    )
    probability_errors = [
        abs(math.exp(actual_logprobs[token_id]) - math.exp(expected_logprob))
        for token_id, expected_logprob in expected_logprobs.items()
    ]
    max_probability_error = max(probability_errors)
    top_probability_l1_error = sum(probability_errors)
    metric = InferenceMetrics(
        lane=lane,
        max_abs_logprob_error=max_logprob_error,
        max_abs_probability_error=max_probability_error,
        top_probability_l1_error=top_probability_l1_error,
    )
    assert max_probability_error <= MAX_PROBABILITY_ERROR, metric
    assert top_probability_l1_error <= TOP_PROBABILITY_L1_ERROR, metric
    return metric
