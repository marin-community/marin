# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload the region-local full A/B export to a private HF repo, then publish it."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)
RELEASE_DIR = Path(__file__).with_name("hf_release_20261005")
EXPECTED_TENSORS = 502
EXPECTED_TENSOR_BYTES = 134_157_765_632
STEP = 161_750
EXPORT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)


def copy_from_gcs(source: str, destination: Path) -> None:
    subprocess.run(["fsutil", "cp", source, str(destination)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--wait-timeout", type=int, default=172_800)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    repo = args.repo
    source = args.source.rstrip("/")
    index_path = StoragePath(f"{source}/model.safetensors.index.json")
    deadline = time.monotonic() + args.wait_timeout
    while not index_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Export index did not appear at {index_path}")
        logger.info("Waiting for completed export index at %s", index_path)
        time.sleep(120)

    index = json.loads(index_path.read_text())
    if len(index["weight_map"]) != EXPECTED_TENSORS:
        raise ValueError(f"Expected {EXPECTED_TENSORS} tensors, got {len(index['weight_map'])}")
    if index["metadata"]["total_size"] != EXPECTED_TENSOR_BYTES:
        raise ValueError(f"Unexpected tensor bytes: {index['metadata']['total_size']}")
    shards = sorted(set(index["weight_map"].values()))
    for name in (*EXPORT_FILES, *shards):
        if not StoragePath(f"{source}/{name}").exists():
            raise FileNotFoundError(f"Incomplete export: {source}/{name}")

    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo).private:
        raise ValueError(f"Expected a private upload target: {repo}")
    api.upload_folder(folder_path=RELEASE_DIR, repo_id=repo, repo_type="model", commit_message="Add model card")

    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        for name in EXPORT_FILES:
            local = staging / name
            copy_from_gcs(f"{source}/{name}", local)
            if name == "chat_template.jinja" and local.read_bytes().rstrip(b"\n") != (
                RELEASE_DIR / name
            ).read_bytes().rstrip(b"\n"):
                raise ValueError("Exported inference template differs from the release template")
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=name,
                repo_id=repo,
                repo_type="model",
                commit_message=f"Add {name}",
            )
            local.unlink()

        present = set(api.list_repo_files(repo, repo_type="model"))
        for number, name in enumerate(shards, 1):
            if name in present:
                logger.info("Already uploaded shard %d/%d: %s", number, len(shards), name)
                continue
            local = staging / name
            copy_from_gcs(f"{source}/{name}", local)
            logger.info("Uploading shard %d/%d: %s (%d bytes)", number, len(shards), name, local.stat().st_size)
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=name,
                repo_id=repo,
                repo_type="model",
                commit_message=f"Add weight shard {number}/{len(shards)}",
            )
            local.unlink()

    info = api.model_info(repo, files_metadata=True)
    remote = {sibling.rfilename: sibling.size for sibling in info.siblings}
    expected = set(EXPORT_FILES) | set(shards) | {path.name for path in RELEASE_DIR.iterdir()}
    if missing := expected - remote.keys():
        raise ValueError(f"Missing Hugging Face files: {sorted(missing)}")
    uploaded_bytes = sum(remote[name] for name in shards)
    if not EXPECTED_TENSOR_BYTES <= uploaded_bytes <= EXPECTED_TENSOR_BYTES + 1_000_000:
        raise ValueError(f"Unexpected uploaded shard bytes: {uploaded_bytes}")

    complete = {
        "step": STEP,
        "dtype": "bfloat16",
        "pending_qb_betas_applied": True,
        "shards": len(shards),
        "total_size": EXPECTED_TENSOR_BYTES,
        "default_enable_thinking": True,
        "generation_eos_token_ids": [128001, 128009],
        "training_template_sha256": (
            hashlib.sha256((RELEASE_DIR / "training_chat_template.jinja").read_bytes()).hexdigest()
        ),
        "inference_template_sha256": (
            hashlib.sha256((RELEASE_DIR / "chat_template.jinja").read_bytes().rstrip(b"\n")).hexdigest()
        ),
    }
    api.upload_file(
        path_or_fileobj=json.dumps(complete, indent=2).encode(),
        path_in_repo="export_complete.json",
        repo_id=repo,
        repo_type="model",
        commit_message="Record verified export",
    )
    api.update_repo_visibility(repo, private=False, repo_type="model")
    logger.info("Published %s with %d shards (%d bytes)", repo, len(shards), uploaded_bytes)


if __name__ == "__main__":
    main()
