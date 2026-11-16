# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prewarm a published merge from its checksum-pinned regional upload source."""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import fsspec
from huggingface_hub import hf_hub_download, model_info
from levanter.model_cache import CacheMetadata, _cache_slug, cache_to_prefix
from marin.evaluation.model_config import load_model_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifiedWeight:
    filename: str
    sha256: str


def copy_verified_weight(fs: fsspec.AbstractFileSystem, source: str, destination: str, weight: VerifiedWeight) -> None:
    """Copy a regional weight only after its bytes match the published HF checksum."""
    with TemporaryDirectory(prefix="merge_warmup_") as scratch:
        local = Path(scratch) / "weight.safetensors"
        fs.get_file(f"{source}/{weight.filename}", str(local))
        with local.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != weight.sha256:
            raise ValueError(f"Published checksum mismatch for {weight.filename}: {digest}")
        remote = f"{destination}/{weight.filename}"
        fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
        fs.put_file(str(local), remote)
        logger.info("Verified and cached %s", weight.filename)


def warm_uploaded_model(
    model_config: Path, cache_path: str, manifest_path: str, workers: int, source: str | None = None
) -> str:
    """Populate the canonical cache from verified source weights and pinned HF metadata."""
    model = load_model_config(model_config)
    if model.revision is None:
        raise ValueError("Uploaded-model warmup requires an immutable HF revision")
    if cache_path.rsplit("/", 1)[-1] != _cache_slug(f"{model.location}@{model.revision}"):
        raise ValueError("Cache destination differs from the pinned model identity")
    info = model_info(model.location, revision=model.revision, files_metadata=True)
    if info.sha != model.revision or info.siblings is None:
        raise ValueError("HF metadata does not identify the pinned snapshot")
    manifest_file = Path(hf_hub_download(model.location, manifest_path, revision=model.revision))
    manifest = json.loads(manifest_file.read_text())
    if source is None:
        source = manifest["checkpoint"]
        objects = {item["name"]: item for item in manifest["objects"]}
    else:
        objects = {item["path"]: item for item in manifest["files"]}
    if urlsplit(source).netloc != urlsplit(cache_path).netloc:
        raise ValueError("Warmup source and cache must use the same regional bucket")
    weights = []
    for sibling in info.siblings:
        if not sibling.rfilename.endswith(".safetensors"):
            continue
        if sibling.lfs is None or objects[sibling.rfilename]["sha256"] != sibling.lfs.sha256:
            raise ValueError(f"Upload manifest disagrees with pinned HF weights: {sibling.rfilename}")
        weights.append(VerifiedWeight(sibling.rfilename, sibling.lfs.sha256))
    weight_names = {weight.filename for weight in weights}
    siblings = info.siblings

    def populate(fs: fsspec.AbstractFileSystem, destination: str) -> None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(copy_verified_weight, fs, source, destination, weight) for weight in weights]
            for future in futures:
                future.result()
        with TemporaryDirectory(prefix="merge_metadata_") as scratch:
            for sibling in siblings:
                if sibling.rfilename in weight_names:
                    continue
                local = hf_hub_download(model.location, sibling.rfilename, revision=model.revision, local_dir=scratch)
                remote = f"{destination}/{sibling.rfilename}"
                fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
                fs.put_file(local, remote)

    return cache_to_prefix(cache_path, populate, metadata=CacheMetadata(source_revision=model.revision))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", help="Regional export for a published producer manifest with a files list")
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    configure_logging()
    configure_coreweave_s3()
    print(warm_uploaded_model(args.model_config, args.cache_path, args.manifest, args.workers, args.source))


if __name__ == "__main__":
    main()
