# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the exact 39e ShortConv experiment before importing vLLM.

This belongs only to the isolated Hero experiment snapshot. It does not qualify
CUDA graphs or change the managed dependency lock or shared uv wheel cache.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote

MANIFEST_SHA256 = "266cb34f775746ac050b13f4dcce38a615ae1718b450c546fba4ca4c06dfca8d"
PAYLOAD_DIRECTORY = Path("experiments/post_training/hero_rl/piecewise_payload")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def extension_hashes(package: Path) -> dict[str, str]:
    hashes = {}
    for directory, _, files in os.walk(package, followlinks=True):
        for name in files:
            path = Path(directory) / name
            if ".so" in name or path.suffix in (".pyd", ".dylib"):
                hashes[str(path.relative_to(package))] = digest(path)
    return hashes


def load_manifest(workspace: Path) -> dict:
    """Verify the fixed source payload and frozen dependency inputs."""
    payload = workspace / PAYLOAD_DIRECTORY
    if digest(payload / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Hero experimental runtime manifest changed")
    manifest = json.loads((payload / "manifest.json").read_text())
    for path, expected in manifest["managed_inputs"].items():
        if digest(workspace / path) != expected:
            raise ValueError(f"Hero managed dependency input changed: {path}")
    for item in manifest["overlay"]:
        if digest(payload / item["payload_path"]) != item["patched_sha256"]:
            raise ValueError(f"Hero experimental payload changed: {item['path']}")
    return manifest


def stage_runtime(workspace: Path, environment: Path, receipt_path: Path) -> dict:
    """Install the verified Python overlay into an owned environment package."""
    if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
        raise RuntimeError("Hero experimental staging must precede every vLLM import")
    manifest = load_manifest(workspace)
    core = {name: metadata.version(name) for name in manifest["expected_core"]}
    if core != manifest["expected_core"]:
        raise ValueError(f"Hero experimental core dependencies differ from the lock: {core}")
    distribution = metadata.distribution("vllm")
    direct = json.loads(distribution.read_text("direct_url.json"))
    wheel = manifest["wheel"]
    if (
        distribution.version != wheel["version"]
        or unquote(direct["url"]) != unquote(wheel["url"])
        or direct["archive_info"]["hashes"]["sha256"] != wheel["hash"].removeprefix("sha256:")
    ):
        raise ValueError("Installed vLLM is not the exact locked 39e ARM wheel")
    # Keep the package entry lexical: resolving it here would select uv's cache.
    entry = Path(os.path.abspath(distribution.locate_file("vllm")))
    if not entry.parent.resolve().is_relative_to(environment.resolve()):
        raise ValueError("vLLM package entry is outside the task environment")
    original = entry.with_name("vllm.before-hero-piecewise")
    if original.exists() or original.is_symlink():
        raise ValueError("Task environment has already been overlaid")
    for item in manifest["overlay"]:
        relative = Path(item["path"]).relative_to("vllm")
        if digest(entry / relative) != item["base_sha256"]:
            raise ValueError(f"Installed wheel preimage differs from pinned source: {relative}")
    before_extensions = extension_hashes(entry)
    if not before_extensions:
        raise ValueError("Pinned vLLM wheel has no compiled extensions")
    with tempfile.TemporaryDirectory(prefix=".hero-piecewise-", dir=entry.parent) as temporary:
        owned = Path(temporary) / "vllm"
        shutil.copytree(entry, owned, symlinks=False)
        for item in manifest["overlay"]:
            relative = Path(item["path"]).relative_to("vllm")
            target = owned / relative
            content = workspace / PAYLOAD_DIRECTORY / item["payload_path"]
            # Replacing the file also breaks any hard link in a future copier.
            replacement = target.with_name(target.name + ".hero-new")
            replacement.write_bytes(content.read_bytes())
            replacement.chmod(target.stat().st_mode)
            os.replace(replacement, target)
            if digest(target) != item["patched_sha256"]:
                raise ValueError(f"Hero overlay verification failed: {relative}")
        if any(path.is_symlink() for path in owned.rglob("*")):
            raise ValueError("Owned vLLM package still contains symlinks")
        after_extensions = extension_hashes(owned)
        if after_extensions != before_extensions:
            raise ValueError("A compiled vLLM extension changed during staging")
        # No vLLM process exists yet. Publish all three changes together, and
        # restore the environment entry if the second rename fails.
        entry.rename(original)
        try:
            os.replace(owned, entry)
        except OSError:
            os.replace(original, entry)
            raise
    receipt = {
        "qualification": manifest["qualification"],
        "manifest_sha256": MANIFEST_SHA256,
        "stager_sha256": digest(Path(__file__)),
        "hostname": socket.gethostname(),
        "task_id": os.environ.get("IRIS_TASK_ID"),
        "source_revision": manifest["source_revision"],
        "patch_sha256": manifest["patch_sha256"],
        "installed_package": str(entry),
        "original_package_entry": str(original),
        "wheel_direct_url": direct,
        "core_dependencies": core,
        "overlay": manifest["overlay"],
        "extensions_before": before_extensions,
        "extensions_after": after_extensions,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt_path.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_receipt, receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("Hero PIECEWISE experiment requires Linux aarch64")
    receipt = stage_runtime(args.workspace.resolve(), Path(sys.prefix), args.receipt)
    print(json.dumps({"hero_piecewise_runtime": receipt}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
