# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the original tokenizer separately from immutable checkpoint exports."""

import hashlib
import os
import re
from pathlib import Path

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.rate import MODEL_PROFILES

EAST_PREFIX = "s3://marin-us-east-02a/marin/"
MAX_TOKENIZER_BYTES = 32 * 1024**2
TOKENIZER_STAGE_ROOT = Path("/tmp")
TOKENIZER_FILES = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    }
)
REQUIRED_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}


def validate_tokenizer_source(source):
    """Require a byte-bound tokenizer source; exported tokenizer equivalence is not assumed."""
    files = source["files"]
    if not source["uri"].startswith(EAST_PREFIX) or not REQUIRED_FILES <= files.keys() <= TOKENIZER_FILES:
        raise ValueError("Calibration tokenizer source or metadata filenames changed")
    if any(
        type(item["bytes"]) is not int or item["bytes"] <= 0 or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
        for item in files.values()
    ):
        raise ValueError("Calibration tokenizer metadata has invalid bytes or hash")
    size = sum(item["bytes"] for item in files.values())
    if not 0 < size <= MAX_TOKENIZER_BYTES or source["total_bytes"] != size:
        raise ValueError("Calibration tokenizer metadata exceeded its bound")
    if source["files_sha256"] != audit.canonical_sha(files):
        raise ValueError("Calibration tokenizer metadata digest changed")
    if files["tokenizer.json"]["sha256"] != MODEL_PROFILES["qwen"]["tokenizer_sha256"]:
        raise ValueError("Calibration tokenizer differs from the frozen answer contract")


def _read_metadata(filesystem, path, expected_size):
    if not 0 < expected_size <= MAX_TOKENIZER_BYTES or filesystem.info(path)["size"] != expected_size:
        raise ValueError("Tokenizer metadata size changed before read")
    with filesystem.open(path, "rb") as stream:
        content = stream.read(expected_size + 1)
    if len(content) != expected_size:
        raise ValueError("Tokenizer metadata bytes changed during read")
    return content


def snapshot_tokenizer_source(uri):
    """Hash only tokenizer metadata in east, without copying model weights."""
    if not uri.startswith(EAST_PREFIX) or not os.environ.get("IRIS_TASK_ID"):
        raise ValueError("Read calibration tokenizer only in an east Iris task")
    filesystem, root = audit.fs_path(uri)
    files = {}
    for name in sorted(TOKENIZER_FILES):
        path = root.rstrip("/") + "/" + name
        if filesystem.exists(path):
            content = _read_metadata(filesystem, path, filesystem.info(path)["size"])
            files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    source = {
        "uri": uri,
        "files": files,
        "total_bytes": sum(v["bytes"] for v in files.values()),
        "files_sha256": audit.canonical_sha(files),
    }
    validate_tokenizer_source(source)
    return source


def tokenizer_stage_path(source):
    validate_tokenizer_source(source)
    return str(TOKENIZER_STAGE_ROOT / ("math-eval-tokenizer-" + source["files_sha256"]))


def stage_tokenizer(source):
    """Stage exact metadata bytes for the native --tokenizer path, verifying every file."""
    if not os.environ.get("IRIS_TASK_ID"):
        raise ValueError("Stage calibration tokenizer only inside Iris")
    destination = Path(tokenizer_stage_path(source))
    filesystem, root = audit.fs_path(source["uri"])
    destination.mkdir(exist_ok=True)
    if any(path.name not in source["files"] or not path.is_file() for path in destination.iterdir()):
        raise ValueError("Unexpected file in the tokenizer staging directory")
    for name, item in source["files"].items():
        content = _read_metadata(filesystem, root.rstrip("/") + "/" + name, item["bytes"])
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Original tokenizer metadata differs from the frozen byte inventory")
        path = destination / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Existing staged tokenizer metadata differs")
        path.write_bytes(content)
        if path.read_bytes() != content:
            raise ValueError("Staged tokenizer metadata failed readback")
    return str(destination)
