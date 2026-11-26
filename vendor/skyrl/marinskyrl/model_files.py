"""Identify portable Hugging Face exports and their completion manifests."""

import hashlib
import json

from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

from .hf_model import MODEL_CACHE_SUFFIXES, validate_hf_model_weights, validate_portable_hf_model_files

MODEL_MANIFEST_FILENAME = ".marinskyrl-model-manifest.json"


def hugging_face_export_digest(uri: str, *, embedded_tokenizer: bool) -> str:
    """Bind HF files by cloud object identity or local content without downloading cloud weights."""
    root = StoragePath(uri)
    fs, path = url_to_fs(uri)
    files = {}
    for entry in fs.ls(path, detail=True):
        name = entry["name"].rsplit("/", 1)[-1]
        if entry["type"] != "file" or not name.endswith((*MODEL_CACHE_SUFFIXES, ".bin")):
            continue
        checksum = entry.get("generation") or entry.get("ETag")
        if root.scheme in (None, "file"):
            with (root / name).open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if not checksum:
            raise ValueError(f"Cannot identify the model export file {root / name}")
        files[name] = (entry["size"], checksum)
    validate = validate_portable_hf_model_files if embedded_tokenizer else validate_hf_model_weights
    validate(set(files), uri)
    return hashlib.sha256(json.dumps((uri, files), sort_keys=True).encode()).hexdigest()
