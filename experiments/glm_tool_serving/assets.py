# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize and verify the tokenizer assets required by a fresh fleet checkout."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from experiments.post_training.task_curation.reference_data import read_reference_index, read_referenced_document


def provision_tokenizers(data: Path) -> None:
    """Fetch missing assets and reject cached bytes that differ from the committed pins."""
    index_path = data / "reference-index.json"
    reference = next(item for item in read_reference_index(index_path).references if item.path == "sft_tokenizer")
    sft = data / reference.path
    raw = sft.read_bytes() if sft.exists() else read_referenced_document(index_path, reference.path)
    if hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("SFT tokenizer differs from the committed pin")
    if not sft.exists():
        sft.write_bytes(raw)
    serving = data / "serving_tokenizer"
    pins = json.loads((serving / "pins.json").read_text())
    for name, expected in pins["files"].items():
        target = serving / name
        source = (
            target
            if target.exists()
            else Path(
                hf_hub_download(
                    repo_id=pins["repository"],
                    revision=pins["declared_revision"],
                    filename=name,
                )
            )
        )
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Serving tokenizer asset differs from the committed pin: {name}")
        if not target.exists():
            target.write_bytes(raw)
