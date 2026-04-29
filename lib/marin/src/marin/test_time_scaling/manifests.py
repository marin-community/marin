# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import fsspec
from fsspec.core import url_to_fs

from marin.test_time_scaling.config import ScoringMode

MANIFEST_FILENAME = "manifest.json"
PROMPTS_FILENAME = "prompts.jsonl"
MANIFEST_FORMAT_VERSION = 1


def _artifact_path(base_path: str, filename: str) -> str:
    if not base_path:
        return filename
    return f"{base_path.rstrip('/')}/{filename}"


def _ensure_parent_dir(path: str) -> None:
    fs, fs_path = url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.mkdirs(parent, exist_ok=True)


@dataclass(frozen=True)
class PromptMessage:
    """Single chat message for a prompt."""

    role: str
    content: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptMessage:
        return cls(role=str(data["role"]), content=str(data["content"]))

    def to_openai_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, str]:
        return self.to_openai_dict()


@dataclass(frozen=True)
class PromptManifestRecord:
    """Prompt record written to and read from `prompts.jsonl`."""

    prompt_id: str
    messages: tuple[PromptMessage, ...]
    expected_answer: str | None = None
    scoring_mode: ScoringMode = ScoringMode.UNSCORED
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptManifestRecord:
        return cls(
            prompt_id=str(data["prompt_id"]),
            messages=tuple(PromptMessage.from_dict(message) for message in data["messages"]),
            expected_answer=data.get("expected_answer"),
            scoring_mode=ScoringMode(data.get("scoring_mode", ScoringMode.UNSCORED)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "messages": [message.to_dict() for message in self.messages],
            "expected_answer": self.expected_answer,
            "scoring_mode": self.scoring_mode.value,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PromptManifest:
    """Prompt manifest metadata plus ordered prompt records."""

    manifest_id: str
    task_name: str
    records: tuple[PromptManifestRecord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def header_dict(self) -> dict[str, Any]:
        return {
            "format_version": MANIFEST_FORMAT_VERSION,
            "manifest_id": self.manifest_id,
            "task_name": self.task_name,
            "num_prompts": len(self.records),
            "metadata": self.metadata,
        }


def write_prompt_manifest(output_dir: str, manifest: PromptManifest) -> None:
    """Write `manifest.json` and `prompts.jsonl` for a prompt manifest."""

    manifest_path = _artifact_path(output_dir, MANIFEST_FILENAME)
    prompts_path = _artifact_path(output_dir, PROMPTS_FILENAME)
    _ensure_parent_dir(manifest_path)

    with fsspec.open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.header_dict(), handle, indent=2, sort_keys=True)

    _ensure_parent_dir(prompts_path)
    with fsspec.open(prompts_path, "w", encoding="utf-8") as handle:
        for record in manifest.records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def load_prompt_manifest(path: str) -> PromptManifest:
    """Load a prompt manifest from a directory, `manifest.json`, or `prompts.jsonl`."""

    fs, fs_path = url_to_fs(path)
    if fs.isdir(fs_path):
        manifest_path = _artifact_path(path, MANIFEST_FILENAME)
        prompts_path = _artifact_path(path, PROMPTS_FILENAME)
        with fsspec.open(manifest_path, "r", encoding="utf-8") as handle:
            header = json.load(handle)
    elif fs_path.endswith(MANIFEST_FILENAME):
        manifest_path = path
        prompts_path = _artifact_path(path.rpartition("/")[0], PROMPTS_FILENAME)
        with fsspec.open(manifest_path, "r", encoding="utf-8") as handle:
            header = json.load(handle)
    elif fs_path.endswith(".jsonl"):
        prompts_path = path
        header = {
            "manifest_id": fs_path.rsplit("/", 1)[-1].removesuffix(".jsonl"),
            "task_name": fs_path.rsplit("/", 1)[-1].removesuffix(".jsonl"),
            "metadata": {},
        }
    else:
        raise ValueError(f"Unsupported manifest path: {path}")

    records: list[PromptManifestRecord] = []
    with fsspec.open(prompts_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(PromptManifestRecord.from_dict(json.loads(stripped)))

    return PromptManifest(
        manifest_id=str(header["manifest_id"]),
        task_name=str(header.get("task_name", header["manifest_id"])),
        records=tuple(records),
        metadata=dict(header.get("metadata", {})),
    )
