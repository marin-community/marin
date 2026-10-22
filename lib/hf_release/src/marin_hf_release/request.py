"""Resolved, retry-stable inference export requests and metadata."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from marin_hf_release.policy import ReleasePolicy, TensorSpec

REQUEST_NAME = "hf_release_request.json"
MANIFEST_NAME = "hf_release_manifest.json"


def exporter_revision(backend_file: Path) -> str:
    """Identify the exporter commit or hash its source when Git metadata is absent."""
    commit = os.environ.get("GIT_COMMIT")
    if commit is None and shutil.which("git") is not None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=backend_file.parent, capture_output=True, text=True, check=False
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    if commit is not None and len(commit) == 40 and all(char in "0123456789abcdef" for char in commit):
        return commit
    digest = hashlib.sha256()
    for path in (backend_file, Path(__file__), Path(__file__).with_name("policy.py"), Path(__file__).with_name("validation.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return f"source-sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class TokenizerMetadata:
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int
    chat_template: str | None
    special_token_ids: dict[str, int] = field(default_factory=dict)


def tokenizer_special_token_ids(tokenizer: object) -> dict[str, int]:
    """Capture the named special-token IDs saved with an HF tokenizer."""
    names = ("bos_token", "eos_token", "pad_token", "unk_token", "sep_token", "cls_token", "mask_token")
    result = {}
    for name in names:
        token_id = getattr(tokenizer, f"{name}_id")
        if token_id is not None:
            result[name] = token_id
    for index, token_id in enumerate(getattr(tokenizer, "extra_special_tokens_ids")):
        result[f"extra_special_tokens_{index}"] = token_id
    return result


def resolve_metadata(
    explicit: Mapping[str, object],
    checkpoint: Mapping[str, object],
    source: Mapping[str, object],
    *,
    require_chat_template: bool,
) -> tuple[TokenizerMetadata, dict[str, object]]:
    """Resolve behavior metadata fieldwise, with request values taking precedence."""
    layers = (explicit, checkpoint, source)

    def value(key: str) -> object | None:
        return next((layer[key] for layer in layers if layer.get(key) is not None), None)

    vocab_size = value("vocab_size")
    eos_token_id = value("eos_token_id")
    bos_token_id = value("bos_token_id")
    chat_template = value("chat_template")
    special_token_ids = value("special_token_ids") or {}
    eos_token_ids = value("eos_token_ids")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("HF release requires tokenizer vocab_size")
    if not isinstance(eos_token_id, int) or not 0 <= eos_token_id < vocab_size:
        raise ValueError("HF release requires a valid tokenizer eos_token_id")
    if bos_token_id is not None and (not isinstance(bos_token_id, int) or not 0 <= bos_token_id < vocab_size):
        raise ValueError("HF release bos_token_id is outside tokenizer vocabulary")
    if not isinstance(eos_token_ids, list) or not eos_token_ids:
        raise ValueError("HF release requires a nonempty eos_token_ids list")
    if any(not isinstance(token_id, int) or not 0 <= token_id < vocab_size for token_id in eos_token_ids):
        raise ValueError("HF release EOS token IDs are outside tokenizer vocabulary")
    if eos_token_id not in eos_token_ids:
        raise ValueError("tokenizer eos_token_id must appear in generation eos_token_ids")
    if require_chat_template and (not isinstance(chat_template, str) or not chat_template):
        raise ValueError("Grug HF release requires a chat template")
    if chat_template is not None and not isinstance(chat_template, str):
        raise ValueError("HF release chat_template must be a string")
    if not isinstance(special_token_ids, dict) or any(
        not isinstance(name, str) or not isinstance(token_id, int) or not 0 <= token_id < vocab_size
        for name, token_id in special_token_ids.items()
    ):
        raise ValueError("HF release special token IDs are outside tokenizer vocabulary")
    if special_token_ids.get("eos_token") not in (None, eos_token_id):
        raise ValueError("HF release tokenizer EOS ID conflicts with special token metadata")
    if special_token_ids.get("bos_token") not in (None, bos_token_id):
        raise ValueError("HF release tokenizer BOS ID conflicts with special token metadata")
    controls: dict[str, object] = {}
    for layer in reversed(layers):
        controls.update(layer.get("generation_controls", {}))
    generation = {**controls, "eos_token_id": eos_token_ids}
    if bos_token_id is not None:
        generation["bos_token_id"] = bos_token_id
    return TokenizerMetadata(vocab_size, bos_token_id, eos_token_id, chat_template, special_token_ids), generation


@dataclass(frozen=True)
class ReleaseRequest:
    """All decisions that affect one public HF inference artifact."""

    policy: ReleasePolicy
    schema_id: str
    tensors: dict[str, TensorSpec]
    config: dict[str, object]
    generation: dict[str, object]
    tokenizer: TokenizerMetadata
    source_checkpoint: str
    source_step: int | None
    checkpoint_identity: str
    source_model: str | None
    source_model_commit: str | None
    exporter_revision: str

    def __post_init__(self) -> None:
        if not self.schema_id or not self.tensors or not self.source_checkpoint or not self.checkpoint_identity:
            raise ValueError("HF release request requires schema, tensors, and checkpoint identity")
        if not self.exporter_revision:
            raise ValueError("HF release request requires exporter code revision")
        if self.source_model is not None and self.source_model_commit is None:
            raise ValueError("HF release source model must be pinned to a commit")
        if self.source_model_commit is not None and (
            len(self.source_model_commit) != 40
            or any(char not in "0123456789abcdef" for char in self.source_model_commit)
        ):
            raise ValueError("HF release source model commit must be a 40-character SHA")
        for field_name, expected in (
            ("bos_token_id", self.tokenizer.bos_token_id),
            ("eos_token_id", self.tokenizer.eos_token_id),
            ("pad_token_id", self.tokenizer.special_token_ids.get("pad_token")),
        ):
            declared = self.config.get(field_name)
            if declared is not None and declared != expected:
                raise ValueError(f"HF release config {field_name} conflicts with tokenizer metadata")

    def payload(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def persist_request(root: Path, request: ReleaseRequest) -> None:
    """Store the resolved request before weight export; reject changed retries."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / REQUEST_NAME
    payload = json.dumps(request.payload(), sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"HF release request changed for existing destination: {root}")
        return
    path.write_text(payload)


def validate_release_metadata(
    root: Path, request: ReleaseRequest, file_names: Iterable[str] | None = None
) -> dict[str, str]:
    """Check saved config, tokenizer IDs/template, and generation behavior."""
    if json.loads((root / "config.json").read_text()) != request.config:
        raise ValueError("HF release config.json differs from resolved request")
    if json.loads((root / "generation_config.json").read_text()) != request.generation:
        raise ValueError("HF release generation_config.json differs from resolved request")
    tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
    template = request.tokenizer.chat_template
    if template is not None:
        if tokenizer_config.get("chat_template") != template:
            raise ValueError("HF release tokenizer_config.json has a different chat template")
        template_path = root / "chat_template.jinja"
        if template_path.is_file() and template_path.read_text() != template:
            raise ValueError("HF release chat_template.jinja differs from tokenizer_config.json")
    tokenizer_json = json.loads((root / "tokenizer.json").read_text())
    vocabulary = tokenizer_json["model"]["vocab"]
    tokens_to_ids = (
        dict(vocabulary) if isinstance(vocabulary, dict)
        else {entry[0]: index for index, entry in enumerate(vocabulary)}
    )
    tokens_to_ids.update({token["content"]: token["id"] for token in tokenizer_json.get("added_tokens", [])})
    ids = set(tokens_to_ids.values())
    if len(ids) != request.tokenizer.vocab_size:
        raise ValueError("HF release tokenizer vocabulary size differs from resolved request")
    required_ids = [request.tokenizer.eos_token_id, *request.generation["eos_token_id"]]
    if request.tokenizer.bos_token_id is not None:
        required_ids.append(request.tokenizer.bos_token_id)
    if any(token_id not in ids for token_id in required_ids):
        raise ValueError("HF release special or EOS token ID is missing from tokenizer vocabulary")
    for name, token_id in request.tokenizer.special_token_ids.items():
        if name.startswith("extra_special_tokens_"):
            index = int(name.removeprefix("extra_special_tokens_"))
            extra_tokens = tokenizer_config.get("extra_special_tokens", [])
            token_value = extra_tokens[index] if index < len(extra_tokens) else None
        else:
            token_value = tokenizer_config.get(name)
        if isinstance(token_value, dict):
            token_value = token_value.get("content")
        if not isinstance(token_value, str) or tokens_to_ids.get(token_value) != token_id:
            raise ValueError(f"HF release {name} differs from resolved tokenizer metadata")
    names = (
        file_names
        if file_names is not None
        else (
            path.name
            for path in root.iterdir()
            if path.is_file() and not path.name.endswith(".safetensors") and path.name != MANIFEST_NAME
        )
    )
    return {name: _file_sha256(root / name) for name in names}


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def completion_manifest(request: ReleaseRequest, file_hashes: Mapping[str, str]) -> dict[str, object]:
    """Return a provenance marker for an already validated release."""
    return {
        "policy_version": request.policy.version,
        "request_sha256": request.digest(),
        "source_checkpoint": request.source_checkpoint,
        "source_step": request.source_step,
        "checkpoint_identity": request.checkpoint_identity,
        "source_model": request.source_model,
        "source_model_commit": request.source_model_commit,
        "exporter_revision": request.exporter_revision,
        "files": dict(sorted(file_hashes.items())),
    }


def write_completion_manifest(root: Path, request: ReleaseRequest, file_hashes: Mapping[str, str]) -> None:
    """Write the completion marker after every file has passed validation."""
    marker = root / MANIFEST_NAME
    payload = json.dumps(completion_manifest(request, file_hashes), sort_keys=True, indent=2) + "\n"
    if marker.exists():
        if marker.read_text() != payload:
            raise FileExistsError(f"HF release is already complete with different contents: {root}")
        return
    marker.write_text(payload)
