"""Shared Hugging Face model lifecycle utilities."""

import contextlib
import hashlib
import json
import os
from pathlib import Path

import huggingface_hub.constants

from marinskyrl.environment_contract import HF_HUB_OFFLINE_ENV, TRANSFORMERS_OFFLINE_ENV

TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
TOKENIZER_JSON_NAME = "tokenizer.json"
_OFFLINE_ENVIRONMENT_VARIABLES = (HF_HUB_OFFLINE_ENV, TRANSFORMERS_OFFLINE_ENV)


def normalize_fast_tokenizer_metadata_bytes(payload: bytes, *, has_tokenizer_json: bool, source: str) -> bytes:
    """Rewrite Transformers 5 fast-tokenizer metadata for Transformers 4."""
    config = json.loads(payload)
    if config.get("tokenizer_class") != "TokenizersBackend":
        return payload
    if not has_tokenizer_json:
        raise ValueError(f"TokenizersBackend export is missing {TOKENIZER_JSON_NAME}: {source}")
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    return (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode()


def immutable_model_cache_key(source: str, identity: str) -> str:
    """Return a stable cache key for one immutable model source."""
    return hashlib.sha256(f"{source}@{identity}".encode()).hexdigest()


@contextlib.contextmanager
def hugging_face_hub_online():
    """Temporarily allow an explicit Hub operation from an offline runtime."""
    previous_environment = {name: os.environ.pop(name) for name in _OFFLINE_ENVIRONMENT_VARIABLES if name in os.environ}
    previous_offline = huggingface_hub.constants.HF_HUB_OFFLINE
    huggingface_hub.constants.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        huggingface_hub.constants.HF_HUB_OFFLINE = previous_offline
        for name in _OFFLINE_ENVIRONMENT_VARIABLES:
            os.environ.pop(name, None)
        os.environ.update(previous_environment)


def normalize_fast_tokenizer_metadata(model_dir: Path) -> bool:
    """Rewrite Transformers 5 fast-tokenizer metadata for Transformers 4; return whether it changed."""
    config_path = model_dir / TOKENIZER_CONFIG_NAME
    if not config_path.is_file():
        return False
    original = config_path.read_bytes()
    normalized = normalize_fast_tokenizer_metadata_bytes(
        original,
        has_tokenizer_json=(model_dir / TOKENIZER_JSON_NAME).is_file(),
        source=str(model_dir),
    )
    if normalized == original:
        return False
    config_path.write_bytes(normalized)
    return True


def validate_hf_model_weights(names: set[str], source: str) -> None:
    """Validate the config and weights required by an auxiliary Hugging Face model."""
    if "config.json" not in names:
        raise ValueError(f"Model export is missing config.json: {source}")
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"Model export has no weight shards: {source}")


def validate_portable_hf_model_files(names: set[str], source: str) -> None:
    """Validate the minimum portable Hugging Face model export contract."""
    validate_hf_model_weights(names, source)
    if not any(name.startswith("tokenizer") or name.endswith(".model") for name in names):
        raise ValueError(f"Model export has no tokenizer files: {source}")
