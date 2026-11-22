"""Shared Hugging Face model lifecycle utilities."""

import contextlib
from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any, Protocol

import huggingface_hub.constants

from .environment_contract import HF_HUB_OFFLINE_ENV, TRANSFORMERS_OFFLINE_ENV

TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
TOKENIZER_JSON_NAME = "tokenizer.json"
GENERATION_CONFIG_NAME = "generation_config.json"
END_OF_TURN_TOKENS = ("<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<end_of_turn>")
DEFAULT_GENERATION_TEMPERATURE = 0.6
DEFAULT_GENERATION_TOP_P = 0.95
_OFFLINE_ENVIRONMENT_VARIABLES = (HF_HUB_OFFLINE_ENV, TRANSFORMERS_OFFLINE_ENV)


MODEL_CACHE_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".py", ".jinja")


class GenerationConfigTokenizer(Protocol):
    """Tokenizer fields needed to derive generation stop tokens."""

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    def get_vocab(self) -> dict[str, int]: ...

    def __len__(self) -> int: ...


def _token_ids(value: int | list[int] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return list(value)


def export_generation_config(
    tokenizer: GenerationConfigTokenizer,
    *,
    chat_template: str | None,
    explicit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete preserved generation settings for a standalone SkyRL export."""
    config = dict(explicit or {})
    eos_ids = set(_token_ids(config.get("eos_token_id")))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    if chat_template:
        vocab = tokenizer.get_vocab()
        eos_ids.update(vocab[token] for token in END_OF_TURN_TOKENS if token in chat_template and token in vocab)
    vocab_size = len(tokenizer)
    for token_id in eos_ids:
        if not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
            raise ValueError(f"eos_token_id {token_id!r} is outside the tokenizer vocabulary [0, {vocab_size})")
    if eos_ids:
        config["eos_token_id"] = sorted(eos_ids)
    if "bos_token_id" not in config and tokenizer.bos_token_id is not None:
        config["bos_token_id"] = tokenizer.bos_token_id
    if config.get("do_sample", True):
        config.setdefault("do_sample", True)
        config.setdefault("temperature", DEFAULT_GENERATION_TEMPERATURE)
        config.setdefault("top_p", DEFAULT_GENERATION_TOP_P)
    return config


def write_generation_config(directory: str, config: Mapping[str, Any]) -> Path:
    """Write a deterministic generation config in a local model directory."""
    path = Path(directory) / GENERATION_CONFIG_NAME
    path.write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n")
    return path


def normalize_fast_tokenizer_metadata_bytes(payload: bytes, *, has_tokenizer_json: bool, source: str) -> bytes:
    """Rewrite Transformers 5 fast-tokenizer metadata for Transformers 4."""
    config = json.loads(payload)
    if config.get("tokenizer_class") != "TokenizersBackend":
        return payload
    if not has_tokenizer_json:
        raise ValueError(f"TokenizersBackend export is missing {TOKENIZER_JSON_NAME}: {source}")
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    return (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode()


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
