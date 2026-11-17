"""Shared tokenizer construction for training and checkpoint conversion."""

import hashlib
import json

from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from skyrl_train.inference_engines.chat_template import get_custom_chat_template


def tokenizer_vocabulary_fingerprint(tokenizer: PreTrainedTokenizerBase) -> str:
    """Hash token-ID semantics without coupling identity to a model or chat template."""
    vocabulary = tokenizer.get_vocab()
    ordered_tokens: list[str | None] = [None] * (max(vocabulary.values(), default=-1) + 1)
    for token, token_id in vocabulary.items():
        if token_id < 0 or token_id >= len(ordered_tokens) or ordered_tokens[token_id] is not None:
            raise ValueError("tokenizer vocabulary must map each non-negative token ID to exactly one token")
        ordered_tokens[token_id] = token
    if any(token is None for token in ordered_tokens):
        raise ValueError("tokenizer vocabulary token IDs must be contiguous")
    encoded = json.dumps(ordered_tokens, ensure_ascii=False, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def create_tokenizer(
    model_path: str,
    *,
    disable_fast_tokenizer: bool,
    padding_side: str = "left",
    revision: str | None = None,
) -> PreTrainedTokenizerBase:
    """Create a model tokenizer with the repository's padding contract."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=not disable_fast_tokenizer,
        revision=revision,
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def validate_checkpoint_vocabulary(
    tokenizer: PreTrainedTokenizerBase,
    *,
    checkpoint_path: str,
    checkpoint_revision: str | None,
    checkpoint_identity: str,
    tokenizer_identity: str,
    disable_fast_tokenizer: bool,
) -> None:
    """Require checkpoint vocabulary evidence that matches the tokenizer's token-ID mapping."""
    try:
        checkpoint_tokenizer = create_tokenizer(
            checkpoint_path, revision=checkpoint_revision, disable_fast_tokenizer=disable_fast_tokenizer
        )
        expected = tokenizer_vocabulary_fingerprint(checkpoint_tokenizer)
        actual = tokenizer_vocabulary_fingerprint(tokenizer)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot verify checkpoint vocabulary {checkpoint_identity!r} "
            f"against tokenizer {tokenizer_identity!r}: {error}"
        ) from error
    if expected != actual:
        raise ValueError(
            f"Checkpoint vocabulary {checkpoint_identity!r} differs from loaded tokenizer {tokenizer_identity!r}"
        )


def tokenizer_from_config(config: DictConfig, padding_side: str = "left") -> PreTrainedTokenizerBase:
    """Create the policy tokenizer that a training config names."""
    model = config.trainer.policy.model
    tokenizer = create_tokenizer(
        model_path=model.tokenizer_path,
        disable_fast_tokenizer=config.trainer.disable_fast_tokenizer,
        padding_side=padding_side,
        revision=model.get("tokenizer_revision"),
    )
    template = get_custom_chat_template(config.generator.chat_template)
    if template is not None:
        tokenizer.chat_template = template
    return tokenizer
