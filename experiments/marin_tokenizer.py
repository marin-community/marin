# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Saves a modified version of the Marin tokenizer with:
1) a simple Olmo2-inspired chat format and
2) special tokens defined in this module.

The script uses temporary in-memory storage for intermediate operations.

"""

import json
import os
import tempfile
from typing import cast

from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from transformers import AutoTokenizer, PreTrainedTokenizer

marin_tokenizer = "marin-community/marin-tokenizer"
"""
The HF Hub name for the Marin tokenizer.
The Marin tokenizer is (currently) just the Llama 3 tokenizer with a custom chat template (MARIN_CHAT_TEMPLATE).
"""
MARIN_TOKENIZER_BASE_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
MARIN_TOKENIZER_TOOLS_REPO = "marin-community/marin-tokenizer-tools"

MARIN_CUSTOM_SPECIAL_TOKENS = {
    128002: "<|start_think|>",  # Originally "<|reserved_special_token_0|>"
    128003: "<|end_think|>",  # Originally "<|reserved_special_token_1|>"
    128005: "<tool_call>",  # Originally "<|reserved_special_token_2|>"
    128011: "</tool_call>",  # Originally "<|reserved_special_token_3|>"
}


def inject_special_tokens(
    tokenizer: PreTrainedTokenizer,
    new_tokens: dict[int, str],
) -> PreTrainedTokenizer:
    """
    Rename existing vocabulary slots to new special-token strings.

    Rewrites ``added_tokens_decoder`` (``tokenizer_config.json``) and the fast tokenizer's
    ``added_tokens`` (``tokenizer.json``) so each ``token_id`` decodes to ``token_str`` and the
    string round-trips to that single id. A slot that does not exist yet is added.

    Args:
        tokenizer: The tokenizer to modify
        new_tokens: A dictionary of token_id -> token_str

    Returns:
        A new tokenizer instance
    """
    # Create a temporary directory that may be RAM-based
    with tempfile.TemporaryDirectory() as temp_path:
        # Save the original tokenizer to temp directory
        tokenizer.save_pretrained(temp_path)

        # Modify and save tokenizer_config.json
        tokenizer_config_path = os.path.join(temp_path, "tokenizer_config.json")
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        added_tokens_decoder = tokenizer_config.setdefault("added_tokens_decoder", {})
        for token_id, token_str in new_tokens.items():
            token_config = added_tokens_decoder.setdefault(
                str(token_id),
                {
                    "content": token_str,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
            )
            token_config["content"] = token_str
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f)

        # Also update tokenizer.json so the underlying fast tokenizer knows about the new strings
        tokenizer_json_path = os.path.join(temp_path, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r") as f:
                tokenizer_json = json.load(f)
            # Update the added_tokens list (id -> content)
            if "added_tokens" in tokenizer_json:
                seen_token_ids = set()
                for at in tokenizer_json["added_tokens"]:
                    tid = at.get("id")
                    if tid in new_tokens:
                        seen_token_ids.add(tid)
                        at["content"] = new_tokens[tid]
                for tid, token_str in new_tokens.items():
                    if tid not in seen_token_ids:
                        tokenizer_json["added_tokens"].append(
                            {
                                "id": tid,
                                "content": token_str,
                                "single_word": False,
                                "lstrip": False,
                                "rstrip": False,
                                "normalized": False,
                                "special": True,
                            }
                        )
            # Persist the file back
            with open(tokenizer_json_path, "w") as f:
                json.dump(tokenizer_json, f)

        # Load the modified tokenizer
        return cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(temp_path))


def create_marin_tokenizer(
    tokenizer: PreTrainedTokenizer,
    token_renames: dict[int, str],
) -> PreTrainedTokenizer:
    """
    Create a modified version of tokenizer with custom chat format and special tokens.

    Args:
        tokenizer: The base tokenizer to modify (typically llama3)
        token_renames: Existing token IDs mapped to their Marin token strings

    Returns:
        A new tokenizer instance
    """
    # Inject special tokens
    marin_tokenizer = inject_special_tokens(tokenizer, token_renames)

    # Assign marin template
    marin_tokenizer.chat_template = MARIN_CHAT_TEMPLATE

    return marin_tokenizer


def load_base_marin_tokenizer() -> PreTrainedTokenizer:
    """
    Load the pinned Marin tokenizer used for the tool-token variant.

    Returns:
        The base Marin tokenizer instance

    Raises:
        OSError, GatedRepoError, HTTPError: If access to the tokenizer is not available
    """
    return cast(
        PreTrainedTokenizer, AutoTokenizer.from_pretrained(marin_tokenizer, revision=MARIN_TOKENIZER_BASE_REVISION)
    )


def main(dry_run: bool = False):
    """
    Create the Marin tokenizer and push it to the Hugging Face Hub.

    Loads the pinned Marin tokenizer, applies the custom chat format and special
    tokens, performs a roundtrip write-read to ensure consistency, and (unless
    ``dry_run``) pushes the result to the Hub.
    """
    base_tokenizer = load_base_marin_tokenizer()
    tokenizer = create_marin_tokenizer(base_tokenizer, MARIN_CUSTOM_SPECIAL_TOKENS)

    # Roundtrip write-read to ensure consistency.
    with tempfile.TemporaryDirectory() as temp_path:
        tokenizer.save_pretrained(temp_path)
        tokenizer = AutoTokenizer.from_pretrained(temp_path, local_files_only=True)

    if not dry_run:
        tokenizer.push_to_hub(MARIN_TOKENIZER_TOOLS_REPO)


if __name__ == "__main__":
    import sys

    main(dry_run="--dry-run" in sys.argv)
