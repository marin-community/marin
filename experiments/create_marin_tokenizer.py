"""
Saves a modified version of the llama3 tokenizer with:
1) a simple Olmo2-inspired chat format and
2) special tokens as defined in marin_models.py

The script uses temporary in-memory storage for intermediate operations.

"""

import json
import os
import tempfile
from urllib.error import HTTPError

import numpy as np
import pytest
from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer, PreTrainedTokenizer

from experiments.llama import llama3_tokenizer as llama3_tokenizer_hf_path
from experiments.marin_models import MARIN_CHAT_TEMPLATE, MARIN_CUSTOM_SPECIAL_TOKENS
from experiments.marin_models import marin_tokenizer as marin_tokenizer_hf_path


def _inject_special_tokens(
    tokenizer: PreTrainedTokenizer,
    new_tokens: dict[int, str],
):
    """
    Inject special tokens into the tokenizer config.

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
        for token_id, token_str in new_tokens.items():
            tokenizer_config["added_tokens_decoder"][str(token_id)]["content"] = token_str
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f)

        # Also update tokenizer.json so the underlying fast tokenizer knows about the new strings
        tokenizer_json_path = os.path.join(temp_path, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r") as f:
                tokenizer_json = json.load(f)
            # Update the added_tokens list (id -> content)
            if "added_tokens" in tokenizer_json:
                for at in tokenizer_json["added_tokens"]:
                    tid = at.get("id")
                    if tid in new_tokens:
                        at["content"] = new_tokens[tid]
            # Persist the file back
            with open(tokenizer_json_path, "w") as f:
                json.dump(tokenizer_json, f)

        # Load the modified tokenizer
        return AutoTokenizer.from_pretrained(temp_path)


def create_marin_tokenizer(
    tokenizer: PreTrainedTokenizer,
) -> PreTrainedTokenizer:
    """
    Create a modified version of tokenizer with custom chat format and special tokens.

    Args:
        tokenizer: The base tokenizer to modify (typically llama3)

    Returns:
        A new tokenizer instance
    """
    # Inject special tokens
    marin_tokenizer = _inject_special_tokens(tokenizer, MARIN_CUSTOM_SPECIAL_TOKENS)

    # Assign marin template
    marin_tokenizer.chat_template = MARIN_CHAT_TEMPLATE

    return marin_tokenizer


def load_llama3_tokenizer() -> PreTrainedTokenizer:
    """
    Load the base llama3 tokenizer.

    Returns:
        The llama3 tokenizer instance

    Raises:
        OSError, GatedRepoError, HTTPError: If access to the tokenizer is not available
    """
    try:
        return AutoTokenizer.from_pretrained(llama3_tokenizer_hf_path)
    except (OSError, GatedRepoError, HTTPError) as e:
        print("You need to request access to the llama3 tokenizer")
        if os.getenv("CI", False) in ["true", "1"]:
            print("Skipping test in CI")
            return
        raise e


# ============ Test data and functions ============
REASONING_TRACE_EXAMPLE = "<|start_think|>User is asking how am I doing. \
    This should be straightforward. \
    I should reply politely.<|end_think|>"

TEST_CONVERSATION = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": REASONING_TRACE_EXAMPLE + "I'm doing well, thanks!"},
    {"role": "user", "content": "That's good to hear!"},
    {"role": "assistant", "content": "Great!"},
]


@pytest.fixture
def marin_tokenizer():
    """Fixture that provides a configured marin tokenizer for testing."""
    llama3_tokenizer = load_llama3_tokenizer()
    tokenizer = create_marin_tokenizer(llama3_tokenizer)

    # Roundtrip write-read to ensure consistency
    with tempfile.TemporaryDirectory() as temp_path:
        tokenizer.save_pretrained(temp_path)
        tokenizer = AutoTokenizer.from_pretrained(temp_path, local_files_only=True)

    return tokenizer


def test_special_tokens(marin_tokenizer: PreTrainedTokenizer):
    """Test that special tokens are correctly replaced."""
    for token_id, token_str in MARIN_CUSTOM_SPECIAL_TOKENS.items():
        assert marin_tokenizer.decode(token_id) == token_str
        assert marin_tokenizer.convert_tokens_to_ids([token_str]) == [token_id]


def test_chat_template(marin_tokenizer: PreTrainedTokenizer):
    """Test that chat template is correctly set."""
    assert marin_tokenizer.chat_template == MARIN_CHAT_TEMPLATE


def test_normal_tokenization(marin_tokenizer: PreTrainedTokenizer):
    """Test that normal tokenization is preserved."""
    llama3_tokenizer = load_llama3_tokenizer()
    assert marin_tokenizer.tokenize("Hello, how are you?") == llama3_tokenizer.tokenize("Hello, how are you?")


def test_chat_template_special_tokens(marin_tokenizer: PreTrainedTokenizer):
    """Test that special tokens are used in chat template."""
    out = marin_tokenizer.apply_chat_template(
        TEST_CONVERSATION, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    assert all(
        token in out["input_ids"]
        for token in marin_tokenizer.convert_tokens_to_ids(["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    )


def test_assistant_token_masking(marin_tokenizer: PreTrainedTokenizer):
    """Test that assistant tokens are masked correctly."""
    out = marin_tokenizer.apply_chat_template(
        TEST_CONVERSATION, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    expected_length = len(marin_tokenizer(REASONING_TRACE_EXAMPLE + "I'm doing well, thanks!")["input_ids"]) + len(
        marin_tokenizer("Great!")["input_ids"]
    )
    assert np.sum(out["assistant_masks"]) == expected_length


def test_assistant_token_decoding(marin_tokenizer: PreTrainedTokenizer):
    """Test that decoding of assistant tokens is correct."""
    out = marin_tokenizer.apply_chat_template(
        TEST_CONVERSATION, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    ids = np.array(out["input_ids"])
    expected_text = REASONING_TRACE_EXAMPLE + "I'm doing well, thanks!<|eot_id|>Great!<|eot_id|>"
    assert marin_tokenizer.decode(ids[np.array(out["assistant_masks"]).astype(bool)]) == expected_text


def test_generation_prompt(marin_tokenizer: PreTrainedTokenizer):
    """Test that add_generation_prompt adds the final newline."""
    assert marin_tokenizer.apply_chat_template(TEST_CONVERSATION, tokenize=False, add_generation_prompt=True).endswith(
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def run_all_tests(marin_tokenizer: PreTrainedTokenizer):
    """Run all tests on the modified tokenizer."""
    test_special_tokens(marin_tokenizer)
    test_chat_template(marin_tokenizer)
    test_normal_tokenization(marin_tokenizer)
    test_chat_template_special_tokens(marin_tokenizer)
    test_assistant_token_masking(marin_tokenizer)
    test_assistant_token_decoding(marin_tokenizer)
    test_generation_prompt(marin_tokenizer)
    print("All tests passed!")


# ============ Main function ============
def main():
    """
    Create and save a modified version of the llama3 tokenizer.

    The script:
    1) Loads the base llama3 tokenizer
    2) Creates a modified version with custom chat format and special tokens
    3) Performs a roundtrip write-read to ensure consistency
    4) Runs validation tests
    5) Optionally pushes to Hugging Face Hub
    """
    # Load llama3 tokenizer
    llama3_tokenizer = load_llama3_tokenizer()

    # Create marin tokenizer from llama3 tokenizer
    marin_tokenizer = create_marin_tokenizer(
        llama3_tokenizer,
    )

    # Roundtrip write-read
    with tempfile.TemporaryDirectory() as temp_path:
        marin_tokenizer.save_pretrained(temp_path)
        marin_tokenizer = AutoTokenizer.from_pretrained(temp_path, local_files_only=True)

    # Run tests to make sure that the tokenizer is modified correctly
    run_all_tests(marin_tokenizer)

    # Push to huggingface
    marin_tokenizer.push_to_hub(marin_tokenizer_hf_path)


if __name__ == "__main__":
    main()
