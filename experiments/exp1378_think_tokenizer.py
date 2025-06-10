"""
Saves a modified version of the llama3 tokenizer with
1) a simple Olmo2-inspired chat format and
2) special tokens as defined in marin_models.py
"""

import json
import os
import shutil
from urllib.error import HTTPError

import numpy as np
from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer

from experiments.llama import llama3_tokenizer
from experiments.marin_models import MARIN_CHAT_TEMPLATE, MARIN_CUSTOM_SPECIAL_TOKENS, marin_tokenizer

def test_special_reserved_tokens(tokenizer):
    """
    Test that the tokenizer correctly handles the custom special tokens defined in MARIN_CUSTOM_SPECIAL_TOKENS.
    
    This function verifies that:
    1. Each token ID in MARIN_CUSTOM_SPECIAL_TOKENS decodes to its corresponding token string
    2. Each token string in MARIN_CUSTOM_SPECIAL_TOKENS encodes to its corresponding token ID
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to test
        
    Raises:
        AssertionError: If any token ID/string pair doesn't match between encoding and decoding
    """
    for token_id, token_str in MARIN_CUSTOM_SPECIAL_TOKENS.items():
        assert tokenizer.decode(token_id) == token_str
        assert tokenizer.convert_tokens_to_ids([token_str]) == [token_id]

def main():
    """
    Saves a modified version of the llama3 tokenizer with:
    1) a simple Olmo2-inspired chat format and
    2) special tokens as defined in marin_models.py.

    This script:
    - Downloads the llama3 tokenizer to a temporary directory.
    - Modifies the tokenizer configuration files (tokenizer_config.json and tokenizer.json) to replace the reserved special token strings with new ones (e.g., '<|start_think|>' and '<|end_think|>') while preserving their original token IDs.
    - Loads the modified tokenizer and assigns the MARIN_CHAT_TEMPLATE.
    - Saves the final tokenizer to a local directory.
    - Cleans up the temporary directory.
    - Optionally pushes the tokenizer to the Hugging Face Hub.
    """
    # Create temporary directory for tokenizer
    temp_dir = os.path.join(os.getcwd(), "llama_tokenizer_local")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Download llama3 tokenizer to temp directory
        tokenizer = AutoTokenizer.from_pretrained(llama3_tokenizer, cache_dir=temp_dir)

        # Save the original tokenizer to temp directory
        tokenizer.save_pretrained(temp_dir)

        # Modify and save tokenizer_config.json
        tokenizer_config_path = os.path.join(temp_dir, "tokenizer_config.json")
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        for token_id, token_str in MARIN_CUSTOM_SPECIAL_TOKENS.items():
            tokenizer_config["added_tokens_decoder"][str(token_id)]["content"] = token_str
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f)

        # Also update tokenizer.json so the underlying fast tokenizer knows about the new strings
        tokenizer_json_path = os.path.join(temp_dir, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r") as f:
                tokenizer_json = json.load(f)
            # Update the added_tokens list (id -> content)
            if "added_tokens" in tokenizer_json:
                for at in tokenizer_json["added_tokens"]:
                    tid = at.get("id")
                    if tid in MARIN_CUSTOM_SPECIAL_TOKENS:
                        at["content"] = MARIN_CUSTOM_SPECIAL_TOKENS[tid]
            # Persist the file back
            with open(tokenizer_json_path, "w") as f:
                json.dump(tokenizer_json, f)

        # Load the modified tokenizer
        marin = AutoTokenizer.from_pretrained(temp_dir)

        # Assign marin template
        marin.chat_template = MARIN_CHAT_TEMPLATE

        # Save final version
        final_dir = os.path.join(os.getcwd(), "marin_tokenizer")
        marin.save_pretrained(final_dir)

        # Clean up temp directory
        shutil.rmtree(temp_dir)

    except (OSError, GatedRepoError, HTTPError) as e:
        print("You need to request access to the llama3 tokenizer")
        if os.getenv("CI", False) in ["true", "1"]:
            print("Skipping test in CI")
            return
        raise e

    # Load from final location
    marin = AutoTokenizer.from_pretrained(final_dir, local_files_only=True)

    # Test 1: Tokenizer is modified to use new special tokens
    test_special_reserved_tokens(tokenizer)

    # Test 2: check chat template
    assert marin.chat_template == MARIN_CHAT_TEMPLATE
    
    reasoning_trace_example = "<|start_think|>User is asking how am I doing. \
        This should be straightforward. \
        I should reply politely.<|end_think|>"
    convo = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": reasoning_trace_example + "I'm doing well, thanks!"},
        {"role": "user", "content": "That's good to hear!"},
        {"role": "assistant", "content": "Great!"},
    ]

    # Test 3: Ensure that we didn't mess up normal tokenization
    llama3 = AutoTokenizer.from_pretrained(llama3_tokenizer)
    assert marin.tokenize("Hello, how are you?") == llama3.tokenize("Hello, how are you?")

    # Test 4: Make sure we use the special tokens in _chat_ template
    out = marin.apply_chat_template(convo, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    assert all(
        token in out["input_ids"]
        for token in marin.convert_tokens_to_ids(["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    )

    # Test 5: Ensure masking of assistant tokens works correctly
    ids = np.array(out["input_ids"])
    assert np.sum(out["assistant_masks"]) == (
        len(marin(reasoning_trace_example + "I'm doing well, thanks!")["input_ids"]) + len(marin("Great!")["input_ids"])
    )

    assert (
        marin.decode(ids[np.array(out["assistant_masks"]).astype(bool)])
        == reasoning_trace_example + "I'm doing well, thanks!<|eot_id|>Great!<|eot_id|>"
    )

    # Test 6: Ensure that when we use add_generation_prompt, we add the final newline (and other bits)
    assert marin.apply_chat_template(convo, tokenize=False, add_generation_prompt=True).endswith(
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    # Push to huggingface
    marin.push_to_hub(marin_tokenizer)


if __name__ == "__main__":
    main()
