"""
Saves a modified version of the llama3 tokenizer with a simple Olmo2-inspired chat format.
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

# Olmo 2 template modified so we can use with levanter
MARIN_OLMO2_CHAT_TEMPLATE = """
{{ bos_token }}
{%- for message in messages -%}
  {%- if message['role'] == 'system' -%}
<|system|>\n{{ message['content'] | trim }}\n
  {%- elif message['role'] == 'user' -%}
<|user|>\n{{ message['content'] | trim }}\n
  {%- elif message['role'] == 'assistant' -%}
    {%- if not loop.last -%}
<|assistant|>\n{% generation %}{{ message['content'] | trim }}{{ eos_token }}{% endgeneration %}\n
    {%- else -%}
<|assistant|>\n{% generation %}{{ message['content'] | trim }}{{ eos_token }}{% endgeneration %}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>\n
{%- endif -%}""".strip()


def main():
    # Create temporary directory for tokenizer
    temp_dir = os.path.join(os.getcwd(), "experiments/exp9999/temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Download llama3 tokenizer to temp directory
        tokenizer = AutoTokenizer.from_pretrained(llama3_tokenizer, cache_dir=temp_dir)
        tokenizer.save_pretrained(temp_dir)

        # Modify tokenizer_config.json
        config_path = os.path.join(temp_dir, "tokenizer_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        for id, t in MARIN_CUSTOM_SPECIAL_TOKENS.items():
            config["added_tokens_decoder"][str(id)]["content"] = t

        # Save the config
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Load the modified tokenizer
        marin = AutoTokenizer.from_pretrained(temp_dir, local_files_only=True)

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

    assert marin.chat_template == MARIN_CHAT_TEMPLATE

    # Load from final location
    marin = AutoTokenizer.from_pretrained(final_dir, local_files_only=True)

    # Test 1: Tokenizer is modified to use new special tokens
    for id, t in MARIN_CUSTOM_SPECIAL_TOKENS.items():
        assert marin.decode([id]) == t
        assert marin.tokenize(t) == id

    # Tests (including those from exp964 since we are starting from base llama3)
    reasoning_trace_example = "<|start_think|>User is asking how am I doing. This should be straightforward. I should reply politely.<|end_think|>"
    convo = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": reasoning_trace_example + "I'm doing well, thanks!"},
        {"role": "user", "content": "That's good to hear!"},
        {"role": "assistant", "content": "Great!"},
    ]

    # Test 2: Ensure that we didn't mess up normal tokenization
    llama3 = AutoTokenizer.from_pretrained(llama3_tokenizer)
    assert marin.tokenize("Hello, how are you?") == llama3.tokenize("Hello, how are you?")

    # Test 3: Make sure we use the special tokens in _chat_ template
    out = marin.apply_chat_template(convo, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    assert all(
        token in out["input_ids"]
        for token in marin.convert_tokens_to_ids(["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    )

    # Test 4: Ensure masking of assistant tokens works correctly
    ids = np.array(out["input_ids"])
    assert np.sum(out["assistant_masks"]) == (
        len(marin(reasoning_trace_example + "I'm doing well, thanks!")["input_ids"]) + len(marin("Great!")["input_ids"])
    )

    assert (
        marin.decode(ids[np.array(out["assistant_masks"]).astype(bool)])
        == reasoning_trace_example + "I'm doing well, thanks!<|eot_id|>Great!<|eot_id|>"
    )

    # Test 5: Ensure that when we use add_generation_prompt, we add the final newline (and other bits)
    assert marin.apply_chat_template(convo, tokenize=False, add_generation_prompt=True).endswith(
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    # Push to huggingface
    marin.push_to_hub(marin_tokenizer)


if __name__ == "__main__":
    main()
