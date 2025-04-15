"""
Saves a modified version of the llama3 tokenizer with a simple Olmo2-inspired chat format.
"""

import os
from urllib.error import HTTPError

import numpy as np
from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer

from experiments.llama import llama3_instruct_tokenizer, llama3_tokenizer

# name for the hf hub. cf llama3_tokenizer
marin_tokenizer = "stanford-crfm/marin-tokenizer"

# to be clear this is the Olmo 2 template except we use llama3's special tokens
# we also add the {% generation -%} tag stuff that makes the assistant_mask work
MARIN_TEMPLATE = """
{{ bos_token }}
{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>
{% generation %}{{- message['content'] | trim }}<|eot_id|>{% endgeneration %}\n
{% else %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] | trim }}<|eot_id|>
{% endif %}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{%- endif -%}
""".strip()


def main():
    try:
        marin = AutoTokenizer.from_pretrained(llama3_tokenizer)
    except (OSError, GatedRepoError, HTTPError) as e:
        print("You need to request access to the llama3 tokenizer")
        if os.getenv("CI", False) in ["true", "1"]:
            print("Skipping test in CI")
            return
        raise e

    marin.chat_template = MARIN_TEMPLATE

    marin.save_pretrained(os.path.join(os.getcwd(), "marin_tokenizer"))

    marin = AutoTokenizer.from_pretrained(os.path.join(os.getcwd(), "marin_tokenizer"))

    assert marin.chat_template == MARIN_TEMPLATE

    olmo2 = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-SFT")
    llama3_instruct = AutoTokenizer.from_pretrained(llama3_instruct_tokenizer)
    llama3 = AutoTokenizer.from_pretrained(llama3_tokenizer)

    # try it out a bit
    convo = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
        {"role": "user", "content": "That's good to hear!"},
        {"role": "assistant", "content": "Great!"},
    ]
    print("======")
    print("olmo2")
    print(olmo2.apply_chat_template(convo, tokenize=False))
    print("=======")
    print("llama3_instruct")
    print(llama3_instruct.apply_chat_template(convo, tokenize=False))
    print("======")
    print("marin")
    print(marin.apply_chat_template(convo, tokenize=False))

    # ensure it didn't mess up normal tokenization
    assert marin.tokenize("Hello, how are you?") == llama3.tokenize("Hello, how are you?")

    # make sure we use the special tokens in chat template
    out = marin.apply_chat_template(convo, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    assert all(
        token in out["input_ids"]
        for token in marin.convert_tokens_to_ids(["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    )

    # print the assistant tokens
    ids = np.array(out["input_ids"])
    assert np.sum(out["assistant_masks"]) == (
        len(marin("I'm doing well, thanks!")["input_ids"]) + len(marin("Great!")["input_ids"])
    )

    assert (
        marin.decode(ids[np.array(out["assistant_masks"]).astype(bool)])
        == "I'm doing well, thanks!<|eot_id|>Great!<|eot_id|>"
    )

    # upload marin to hf hub
    marin.push_to_hub(marin_tokenizer)


if __name__ == "__main__":
    main()
