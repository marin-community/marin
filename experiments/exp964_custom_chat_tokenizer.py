"""
Saves a modified version of the llama3 tokenizer with a simple Olmo2-inspired chat format.
"""

import os
from urllib.error import HTTPError

import numpy as np
from huggingface_hub.errors import GatedRepoError
from levanter.data.text import ChatLmDatasetFormat
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
<|start_header_id|>assistant<|end_header_id|>\n{% endif -%}
""".strip()


# Olmo 2 template modified so we can use with with levanter
MARIN_OLMO2_TEMPLATE = """
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

    # now do the olmo2 template
    marin_olmo2_tokenizer = "stanford-crfm/marin-olmo2-tokenizer"
    marin_olmo2 = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-SFT")
    marin_olmo2.chat_template = MARIN_OLMO2_TEMPLATE
    marin_olmo2.save_pretrained(os.path.join(os.getcwd(), "marin_olmo2_tokenizer"))

    marin_olmo2 = AutoTokenizer.from_pretrained(os.path.join(os.getcwd(), "marin_olmo2_tokenizer"))
    assert marin_olmo2.chat_template == MARIN_OLMO2_TEMPLATE

    # try it out a bit
    convo = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
        {"role": "user", "content": "That's good to hear!"},
        {"role": "assistant", "content": "Great!"},
    ]
    print("======")
    print("olmo2")
    print(olmo2.apply_chat_template(convo, tokenize=False, add_generation_prompt=True))
    print("=======")
    print("llama3_instruct")
    print(llama3_instruct.apply_chat_template(convo, tokenize=False, add_generation_prompt=True))
    print("======")
    print("marin")
    print(marin.apply_chat_template(convo, tokenize=False, add_generation_prompt=True))
    print("======")
    print("marin_olmo2")
    print(marin_olmo2.apply_chat_template(convo, tokenize=False, add_generation_prompt=True))
    print("======")

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

    # ensure that when we use add_generation_prompt, we add the final newline (and other bits)
    assert marin.apply_chat_template(convo, tokenize=False, add_generation_prompt=True).endswith(
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    # upload marin to hf hub
    marin.push_to_hub(marin_tokenizer)
    print("Pushed Marin Tokenizer")

    # upload marin_olmo2 to hf hub
    marin_olmo2.push_to_hub(marin_olmo2_tokenizer)
    print("Pushed Marin Olmo2 Tokenizer")


# slight modification of https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/tokenizer_config.json
# to add {% generation %} so we can create the assistant_mask
llama3_instruct_trainable_chat_template = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\\n\\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\\n" }}
{{- "Today Date: " + date_string + "\\n\\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\\n\\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\\n\\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\\n\\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\\n\\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>" }}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {%- if message.role == "assistant" %}
            {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}
            {%- generation %}
            {{- message['content'] | trim }}
            {%- endgeneration %}
            {{- "<|eot_id|>" }}
        {%- else %}
            {{- "<|start_header_id|>" + message['role'] + "<|end_header_id|>\\n\\n" + message['content'] | trim + "<|eot_id|>" }}
        {%- endif %}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}
{%- endif %}"""  # noqa: E501

# this is a chat format compatible with the llama3 instruct tokenizer and Levanter's chat format
llama3_instruct_chat_format = ChatLmDatasetFormat(
    messages_field="messages",
    single_turn=False,
    chat_template=llama3_instruct_trainable_chat_template,
    pack=True,
    mask_user_turns=True,
)


if __name__ == "__main__":
    main()
