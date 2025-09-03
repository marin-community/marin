# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Various models and templates for Marin.
"""

marin_tokenizer = "stanford-crfm/marin-tokenizer"
"""
The HF Hub name for the Marin tokenizer.
The Marin tokenizer is (currently) just the Llama 3 tokenizer with a custom chat template (MARIN_CHAT_TEMPLATE).
"""

# to be clear this is the Olmo 2 template except we use llama3's special tokens
MARIN_CHAT_TEMPLATE = """
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

"""
The chat template for the Marin tokenizer.

This template is used to generate the chat template for the Marin tokenizer.
It is a modified version of the Olmo 2 template.

The modifications are:
- Use the Llama 3 special tokens rather than Olmo (which doesn't use special tokens but rather just literally <|, etc.)
- Add the {% generation -%} tag stuff that makes the assistant_mask in Levanter work.

See [Levanter's documentation on Chat Templates](https://levanter.readthedocs.io/en/latest/reference/Data-Formats/#chat-format)
for more information on how this works.
"""

MARIN_CUSTOM_SPECIAL_TOKENS = {
    128002: "<|start_think|>",  # Originally "<|reserved_special_token_0|>"
    128003: "<|end_think|>",  # Originally "<|reserved_special_token_1|>"
}
