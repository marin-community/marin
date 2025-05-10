# name for the hf hub. cf llama3_tokenizer
marin_tokenizer = "stanford-crfm/marin-tokenizer"

# to be clear this is the Olmo 2 template except we use llama3's special tokens
# we also add the {% generation -%} tag stuff that makes the assistant_mask work
# (Levanter relies on the assistant mask.)
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
