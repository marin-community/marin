"""Render an explicitly selected assistant prefill with the serving tokenizer."""

from typing import Any, cast

from transformers import PreTrainedTokenizerBase


def prefill_token_ids(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, Any]], options: dict[str, Any]
) -> list[int]:
    """Return token IDs continuing the final assistant message.

    Requires explicit continue_final_message=True and add_generation_prompt=False.
    """
    template = dict(options["extra_body"])
    if template.pop("continue_final_message") is not True or template.pop("add_generation_prompt") is not False:
        raise ValueError("Prefill rendering requires explicit assistant continuation without a new generation prompt")
    template_kwargs = {**options.get("chat_template_kwargs", {}), **template.pop("chat_template_kwargs", {})}
    if template:
        raise ValueError(f"Unsupported prefill rendering fields: {sorted(template)}")
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=options.get("tools"),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        continue_final_message=True,
        **template_kwargs,
    )
    # A single conversation with tokenize=True and return_dict=False returns IDs.
    return cast(list[int], rendered)
