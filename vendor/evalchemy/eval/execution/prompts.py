"""Conversation formatting shared by model backends."""

from eval.completion_response import CompletionText, FailedGeneration


def fastchat_prompt(model, messages):
    # The vendored registry preserves benchmark templates without loading torch.
    from eval.chat_benchmarks.LiveBench.livebench.model.model_adapter import get_conversation_template

    if "model_identifier" in model.__dict__:
        name = model.model_identifier.split("=")[1].split(",")[0].split("__")[-1]
        name = name.replace("-", "_").lower().replace(".", "")
    else:
        name = model.model.__class__.__name__
    conversation = get_conversation_template(name)
    system = []
    for message in messages:
        if message["role"] == "system":
            # A caller's system instruction surrounds the benchmark template.
            system.append(message)
        else:
            role = {"user": conversation.roles[0], "assistant": conversation.roles[1]}[message["role"]]
            conversation.append_message(role, message["content"])
    conversation.append_message(conversation.roles[1], None)
    return [*system, {"role": "user", "content": conversation.get_prompt()}], conversation


def clean_fastchat_response(model, conversation, output):
    """Apply the template's answer cleanup before scoring or reusing a turn."""
    if conversation is None or isinstance(output, FailedGeneration):
        return output
    text = str(output).strip()
    stops = conversation.stop_str
    stops = stops if isinstance(stops, list) else [stops] if stops else []
    indices = [text.find(stop) for stop in stops if text.find(stop) > 0]
    if indices:
        text = text[:min(indices)]
    tokenizer = getattr(model, "tokenizer", None)
    for token in getattr(tokenizer, "special_tokens_map", {}).values():
        for value in token if isinstance(token, list) else [token]:
            text = text.replace(value, "")
    if conversation.name == "xgen" and text.startswith("Assistant:"):
        text = text.replace("Assistant:", "", 1).strip()
    text = text.strip()
    return CompletionText(text, output.response, output.content_policy) if isinstance(output, CompletionText) else text
