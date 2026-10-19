from fastchat.conversation import get_conv_template


ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-haiku-20240307-vertex",
    "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229-vertex",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-instant-1",
    "claude-instant-1.2",
)

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-browsing",
    "gpt-4-turbo-2024-04-09",
    "gpt2-chatbot",
    "im-also-a-good-gpt2-chatbot",
    "im-a-good-gpt2-chatbot",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "chatgpt-4o-latest-20240903",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-mini",
)


def get_api_conversation_template(model_path):
    if model_path in OPENAI_MODEL_LIST:
        if "browsing" in model_path:
            return get_conv_template("api_based_default")
        if "gpt-4-turbo-2024-04-09" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "gpt2-chatbot" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "gpt-4o-2024-05-13" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "gpt-4o-2024-08-06" in model_path:
            return get_conv_template("gpt-mini")
        if "anonymous-chatbot" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "chatgpt-4o-latest" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "gpt-mini" in model_path:
            return get_conv_template("gpt-mini")
        if "gpt-4o-mini-2024-07-18" in model_path:
            return get_conv_template("gpt-mini")
        if "o1" in model_path:
            return get_conv_template("api_based_default")
        return get_conv_template("chatgpt")

    if "claude-3-haiku" in model_path:
        return get_conv_template("claude-3-haiku-20240307")
    if "claude-3-sonnet" in model_path:
        return get_conv_template("claude-3-sonnet-20240229")
    if "claude-3-5-sonnet" in model_path:
        return get_conv_template("claude-3-5-sonnet-20240620-v2")
    if "claude-3-opus" in model_path:
        return get_conv_template("claude-3-opus-20240229")
    if model_path in ANTHROPIC_MODEL_LIST:
        return get_conv_template("claude")
    raise ValueError(f"Unknown API model: {model_path}")
