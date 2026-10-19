def __getattr__(name):
    if name in {"load_model", "get_conversation_template", "add_model_args"}:
        from fastchat.model import model_adapter

        return getattr(model_adapter, name)
    raise AttributeError(name)
