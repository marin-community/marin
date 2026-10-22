def split_provider_model_name(model_name: str) -> tuple[str | None, str]:
    """Split a model identifier into its provider prefix and remaining name."""
    if "/" not in model_name:
        return None, model_name

    provider, name = model_name.split("/", 1)
    return provider.lower(), name
