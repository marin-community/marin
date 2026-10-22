PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "azure": ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"],
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel_ai_gateway": "VERCEL_AI_GATEWAY_API_KEY",
    "datarobot": "DATAROBOT_API_TOKEN",
    "vertex_ai": "VERTEXAI_PROJECT",
    "huggingface": "HUGGINGFACE_API_KEY",
    "ai21": "AI21_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "aleph_alpha": "ALEPH_ALPHA_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "nlp_cloud": "NLP_CLOUD_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "sagemaker": "AWS_SECRET_ACCESS_KEY",
    "ollama": "OLLAMA_API_BASE",
    "anyscale": "ANYSCALE_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "featherless_ai": "FEATHERLESS_AI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "ai21_chat": "AI21_API_KEY",
    "volcengine": "VOLCENGINE_API_KEY",
    "codestral": "CODESTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "palm": "PALM_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "infinity": "INFINITY_API_KEY",
    "fireworks_ai": "FIREWORKS_AI_API_KEY",
    "cloudflare": "CLOUDFLARE_API_KEY",
    "novita": "NOVITA_API_KEY",
    "nebius": "NEBIUS_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zai": "ZAI_API_KEY",
}


def get_api_key_var_names_from_model_name(model_name: str) -> list[str]:
    provider, separator, _ = model_name.partition("/")
    provider = provider.lower()
    if not separator:
        if model_name.startswith(("claude-",)):
            provider = "anthropic"
        elif model_name.startswith(("gemini-",)):
            provider = "gemini"
        elif model_name.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            provider = "openai"
        else:
            raise ValueError(f"Specify a provider prefix for model: {model_name}")
    if provider == "hosted_vllm":
        return []
    if provider not in PROVIDER_KEYS:
        raise ValueError(f"Unknown provider: {provider}")
    keys = PROVIDER_KEYS[provider]
    return [keys] if isinstance(keys, str) else keys
