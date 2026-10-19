import sys
import warnings

from eval.chat_benchmarks.LiveBench.livebench.model.completions import chat_completion_openai, chat_completion_palm
from eval.chat_benchmarks.LiveBench.livebench.model.model_adapter import BaseModelAdapter, PaLM2Adapter, get_model_adapter
from eval.chat_benchmarks.LiveBench.livebench.model.models import (
    AnthropicModel,
    AWSModel,
    CohereModel,
    DeepseekModel,
    GeminiModel,
    GemmaModel,
    LlamaModel,
    MistralModel,
    Model,
    NvidiaModel,
    OpenAIModel,
    QwenModel,
    XAIModel,
)


if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache


# Anthropic Models
ANTHROPIC_MODELS = [
    AnthropicModel(api_name="claude-1", display_name="claude-1", aliases=[]),
    AnthropicModel(api_name="claude-2", display_name="claude-2", aliases=[]),
    AnthropicModel(api_name="claude-2.0", display_name="claude-2.0", aliases=[]),
    AnthropicModel(api_name="claude-2.1", display_name="claude-2.1", aliases=[]),
    AnthropicModel(api_name="claude-instant-1", display_name="claude-instant-1", aliases=[]),
    AnthropicModel(api_name="claude-instant-1.2", display_name="claude-instant-1.2", aliases=[]),
    AnthropicModel(
        api_name="claude-3-opus-20240229",
        display_name="claude-3-opus-20240229",
        aliases=[],
    ),
    AnthropicModel(
        api_name="claude-3-sonnet-20240229",
        display_name="claude-3-sonnet-20240229",
        aliases=[],
    ),
    AnthropicModel(
        api_name="claude-3-haiku-20240307",
        display_name="claude-3-haiku-20240307",
        aliases=[],
    ),
    AnthropicModel(
        api_name="claude-3-5-sonnet-20240620",
        display_name="claude-3-5-sonnet-20240620",
        aliases=[],
    ),
    AnthropicModel(
        api_name="claude-3-5-sonnet-20241022",
        display_name="claude-3-5-sonnet-20241022",
        aliases=["claude-3-5-sonnet"],
    ),
    AnthropicModel(
        api_name="claude-3-5-haiku-20241022",
        display_name="claude-3-5-haiku-20241022",
        aliases=["claude-3-5-haiku"],
    ),
]

# OpenAI Models
OPENAI_MODELS = [
    OpenAIModel(api_name="gpt-3.5-turbo", display_name="gpt-3.5-turbo", aliases=[]),
    OpenAIModel(api_name="gpt-3.5-turbo-0301", display_name="gpt-3.5-turbo-0301", aliases=[]),
    OpenAIModel(api_name="gpt-3.5-turbo-0613", display_name="gpt-3.5-turbo-0613", aliases=[]),
    OpenAIModel(api_name="gpt-3.5-turbo-1106", display_name="gpt-3.5-turbo-1106", aliases=[]),
    OpenAIModel(api_name="gpt-3.5-turbo-0125", display_name="gpt-3.5-turbo-0125", aliases=[]),
    OpenAIModel(api_name="gpt-4", display_name="gpt-4", aliases=[]),
    OpenAIModel(api_name="gpt-4-0314", display_name="gpt-4-0314", aliases=[]),
    OpenAIModel(api_name="gpt-4-0613", display_name="gpt-4-0613", aliases=[]),
    OpenAIModel(api_name="gpt-4-turbo", display_name="gpt-4-turbo", aliases=[]),
    OpenAIModel(
        api_name="gpt-4-turbo-2024-04-09",
        display_name="gpt-4-turbo-2024-04-09",
        aliases=[],
    ),
    OpenAIModel(api_name="gpt-4-1106-preview", display_name="gpt-4-1106-preview", aliases=[]),
    OpenAIModel(api_name="gpt-4-0125-preview", display_name="gpt-4-0125-preview", aliases=[]),
    OpenAIModel(api_name="gpt-4o-2024-05-13", display_name="gpt-4o-2024-05-13", aliases=[]),
    OpenAIModel(
        api_name="gpt-4o-mini-2024-07-18",
        display_name="gpt-4o-mini-2024-07-18",
        aliases=["gpt-4o-mini"],
    ),
    OpenAIModel(api_name="gpt-4o-2024-08-06", display_name="gpt-4o-2024-08-06", aliases=[]),
    OpenAIModel(api_name="chatgpt-4o-latest", display_name="chatgpt-4o-latest", aliases=[]),
]

INFERENCE_OPENAI_MODELS = [
    OpenAIModel(
        api_name="o1-mini-2024-09-12",
        display_name="o1-mini-2024-09-12",
        aliases=["o1-mini"],
        inference_api=True,
    ),
    OpenAIModel(
        api_name="o1-preview-2024-09-12",
        display_name="o1-preview-2024-09-12",
        aliases=["o1-preview"],
        inference_api=True,
    ),
    OpenAIModel(
        api_name="o1-2024-12-17",
        display_name="o1-2024-12-17-high",
        aliases=["o1", "o1-high", "o1-2024-12-17"],
        inference_api=True,
        api_kwargs={"reasoning_effort": "high"},
    ),
    OpenAIModel(
        api_name="o1-2024-12-17",
        display_name="o1-2024-12-17-low",
        aliases=["o1-low"],
        inference_api=True,
        api_kwargs={"reasoning_effort": "low"},
    ),
]

# Together Models
TOGETHER_MODELS = [
    LlamaModel(
        api_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        display_name="meta-llama-3.1-405b-instruct-turbo",
        aliases=[],
    ),
    LlamaModel(
        api_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        display_name="meta-llama-3.1-70b-instruct-turbo",
        aliases=[],
    ),
    LlamaModel(
        api_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        display_name="meta-llama-3.1-8b-instruct-turbo",
        aliases=[],
    ),
    LlamaModel(
        api_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        display_name="llama-3.3-70b-instruct-turbo",
        aliases=[],
    ),
    QwenModel(
        api_name="qwen/Qwen2.5-7B-Instruct-Turbo",
        display_name="Qwen-2.5-7B-Instruct-Turbo",
        aliases=[],
    ),
    QwenModel(
        api_name="qwen/Qwen2.5-72B-Instruct-Turbo",
        display_name="Qwen-2.5-72B-Instruct-Turbo",
        aliases=[],
    ),
    LlamaModel(
        api_name="Llama-3.1-Nemotron-70B-Instruct-HF",
        display_name="llama-3.1-nemotron-70b-instruct",
        aliases=["llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"],
    ),
    GemmaModel(api_name="google/gemma-2-27b-it", display_name="gemma-2-27b-it", aliases=[]),
    GemmaModel(api_name="google/gemma-2-9b-it", display_name="gemma-2-9b-it", aliases=[]),
    QwenModel(api_name="qwen/qwq-32b-preview", display_name="Qwen-32B-Preview", aliases=[]),
]

# Google GenerativeAI Models
GOOGLE_GENERATIVEAI_MODELS = [
    GeminiModel(api_name="gemini-1.5-pro-001", display_name="gemini-1.5-pro-001", aliases=[]),
    GeminiModel(api_name="gemini-1.5-flash-001", display_name="gemini-1.5-flash-001", aliases=[]),
    GeminiModel(
        api_name="gemini-1.5-pro-exp-0801",
        display_name="gemini-1.5-pro-exp-0801",
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-1.5-pro-exp-0827",
        display_name="gemini-1.5-pro-exp-0827",
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-1.5-flash-exp-0827",
        display_name="gemini-1.5-flash-exp-0827",
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-1.5-flash-8b-exp-0827",
        display_name="gemini-1.5-flash-8b-exp-0827",
        aliases=[],
    ),
    GeminiModel(api_name="gemini-1.5-pro-002", display_name="gemini-1.5-pro-002", aliases=[]),
    GeminiModel(api_name="gemini-1.5-flash-002", display_name="gemini-1.5-flash-002", aliases=[]),
    GeminiModel(api_name="gemini-exp-1114", display_name="gemini-exp-1114", aliases=[]),
    GeminiModel(api_name="gemini-exp-1121", display_name="gemini-exp-1121", aliases=[]),
    GeminiModel(
        api_name="gemini-1.5-flash-8b-exp-0924",
        display_name="gemini-1.5-flash-8b-exp-0924",
        aliases=[],
    ),
    GeminiModel(
        api_name="learnlm-1.5-pro-experimental",
        display_name="learnlm-1.5-pro-experimental",
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-2.0-flash-thinking-exp-1219",
        display_name="gemini-2.0-flash-thinking-exp-1219",
        aliases=["gemini-2.0-flash-thinking-exp"],
        api_kwargs={"max_tokens": None},
    ),
]

# Vertex Models
VERTEX_MODELS = [
    GeminiModel(
        api_name="gemini-1.5-pro-preview-0409",
        display_name="gemini-1.5-pro-preview-0409",
        aliases=[],
    ),
]

# Mistral Models
MISTRAL_MODELS = [
    MistralModel(api_name="mistral-large-latest", display_name="mistral-large-latest", aliases=[]),
    MistralModel(api_name="mistral-large-2402", display_name="mistral-large-2402", aliases=[]),
    MistralModel(api_name="mistral-large", display_name="mistral-large", aliases=[]),
    MistralModel(api_name="mistral-medium-23-12", display_name="mistral-medium-23-12", aliases=[]),
    MistralModel(api_name="mistral-medium", display_name="mistral-medium", aliases=[]),
    MistralModel(api_name="mistral-small-2402", display_name="mistral-small-2402", aliases=[]),
    MistralModel(api_name="mistral-small", display_name="mistral-small", aliases=[]),
    MistralModel(api_name="open-mixtral-8x7b", display_name="open-mixtral-8x7b", aliases=[]),
    MistralModel(api_name="open-mixtral-8x22b", display_name="open-mixtral-8x22b", aliases=[]),
    MistralModel(api_name="mistral-large-2407", display_name="mistral-large-2407", aliases=[]),
    MistralModel(api_name="open-mistral-nemo", display_name="open-mistral-nemo", aliases=[]),
    MistralModel(api_name="mistral-large-2411", display_name="mistral-large-2411", aliases=[]),
    MistralModel(api_name="mistral-small-2409", display_name="mistral-small-2409", aliases=[]),
]

# Cohere Models
COHERE_MODELS = [
    CohereModel(
        api_name="command-r-plus-04-2024",
        display_name="command-r-plus-04-2024",
        aliases=[],
    ),
    CohereModel(api_name="command-r-03-2024", display_name="command-r-03-2024", aliases=[]),
    CohereModel(api_name="command", display_name="command", aliases=[]),
    CohereModel(api_name="command-r-08-2024", display_name="command-r-08-2024", aliases=[]),
    CohereModel(
        api_name="command-r-plus-08-2024",
        display_name="command-r-plus-08-2024",
        aliases=[],
    ),
]

# Deepseek Models
DEEPSEEK_MODELS = [
    DeepseekModel(api_name="deepseek-chat", display_name="deepseek-v3", aliases=[]),
]

# Nvidia Models
NVIDIA_MODELS = [
    NvidiaModel(
        api_name="nvidia/nemotron-4-340b-instruct",
        display_name="nemotron-4-340b-instruct",
        aliases=[],
    ),
]

# XAI Models
XAI_MODELS = [
    XAIModel(api_name="grok-beta", display_name="grok-beta", aliases=[]),
    XAIModel(api_name="grok-2-1212", display_name="grok-2-1212", aliases=[]),
]

# AWS Models
AWS_MODELS = [
    AWSModel(
        api_name="amazon.nova-micro-v1:0",
        display_name="amazon.nova-micro-v1:0",
        aliases=[],
    ),
    AWSModel(
        api_name="amazon.nova-micro-v1:0:128k",
        display_name="amazon.nova-micro-v1:0:128k",
        aliases=[],
    ),
    AWSModel(
        api_name="amazon.nova-lite-v1:0",
        display_name="amazon.nova-lite-v1:0",
        aliases=[],
    ),
    AWSModel(
        api_name="amazon.nova-lite-v1:0:300k",
        display_name="amazon.nova-lite-v1:0:300k",
        aliases=[],
    ),
    AWSModel(api_name="amazon.nova-pro-v1:0", display_name="amazon.nova-pro-v1:0", aliases=[]),
    AWSModel(
        api_name="amazon.nova-pro-v1:0:300k",
        display_name="amazon.nova-pro-v1:0:300k",
        aliases=[],
    ),
]

ALL_MODELS = (
    ANTHROPIC_MODELS
    + OPENAI_MODELS
    + INFERENCE_OPENAI_MODELS
    + MISTRAL_MODELS
    + COHERE_MODELS
    + DEEPSEEK_MODELS
    + NVIDIA_MODELS
    + XAI_MODELS
    + AWS_MODELS
    + GOOGLE_GENERATIVEAI_MODELS
)


@cache
def get_model(name: str) -> Model:
    matches = []
    for model in ALL_MODELS:
        if model.display_name.lower() == name.lower() or any(alias.lower() == name.lower() for alias in model.aliases):
            matches.append(model)
    if len(matches) > 1:
        raise ValueError(f"Multiple models found for {name}")
    elif len(matches) == 0:
        warnings.warn(f"No model found for {name}")
        return Model(
            api_name=name,
            display_name=name,
            aliases=[],
            adapter=get_model_adapter(name),
            api_function=chat_completion_openai,
        )
    else:
        return matches[0]
