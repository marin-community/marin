from collections.abc import Callable
from dataclasses import dataclass, field
from eval.chat_benchmarks.LiveBench.livebench.conversation import Conversation
from eval.chat_benchmarks.LiveBench.livebench.model.model_adapter import (
    BaseModelAdapter,
    ClaudeAdapter,
    ChatGPTAdapter,
    Llama3Adapter,
    QwenChatAdapter,
    GeminiAdapter,
    MistralAdapter,
    CohereAdapter,
    DeepseekChatAdapter,
    NvidiaChatAdapter,
    GemmaAdapter,
)
from eval.chat_benchmarks.LiveBench.livebench.model.completions import (
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_google_generativeai,
    chat_completion_mistral,
    chat_completion_cohere,
    chat_completion_aws,
    chat_completion_xai,
    chat_completion_deepseek,
    chat_completion_nvidia,
    chat_completion_together,
)

model_api_function = Callable[["Model", Conversation, float, int, dict | None], tuple[str, int]]


@dataclass(kw_only=True, frozen=True)
class Model:
    api_name: str
    display_name: str
    aliases: list[str]
    adapter: BaseModelAdapter
    api_function: model_api_function | None = None
    api_kwargs: dict | None = None


@dataclass(kw_only=True, frozen=True)
class AnthropicModel(Model):
    adapter: BaseModelAdapter = field(default=ClaudeAdapter())
    api_function: model_api_function = field(default=chat_completion_anthropic)


@dataclass(kw_only=True, frozen=True)
class OpenAIModel(Model):
    adapter: BaseModelAdapter = field(default=ChatGPTAdapter())
    inference_api: bool = False
    api_function: model_api_function = field(default=chat_completion_openai)


@dataclass(kw_only=True, frozen=True)
class LlamaModel(Model):
    adapter: BaseModelAdapter = field(default=Llama3Adapter())
    api_function: model_api_function = field(default=chat_completion_together)


@dataclass(kw_only=True, frozen=True)
class QwenModel(Model):
    adapter: BaseModelAdapter = field(default=QwenChatAdapter())
    api_function: model_api_function = field(default=chat_completion_together)


@dataclass(kw_only=True, frozen=True)
class GemmaModel(Model):
    adapter: BaseModelAdapter = field(default=GemmaAdapter())
    api_function: model_api_function = field(default=chat_completion_together)


@dataclass(kw_only=True, frozen=True)
class GeminiModel(Model):
    adapter: BaseModelAdapter = field(default=GeminiAdapter())
    api_function: model_api_function = field(default=chat_completion_google_generativeai)


@dataclass(kw_only=True, frozen=True)
class MistralModel(Model):
    adapter: BaseModelAdapter = field(default=MistralAdapter())
    api_function: model_api_function = field(default=chat_completion_mistral)


@dataclass(kw_only=True, frozen=True)
class CohereModel(Model):
    adapter: BaseModelAdapter = field(default=CohereAdapter())
    api_function: model_api_function = field(default=chat_completion_cohere)


@dataclass(kw_only=True, frozen=True)
class DeepseekModel(Model):
    adapter: BaseModelAdapter = field(default=DeepseekChatAdapter())
    api_function: model_api_function = field(default=chat_completion_deepseek)


@dataclass(kw_only=True, frozen=True)
class NvidiaModel(Model):
    adapter: BaseModelAdapter = field(default=NvidiaChatAdapter())
    api_function: model_api_function = field(default=chat_completion_nvidia)


@dataclass(kw_only=True, frozen=True)
class XAIModel(Model):
    adapter: BaseModelAdapter = field(default=ChatGPTAdapter())
    api_function: model_api_function = field(default=chat_completion_xai)


@dataclass(kw_only=True, frozen=True)
class AWSModel(Model):
    adapter: BaseModelAdapter = field(default=ChatGPTAdapter())
    api_function: model_api_function = field(default=chat_completion_aws)
