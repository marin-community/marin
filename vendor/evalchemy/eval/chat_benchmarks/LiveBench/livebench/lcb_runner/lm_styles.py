from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class LMStyle(Enum):
    OpenAIChat = "OpenAIChat"
    Claude = "Claude"  # Claude 1 and Claude 2
    Claude3 = "Claude3"
    Gemini = "Gemini"
    MistralWeb = "MistralWeb"

    GenericBase = "GenericBase"

    DeepSeekCodeInstruct = "DeepSeekCodeInstruct"
    CodeLLaMaInstruct = "CodeLLaMaInstruct"
    StarCoderInstruct = "StarCoderInstruct"

    Phind = "Phind"
    WizardCoder = "WizardCoder"
    MagiCoder = "MagiCoder"
    OC = "OC"

    Qwen1point5 = "Qwen1point5"
    Smaug2 = "Smaug2"

    LLaMa3 = "LLaMa3"


@dataclass
class LanguageModel:
    model_name: str
    model_repr: str
    model_style: LMStyle
    release_date: datetime | None  # XXX Should we use timezone.utc?
    link: str | None = None

    def __hash__(self) -> int:
        return hash(self.model_name)


LanguageModelList: list[LanguageModel] = [
    LanguageModel(
        "deepseek-ai/deepseek-coder-33b-instruct",
        "DSCoder-33b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 9, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "DSCoder-6.7b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 9, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "DSCoder-1.3b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 8, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct",
    ),
    LanguageModel(
        "codellama/CodeLlama-34b-Instruct-hf",
        "Cllama-34b-Ins",
        LMStyle.CodeLLaMaInstruct,
        datetime(2023, 1, 1),
        link="https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf",
    ),
    LanguageModel(
        "codellama/CodeLlama-13b-Instruct-hf",
        "Cllama-13b-Ins",
        LMStyle.CodeLLaMaInstruct,
        datetime(2023, 1, 1),
        link="https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf",
    ),
    LanguageModel(
        "codellama/CodeLlama-7b-Instruct-hf",
        "Cllama-7b-Ins",
        LMStyle.CodeLLaMaInstruct,
        datetime(2023, 1, 1),
        link="https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf",
    ),
    LanguageModel(
        "WizardCoderLM/WizardCoderCoder-Python-34B-V1.0",
        "WCoder-34B-V1",
        LMStyle.WizardCoder,
        datetime(2023, 1, 1),
        link="https://huggingface.co/WizardCoderLM/WizardCoderCoder-Python-34B-V1.0",
    ),
    LanguageModel(
        "WizardCoderLM/WizardCoderCoder-33B-V1.1",
        "WCoder-33B-V1.1",
        LMStyle.WizardCoder,
        datetime(2023, 9, 1),
        link="https://huggingface.co/WizardCoderLM/WizardCoderCoder-33B-V1.1",
    ),
    LanguageModel(
        "Phind/Phind-CodeLlama-34B-v2",
        "Phind-34B-V2",
        LMStyle.Phind,
        datetime(2023, 1, 1),
        link="https://huggingface.co/Phind/Phind-CodeLlama-34B-v2",
    ),
    LanguageModel(
        "gpt-3.5-turbo-0301",
        "GPT-3.5-Turbo-0301",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
    ),
    LanguageModel(
        "gpt-3.5-turbo-0125",
        "GPT-3.5-Turbo-0125",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://openai.com/blog/new-embedding-models-and-api-updates#:~:text=Other%20new%20models%20and%20lower%20pricing",
    ),
    LanguageModel(
        "gpt-4-0613",
        "GPT-4-0613",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4",
    ),
    LanguageModel(
        "gpt-4-1106-preview",
        "GPT-4-Turbo-1106",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
    ),
    LanguageModel(
        "gpt-4-turbo-2024-04-09",
        "GPT-4-Turbo-2024-04-09",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4",
    ),
    LanguageModel(
        "claude-2",
        "Claude-2",
        LMStyle.Claude,
        datetime(2022, 12, 31),
        link="https://www.anthropic.com/index/claude-2",
    ),
    LanguageModel(
        "claude-instant-1",
        "Claude-Instant-1",
        LMStyle.Claude,
        datetime(2022, 12, 31),
        link="https://www.anthropic.com/index/introducing-claude",
    ),
    LanguageModel(
        "claude-3-opus-20240229",
        "Claude-3-Opus",
        LMStyle.Claude3,
        datetime(2023, 9, 1),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "claude-3-sonnet-20240229",
        "Claude-3-Sonnet",
        LMStyle.Claude3,
        datetime(2023, 9, 1),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "claude-3-haiku-20240307",
        "Claude-3-Haiku",
        LMStyle.Claude3,
        datetime(2023, 4, 30),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "gemini-pro",
        "Gemini-Gemini-Pro",
        LMStyle.Gemini,
        datetime(2023, 5, 1),
        link="https://blog.Gemini/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "ise-uiuc/Magicoder-S-DS-6.7B",
        "MagiCoderS-DS-6.7B",
        LMStyle.MagiCoder,
        datetime(2023, 7, 30),
        link="https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B",
    ),
    LanguageModel(
        "ise-uiuc/Magicoder-S-CL-7B",
        "MagiCoderS-CL-7B",
        LMStyle.MagiCoder,
        datetime(2023, 1, 1),
        link="https://huggingface.co/ise-uiuc/Magicoder-S-CL-7B",
    ),
    LanguageModel(
        "bigcode/starcoder2-3b",
        "StarCoder2-3b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "bigcode/starcoder2-7b",
        "StarCoder2-7b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "bigcode/starcoder2-15b",
        "StarCoder2-15b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "codellama/CodeLlama-34b-hf",
        "CodeLlama-34b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/codellama/CodeLlama-34b-hf",
    ),
    LanguageModel(
        "codellama/CodeLlama-13b-hf",
        "CodeLlama-13b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/codellama/CodeLlama-13b-hf",
    ),
    LanguageModel(
        "codellama/CodeLlama-7b-hf",
        "CodeLlama-7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/codellama/CodeLlama-7b-hf",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-33b-base",
        "DSCoder-33b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-33b-base",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-6.7b-base",
        "DSCoder-6.7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-1.3b-base",
        "DSCoder-1.3b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base",
    ),
    LanguageModel(
        "google/codegemma-7b",
        "CodeGemma-7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/codegemma-7b",
    ),
    LanguageModel(
        "google/codegemma-2b",
        "CodeGemma-2b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/codegemma-2b",
    ),
    LanguageModel(
        "google/gemma-7b",
        "Gemma-7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/gemma-7b",
    ),
    LanguageModel(
        "google/gemma-2b",
        "Gemma-2b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/gemma-2b",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-70B",
        "LLama3-70b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-70B",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-8B",
        "LLama3-8b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-8B",
    ),
    LanguageModel(
        "mistral-large-latest",
        "Mistral-Large",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mistral-large/",
    ),
    LanguageModel(
        "m-a-p/OpenCodeInterpreter-DS-33B",
        "OC-DS-33B",
        LMStyle.OC,
        datetime(2023, 1, 1),
        link="https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-33B/",
    ),
    LanguageModel(
        "m-a-p/OpenCodeInterpreter-DS-6.7B",
        "OC-DS-6.7B",
        LMStyle.OC,
        datetime(2023, 9, 1),
        link="https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-6.7B/",
    ),
    LanguageModel(
        "m-a-p/OpenCodeInterpreter-DS-1.3B",
        "OC-DS-1.3B",
        LMStyle.OC,
        datetime(2023, 9, 1),
        link="https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-1.3B/",
    ),
    LanguageModel(
        "stabilityai/stable-code-3b",
        "StableCode-3B",
        LMStyle.GenericBase,
        datetime(2023, 9, 1),
        link="https://huggingface.co/stabilityai/stable-code-3b/",
    ),
    LanguageModel(
        "qwen/Qwen1.5-72B-Chat",
        "Qwen-1.5-72B-Chat ",
        LMStyle.Qwen1point5,
        datetime(2024, 3, 31),
        link="https://huggingface.co/qwen/Qwen1.5-72B-Chat/",
    ),
    LanguageModel(
        "abacusai/Smaug-2-72B",
        "Smaug-2-72B ",
        LMStyle.Smaug2,
        datetime(2024, 3, 31),
        link="https://huggingface.co/abacusai/Smaug-2-72B/",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "LLama3-8b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "LLama3-70b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf",
    ),
    LanguageModel(
        "bigcode/starcoder2-instruct-15b-v0.1",
        "StarCoder2-Ins-v0.1",
        LMStyle.LLaMa3,
        datetime(2023, 4, 30),
        link="https://huggingface.co/bigcode/starcoder2-instruct-15b-v0.1",
    ),
]

LanguageModelStore: dict[str, LanguageModel] = {lm.model_name: lm for lm in LanguageModelList}

if __name__ == "__main__":
    print(list(LanguageModelStore.keys()))
