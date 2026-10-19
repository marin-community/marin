"""LiveBench model-name matching and conversation templates.

Adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py.

Local inference belongs to the served endpoint. Keep the corpus's adapter order
and template selection without importing its local model loaders.
"""

from __future__ import annotations

import os
import re
from functools import cache
from typing import List

from eval.chat_benchmarks.LiveBench.livebench.conversation import Conversation, get_conv_template


ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1",
    "claude-instant-1.2",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
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
    "gpt-4-turbo-2024-04-09",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "chatgpt-4o-latest",
)

INFERENCE_OPENAI_MODEL_LIST = (
    "o1-mini-2024-09-12",
    "o1-mini",
    "o1-preview",
    "o1-preview-2024-09-12",
)

TOGETHER_MODEL_LIST = (
    "Meta-Llama-3.1-405B-Instruct-Turbo",
    "Meta-Llama-3.1-70B-Instruct-Turbo",
    "Meta-Llama-3.1-8B-Instruct-Turbo",
    "Llama-3.3-70B-Instruct-Turbo",
    "Qwen2.5-7B-Instruct-Turbo",
    "Qwen2.5-72B-Instruct-Turbo",
    "Llama-3.1-Nemotron-70B-Instruct-HF",
    "gemma-2-27b-it",
    "gemma-2-9b-it",
    "qwq-32b-preview",
)
TOGETHER_MODEL_LIST = tuple([m.lower() for m in TOGETHER_MODEL_LIST])

GOOGLE_GENERATIVEAI_MODEL_LIST = (
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-exp-0801",
    "gemini-1.5-pro-exp-0827",
    "gemini-1.5-flash-exp-0827",
    "gemini-1.5-flash-8b-exp-0827",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-exp-1114",
    "gemini-exp-1121",
    "gemini-1.5-flash-8b-exp-0924",
    "gemini-exp-1206",
    "gemini-2.0-flash-exp",
    "learnlm-1.5-pro-experimental",
)

VERTEX_MODEL_LIST = ("gemini-1.5-pro-preview-0409",)

MISTRAL_MODEL_LIST = (
    "mistral-large-latest",
    "mistral-large-2402",
    "mistral-large",
    "mistral-medium-23-12",
    "mistral-medium",
    "mistral-small-2402",
    "mistral-small",
    "open-mixtral-8x7b",
    "open-mixtral-8x22b",
    "mistral-large-2407",
    "open-mistral-nemo",
    "mistral-large-2411",
    "mistral-small-2409",
)

COHERE_MODEL_LIST = (
    "command-r-plus-04-2024",
    "command-r-03-2024",
    "command",
    "command-r-08-2024",
    "command-r-plus-08-2024",
)

DEEPSEEK_MODEL_LIST = (
    "deepseek-coder",
    "deepseek-v3",
)

NVIDIA_MODEL_LIST = (
    "nemotron-4-340b-instruct",
    "llama-3.1-nemotron-70b-instruct",
)

XAI_MODEL_LIST = "grok-beta"

AWS_MODEL_LIST = (
    "amazon.nova-micro-v1:0",
    "amazon.nova-micro-v1:0:128k",
    "amazon.nova-lite-v1:0",
    "amazon.nova-lite-v1:0:300k",
    "amazon.nova-pro-v1:0",
    "amazon.nova-pro-v1:0:300k",
)


class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        """Return whether this adapter is suitable for the given model"""
        return True


    def get_default_conv_template(self, model_path: str) -> Conversation:
        """Retrieve the default conversation template for the model"""
        return get_conv_template("one_shot")


# A global registry for all model adapters
# TODO (lmzheng): make it a priority queue.
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get the suitable model adapter for a model specified by model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            print("Using adapter:", type(adapter))
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            print("Using adapter:", type(adapter))
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")


def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template for the given model."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)


def remove_parent_directory_name(model_path):
    """Remove parent directory name."""
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    return model_path.split("/")[-1]


class VicunaAdapter(BaseModelAdapter):
    "Model adapter for Vicuna models (e.g., lmsys/vicuna-7b-v1.5)" ""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vicuna" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "v0" in remove_parent_directory_name(model_path):
            return get_conv_template("one_shot")
        return get_conv_template("vicuna_v1.1")


class AiroborosAdapter(BaseModelAdapter):
    """The model adapter for jondurbin/airoboros-*"""

    def match(self, model_path: str):
        if re.search(r"airoboros|spicyboros", model_path, re.I):
            return True
        return False

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "-3." in model_path or "-3p" in model_path:
            return get_conv_template("airoboros_v3")
        if "spicyboros" in model_path or re.search(r"-(2\.[2-9]+)", model_path):
            return get_conv_template("airoboros_v2")
        return get_conv_template("airoboros_v1")


class LongChatAdapter(BaseModelAdapter):
    "Model adapter for LongChat models (e.g., lmsys/longchat-7b-16k)."

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "longchat" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.1")


class GoogleT5Adapter(BaseModelAdapter):
    """The model adapter for google/Flan based models, such as Salesforce/codet5p-6b, lmsys/fastchat-t5-3b-v1.0, flan-t5-*, flan-ul2"""

    def match(self, model_path: str):
        return any(model_str in model_path.lower() for model_str in ["flan-", "fastchat-t5", "codet5p"])


class KoalaAdapter(BaseModelAdapter):
    """The model adapter for Koala"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "koala" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("koala_v1")


class AlpacaAdapter(BaseModelAdapter):
    """The model adapter for Alpaca"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "alpaca" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")


class ChatGLMAdapter(BaseModelAdapter):
    """The model adapter for THUDM/chatglm-6b, THUDM/chatglm2-6b"""

    def match(self, model_path: str):
        return "chatglm" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "chatglm2" in model_path.lower():
            return get_conv_template("chatglm2")
        if "chatglm3" in model_path.lower():
            return get_conv_template("chatglm3")
        return get_conv_template("chatglm")


class CodeGeexAdapter(BaseModelAdapter):
    """The model adapter for THUDM/codegeex-6b, THUDM/codegeex2-6b"""

    def match(self, model_path: str):
        return "codegeex" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("codegeex")


class DollyV2Adapter(BaseModelAdapter):
    """The model adapter for databricks/dolly-v2-12b"""

    def match(self, model_path: str):
        return "dolly-v2" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("dolly_v2")


class OasstPythiaAdapter(BaseModelAdapter):
    """The model adapter for OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "oasst" in model_path and "pythia" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("oasst_pythia")


class OasstLLaMAAdapter(BaseModelAdapter):
    """The model adapter for OpenAssistant/oasst-sft-7-llama-30b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        model_path = model_path.lower()
        if "openassistant-sft-7-llama-30b-hf" in model_path:
            return True
        return "oasst" in model_path and "pythia" not in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("oasst_llama")


class OpenChat35Adapter(BaseModelAdapter):
    """The model adapter for OpenChat 3.5 (e.g. openchat/openchat_3.5)"""

    def match(self, model_path: str):
        if "openchat" in model_path.lower() and "3.5" in model_path.lower():
            return True
        elif "starling-lm" in model_path.lower():
            return True
        return False

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openchat_3.5")


class TenyxChatAdapter(BaseModelAdapter):
    """The model adapter for TenyxChat (e.g. tenyx/TenyxChat-7B-v1)"""

    def match(self, model_path: str):
        return "tenyxchat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tenyxchat")


class PythiaAdapter(BaseModelAdapter):
    """The model adapter for any EleutherAI/pythia model"""

    def match(self, model_path: str):
        return "pythia" in model_path.lower()


class StableLMAdapter(BaseModelAdapter):
    """The model adapter for StabilityAI/stablelm-tuned-alpha-7b"""

    def match(self, model_path: str):
        return "stablelm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("stablelm")


class MPTAdapter(BaseModelAdapter):
    """The model adapter for MPT series (mosaicml/mpt-7b-chat, mosaicml/mpt-30b-chat)"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "mpt" in model_path and not "airoboros" in model_path


    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "mpt-7b-chat" in model_path:
            return get_conv_template("mpt-7b-chat")
        elif "mpt-30b-chat" in model_path:
            return get_conv_template("mpt-30b-chat")
        elif "mpt-30b-instruct" in model_path:
            return get_conv_template("mpt-30b-instruct")
        else:
            print(
                "Warning: Loading base MPT model with `zero_shot` conversation configuration.  "
                "If this is not desired, inspect model configurations and names."
            )
            return get_conv_template("zero_shot")


class BaizeAdapter(BaseModelAdapter):
    """The model adapter for project-baize/baize-v2-7b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "baize" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("baize")


class RwkvAdapter(BaseModelAdapter):
    """The model adapter for BlinkDL/RWKV-4-Raven"""

    def match(self, model_path: str):
        return "rwkv-4" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("rwkv")


class OpenBuddyAdapter(BaseModelAdapter):
    """The model adapter for OpenBuddy/openbuddy-7b-v1.1-bf16-enc"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "openbuddy" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openbuddy")


class PhoenixAdapter(BaseModelAdapter):
    """The model adapter for FreedomIntelligence/phoenix-inst-chat-7b"""

    def match(self, model_path: str):
        return "phoenix" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phoenix")


class ReaLMAdapter(BaseModelAdapter):
    """The model adapter for FreedomIntelligence/ReaLM-7b"""

    def match(self, model_path: str):
        return "ReaLM" in model_path


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ReaLM-7b-v1")


class ChatGPTAdapter(BaseModelAdapter):
    """The model adapter for ChatGPT"""

    def match(self, model_path: str):
        return (
            model_path in OPENAI_MODEL_LIST
            or model_path in INFERENCE_OPENAI_MODEL_LIST
            or model_path in XAI_MODEL_LIST
            or model_path in AWS_MODEL_LIST
        )


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("chatgpt")


class AzureOpenAIAdapter(BaseModelAdapter):
    """The model adapter for Azure OpenAI"""

    def match(self, model_path: str):
        return model_path in ("azure-gpt-35-turbo", "azure-gpt-4")


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("chatgpt")


class PplxAIAdapter(BaseModelAdapter):
    """The model adapter for Perplexity AI"""

    def match(self, model_path: str):
        return model_path in (
            "pplx-7b-online",
            "pplx-70b-online",
        )


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("pplxai")


class ClaudeAdapter(BaseModelAdapter):
    """The model adapter for Claude"""

    def match(self, model_path: str):
        return model_path in ANTHROPIC_MODEL_LIST


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("claude")


class BardAdapter(BaseModelAdapter):
    """The model adapter for Bard"""

    def match(self, model_path: str):
        return model_path == "bard"


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")


class PaLM2Adapter(BaseModelAdapter):
    """The model adapter for PaLM2"""

    def match(self, model_path: str):
        return model_path == "palm-2"


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")


class GeminiAdapter(BaseModelAdapter):
    """The model adapter for Gemini"""

    def match(self, model_path: str):
        return "gemini" in model_path.lower() or "bard" in model_path.lower() or "learnlm" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("gemini")


class GeminiDevAdapter(BaseModelAdapter):
    """The model adapter for Gemini 1.5 Pro"""

    def match(self, model_path: str):
        return "gemini-1.5-pro" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("gemini-dev")


class BiLLaAdapter(BaseModelAdapter):
    """The model adapter for Neutralzz/BiLLa-7B-SFT"""

    def match(self, model_path: str):
        return "billa" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("billa")


class RedPajamaINCITEAdapter(BaseModelAdapter):
    """The model adapter for togethercomputer/RedPajama-INCITE-7B-Chat"""

    def match(self, model_path: str):
        return "redpajama-incite" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("redpajama-incite")


class H2OGPTAdapter(BaseModelAdapter):
    """The model adapter for h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "h2ogpt" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("h2ogpt")


class RobinAdapter(BaseModelAdapter):
    """The model adapter for LMFlow/Full-Robin-7b-v2"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "robin" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("Robin")


class SnoozyAdapter(BaseModelAdapter):
    """The model adapter for nomic-ai/gpt4all-13b-snoozy"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "gpt4all" in model_path and "snoozy" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("snoozy")


class WizardLMAdapter(BaseModelAdapter):
    """The model adapter for WizardLM/WizardLM-13B-V1.0"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "wizardlm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "13b" in model_path or "30b" in model_path or "70b" in model_path:
            return get_conv_template("vicuna_v1.1")
        else:
            # TODO: use the recommended template for 7B
            # (https://huggingface.co/WizardLM/WizardLM-13B-V1.0)
            return get_conv_template("one_shot")


class ManticoreAdapter(BaseModelAdapter):
    """The model adapter for openaccess-ai-collective/manticore-13b-chat-pyg"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "manticore" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("manticore")


class GuanacoAdapter(BaseModelAdapter):
    """The model adapter for timdettmers/guanaco-33b-merged"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "guanaco" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("zero_shot")


class ChangGPTAdapter(BaseModelAdapter):
    """The model adapter for lcw99/polyglot-ko-12.8b-chang-instruct-chat"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "polyglot" in model_path and "chang" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("polyglot_changgpt")


class CamelAdapter(BaseModelAdapter):
    """The model adapter for camel-ai/CAMEL-13B-Combined-Data"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "camel" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.1")


class TuluAdapter(BaseModelAdapter):
    """The model adapter for allenai/tulu-30b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "tulu" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tulu")


class FalconAdapter(BaseModelAdapter):
    """The model adapter for tiiuae/falcon-40b"""

    def match(self, model_path: str):
        return "falcon" in model_path.lower() and "chat" not in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("falcon")


class FalconChatAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "falcon" in model_path.lower() and "chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("falcon-chat")


class TigerBotAdapter(BaseModelAdapter):
    """The model adapter for TigerResearch/tigerbot-7b-sft"""

    def match(self, model_path: str):
        return "tigerbot" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tigerbot")


class BaichuanAdapter(BaseModelAdapter):
    """The model adapter for Baichuan models (e.g., baichuan-inc/Baichuan-7B)"""

    def match(self, model_path: str):
        return "baichuan" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        # for Baichuan-13B-Chat
        if "chat" in model_path.lower():
            if "baichuan2" in model_path.lower():
                return get_conv_template("baichuan2-chat")
            return get_conv_template("baichuan-chat")
        return get_conv_template("zero_shot")


class XGenAdapter(BaseModelAdapter):
    """The model adapter for Salesforce/xgen-7b"""

    def match(self, model_path: str):
        return "xgen" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("xgen")


class NousHermesAdapter(BaseModelAdapter):
    """The model adapter for NousResearch/Nous-Hermes-13b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "nous-hermes" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")


class InternLMChatAdapter(BaseModelAdapter):
    """The model adapter for internlm/internlm-chat-7b"""

    def match(self, model_path: str):
        return "internlm" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("internlm-chat")


class StarChatAdapter(BaseModelAdapter):
    """The model adapter for HuggingFaceH4/starchat-beta"""

    def match(self, model_path: str):
        return "starchat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("starchat")


class MistralAdapter(BaseModelAdapter):
    """The model adapter for Mistral AI models"""

    def match(self, model_path: str):
        return "mistral" in model_path.lower() or "mixtral" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        # return get_conv_template("mistral")
        return get_conv_template("chatgpt")


class Llama2Adapter(BaseModelAdapter):
    """The model adapter for Llama-2 (e.g., meta-llama/Llama-2-7b-hf)"""

    def match(self, model_path: str):
        return "llama-2" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")


class Llama3Adapter(BaseModelAdapter):
    """The model adapter for Llama-3 (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"""

    def match(self, model_path: str):
        return "llama-3" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-3")


class CuteGPTAdapter(BaseModelAdapter):
    """The model adapter for CuteGPT"""

    def match(self, model_path: str):
        return "cutegpt" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("cutegpt")


class OpenOrcaAdapter(BaseModelAdapter):
    """Model adapter for Open-Orca models which may use different prompt templates
    - (e.g. Open-Orca/OpenOrcaxOpenChat-Preview2-13B, Open-Orca/Mistral-7B-OpenOrca)
    - `OpenOrcaxOpenChat-Preview2-13B` uses their "OpenChat Llama2 V1" prompt template.
        - [Open-Orca/OpenOrcaxOpenChat-Preview2-13B #Prompt Template](https://huggingface.co/Open-Orca/OpenOrcaxOpenChat-Preview2-13B#prompt-template)
    - `Mistral-7B-OpenOrca` uses the [OpenAI's Chat Markup Language (ChatML)](https://github.com/openai/openai-python/blob/main/chatml.md)
        format, with <|im_start|> and <|im_end|> tokens added to support this.
        - [Open-Orca/Mistral-7B-OpenOrca #Prompt Template](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca#prompt-template)
    """

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "mistral-7b-openorca" in model_path.lower() or "openorca" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "mistral-7b-openorca" in model_path.lower():
            return get_conv_template("mistral-7b-openorca")
        return get_conv_template("open-orca")


class DolphinAdapter(OpenOrcaAdapter):
    """Model adapter for ehartford/dolphin-2.2.1-mistral-7b"""

    def match(self, model_path: str):
        return "dolphin" in model_path.lower() and "mistral" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("dolphin-2.2.1-mistral-7b")


class Hermes2Adapter(BaseModelAdapter):
    """Model adapter for teknium/OpenHermes-2.5-Mistral-7B and teknium/OpenHermes-2-Mistral-7B models"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return any(
            model_str in model_path.lower() for model_str in ["openhermes-2.5-mistral-7b", "openhermes-2-mistral-7b"]
        )


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("OpenHermes-2.5-Mistral-7B")


class NousHermes2MixtralAdapter(BaseModelAdapter):
    """Model adapter for NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO model"""

    def match(self, model_path: str):
        return any(
            model_str in model_path.lower()
            for model_str in [
                "nous-hermes-2-mixtral-8x7b-dpo",
                "nous-hermes-2-mixtral-8x7b-sft",
            ]
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("Nous-Hermes-2-Mixtral-8x7B-DPO")


class WizardCoderAdapter(BaseModelAdapter):
    """The model adapter for WizardCoder (e.g., WizardLM/WizardCoder-Python-34B-V1.0)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "wizardcoder" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        # Same as Alpaca, see :
        # https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/inference_wizardcoder.py#L60
        return get_conv_template("alpaca")


class QwenChatAdapter(BaseModelAdapter):
    """The model adapter for Qwen/Qwen-7B-Chat
    To run this model, you need to ensure additional flash attention installation:
    ``` bash
    git clone https://github.com/Dao-AILab/flash-attention
    cd flash-attention && pip install .
    pip install csrc/layer_norm
    pip install csrc/rotary
    ```

    Since from 2.0, the following change happened
    - `flash_attn_unpadded_func` -> `flash_attn_varlen_func`
    - `flash_attn_unpadded_qkvpacked_func` -> `flash_attn_varlen_qkvpacked_func`
    - `flash_attn_unpadded_kvpacked_func` -> `flash_attn_varlen_kvpacked_func`
    You may need to revise the code in: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/modeling_qwen.py#L69
    to from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    """

    def match(self, model_path: str):
        return (
            "qwen" in model_path.lower()
            or ("dracarys" in model_path.lower() and "llama" not in model_path.lower())
            or "qwq" in model_path.lower()
        )


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen-7b-chat")


class SmaugChatAdapter(BaseModelAdapter):
    """The model adapter for abacusai/Smaug-2-72B."""

    def match(self, model_path: str):
        return "smaug-2" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen-7b-chat")


class BGEAdapter(BaseModelAdapter):
    """The model adapter for BGE (e.g., BAAI/bge-large-en-v1.5)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "bge" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


class E5Adapter(BaseModelAdapter):
    """The model adapter for E5 (e.g., intfloat/e5-large-v2)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "e5-" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


class AquilaChatAdapter(BaseModelAdapter):
    """The model adapter for BAAI/Aquila

    Now supports:
    - BAAI/AquilaChat-7B
    - BAAI/AquilaChat2-7B
    - BAAI/AquilaChat2-34B
    """

    def match(self, model_path: str):
        return "aquila" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        # See: https://huggingface.co/BAAI/AquilaChat2-34B/blob/4608b75855334b93329a771aee03869dbf7d88cc/predict.py#L347
        if "aquilachat2" in model_path:
            if "16k" in model_path:
                return get_conv_template("aquila")
            elif "34b" in model_path:
                return get_conv_template("aquila-legacy")
            else:
                return get_conv_template("aquila-v1")
        else:
            return get_conv_template("aquila-chat")


class Lamma2ChineseAdapter(BaseModelAdapter):
    """The model adapter for FlagAlpha/LLama2-Chinese sft"""

    def match(self, model_path: str):
        return "llama2-chinese" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama2-chinese")


class Lamma2ChineseAlpacaAdapter(BaseModelAdapter):
    """The model adapter for ymcui/Chinese-LLaMA-Alpaca sft"""

    def match(self, model_path: str):
        return "chinese-alpaca" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("chinese-alpaca2")


class VigogneAdapter(BaseModelAdapter):
    """The model adapter for vigogne (e.g., bofenghuang/vigogne-2-7b-chat)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return bool(re.search(r"vigogne|vigostral", model_path, re.I))


    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "chat" in model_path.lower():
            if "vigostral" in model_path.lower():
                return get_conv_template("vigogne_chat_v3")
            return get_conv_template("vigogne_chat_v2")
        return get_conv_template("vigogne_instruct")


class OpenLLaMaOpenInstructAdapter(BaseModelAdapter):
    """The model adapter for OpenLLaMa-Open-Instruct (e.g., VMware/open-llama-7b-open-instruct)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "open-llama" in model_path.lower() and "open-instruct" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")


class CodeLlamaAdapter(BaseModelAdapter):
    """The model adapter for CodeLlama (e.g., codellama/CodeLlama-34b-hf)"""

    def match(self, model_path: str):
        return "codellama" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")


class StableVicunaAdapter(BaseModelAdapter):
    """The model adapter for StableVicuna"""

    def match(self, model_path: str):
        return "stable-vicuna" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("stable-vicuna")


class PhindCodeLlamaAdapter(CodeLlamaAdapter):
    """The model adapter for Phind-CodeLlama (e.g., Phind/Phind-CodeLlama-34B-v2)"""

    def match(self, model_path: str):
        return "phind-codellama-" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phind")


class Llama2ChangAdapter(Llama2Adapter):
    """The model adapter for Llama2-ko-chang (e.g., lcw99/llama2-ko-chang-instruct-chat)"""

    def match(self, model_path: str):
        return "llama2-ko-chang" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("polyglot_changgpt")


class ZephyrAdapter(BaseModelAdapter):
    """The model adapter for Zephyr (e.g. HuggingFaceH4/zephyr-7b-alpha)"""

    def match(self, model_path: str):
        return "zephyr" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("zephyr")


class NotusAdapter(BaseModelAdapter):
    """The model adapter for Notus (e.g. argilla/notus-7b-v1)"""

    def match(self, model_path: str):
        return "notus" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("zephyr")


class CatPPTAdapter(BaseModelAdapter):
    """The model adapter for CatPPT (e.g. rishiraj/CatPPT)"""

    def match(self, model_path: str):
        return "catppt" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("catppt")


class TinyLlamaAdapter(BaseModelAdapter):
    """The model adapter for TinyLlama (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)"""

    def match(self, model_path: str):
        return "tinyllama" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("TinyLlama")


class XwinLMAdapter(BaseModelAdapter):
    """The model adapter for Xwin-LM V0.1 and V0.2 series of models(e.g., Xwin-LM/Xwin-LM-70B-V0.1)"""

    # use_fast_tokenizer = False

    def match(self, model_path: str):
        return "xwin-lm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.1")


class LemurAdapter(BaseModelAdapter):
    """The model adapter for OpenLemur/lemur-70b-chat-v1"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "lemur-70b-chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("lemur-70b-chat")


class PygmalionAdapter(BaseModelAdapter):
    """The model adapter for Pygmalion/Metharme series of models(e.g., PygmalionAI/mythalion-13b)"""

    # use_fast_tokenizer = False

    def match(self, model_path: str):
        return bool(re.search(r"pygmalion|mythalion|metharme", model_path.lower(), re.I))

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("metharme")


class XdanAdapter(BaseModelAdapter):
    """The model adapter for xDAN-AI (e.g. xDAN-AI/xDAN-L1-Chat-RL-v1)"""

    def match(self, model_path: str):
        return "xdan" in model_path.lower() and "v1" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("xdan-v1")


class MicrosoftOrcaAdapter(BaseModelAdapter):
    """The model adapter for Microsoft/Orca-2 series of models (e.g. Microsoft/Orca-2-7b, Microsoft/Orca-2-13b)"""

    use_fast_tokenizer = (
        False  # Flag neeeded since tokenizers>=0.13.3 is required for a normal functioning of this module
    )

    def match(self, model_path: str):
        return "orca-2" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("orca-2")


class YiAdapter(BaseModelAdapter):
    """The model adapter for Yi models"""

    def match(self, model_path: str):
        return "yi-" in model_path.lower() and "chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("Yi-34b-chat")


class DeepseekCoderAdapter(BaseModelAdapter):
    """The model adapter for deepseek-ai's coder models"""

    def match(self, model_path: str):
        return "deepseek-coder" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("deepseek-coder")


class DeepseekChatAdapter(BaseModelAdapter):
    """The model adapter for deepseek-ai's chat models"""

    # Note: that this model will require tokenizer version >= 0.13.3 because the tokenizer class is LlamaTokenizerFast

    def match(self, model_path: str):
        return (
            "deepseek-llm" in model_path.lower() and "chat" in model_path.lower()
        ) or "deepseek-chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("deepseek-chat")


class NvidiaChatAdapter(BaseModelAdapter):
    """The model adapter for nemotron's chat models"""

    def match(self, model_path: str):
        return "nemotron" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("api_based_default")


class Yuan2Adapter(BaseModelAdapter):
    """The model adapter for Yuan2.0"""

    def match(self, model_path: str):
        return "yuan2" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("yuan2")


class MetaMathAdapter(BaseModelAdapter):
    """The model adapter for MetaMath models"""

    def match(self, model_path: str):
        return "metamath" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("metamath")


class BagelAdapter(BaseModelAdapter):
    """Model adapter for jondurbin/bagel-* models"""

    def match(self, model_path: str):
        return "bagel" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("airoboros_v3")


class SolarAdapter(BaseModelAdapter):
    """The model adapter for upstage/SOLAR-10.7B-Instruct-v1.0"""

    def match(self, model_path: str):
        return "solar-" in model_path.lower() and "instruct" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("solar")


class SteerLMAdapter(BaseModelAdapter):
    """The model adapter for nvidia/Llama2-70B-SteerLM-Chat"""

    def match(self, model_path: str):
        return "steerlm-chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("steerlm")


class LlavaAdapter(BaseModelAdapter):
    """The model adapter for liuhaotian/llava-v1.5 series of models"""


    def match(self, model_path: str):
        return "llava" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "34b" in model_path:
            return get_conv_template("llava-chatml")

        return get_conv_template("vicuna_v1.1")


class YuanAdapter(BaseModelAdapter):
    """The model adapter for Yuan"""

    def match(self, model_path: str):
        return "yuan" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("yuan")


class GemmaAdapter(BaseModelAdapter):
    """The model adapter for Gemma"""

    def match(self, model_path: str):
        return "gemma" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("gemma")


class PhiAdapter(BaseModelAdapter):
    """The model adapter for microsoft/phi"""

    def match(self, model_path: str):
        return "phi-3" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phi-3")


class CllmAdapter(BaseModelAdapter):
    """The model adapter for CLLM"""

    def match(self, model_path: str):
        return "consistency-llm" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("cllm")


class CohereAdapter(BaseModelAdapter):
    """The model adapter for Cohere"""

    def match(self, model_path: str):
        return "command" in model_path.lower()


    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("api_based_default")


# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
register_model_adapter(StableVicunaAdapter)
register_model_adapter(VicunaAdapter)
register_model_adapter(AiroborosAdapter)
register_model_adapter(LongChatAdapter)
register_model_adapter(GoogleT5Adapter)
register_model_adapter(KoalaAdapter)
register_model_adapter(AlpacaAdapter)
register_model_adapter(ChatGLMAdapter)
register_model_adapter(CodeGeexAdapter)
register_model_adapter(DollyV2Adapter)
register_model_adapter(OasstPythiaAdapter)
register_model_adapter(OasstLLaMAAdapter)
register_model_adapter(OpenChat35Adapter)
register_model_adapter(TenyxChatAdapter)
register_model_adapter(StableLMAdapter)
register_model_adapter(BaizeAdapter)
register_model_adapter(RwkvAdapter)
register_model_adapter(OpenBuddyAdapter)
register_model_adapter(PhoenixAdapter)
register_model_adapter(BardAdapter)
register_model_adapter(PaLM2Adapter)
register_model_adapter(GeminiAdapter)
register_model_adapter(GeminiDevAdapter)
register_model_adapter(ChatGPTAdapter)
register_model_adapter(AzureOpenAIAdapter)
register_model_adapter(ClaudeAdapter)
register_model_adapter(MPTAdapter)
register_model_adapter(BiLLaAdapter)
register_model_adapter(RedPajamaINCITEAdapter)
register_model_adapter(H2OGPTAdapter)
register_model_adapter(RobinAdapter)
register_model_adapter(SnoozyAdapter)
register_model_adapter(WizardLMAdapter)
register_model_adapter(ManticoreAdapter)
register_model_adapter(GuanacoAdapter)
register_model_adapter(CamelAdapter)
register_model_adapter(ChangGPTAdapter)
register_model_adapter(TuluAdapter)
register_model_adapter(FalconChatAdapter)
register_model_adapter(FalconAdapter)
register_model_adapter(TigerBotAdapter)
register_model_adapter(BaichuanAdapter)
register_model_adapter(XGenAdapter)
register_model_adapter(PythiaAdapter)
register_model_adapter(InternLMChatAdapter)
register_model_adapter(StarChatAdapter)
register_model_adapter(Llama2Adapter)
register_model_adapter(Llama3Adapter)
register_model_adapter(CuteGPTAdapter)
register_model_adapter(OpenOrcaAdapter)
register_model_adapter(DolphinAdapter)
register_model_adapter(Hermes2Adapter)
register_model_adapter(NousHermes2MixtralAdapter)
register_model_adapter(NousHermesAdapter)
register_model_adapter(MistralAdapter)
register_model_adapter(WizardCoderAdapter)
register_model_adapter(QwenChatAdapter)
register_model_adapter(AquilaChatAdapter)
register_model_adapter(BGEAdapter)
register_model_adapter(E5Adapter)
register_model_adapter(Lamma2ChineseAdapter)
register_model_adapter(Lamma2ChineseAlpacaAdapter)
register_model_adapter(VigogneAdapter)
register_model_adapter(OpenLLaMaOpenInstructAdapter)
register_model_adapter(ReaLMAdapter)
register_model_adapter(PhindCodeLlamaAdapter)
register_model_adapter(CodeLlamaAdapter)
register_model_adapter(Llama2ChangAdapter)
register_model_adapter(ZephyrAdapter)
register_model_adapter(NotusAdapter)
register_model_adapter(CatPPTAdapter)
register_model_adapter(TinyLlamaAdapter)
register_model_adapter(XwinLMAdapter)
register_model_adapter(LemurAdapter)
register_model_adapter(PygmalionAdapter)
register_model_adapter(MicrosoftOrcaAdapter)
register_model_adapter(XdanAdapter)
register_model_adapter(YiAdapter)
register_model_adapter(PplxAIAdapter)
register_model_adapter(DeepseekCoderAdapter)
register_model_adapter(NvidiaChatAdapter)
register_model_adapter(DeepseekChatAdapter)
register_model_adapter(Yuan2Adapter)
register_model_adapter(MetaMathAdapter)
register_model_adapter(BagelAdapter)
register_model_adapter(SolarAdapter)
register_model_adapter(SteerLMAdapter)
register_model_adapter(LlavaAdapter)
register_model_adapter(YuanAdapter)
register_model_adapter(GemmaAdapter)
register_model_adapter(CllmAdapter)
register_model_adapter(CohereAdapter)
register_model_adapter(SmaugChatAdapter)
register_model_adapter(PhiAdapter)

# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)
