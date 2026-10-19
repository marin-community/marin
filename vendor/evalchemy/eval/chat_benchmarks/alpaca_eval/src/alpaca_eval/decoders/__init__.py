import logging
from typing import Callable, Union


def get_fn_completions(name: Union[str, Callable]) -> Callable:
    """Get a decoder by name."""
    if not isinstance(name, str):
        return name

    if name == "anthropic_completions":
        try:
            from .anthropic import anthropic_completions
        except ImportError as e:
            packages = ["anthropic"]
            logging.exception(f"You need {packages} to use anthropic_completions. Error:")
            raise e

        return anthropic_completions

    elif name == "google_completions":
        try:
            from .google import google_completions
        except ImportError as e:
            packages = ["google.generativeai"]
            logging.exception(f"You need {packages} to use google_completions. Error:")
            raise e

        return google_completions

    elif name == "openai_completions":
        try:
            from .openai import openai_completions
        except ImportError as e:
            packages = ["openai"]
            logging.exception(f"You need {packages} to use openai_completions. Error:")
            raise e

        return openai_completions

    elif name == "huggingface_api_completions":
        try:
            from .huggingface_api import huggingface_api_completions
        except ImportError as e:
            packages = ["huggingface_hub"]
            logging.exception(f"You need {packages} to use huggingface_api_completions. Error:")
            raise e

        return huggingface_api_completions

    elif name == "huggingface_local_completions":
        try:
            from .huggingface_local import huggingface_local_completions
        except ImportError as e:
            packages = ["accelerate", "transformers", "bitsandbytes", "xformers", "peft", "optimum", "scipy"]
            logging.exception(f"You need {packages} to use huggingface_local_completions. Error:")
            raise e

        return huggingface_local_completions

    elif name == "cohere_completions":
        try:
            from .cohere import cohere_completions
        except ImportError as e:
            packages = ["cohere"]
            logging.exception(f"You need {packages} to use cohere_completions. Error:")
            raise e

        return cohere_completions

    elif name == "replicate_completions":
        try:
            from .replicate import replicate_completions
        except ImportError as e:
            packages = ["replicate"]
            logging.exception(f"You need {packages} to use replicate_completions. Error:")
            raise e

        return replicate_completions

    elif name == "jina_chat_completions":
        from .jinachat import jina_chat_completions

        return jina_chat_completions

    elif name == "vllm_local_completions":
        try:
            from .vllm_local import vllm_local_completions

            return vllm_local_completions
        except ImportError as e:
            packages = ["vllm", "ray", "transformers"]
            logging.exception(f"You need {packages} to use vllm_completions. Error:")
            raise e

    elif name == "sambanova_completions":
        try:
            from .sambanova_api import sambanova_completions

            return sambanova_completions
        except ImportError as e:
            packages = ["vllm", "ray", "transformers"]
            logging.exception(f"You need {packages} to use vllm_completions. Error:")
            raise e

    elif name == "bedrock_anthropic_completions":
        try:
            from .bedrock_anthropic import bedrock_anthropic_completions

            return bedrock_anthropic_completions
        except ImportError as e:
            packages = ["boto3"]
            logging.exception(f"You need {packages} to use bedrock_anthropic. Error:")
            raise e

    else:
        raise ValueError(f"Unknown decoder: {name}")
