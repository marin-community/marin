import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for Marin agents that interact with LLMs (OpenAI, HuggingFace, etc.).
    Supports three modes:
      - 'auto': fully automatic, returns only valid outputs or raises errors
      - 'manual': interactively confirms/edits with user via callback
      - 'suggest': provides suggestions, user must confirm/modify
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        mode: str = "auto",
        user_interact: Callable[[str, Any], Any] | None = None,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Args:
            model: Model name or path (e.g., 'gpt-4o', 'llama-3', 'meta-llama/Llama-3-8B')
            provider: 'openai', 'huggingface', or other supported providers
            mode: 'auto', 'manual', or 'suggest'
            user_interact: Optional callback for user interaction (prompt, data) -> new_data
            max_retries: Maximum retries for prompt on invalid output
            kwargs: Additional provider/model-specific arguments
        Note: For OpenAI, the API key must be set via the OPENAI_API_KEY environment variable.
        """
        self.model = model
        self.provider = provider
        self.kwargs = kwargs
        self.mode = mode
        self.user_interact = user_interact
        self.max_retries = max_retries
        self.llm = self._init_llm()

    def _init_llm(self):
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider.")
            return OpenAI()
        elif self.provider == "huggingface":
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("transformers package is required for HuggingFace provider.")
            return pipeline("text-generation", model=self.model, **self.kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def prompt(self, prompt: str, use_json_mode: bool = False, **kwargs) -> str:
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    extra_kwargs = {"response_format": {"type": "json_object"}} if use_json_mode else {}
                    response = self.llm.chat.completions.create(
                        model=self.model, messages=[{"role": "user", "content": prompt}], **extra_kwargs, **kwargs
                    )
                    output = response.choices[0].message.content
                elif self.provider == "huggingface":
                    outputs = self.llm(prompt, **kwargs)
                    output = outputs[0]["generated_text"]
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                logger.info(f"Full LLM response (attempt {attempt+1}): {output}")
                return output
            except Exception as e:
                logger.warning(f"Prompt failed (attempt {attempt+1}): {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Prompt failed after {self.max_retries} attempts: {e}")

    def interact(self, prompt: str, data: Any, context: dict = None) -> Any:
        """
        If in manual/suggest mode and user_interact is set, call it. Otherwise, return data.
        """
        if self.mode in ("manual", "suggest") and self.user_interact is not None:
            return self.user_interact(prompt, data, context)
        return data

    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement the run() method.")
