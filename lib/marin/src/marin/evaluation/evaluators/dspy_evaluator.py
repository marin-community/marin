import json
import os
import re
from typing import Any
from collections.abc import Callable, Iterable
from datetime import datetime

import dspy
import requests
from openai import AsyncOpenAI

from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig


class DspyEvaluator(Evaluator):
    """
    Marin <-> DSPy evaluator wrapper.

    Adds:
      - endpoint healthcheck + (optional) local OAI server bootstrap
      - format error rate measurement (JSON / regex / custom validator)
      - writes metrics to output_path (json)
      - versions outputs with timestamps
      - saves raw responses

    To enable format checks, pass one of:
      - format_json=True
      - format_regex="^...$"
      - format_validator=<callable taking (text:str)->bool>

    NOTE:
      - langprobe is an optional dependency. We lazy-import it.
    """

    def __init__(
        self,
        model: ModelConfig,
        endpoint: str,
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        # format checks
        format_json: bool = False,
        format_regex: str | None = None,
        format_validator: Callable[[str], bool] | None = None,
        # misc
        request_timeout_s: float = 3.0,
        **kwargs,
    ):
        super().__init__()

        self.endpoint = endpoint.rstrip("/")
        
        # Versioning: Append timestamp to output_path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f"{output_path}_{timestamp}"
        
        self.max_eval_instances = max_eval_instances
        self.wandb_tags = wandb_tags
        self.model = model

        self.format_json = format_json
        self.format_regex = format_regex
        self.format_validator = format_validator
        self.request_timeout_s = request_timeout_s

        # Validate / bootstrap endpoint
        self.endpoint = self._validate_endpoint()

        # DSPy/OpenAI-style client
        self.client = AsyncOpenAI(base_url=self.endpoint, api_key="dspy")

        # Lazy import langprobe here (optional dependency)
        self.langprobe = self._make_langprobe(**kwargs)

    def _make_langprobe(self, **kwargs):
        """
        Create EvaluateBench if langprobe is available.
        If not installed, keep None and raise a clear error when evaluate() is called.
        """
        try:
            from langprobe import EvaluateBench  # type: ignore
        except Exception:
            return None
        return EvaluateBench(**kwargs)

    def _validate_endpoint(self) -> str:
        health_url = f"{self.endpoint}/health"
        try:
            resp = requests.get(health_url, timeout=self.request_timeout_s)
            if resp.status_code == 200:
                return self.endpoint
        except Exception:
            pass

        return self._run_oai_server()

    def _run_oai_server(self) -> str:
        """
        Try to start a local OpenAI-compatible server via Levanter InferenceServer.
        """
        default_endpoint = "http://localhost:8000"

        cfg = InferenceServerConfig(host="0.0.0.0", port=8000)
        server = InferenceServer(cfg)

        if hasattr(server, "start"):
            server.start()
        elif hasattr(server, "run"):
            server.run()
        elif hasattr(server, "serve"):
            server.serve()
        else:
            raise RuntimeError(
                "InferenceServer has no start/run/serve method. "
                "Please check levanter.inference.openai.InferenceServer API."
            )

        return default_endpoint

    def _iter_text_outputs(self, result: Any) -> list[str]:
        keys = ("outputs", "predictions", "responses", "generations", "texts")

        def flatten_texts(x: Any) -> Iterable[str]:
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            if isinstance(x, dict):
                for k in ("text", "output", "prediction", "response", "generation"):
                    v = x.get(k)
                    if isinstance(v, str):
                        return [v]
                return []
            if isinstance(x, (list, tuple)):
                out: list[str] = []
                for item in x:
                    out.extend(list(flatten_texts(item)))
                return out
            return []

        if isinstance(result, dict):
            for k in keys:
                if k in result:
                    texts = list(flatten_texts(result.get(k)))
                    if texts:
                        return texts
            for k in ("result", "results", "metrics"):
                if k in result:
                    texts = self._iter_text_outputs(result.get(k))
                    if texts:
                        return texts
            return []

        if isinstance(result, (list, tuple)):
            return list(flatten_texts(result))

        for k in keys:
            if hasattr(result, k):
                texts = list(flatten_texts(getattr(result, k)))
                if texts:
                    return texts

        return []

    def _is_format_valid(self, text: str) -> bool | None:
        if self.format_validator is not None:
            try:
                return bool(self.format_validator(text))
            except Exception:
                return False

        if self.format_json:
            try:
                json.loads(text)
                return True
            except Exception:
                return False

        if self.format_regex is not None:
            try:
                return re.fullmatch(self.format_regex, text.strip()) is not None
            except re.error:
                return False

        return None

    def _write_metrics(self, payload: dict[str, Any]) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_responses(self, evals: list[dspy.Example]) -> None:
        """Save ID and Response for each example to responses.json."""
        if not evals:
            return

        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, "responses.json")
        
        responses_data = []
        for ex in evals:
            # Try to grab ID
            # Use getattr with default None, checking common ID fields
            item_id = getattr(ex, "did", None) or getattr(ex, "id", None) or getattr(ex, "record_id", None)
            
            # Try to grab prediction/response
            # DSPy typically attaches the prediction object to the example or returns it separately.
            # If dspy.Evaluate modified the example in place (common), it might be under 'prediction' or implicit.
            # We'll check for common attributes or just serialize the whole thing if unsure, 
            # but strictly requesting 'did' and 'response'.
            
            # Assuming the 'prediction' is attached or is the example itself if transformed.
            # Use safe serialization for the 'response' part.
            response_val = "N/A"
            # Check if example has a prediction attribute (standard in some flows)
            if hasattr(ex, "prediction"):
                response_val = str(ex.prediction)
            else:
                # Fallback: try to find the 'output' field from signature if possible, or just dump basic fields
                # For now, we'll try to capture the 'answer' or 'prediction' key if it exists in the dict
                ex_dict = ex.toDict() if hasattr(ex, "toDict") else ex.__dict__
                # Filter for keys that look like outputs
                possible_outputs = [v for k, v in ex_dict.items() if k in ["answer", "prediction", "output", "response"]]
                if possible_outputs:
                    response_val = str(possible_outputs[0])

            responses_data.append({
                "did": str(item_id) if item_id is not None else "unknown",
                "response": response_val
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(responses_data, f, ensure_ascii=False, indent=2)

    def evaluate(
            self,
            modules: dspy.Module,
            evals: list[dspy.Example],
            optimizer: Any,
            **kwargs,
    ) -> Any:

        if self.langprobe is None:
            raise RuntimeError(
                "langprobe is not installed, but DspyEvaluator.evaluate() was called.\n"
                "Install/add langprobe as a dependency (likely a git/internal package), "
                "or vendor it into the workspace."
            )

        result = self.langprobe.evaluate(modules, evals, optimizer, **kwargs)

        texts = self._iter_text_outputs(result)
        fmt_checks = [self._is_format_valid(t) for t in texts] if texts else []

        format_error_rate: float | None = None
        n_checked = 0
        n_invalid = 0

        configured_checks = [c for c in fmt_checks if c is not None]
        if configured_checks:
            n_checked = len(configured_checks)
            n_invalid = sum(1 for c in configured_checks if c is False)
            format_error_rate = (n_invalid / n_checked) if n_checked > 0 else None

        metrics_payload = {
            "endpoint": self.endpoint,
            "max_eval_instances": self.max_eval_instances,
            "format_json": self.format_json,
            "format_regex": self.format_regex,
            "format_error_rate": format_error_rate,
            "format_checked": n_checked,
            "format_invalid": n_invalid,
        }
        self._write_metrics(metrics_payload)
        
        # New: Save responses (did, response)
        self._write_responses(evals)

        return result
