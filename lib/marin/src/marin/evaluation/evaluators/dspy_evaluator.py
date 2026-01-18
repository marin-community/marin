import dspy
import requests

from openai import AsyncOpenAI
from langprobe import EvaluateBench

from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig


class DspyEvaluator(Evaluator):
    def __init__(
        self,
        model: ModelConfig,
        endpoint: str,
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        **kwargs,
    ):
        super().__init__()

        self.endpoint = endpoint
        self.output_path = output_path
        self.max_eval_instances = max_eval_instances
        self.wandb_tags = wandb_tags

        self.model = model
        self.endpoint = self._validate_endpoint()
        self.client = AsyncOpenAI(base_url=self.endpoint, api_key="dspy")
        # IMPORTANT: pass keyword arguments correctly
        self.langprobe = EvaluateBench(**kwargs)

    def _validate_endpoint(self) -> str:
        response = requests.get(self.endpoint + "/health")
        if response.status_code == 200:
            return self.endpoint
        else:
            return self._run_oai_server()

    def _run_oai_server(self) -> str:
        # Run oai server and return the endpoint
        return "http://localhost:8000"

    def evaluate(
        self,
        model: dspy.Module,
        dataset: list[dspy.Example],
        optimizer: dspy.Teleprompter,
        **kwargs,
    ) -> None:
        return self.langprobe.evaluate(model, dataset, optimizer, **kwargs)
