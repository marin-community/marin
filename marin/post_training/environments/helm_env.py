import json
import os
import uuid
from typing import Any
from yaml import dump

from transformers import AutoTokenizer
import numpy as np

from marin_env import MarinEnv, EnvStep
from marin.evaluation.utils import run_bash_command


class HELMEnv(MarinEnv):
    """
    Runs HELM with a single run spec.

    HELM Branch: marin_rl
    """

    # Where model/tokenizer YAMLs go
    _PROD_ENV_DIRNAME = "prod_env"
    # Top‑level for all helm‑run output
    _BENCHMARK_OUTPUT_ROOT = "benchmark_output"
    # helm‑run creates this inside the root folder
    _RUNS_DIRNAME = "runs"

    _VLLM_CLIENT_CLASS = "helm.clients.vllm_client.LocalVLLMClient"

    @staticmethod
    def write_run_entries(eval_name: str, dest_dir: str, n_generations: int = 1) -> str:
        """Creates `run_entries_*.conf` containing a single HELM run entry."""
        filename = f"run_entries_{eval_name}.conf"
        path = os.path.join(dest_dir, filename)
        if not os.path.exists(path):
            entry = f'{{description: "{eval_name}:num_outputs={n_generations},model=text", priority: 1}}'
            with open(path, "w") as f:
                f.write("entries: [\n  " + entry + "\n]\n")
        return path

    def __init__(self, output_dir_path: str, **kwargs):
        self._environment_id = uuid.uuid4().hex
        self._root = os.path.join(output_dir_path, self._environment_id)
        os.makedirs(self._root, exist_ok=True)

        self._model_name: str = kwargs["model_name"]
        self._eval_name: str = kwargs.get("eval", "ifeval")

        # Determines which metric is uses as the primary reward score
        self._primary_score_key = kwargs.get("primary_score_key", "exact_match")

        # Prepare shared folders once per environment
        self._prod_env_path = os.path.join(self._root, self._PROD_ENV_DIRNAME)
        os.makedirs(self._prod_env_path, exist_ok=True)

        # Single benchmark output dir for all steps (HELM requires this)
        self._benchmark_output = os.path.join(self._root, self._BENCHMARK_OUTPUT_ROOT)
        os.makedirs(self._benchmark_output, exist_ok=True)

        self._write_model_config_files()

    def step(
        self, sampler, params, n_examples: int, prng_key, mode: str = "train", n_generations: int = 1, **kwargs
    ) -> EnvStep:
        step_id = uuid.uuid4().hex
        suite_path = os.path.join(self._benchmark_output, self._RUNS_DIRNAME, step_id)
        os.makedirs(suite_path, exist_ok=True)

        # Write out the run entries if one does not exist
        eval_conf_path: str = self.write_run_entries(self._eval_name, self._root, n_generations)

        # Construct the command with correct parameters and launch HELM
        command: list[str] = [
            "helm-run",
            "--conf-paths",
            eval_conf_path,
            "--models-to-run",
            self._model_name,
            # isolate each run under unique suite
            "--suite",
            step_id,
            "--output-path",
            self._benchmark_output,
            "--local-path",
            self._prod_env_path,
            "--exit-on-error",
            "--max-eval-instances",
            str(n_examples),
        ]
        run_bash_command(command)

        # Extract out input, output and metric values (rewards)
        return self._extract_step_result(suite_path)

    def _write_model_config_files(self) -> None:
        """Write out the config files necessary to run the model with vLLM"""
        tok = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        max_len = tok.model_max_length

        # Create model_deployments.yaml to run the model with vLLM
        md = {
            "model_deployments": [
                {
                    "name": self._model_name,
                    "model_name": self._model_name,
                    "tokenizer_name": self._model_name,
                    "max_sequence_length": max_len,
                    "client_spec": {"class_name": self._VLLM_CLIENT_CLASS},
                }
            ]
        }
        with open(os.path.join(self._prod_env_path, "model_deployments.yaml"), "w") as f:
            dump(md, f)

        # model_metadata.yaml
        meta = {
            "models": [
                {
                    "name": self._model_name,
                    "display_name": self._model_name,
                    "access": "open",
                    "description": "",
                    "creator_organization_name": "",
                    "tags": ["TEXT_MODEL_TAG"],
                }
            ]
        }
        with open(os.path.join(self._prod_env_path, "model_metadata.yaml"), "w") as f:
            dump(meta, f)

        # tokenizer_configs.yaml
        tok_cfg = {
            "tokenizer_configs": [
                {
                    "name": self._model_name,
                    "tokenizer_spec": {
                        "class_name": "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
                        "args": {"pretrained_model_name_or_path": self._model_name, "trust_remote_code": True},
                    },
                    "prefix_token": tok.bos_token,
                    "end_of_text_token": tok.eos_token,
                }
            ]
        }
        with open(os.path.join(self._prod_env_path, "tokenizer_configs.yaml"), "w") as f:
            dump(tok_cfg, f)

    def _extract_step_result(self, results_path: str) -> EnvStep:
        """Read HELM outputs/results and returns `EnvStep`."""
        # Locate directory for this eval
        eval_dirs = [
            d
            for d in os.listdir(results_path)
            if d.startswith(self._eval_name) and os.path.isdir(os.path.join(results_path, d))
        ]
        if len(eval_dirs) != 1:
            raise RuntimeError(
                f"Expected exactly one directory starting with '{self._eval_name}' "
                f"in {results_path}, found {eval_dirs}."
            )
        eval_dir = os.path.join(results_path, eval_dirs[0])

        per_instance_stats_path = os.path.join(eval_dir, "per_instance_stats.json")
        scenario_state_path = os.path.join(eval_dir, "scenario_state.json")
        if not (os.path.isfile(per_instance_stats_path) and os.path.isfile(scenario_state_path)):
            raise RuntimeError("per_instance_stats.json or scenario_state.json missing in " + eval_dir)

        with open(per_instance_stats_path, "r") as f:
            per_instance_stats = json.load(f)
        with open(scenario_state_path, "r") as f:
            scenario_state = json.load(f)

        # Build reward map using `sum` value – error if metric or sum missing
        metric_key = self._primary_score_key
        reward_map: dict[str, float] = {}
        for entry in per_instance_stats:
            inst_id = entry.get("instance_id")
            if inst_id is None:
                raise RuntimeError("Missing instance_id in per_instance_stats entry")

            for stat in entry.get("stats"):
                name_obj = stat.get("name")
                if isinstance(name_obj, dict) and name_obj.get("name") == metric_key:
                    if "sum" not in stat:
                        raise RuntimeError(f"Stat '{metric_key}' missing 'sum' for instance {inst_id}")

                    # Add to reward map for this instance
                    reward_map[inst_id] = float(stat["sum"])
                    break

        # Build EnvSteps – ensure every request_state has a reward
        examples: list[dict[str, Any]] = []
        all_responses: list[list[dict[str, np.ndarray]]] = []
        rewards = []
        metrics: dict[str, float] = {}

        for req_state in scenario_state.get("request_states", []):
            instance: dict = req_state.get("instance")
            inst_id: str = instance.get("id")
            if inst_id not in reward_map:
                raise RuntimeError(f"Instance {inst_id} missing corresponding stats entry")

            prompt: str = req_state.get("request").get("prompt")
            examples.append({"prompt": prompt})

            responses = []
            completions = req_state.get("result").get("completions")
            for completion in completions:
                text = completion["text"]
                responses.append({"text": text})
            all_responses.append(responses)

            reward: float = reward_map[inst_id]
            rewards.append(reward)

        return EnvStep(examples, all_responses, np.array(rewards), metrics)


if __name__ == "__main__":
    env = HELMEnv(
        output_dir_path="output",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        eval="ifeval",
        primary_score_key="ifeval_strict_accuracy",
    )
    for i in range(5):
        step_result = env.step(
            # TODO: how to integrate with sampler? Do I need a SamplerClient in HELM?
            sampler=None,
            params=None,
            prng_key=None,
            mode="test",
            n_examples=1,
            n_generations=1,
        )
        print(f"step={i}", step_result)
