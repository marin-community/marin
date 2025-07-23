"""
DatasetAgent: Automated dataset validation, schema inspection, and Marin/Levanter config/recipe generation.

Usage:
    agent = DatasetAgent(model="gpt-4o", provider="openai", mode="auto")
    result = agent.validate("roneneldan/TinyStories", recipe_mode=True)

Author: [Your Name]
"""

import json
import os
import re
from typing import Any

import datasets
import yaml

from marin.execution.executor import ExecutorStep

from .base_agent import BaseAgent


class DatasetAgent(BaseAgent):
    """
    Agent that validates a dataset (Hugging Face ID or local path), inspects schema, samples examples,
    and generates a Marin/Levanter-compatible config snippet or recipe.
    Modes:
      - 'auto': returns only valid config or raises error
      - 'manual'/'suggest': interactively confirms/edits with user
    """

    def validate(
        self,
        dataset_id_or_path: str,
        split: str = "train",
        sample_size: int = 5,
        recipe_mode: bool = False,
        default_tokenizer: str = "gpt2",
    ) -> dict[str, Any]:
        """
        Validate a dataset, inspect schema and samples, and generate a config or recipe.
        Args:
            dataset_id_or_path: Hugging Face dataset ID or local path
            split: Which split to inspect (default: 'train')
            sample_size: Number of examples to sample
            recipe_mode: If True, output a full recipe YAML; else, just the config
            default_tokenizer: Tokenizer to use if agent output is a placeholder
        Returns:
            Dict with keys: 'config_snippet' or 'recipe', 'schema', 'samples', 'rationale', 'agent_steps'
        """
        try:
            ds = datasets.load_dataset(dataset_id_or_path, split=split)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        schema = ds.features if hasattr(ds, "features") else None
        samples = ds.select(range(min(sample_size, len(ds))))
        sample_records = [dict(samples[i]) for i in range(len(samples))]

        prompt = self._build_prompt(dataset_id_or_path, schema, sample_records)
        llm_output = self.prompt(prompt, use_json_mode=True)
        config_snippet = self._extract_and_validate_config(llm_output)

        # Replace placeholder tokenizer with default
        if config_snippet:
            config_snippet = re.sub(
                r"tokenizer:\s*(specified_tokenizer_here|default_tokenizer)",
                f"tokenizer: {default_tokenizer}",
                config_snippet,
            )
            # Add validation_paths if validation split exists
            if "validation" in ds:
                config_snippet = config_snippet.rstrip() + f"\n  validation_paths: [{dataset_id_or_path}/validation]"

        rationale = self._generate_rationale(schema, sample_records, config_snippet)
        agent_steps = [
            "Load dataset",
            "Inspect schema/examples",
            "Validate for LLM suitability",
            "Generate config",
            "Output recipe" if recipe_mode else "Output config",
        ]
        context = {"schema": schema, "dataset_id_or_path": dataset_id_or_path, "samples": sample_records}
        if self.mode in ("manual", "suggest"):
            user_prompt = (
                "Review the config suggestion. "
                "Options: accept (y), edit (e), regenerate (r), clarify (c), or reject (n)."
            )
            config_snippet = self.interact(user_prompt, config_snippet, context)

        if recipe_mode:
            os.makedirs("recipes", exist_ok=True)
            recipe = self._generate_recipe_yaml(
                dataset_id_or_path, rationale, schema, sample_records, config_snippet, agent_steps
            )
            recipe_path = f"recipes/dataset_add_{os.path.basename(dataset_id_or_path)}.yaml"
            with open(recipe_path, "w") as f:
                f.write(recipe)
            return {
                "recipe": recipe,
                "schema": schema,
                "samples": sample_records,
                "rationale": rationale,
                "agent_steps": agent_steps,
            }
        else:
            return {
                "config_snippet": config_snippet,
                "schema": schema,
                "samples": sample_records,
                "rationale": rationale,
                "agent_steps": agent_steps,
            }

    def _build_prompt(self, dataset_id_or_path, schema, samples) -> str:
        return f"""
You are a dataset validation agent for Marin/Levanter LLM training.
Dataset: {dataset_id_or_path}
Schema: {schema}
Sample examples: {samples}

Output ONLY a JSON object with these keys:
- 'valid': boolean (true if suitable for text-based LLM pretraining, false otherwise)
- 'config': YAML string (complete config snippet starting with 'data:', only if valid=true)

Best practices: Ensure text-based, no multimodal. Config must include 'train_paths', 'tokenizer', etc.

Few-shot example:
Input: Dataset: example/text
Schema: {{'text': str}}
Samples: [{{'text': 'hello'}}]
Output: {{"valid": true, "config": "data:\n  train_paths: [example/text]\n  tokenizer: marin"}}

If not valid, set 'valid': false and 'config': null.
"""

    def _extract_and_validate_config(self, llm_output: str) -> str | None:
        try:
            parsed = json.loads(llm_output)
            if not parsed.get("valid", False):
                return None
            config_str = parsed.get("config")
            yaml.safe_load(config_str)  # Validate YAML
            return config_str
        except (json.JSONDecodeError, yaml.YAMLError):
            return None

    def _generate_rationale(self, schema: Any, samples: list[dict], config_snippet: str | None) -> list[str]:
        """Generate a rationale for dataset validation."""
        rationale = [
            "Dataset is text-based" if "text" in schema else "No 'text' field found",
            f"Schema fields: {list(schema.keys())}",
            f"Sample count: {len(samples)}",
            "Config generated successfully" if config_snippet else "Config generation failed",
        ]
        return rationale

    def _generate_recipe_yaml(
        self,
        dataset_id: str,
        rationale: list[str],
        schema: Any,
        samples: list[dict],
        config_snippet: str | None,
        agent_steps: list[str],
    ) -> str:
        """Generate a YAML recipe for dataset addition."""
        # Convert schema to a dict of field names to type names for YAML safety
        if isinstance(schema, dict):
            safe_schema = {k: v.__name__ if hasattr(v, "__name__") else str(v) for k, v in schema.items()}
        else:
            safe_schema = str(schema)
        recipe = {
            "dataset_id": dataset_id,
            "validation_rationale": "\n".join(rationale),
            "schema": safe_schema,
            "sample_examples": samples[:5],
            "config_snippet": config_snippet,
            "agent_steps": agent_steps,
        }
        return yaml.dump(recipe, sort_keys=False)


class DatasetAgentStep(ExecutorStep):
    """Wraps DatasetAgent as an ExecutorStep for Marin pipelines."""

    def __init__(
        self,
        name: str,
        dataset_id_or_path: str,
        split: str = "train",
        sample_size: int = 5,
        agent_kwargs: dict | None = None,
        log_file: str | None = None,
    ):
        self.agent = DatasetAgent(**(agent_kwargs or {}))
        self.dataset_id_or_path = dataset_id_or_path
        self.split = split
        self.sample_size = sample_size
        self.log_file = log_file
        super().__init__(
            name=name,
            description=f"Agentic validation for dataset {dataset_id_or_path}",
            fn=self.run,
            config=None,
        )

    def run(self) -> dict[str, Any]:
        result = self.agent.validate(self.dataset_id_or_path, split=self.split, sample_size=self.sample_size)
        log_msg = f"[DatasetAgentStep] Prompt: {self.agent._build_prompt(self.dataset_id_or_path, None, None)}\nOutput: {result.get('config_snippet', result.get('recipe'))}"
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
        else:
            print(log_msg)
        return result
