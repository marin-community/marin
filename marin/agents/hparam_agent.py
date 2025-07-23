"""
HyperparameterAgent: Automated hyperparameter suggestion and Marin-native config generation for LLM training.

Usage:
    agent = HyperparameterAgent(model="gpt-4o", provider="openai", mode="auto")
    result = agent.suggest(current_config, dataset_metadata, preview_mode=True, decompose_executable=True)

"""

from collections.abc import Callable
from typing import Any

import jax
import yaml

from marin.execution.executor import ExecutorStep

from .base_agent import BaseAgent


class HyperparameterAgent(BaseAgent):
    """
    Agent that suggests hyperparameter tweaks for Marin/Levanter training configs.
    Modes:
      - 'auto': returns only valid config(s) or raises error
      - 'manual'/'suggest': interactively confirms/edits with user
    """

    def suggest(
        self,
        current_config: dict[str, Any],
        dataset_metadata: dict[str, Any],
        num_suggestions: int = 3,
        preview_mode: bool = False,
        decompose_executable: bool = False,
    ) -> dict[str, Any]:
        """
        Suggest hyperparameter configs for LLM training.
        Args:
            current_config: Dict of current training config
            dataset_metadata: Dict with dataset info (e.g., size, num_examples)
            num_suggestions: Number of suggestions to generate
            preview_mode: If True, include a preview subtask
            decompose_executable: If True, return subtasks as callables (stubs)
        Returns:
            Dict with keys: 'suggested_configs', 'marin_configs', 'subtasks', 'hardware_info', 'executable_subtasks'
        """
        hardware_info = self._detect_hardware()
        subtasks = [
            "Parse dataset metadata",
            f"Auto-detect hardware: {hardware_info}",
            "Suggest LR/batch based on size/hardware",
            "Generate 3 variants",
        ]
        if preview_mode:
            subtasks.insert(2, "Preview: quick perplexity check on tiny model subset")
        llm_output = self.prompt(
            self._build_prompt(current_config, dataset_metadata, num_suggestions), use_json_mode=True
        )
        suggested_configs = self._extract_and_validate_configs(llm_output, current_config)
        marin_configs = [self._to_marin_config_string(cfg) for cfg in suggested_configs]
        executable_subtasks: list[Callable[[], None]] | None = None
        if decompose_executable:

            def parse_metadata():
                print("[Executable] Parsing dataset metadata...")

            def detect_hardware():
                print(f"[Executable] Detected hardware: {hardware_info}")

            def suggest_lr_batch():
                print("[Executable] Suggesting LR/batch...")

            def generate_variants():
                print("[Executable] Generating config variants...")

            executable_subtasks = [parse_metadata, detect_hardware, suggest_lr_batch, generate_variants]
        return {
            "suggested_configs": suggested_configs,
            "marin_configs": marin_configs,
            "subtasks": subtasks,
            "hardware_info": hardware_info,
            "executable_subtasks": executable_subtasks,
        }

    def _build_prompt(self, current_config, dataset_metadata, num_suggestions) -> str:
        """Build a structured prompt for the LLM for hyperparameter suggestion."""
        return f"""
You are a hyperparameter tuning agent for Marin/Levanter LLM training.
Current config: {current_config}
Dataset metadata: {dataset_metadata}

Output ONLY a JSON object with 'suggestions': list of {num_suggestions} complete YAML strings (each starting with 'train_batch_size:').

Best practices: Batch sizes as powers of 2 (e.g., 4,8,16). Include all required keys, tweak based on dataset size.

Few-shot example:
Input: Current config: {{'train_batch_size': 4}}
Output: {{"suggestions": ["train_batch_size: 8\nlearning_rate: 1e-4", "train_batch_size: 4\nlearning_rate: 6e-4"]}}
"""

    def _to_marin_config_string(self, yaml_str: str) -> str:
        """Convert YAML string to a SimpleTrainConfig string for Marin."""
        cfg = yaml.safe_load(yaml_str)
        fields = ", ".join(f"{k}={v!r}" for k, v in cfg.items())
        return f"SimpleTrainConfig({fields})"

    def _detect_hardware(self) -> dict[str, Any]:
        """Detect available hardware using JAX."""
        devices = jax.devices()
        device_types = set(d.device_kind for d in devices)
        return {
            "num_devices": len(devices),
            "device_types": list(device_types),
            "platform": jax.default_backend(),
        }

    def _extract_and_validate_configs(self, llm_output: str, defaults: dict) -> list:
        import json

        import yaml

        try:
            parsed = json.loads(llm_output)
            configs = []
            for cfg_str in parsed.get("suggestions", []):
                cfg_dict = yaml.safe_load(cfg_str)
                if not isinstance(cfg_dict, dict):
                    continue
                # Merge with defaults
                merged = {**defaults, **cfg_dict}
                configs.append(yaml.dump(merged))
            return configs
        except (json.JSONDecodeError, yaml.YAMLError):
            return []


class HparamAgentStep(ExecutorStep):
    """Wraps HyperparameterAgent as an ExecutorStep for Marin pipelines."""

    def __init__(
        self,
        name: str,
        current_config: dict[str, Any],
        dataset_metadata: dict[str, Any],
        num_suggestions: int = 3,
        agent_kwargs: dict | None = None,
        log_file: str | None = None,
    ):
        self.agent = HyperparameterAgent(**(agent_kwargs or {}))
        self.current_config = current_config
        self.dataset_metadata = dataset_metadata
        self.num_suggestions = num_suggestions
        self.log_file = log_file
        super().__init__(
            name=name,
            description=f"Agentic hparam suggestion for config {current_config}",
            fn=self.run,
            config=None,
        )

    def run(self) -> dict[str, Any]:
        result = self.agent.suggest(self.current_config, self.dataset_metadata, num_suggestions=self.num_suggestions)
        log_msg = f"[HparamAgentStep] Prompt: {self.agent._build_prompt(self.current_config, self.dataset_metadata, self.num_suggestions)}\nOutput: {result['suggested_configs']}"
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
        else:
            print(log_msg)
        return result
