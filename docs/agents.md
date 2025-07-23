# Agentic Workflows in Marin

## Overview
Marin's agentic workflow system enables automated, transparent, and extensible experiment setup for LLM training. The core agents are:
- **DatasetAgent**: Validates datasets, inspects schema/samples, and generates Marin/Levanter-compatible configs or full onboarding recipes.
- **HyperparameterAgent**: Suggests hyperparameter configs, with hardware awareness and Marin-native output, supporting both auto and manual modes.

Agents can be used in auto, manual, or recipe modes, and support task decomposition for transparency and future tool integration (e.g., LangChain).

## Installation & Requirements
- Ensure you have the following in your `requirements.txt`:
  - `openai` (for OpenAI LLMs)
  - `datasets` (Hugging Face datasets)
  - `pyyaml`
  - `jax`, `equinox`, `haliax` (for hardware detection and Levanter integration)
  - (Optional) `langchain` for advanced agent chains
- Install with:
  ```sh
  pip install -r requirements.txt
  ```

## Usage Flows
### DatasetAgent
```python
from marin.agents.dataset_agent import DatasetAgent
agent = DatasetAgent(model="gpt-4o", provider="openai", mode="auto")
result = agent.validate("roneneldan/TinyStories", recipe_mode=True, default_tokenizer="gpt2")
print(result["recipe"])  # YAML recipe for onboarding
```
- **Modes:**
  - `auto`: Fully automated config/recipe generation
  - `manual`/`suggest`: Human-in-the-loop review/edit/clarify
  - `recipe_mode`: Output a full onboarding recipe (YAML)

### HyperparameterAgent
```python
from marin.agents.hparam_agent import HyperparameterAgent
agent = HyperparameterAgent(model="gpt-4o", provider="openai", mode="auto")
default_hparams = {"resources": "CpuOnlyConfig(num_cpus=1)", "train_batch_size": 4, ...}
dataset_metadata = {"num_examples": 10000, "source": "roneneldan/TinyStories"}
result = agent.suggest(default_hparams, dataset_metadata, preview_mode=True, decompose_executable=True)
print(result["marin_configs"])  # List of SimpleTrainConfig strings
```
- **Features:**
  - Hardware auto-detection
  - Task decomposition (subtasks, executable subtasks)
  - Marin-native config output

## Example Output (Sanitized)
```
[Recipe YAML]
dataset_id: roneneldan/TinyStories
validation_rationale: 'Dataset is text-based\nSchema fields: ["text"]\nSample count: 5\nConfig generated successfully'
schema:
  text: str
sample_examples:
- text: 'Once upon a time...'
config_snippet: "data:\n  train_paths: [roneneldan/TinyStories]\n  tokenizer: gpt2"
agent_steps:
- Load dataset
- Inspect schema/examples
- Validate for LLM suitability
- Generate config
- Output recipe
```

## Extensions & Advanced Usage
- **Playbooks/Recipes:** Use agent-generated YAML recipes for reproducible onboarding and review.
- **Task Decomposition:** Subtasks are logged and can be made executable (e.g., as LangChain tools).
- **LangChain Integration:** Agents can be extended to use LangChain for multi-step chains or tool calls.
- **Manual/Interactive Mode:** Use `mode="manual"` and pass a user_interact callback for human-in-the-loop workflows.

## Contributing New Agents or Recipes
- Add new agent classes to `marin/agents/` with clear docstrings and test coverage.
- Add new recipe templates to `experiments/recipes/`.
- Add tests to `tests/test_agents.py`.
- Document new features in this file and update the main README.

## See Also
- [Agentic End-to-End Experiment Demo](../experiments/tutorials/agentic_end_to_end_experiment.py)
- [Agentic Recipe Demo](../experiments/tutorials/agentic_recipe_demo.py)
