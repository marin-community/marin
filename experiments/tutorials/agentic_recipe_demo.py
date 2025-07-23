"""
Agentic Recipe Demo: How to Use Recipes and Task Decomposition with Marin Agents

- Uses DatasetAgent in recipe_mode to generate a filled dataset_add.yaml recipe for TinyStories
- Prints the recipe, rationale, and agent steps
- Uses HyperparameterAgent with preview_mode to generate suggestions, print subtasks, hardware info, and Marin-native config strings
- No training or tokenization required—just agent logic and output display
"""

from marin.agents.dataset_agent import DatasetAgent
from marin.agents.hparam_agent import HyperparameterAgent
from marin.resources import CpuOnlyConfig

hf_id = "roneneldan/TinyStories"

print("=== DATASET AGENT: RECIPE MODE ===")
dataset_agent = DatasetAgent(model="gpt-4o", provider="openai", mode="auto")
dataset_result = dataset_agent.validate(hf_id, recipe_mode=True)
print("\n[Recipe YAML]\n" + dataset_result["recipe"])
print("[Rationale]\n" + '\n'.join(dataset_result["rationale"]))
print("[Agent Steps]", dataset_result["agent_steps"])

print("\n=== HYPERPARAM AGENT: PREVIEW MODE ===")
default_hparams = {
    "resources": str(CpuOnlyConfig(num_cpus=1)),
    "train_batch_size": 4,
    "num_train_steps": 100,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
}
dataset_metadata = {"num_examples": 10000, "source": hf_id}
hparam_agent = HyperparameterAgent(model="gpt-4o", provider="openai", mode="auto")
hparam_result = hparam_agent.suggest(default_hparams, dataset_metadata, preview_mode=True)
print("[Subtasks]", hparam_result["subtasks"])
print("[Hardware Info]", hparam_result["hardware_info"])
print("[Marin Config Strings]")
for cfg in hparam_result["marin_configs"]:
    print(cfg) 