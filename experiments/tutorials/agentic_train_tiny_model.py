"""
Tutorial: Agentic Workflow for Training a Tiny Model with Marin Agents

This script demonstrates how to use DatasetAgent and HyperparameterAgent in both auto and manual/suggest modes.
- Validates a Hugging Face dataset (TinyStories) using DatasetAgent
- Suggests hyperparameters using HyperparameterAgent
- Runs a training experiment with or without human-in-the-loop
"""

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import CpuOnlyConfig
from marin.agents.dataset_agent import DatasetAgent
from marin.agents.hparam_agent import HyperparameterAgent

# Optional: Example user interaction callback for manual/suggest mode

def cli_user_interact(prompt, data):
    print("\nAGENT SUGGESTION:")
    print(data)
    print("\n" + prompt)
    user_input = input("Accept suggestion? (y/n/edit): ").strip().lower()
    if user_input == "y":
        return data
    elif user_input == "edit":
        print("Paste your edited config (end with a blank line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        return "\n".join(lines)
    else:
        raise RuntimeError("User rejected the agent suggestion.")

# 1. Choose a dataset
hf_id = "roneneldan/TinyStories"

# --- AGENTIC DATASET VALIDATION ---
# AUTO MODE (fully automatic)
dataset_agent_auto = DatasetAgent(model="gpt-4", provider="openai", mode="auto")
dataset_info = dataset_agent_auto.validate(hf_id)
# Use the config_snippet for tokenization (could parse YAML/dict if needed)

# MANUAL/SUGGEST MODE (human-in-the-loop)
dataset_agent_manual = DatasetAgent(model="gpt-4", provider="openai", mode="manual", user_interact=cli_user_interact)
# Uncomment to try manual mode:
# dataset_info = dataset_agent_manual.validate(hf_id)

# 2. Tokenize the dataset (using agent output)
tinystories_tokenized = default_tokenize(
    name=hf_id,
    dataset=hf_id,  # In a real workflow, parse dataset_info['config_snippet'] for custom args
    tokenizer=marin_tokenizer,
)

# --- AGENTIC HYPERPARAMETER SUGGESTION ---
# Current config (could be empty or a default)
default_hparams = {
    "resources": str(CpuOnlyConfig(num_cpus=1)),
    "train_batch_size": 4,
    "num_train_steps": 100,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
}
dataset_metadata = {"num_examples": 10000, "source": hf_id}  # Example metadata

# AUTO MODE (fully automatic)
hparam_agent_auto = HyperparameterAgent(model="gpt-4", provider="openai", mode="auto")
hparam_suggestions = hparam_agent_auto.suggest(default_hparams, dataset_metadata)
# Pick the first valid config (could parse YAML/dict if needed)

# MANUAL/SUGGEST MODE (human-in-the-loop)
hparam_agent_manual = HyperparameterAgent(model="gpt-4", provider="openai", mode="manual", user_interact=cli_user_interact)
# Uncomment to try manual mode:
# hparam_suggestions = hparam_agent_manual.suggest(default_hparams, dataset_metadata)

# 3. Define training configuration (using agent output or default)
# In a real workflow, parse hparam_suggestions['suggested_configs'][0] to a dict and use it
nano_train_config = SimpleTrainConfig(
    resources=CpuOnlyConfig(num_cpus=1),
    train_batch_size=4,
    num_train_steps=100,
    learning_rate=6e-4,
    weight_decay=0.1,
    max_eval_batches=4,
)

# 4. Train the model
nano_tinystories_model = default_train(
    name="marin-nano-tinystories-agentic",
    tokenized=tinystories_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    tags=["llama", "nano", "tinystories", "tutorial", "agentic"],
    eval_harness_tasks=[],
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[nano_tinystories_model]) 