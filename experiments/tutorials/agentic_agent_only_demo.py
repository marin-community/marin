"""
Tutorial: Agentic Workflow Demo (Agent-Only, No Training)

This script demonstrates the DatasetAgent and HyperparameterAgent flows in both auto and manual/suggest modes.
No actual tokenization or training is performed—only agent logic is tested.
Safe to run locally without GPU or model resources.

NOTE: For OpenAI models, set your API key in the environment before running:
    export OPENAI_API_KEY=sk-...
"""

from marin.agents.dataset_agent import DatasetAgent
from marin.agents.hparam_agent import HyperparameterAgent
from marin.resources import CpuOnlyConfig

# Optional: Multi-turn user interaction callback for manual/suggest mode
def cli_user_interact(agent, prompt, data, context=None):
    schema = context.get("schema") if context else None
    dataset_id_or_path = context.get("dataset_id_or_path") if context else None
    samples = context.get("samples") if context else None
    while True:
        print("\nAGENT SUGGESTION:")
        print(data)
        print("\n" + prompt)
        user_input = input("(y) accept, (e) edit, (r) regenerate, (c) clarify, (n) reject: ").strip().lower()
        if user_input == "y":
            return data
        elif user_input == "e":
            print("Paste your edited config (end with a blank line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            return "\n".join(lines)
        elif user_input == "r":
            print("Regenerating...")
            rebuilt_prompt = agent._build_prompt(dataset_id_or_path, schema, samples)
            new_output = agent.prompt(rebuilt_prompt, use_json_mode=True)
            return agent._extract_and_validate_config(new_output)
        elif user_input == "c":
            clarify_prompt = input("Your question/clarification for the agent: ")
            full_clarify_prompt = f"Context: You are analyzing dataset {dataset_id_or_path} with schema {schema} for LLM training. Original prompt: {agent._build_prompt(dataset_id_or_path, schema, samples)}.\nUser question: {clarify_prompt}"
            clarify_response = agent.prompt(full_clarify_prompt)
            print("Agent response:", clarify_response)
            continue
        else:
            raise RuntimeError("User rejected the agent suggestion.")

# 1. Choose a dataset
hf_id = "roneneldan/TinyStories"

print("\n=== DATASET AGENT: AUTO MODE ===")
dataset_agent_auto = DatasetAgent(model="gpt-4o", provider="openai", mode="auto")
dataset_info_auto = dataset_agent_auto.validate(hf_id)
print("\n[Auto] Agent config snippet:")
print(dataset_info_auto["config_snippet"])

print("\n=== DATASET AGENT: MANUAL MODE ===")
# Update lambda to pass context through
# (Assume dataset_agent_manual is created after schema/sample_records are available)
dataset_agent_manual = DatasetAgent(model="gpt-4o", provider="openai", mode="manual", user_interact=lambda p, d, c: cli_user_interact(dataset_agent_manual, p, d, c))
dataset_info_manual = dataset_agent_manual.validate(hf_id)
print("\n[Manual] Agent config snippet:")
print(dataset_info_manual["config_snippet"])

# === DATASET AGENT: AUTO MODE (HuggingFace Local) ===
# NOTE: This example requires a GPU and a local or open HuggingFace model. Uncomment to test with GPU support.
#dataset_agent_hf = DatasetAgent(model="gpt2", provider="huggingface", mode="auto")
#dataset_info_hf = dataset_agent_hf.validate(hf_id)
#print("\n[HuggingFace] Agent config snippet:")
#print(dataset_info_hf["config_snippet"])

# 2. Hyperparameter suggestion
default_hparams = {
    "resources": str(CpuOnlyConfig(num_cpus=1)),
    "train_batch_size": 4,
    "num_train_steps": 100,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
}
dataset_metadata = {"num_examples": 10000, "source": hf_id}

print("\n=== HYPERPARAM AGENT: AUTO MODE ===")
hparam_agent_auto = HyperparameterAgent(model="gpt-4o", provider="openai", mode="auto")
hparam_suggestions_auto = hparam_agent_auto.suggest(default_hparams, dataset_metadata)
print("\n[Auto] Agent hyperparameter suggestions:")
for cfg in hparam_suggestions_auto["suggested_configs"]:
    print(cfg)

print("\n=== HYPERPARAM AGENT: MANUAL MODE ===")
hparam_agent_manual = HyperparameterAgent(model="gpt-4o", provider="openai", mode="manual", user_interact=lambda p, d, c=None: cli_user_interact(hparam_agent_manual, p, d, c))
hparam_suggestions_manual = hparam_agent_manual.suggest(default_hparams, dataset_metadata)
print("\n[Manual] Agent hyperparameter suggestions:")
for cfg in hparam_suggestions_manual["suggested_configs"]:
    print(cfg) 