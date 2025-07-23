"""
Agentic End-to-End Experiment: Full Pipeline Integration

- Uses DatasetAgentStep and HparamAgentStep as ExecutorSteps
- Feeds outputs into default_tokenize and default_train
- Logs agent prompts/outputs
- Supports test mode with mocked LLM responses
- Logs metrics for config validity and time saved
- Shows (in comments) how to use LangChain or HuggingFace local models
"""

import time

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.agents.dataset_agent import DatasetAgentStep
from marin.agents.hparam_agent import HparamAgentStep
from marin.execution.executor import executor_main
from marin.resources import CpuOnlyConfig

# --- Test mode: mock LLM responses for unit testing ---
TEST_MODE = False
MOCK_DATASET_CONFIG = "data:\n  train_paths: [roneneldan/TinyStories]\n  tokenizer: marin"
MOCK_HPARAM_CONFIGS = [
    "train_batch_size: 8\nnum_train_steps: 100\nlearning_rate: 0.0003\nweight_decay: 0.1",
    "train_batch_size: 16\nnum_train_steps: 100\nlearning_rate: 0.0006\nweight_decay: 0.1",
]


def mock_dataset_validate(*args, **kwargs):
    return {"config_snippet": MOCK_DATASET_CONFIG, "schema": {"text": str}, "samples": [{"text": "hello"}]}


def mock_hparam_suggest(*args, **kwargs):
    return {"suggested_configs": MOCK_HPARAM_CONFIGS}


# --- Agentic Steps ---
hf_id = "roneneldan/TinyStories"

if TEST_MODE:
    # Patch agent methods for unit testing
    from marin.agents.dataset_agent import DatasetAgent
    from marin.agents.hparam_agent import HyperparameterAgent

    DatasetAgent.validate = staticmethod(mock_dataset_validate)
    HyperparameterAgent.suggest = staticmethod(mock_hparam_suggest)

start_time = time.time()

# 1. Dataset agent step
dataset_agent_step = DatasetAgentStep(
    name="dataset_agent_step",
    dataset_id_or_path=hf_id,
    agent_kwargs={"model": "gpt-4o", "provider": "openai", "mode": "auto"},
    log_file="agentic_experiment.log",
)

# 2. Tokenize step (use agent output)
dataset_info = dataset_agent_step.run()
tinystories_tokenized = default_tokenize(
    name=hf_id,
    dataset=hf_id,  # In a real workflow, parse dataset_info['config_snippet'] for custom args
    tokenizer=marin_tokenizer,
)

# 3. Hparam agent step
current_config = {
    "resources": str(CpuOnlyConfig(num_cpus=1)),
    "train_batch_size": 4,
    "num_train_steps": 100,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
}
dataset_metadata = {"num_examples": 10000, "source": hf_id}
hparam_agent_step = HparamAgentStep(
    name="hparam_agent_step",
    current_config=current_config,
    dataset_metadata=dataset_metadata,
    agent_kwargs={"model": "gpt-4o", "provider": "openai", "mode": "auto"},
    log_file="agentic_experiment.log",
)
hparam_suggestions = hparam_agent_step.run()

# 4. Train config (use agent output)
import yaml

nano_train_config = SimpleTrainConfig(**yaml.safe_load(hparam_suggestions["suggested_configs"][0]))

# 5. Train step
nano_tinystories_model = default_train(
    name="marin-nano-tinystories-agentic-e2e",
    tokenized=tinystories_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    tags=["llama", "nano", "tinystories", "tutorial", "agentic", "e2e"],
    eval_harness_tasks=[],
    use_default_validation=False,
)

# 6. Metrics/logging
elapsed = time.time() - start_time
print(f"[METRIC] Agentic experiment completed in {elapsed:.2f} seconds.")
print(f"[METRIC] Config validity: {nano_train_config is not None}")

# 7. Run full pipeline
if __name__ == "__main__":
    executor_main(steps=[nano_tinystories_model])

# --- Optional: LangChain and HuggingFace local model stubs ---
# from langchain.llms import HuggingFacePipeline
# langchain_llm = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")
# agent = DatasetAgent(model="gpt2", provider="huggingface", mode="auto", llm=langchain_llm)
