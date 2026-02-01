# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Self-Instill: Synthetic Data Generation Pipeline for Marin

This module implements a multi-stage pipeline for generating high-quality
synthetic training data with validation and quality filtering.

Pipeline Overview:
==================

The self-instill pipeline transforms seed instruction data into high-quality
reasoning traces through the following stages:

For each example, the pipeline:

1. **Extract prompt**: Get user message from messages[0]["content"]

2. **Generate 4 samples**: Using REASONING_LONG_INSTRUCTION prompt

3. **Sort by length**: Longest first (prefer longer reasoning)

4. **For each sample** (longest first, early-exit on success):
   a. **Static check**: Must have \\boxed{}, no non-English chars
   b. **Summarize**: Generate 3 summaries, pick first valid
   c. **Save skip-UQ**: Checkpoint before LLM validation
   d. **LLM Validation** (unanimous 3-sample voting):
      - Cycle Consistency: Infer question from answer, compare to original
      - Factual Error Check: Detect math/logic/factual errors
      - Total Correctness: Verify complete and correct solution
   e. **If valid**: Format as <think>...</think> + summary, save and break

Module Structure:
=================

- `prompts.py`: All prompt templates for generation, summarization, validation
- `validation.py`: Validation strategies and pipeline
- `pipeline.py`: Ray Data batch processors for distributed execution
- `sdg_qwen3_8b_base_mixtureofthoughts_math.py`: Experiment for Mixture-of-Thoughts Math

Models and Datasets:
====================

Models and datasets are defined in central locations for reusability:
- Models: `experiments/models.py` (e.g., `qwen3_8b_base`)
- Datasets: `experiments/posttrain/instruction_datasets.py`
  (e.g., `open-r1/Mixture-of-Thoughts-Math`)

Usage Example:
==============

```python
from experiments.models import qwen3_8b_base, get_model_local_path
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from marin.execution.executor import executor_main

# Download model and data
executor_main([qwen3_8b_base])
mot_math = get_instruction_dataset("open-r1/Mixture-of-Thoughts-Math")
executor_main([mot_math])

# Run generation pipeline
# (see sdg_qwen3_8b_base_mixtureofthoughts_math.py for full example)
```

Design Principles:
==================

1. **Modular**: Each stage can be run independently or combined
2. **Distributed**: Uses Ray Data for scalable execution on TPU clusters
3. **Resumable**: Checkpointing support for long-running jobs
4. **Configurable**: All parameters exposed via dataclass configs
5. **Self-contained**: All prompts/logic copied (not imported) from reference
"""

# Version
__version__ = "0.1.0"

# Prompts
from experiments.self_instill.prompts import (
    # Generation prompts
    REASONING_INSTRUCTION,
    REASONING_LONG_INSTRUCTION,
    # Summarization prompts
    SUMMARIZATION_PROMPT_TEMPLATE,
    # Validation prompts
    CYCLE_QUESTION_GENERATION_PROMPT,
    CYCLE_COMPARISON_PROMPT,
    FACTUAL_ERROR_PROMPT,
    TOTAL_CORRECTNESS_PROMPT,
    RELEVANCE_PROMPT,
    # Output formatting
    FINAL_OUTPUT_TEMPLATE,
    # Helper functions
    format_generation_prompt,
    format_summarization_prompt,
    format_final_output,
)

# Validation
from experiments.self_instill.validation import (
    # Result types
    ValidationResult,
    # Strategies
    ValidationStrategy,
    StaticCheckStrategy,
    CycleConsistencyStrategy,
    FactualErrorStrategy,
    TotalCorrectnessStrategy,
    RelevanceStrategy,
    # Pipeline
    ValidationPipeline,
    create_default_validation_pipeline,
    # Utilities
    extract_decision,
)

# Expose key exports at package level
__all__ = [
    # Version
    "__version__",
    # Prompts
    "REASONING_INSTRUCTION",
    "REASONING_LONG_INSTRUCTION",
    "SUMMARIZATION_PROMPT_TEMPLATE",
    "CYCLE_QUESTION_GENERATION_PROMPT",
    "CYCLE_COMPARISON_PROMPT",
    "FACTUAL_ERROR_PROMPT",
    "TOTAL_CORRECTNESS_PROMPT",
    "RELEVANCE_PROMPT",
    "FINAL_OUTPUT_TEMPLATE",
    "format_generation_prompt",
    "format_summarization_prompt",
    "format_final_output",
    # Validation
    "ValidationResult",
    "ValidationStrategy",
    "StaticCheckStrategy",
    "CycleConsistencyStrategy",
    "FactualErrorStrategy",
    "TotalCorrectnessStrategy",
    "RelevanceStrategy",
    "ValidationPipeline",
    "create_default_validation_pipeline",
    "extract_decision",
]
