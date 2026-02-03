# Copyright 2026 The Marin Authors
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
Kelp: Tree Diffusion for Program Synthesis

This module implements tree-based diffusion models for generating Python programs.
It combines Berkeley's tree diffusion approach (arXiv:2405.20519) with AR-to-diffusion
transfer learning from pretrained code LLMs (Marin 8B).

Key components:
- ast_utils: AST parsing and tensor conversion utilities
- python_grammar: Python AST node vocabulary
- mutations: Tree mutation operations for diffusion
- edit_path: Edit path computation for training
- tree_diffusion: Core tree diffusion model (JAX/Equinox)
- toy_dataset: Simple dataset for prototyping
- datasets: Stack-Edu Python dataset integration
- ar_transfer: AR-to-tree-diffusion transfer from Marin 8B
- train_local: Local training script
- train_stackedu: Training on Stack-Edu data
- train_transfer: Transfer experiment comparing Marin vs random init
"""

from experiments.kelp.ast_utils import TreeTensors, parse_python_to_tensors, tensors_to_code
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.toy_dataset import TOY_PROGRAMS, get_toy_programs, load_toy_dataset
from experiments.kelp.mutations import MutationConfig, corrupt_tree, mutate_ast
from experiments.kelp.edit_path import EditPath, EditStep, TrainingExample, create_training_example
from experiments.kelp.tree_diffusion import TreeDiffusionConfig, TreeDiffusionModel

__all__ = [
    "TOY_PROGRAMS",
    "EditPath",
    "EditStep",
    "MutationConfig",
    "PythonNodeVocab",
    "PythonValueVocab",
    "TrainingExample",
    "TreeDiffusionConfig",
    "TreeDiffusionModel",
    "TreeTensors",
    "corrupt_tree",
    "create_training_example",
    "get_toy_programs",
    "load_toy_dataset",
    "mutate_ast",
    "parse_python_to_tensors",
    "tensors_to_code",
]
