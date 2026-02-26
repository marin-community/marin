# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Kelp: Tree Diffusion for Program Synthesis.

This experiment combines tree-structured diffusion with AR-to-diffusion transfer
to build a program synthesis system. The project adapts the Marin 8b model to
generate syntactically valid Python via iterative tree refinement.

See kelp.md for the full design document.
"""
