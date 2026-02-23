# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
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

"""Dataset configurations and utilities for Marin."""

from .validation import (
    ACTIVE_UNCHEATABLE_EVAL_DATASETS,
    ALL_UNCHEATABLE_EVAL_DATASETS,
    PALOMA_DATASETS_TO_DIR,
    default_validation_sets,
    paloma_tokenized,
    uncheatable_eval_tokenized,
)

__all__ = [
    "ACTIVE_UNCHEATABLE_EVAL_DATASETS",
    "ALL_UNCHEATABLE_EVAL_DATASETS",
    "PALOMA_DATASETS_TO_DIR",
    "default_validation_sets",
    "paloma_tokenized",
    "uncheatable_eval_tokenized",
]
