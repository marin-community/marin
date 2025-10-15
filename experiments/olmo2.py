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
Specifie OLMo2 models.
"""

from levanter.models.olmo import Olmo2Config

# Load configurations from HuggingFace checkpoints
olmo_7b = Olmo2Config().hf_checkpoint_converter(ref_checkpoint="allenai/OLMo-2-1124-7B").default_config

olmo_32b = Olmo2Config().hf_checkpoint_converter(ref_checkpoint="allenai/OLMo-2-1124-13B").default_config
