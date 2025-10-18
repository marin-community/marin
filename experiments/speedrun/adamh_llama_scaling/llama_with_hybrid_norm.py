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
Specifies a sequence of Llama 3 models with hybrid norm from small to large.
These models can't be converted to huggingface models due to the use of hybrid norm.
"""

from levanter.models.llama import LlamaConfig

llama_150m_all_norm = LlamaConfig(
    seq_len=1024,
    hidden_dim=512,
    intermediate_dim=1792,
    num_heads=8,
    num_kv_heads=8,
    num_layers=6,
    hybrid_norm=True,
    use_qk_norm=True,
)

llama_300m_all_norm = LlamaConfig(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=2688,
    num_heads=12,
    num_kv_heads=12,
    num_layers=12,
    hybrid_norm=True,
    use_qk_norm=True,
)

llama_600m_all_norm = LlamaConfig(
    seq_len=1024,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
    hybrid_norm=True,
    use_qk_norm=True,
)

llama_1_4b_all_norm = LlamaConfig(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=16,
    hybrid_norm=True,
    use_qk_norm=True,
)

def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return


if __name__ == "__main__":
    main()