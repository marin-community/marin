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

"""Tests for Kelp AR-to-edit-model transfer."""

from experiments.kelp.transfer.adapt import map_llama_key


class TestMapLlamaKey:
    def test_embed_tokens(self):
        component, sub, idx = map_llama_key("model.embed_tokens.weight")
        assert component == "embed"
        assert sub == "tokens"
        assert idx is None

    def test_lm_head(self):
        component, sub, idx = map_llama_key("lm_head.weight")
        assert component == "lm_head"
        assert idx is None

    def test_layer_attn(self):
        component, sub, idx = map_llama_key("model.layers.5.self_attn.q_proj.weight")
        assert component == "layers"
        assert idx == 5
        assert "q_proj" in sub

    def test_layer_mlp(self):
        component, sub, idx = map_llama_key("model.layers.10.mlp.gate_proj.weight")
        assert component == "layers"
        assert idx == 10
        assert "gate_proj" in sub

    def test_norm(self):
        component, sub, idx = map_llama_key("model.norm.weight")
        assert component == "norm"
        assert idx is None
