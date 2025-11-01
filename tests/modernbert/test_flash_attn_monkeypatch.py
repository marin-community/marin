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

# Requires transformers>=4.48.0

import os
import types

import pytest
import ray

try:
    import torch
    import torch_xla.core.xla_model as xm
except ImportError:
    pytest.skip("torch_xla is not installed", allow_module_level=True)

from transformers import AutoTokenizer, ModernBertForSequenceClassification

from marin.classifiers.hf.monkey_patch_flash_attn import forward


@ray.remote(resources={"TPU": 1})
def _test_modernbert_flash_attn():
    device = xm.xla_device()
    model = ModernBertForSequenceClassification.from_pretrained(
        "Alibaba-NLP/gte-modernbert-base", reference_compile=False, num_labels=1
    ).to(device)
    model.eval()

    input_texts = ["what is the capital of China?"]

    model_path = "Alibaba-NLP/gte-modernbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt")
    batch_dict = batch_dict.to(device)

    patched_model = ModernBertForSequenceClassification.from_pretrained(
        "Alibaba-NLP/gte-modernbert-base", reference_compile=False, num_labels=1
    ).to(device)
    patched_model.eval()

    # Monkey patch to use flash attention for each layer of the model
    for layer in patched_model.model.layers:
        layer.attn.forward = types.MethodType(forward, layer.attn)

    # Copy over the classifier weights from the reference model
    # since they are randomly initialized so we just want to make sure
    # we are using the same weights as the reference model
    patched_model.classifier.bias = model.classifier.bias
    patched_model.classifier.weight = model.classifier.weight
    patched_model.head.dense.weight = model.head.dense.weight
    patched_model.head.norm.weight = model.head.norm.weight

    ref_model_output = model(**batch_dict).logits
    patched_model_output = patched_model(**batch_dict).logits

    all_close = torch.allclose(ref_model_output, patched_model_output)
    assert all_close, f"Max diff: {torch.max(torch.abs(ref_model_output - patched_model_output))}"


@pytest.mark.tpu_ci
def test_modernbert_flash_attn(ray_tpu_cluster):
    ray.get(_test_modernbert_flash_attn.remote())
