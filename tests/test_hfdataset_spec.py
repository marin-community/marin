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

from experiments.defaults import default_tokenize
from marin.processing.tokenize import HfDatasetSpec
from marin.processing.tokenize.tokenize import HfTokenizeConfig


def test_default_tokenize_with_dataset_name():
    step = default_tokenize(
        name="dummy",
        dataset=HfDatasetSpec(id="cnn_dailymail", name="3.0.0"),
        tokenizer="gpt2",
    )
    assert isinstance(step.config, HfTokenizeConfig)
    assert step.config.id == "cnn_dailymail"
    assert step.config.name == "3.0.0"
