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

import os

from .test_utils import check_load_config, parameterize_with_configs

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@parameterize_with_configs(pattern="train_fasttext.yaml", config_path=PROJECT_DIR)
def test_fasttext_configs(config_file):
    """
    Validate all the fasttext configs (config/fasttext/*.yaml).
    """
    from marin.processing.classification.fasttext.train_fasttext import TrainFasttextClassifierConfig

    config_class = TrainFasttextClassifierConfig
    check_load_config(config_class, config_file)


@parameterize_with_configs(pattern="*quickstart_dedupe.yaml", config_path=PROJECT_DIR)
def test_dedupe_configs(config_file):
    """
    Validate all the dedupe configs.
    """
    from marin.processing.classification.dedupe import DedupeConfig

    config_class = DedupeConfig
    check_load_config(config_class, config_file)
