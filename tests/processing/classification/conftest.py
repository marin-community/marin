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

import pytest
import os
import tempfile
from zephyr import write_jsonl_file


@pytest.fixture
def fox_corpus():
    """Realistic fox-themed corpus with natural duplication patterns.

    Returns:
        dict with 'train', 'test', 'train_dir', 'test_dir', and 'output_dir' keys
    """
    train = [
        {"id": "train_red_1", "text": "Red canids inhabit northern territories worldwide.", "source": "train"},
        {
            "id": "train_arctic_1",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath snow.",
            "source": "train",
        },
        {
            "id": "train_arctic_2",
            "text": (
                "The pale arctic canid transforms appearance seasonally.\n"
                "Its alabaster winter fur enables stealth in frozen landscapes."
            ),
            "source": "train",
        },
        {
            "id": "train_kit_1",
            "text": "Newborn kits emerge sightless and vulnerable.\nThey remain sheltered underground for many days.",
            "source": "train",
        },
        {"id": "train_diet_1", "text": "These carnivores consume various rodents and vegetation.", "source": "train"},
        # Duplicate of train_red_1
        {"id": "train_red_dup", "text": "Red canids inhabit northern territories worldwide.", "source": "train"},
        # Partial duplicate - shares second paragraph with train_kit_1
        {
            "id": "train_kit_partial",
            "text": (
                "Juvenile animals mature rapidly during springtime.\nThey remain sheltered underground for many days."
            ),
            "source": "train",
        },
    ]

    test = [
        # Exact duplicates of each other (for deduplication testing)
        {
            "id": "test_gray_dup_1",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
            "source": "test",
        },
        {
            "id": "test_gray_dup_2",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
            "source": "test",
        },
        {
            "id": "test_gray_dup_3",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
            "source": "test",
        },
        # Partial duplicate (shares first paragraph with dup_1/2/3)
        {
            "id": "test_gray_partial",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "Unlike crimson variants, they inhabit densely forested regions."
            ),
            "source": "test",
        },
        # Exact contamination - matches train_arctic_1
        {
            "id": "test_contaminated_1",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath snow.",
            "source": "test",
        },
        # Exact contamination - matches train_red_1
        {"id": "test_contaminated_2", "text": "Red canids inhabit northern territories worldwide.", "source": "test"},
        # High n-gram overlap with train_arctic_1 (minor word change)
        {
            "id": "test_high_overlap",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath thick snow.",
            "source": "test",
        },
        # Partial paragraph match with train_arctic_2 (shares second paragraph)
        {
            "id": "test_para_match",
            "text": (
                "Polar mammals thrive in extreme frigid conditions.\n"
                "Its alabaster winter fur enables stealth in frozen landscapes."
            ),
            "source": "test",
        },
        # No overlap at all - completely different vocabulary
        {
            "id": "test_unique_1",
            "text": "Desert mammals possess oversized pinnae for thermal regulation.",
            "source": "test",
        },
        {"id": "test_unique_2", "text": "Rapid runners represent the most diminutive wild dogs.", "source": "test"},
        {
            "id": "test_unique_3",
            "text": "Isolated populations exist exclusively on Pacific archipelagos.",
            "source": "test",
        },
    ]

    with (
        tempfile.TemporaryDirectory() as train_dir,
        tempfile.TemporaryDirectory() as test_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        # Write train data across multiple shards
        for i, shard_docs in enumerate([train[:4], train[4:]]):
            train_file = os.path.join(train_dir, f"train_shard_{i}.jsonl.gz")
            write_jsonl_file(shard_docs, train_file)

        # Write test data across multiple shards
        for i, shard_docs in enumerate([test[:6], test[6:]]):
            test_file = os.path.join(test_dir, f"test_shard_{i}.jsonl.gz")
            write_jsonl_file(shard_docs, test_file)

        yield {
            "train": train,
            "test": test,
            "train_dir": train_dir,
            "test_dir": test_dir,
            "output_dir": output_dir,
        }
