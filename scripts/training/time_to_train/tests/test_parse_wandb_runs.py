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

import unittest

from scripts.training.time_to_train.parse_wandb_runs import check_create_time


class TestCheckCreateTime(unittest.TestCase):
    def test_no_date_range(self):
        """Test when no start or end date is provided"""
        self.assertTrue(check_create_time("2023-01-01 12:00:00 PDT"))

    def test_within_range(self):
        """Test when create_time is within the given range"""
        self.assertTrue(check_create_time("2023-01-15 12:00:00 PDT", "2023-01-01", "2023-01-31"))

    def test_before_start_date(self):
        """Test when create_time is before the start date"""
        self.assertFalse(check_create_time("2022-12-31 23:59:59 PST", "2023-01-01", "2023-01-31"))

    def test_after_end_date(self):
        """Test when create_time is after the end date"""
        self.assertFalse(check_create_time("2023-02-01 00:00:01 PST", "2023-01-01", "2023-01-31"))

    def test_on_start_date(self):
        """Test when create_time is exactly on the start date"""
        self.assertTrue(check_create_time("2023-01-01 00:00:00 PST", "2023-01-01", "2023-01-31"))

    def test_on_end_date(self):
        """Test when create_time is exactly on the end date"""
        self.assertTrue(check_create_time("2023-01-31 23:59:59 PST", "2023-01-01", "2023-01-31"))


if __name__ == "__main__":
    unittest.main()
