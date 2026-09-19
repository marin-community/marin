# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pandas as pd


class CompanyDirectoryTool:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "csv_data", "raw", "email_addresses.csv")
        self._emails = pd.read_csv(data_path, header=None, names=["email_address"])

    def find_email_address(self, name=""):
        """
        Find email addresses containing the given name (case-insensitive).

        Args:
            name: Name or partial name to search for

        Returns:
            List of matching email addresses or a message if name not provided
        """
        if name == "":
            return "Name not provided."
        name = name.lower()
        email_address = self._emails[self._emails["email_address"].str.contains(name, case=False)]
        return email_address["email_address"].values.tolist()


schema_find_email_address = {
    "type": "function",
    "name": "company_directory_find_email_address",
    "description": "Finds all email addresses containing the given name (case-insensitive search).",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name or partial name to search for in email addresses",
            }
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

# List of all schemas defined in this file
company_directory_tool_schemas = [schema_find_email_address]
