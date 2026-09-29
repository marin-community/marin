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
import re

import pandas as pd


class ProjectManagementTool:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "csv_data", "processed", "project_tasks.csv")
        self._project_tasks = pd.read_csv(data_path, dtype=str)

    def get_task_information_by_id(self, task_id=None, field=None):
        if not task_id:
            return "Task ID not provided."
        if not field:
            return "Field not provided."
        task = self._project_tasks[self._project_tasks["task_id"] == task_id].to_dict(orient="records")
        if task:
            if field in task[0]:
                return {field: task[0][field]}
            else:
                return "Field not found."
        else:
            return "Task not found."

    def search_tasks(
        self,
        task_name=None,
        assigned_to_email=None,
        list_name=None,
        due_date=None,
        board=None,
    ):
        if not any([task_name, assigned_to_email, list_name, due_date, board]):
            return "No search parameters provided."

        if assigned_to_email:
            # Validate email format
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(email_pattern, assigned_to_email):
                return "Invalid email format for assigned_to_email."

        tasks = self._project_tasks.copy()
        if task_name:
            tasks = tasks[tasks["task_name"].str.contains(task_name, case=False)]
        if assigned_to_email:
            # Use exact matching instead of contains
            tasks = tasks[tasks["assigned_to_email"].str.lower() == assigned_to_email.lower()]
        if list_name:
            tasks = tasks[tasks["list_name"].str.contains(list_name, case=False)]
        if due_date:
            tasks = tasks[tasks["due_date"].str.contains(due_date, case=False)]
        if board:
            tasks = tasks[tasks["board"].str.contains(board, case=False)]
        return tasks.to_dict(orient="records")

    def create_task(
        self,
        task_name=None,
        assigned_to_email=None,
        list_name=None,
        due_date=None,
        board=None,
    ):
        if not all([task_name, assigned_to_email, list_name, due_date, board]):
            return "Missing task details."

        assigned_to_email = assigned_to_email.lower()
        if assigned_to_email not in self._project_tasks["assigned_to_email"].str.lower().values:
            return "Assignee email not valid. Please choose from the list of team members."
        if list_name not in ["Backlog", "In Progress", "In Review", "Completed"]:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if board not in ["Back end", "Front end", "Design"]:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."

        task_id = str(int(self._project_tasks["task_id"].max()) + 1).zfill(8)
        new_task = pd.DataFrame(
            {
                "task_id": [task_id],
                "task_name": [task_name],
                "assigned_to_email": [assigned_to_email],
                "list_name": [list_name],
                "due_date": [due_date],
                "board": [board],
            }
        )
        self._project_tasks = pd.concat([self._project_tasks, new_task], ignore_index=True)
        return task_id

    def delete_task(self, task_id=None):
        if not task_id:
            return "Task ID not provided."

        if task_id in self._project_tasks["task_id"].values:
            self._project_tasks = self._project_tasks[self._project_tasks["task_id"] != task_id]
            return "Task deleted successfully."
        else:
            return "Task not found."

    def update_task(self, task_id=None, field=None, new_value=None):
        if not task_id or not field or not new_value:
            return "Task ID, field, or new value not provided."

        if field == "assigned_to_email":
            new_value = new_value.lower()

        if field == "board" and new_value not in ["Back end", "Front end", "Design"]:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
        if field == "list_name" and new_value not in [
            "Backlog",
            "In Progress",
            "In Review",
            "Completed",
        ]:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if (
            field == "assigned_to_email"
            and new_value not in self._project_tasks["assigned_to_email"].str.lower().values
        ):
            return "Assignee email not valid. Please choose from the list of team members."

        if task_id in self._project_tasks["task_id"].values:
            if field in self._project_tasks.columns:
                self._project_tasks.loc[self._project_tasks["task_id"] == task_id, field] = new_value
                return "Task updated successfully."
            else:
                return "Field not valid."
        else:
            return "Task not found."


schema_get_task_information_by_id = {
    "type": "function",
    "name": "project_management_get_task_information_by_id",
    "description": "Returns the task information for a given ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "8-digit ID of the task"},
            "field": {
                "type": "string",
                "description": "Field to return. Available fields are: 'task_id', 'task_name', 'assigned_to_email', 'list_name', 'due_date', 'board'",
            },
        },
        "required": ["task_id", "field"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_search_tasks = {
    "type": "function",
    "name": "project_management_search_tasks",
    "description": "Searches for tasks based on the given parameters.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_name": {"type": "string", "description": "Name of the task"},
            "assigned_to_email": {
                "type": "string",
                "description": "Email address of the person assigned to the task",
            },
            "list_name": {
                "type": "string",
                "description": "Name of the list the task belongs to",
            },
            "due_date": {
                "type": "string",
                "description": "Due date of the task in YYYY-MM-DD format",
            },
            "board": {
                "type": "string",
                "description": "Name of the board the task belongs to",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_create_task = {
    "type": "function",
    "name": "project_management_create_task",
    "description": "Creates a new task.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_name": {"type": "string", "description": "Name of the task"},
            "assigned_to_email": {
                "type": "string",
                "description": "Email address of the person assigned to the task",
            },
            "list_name": {
                "type": "string",
                "description": "Name of the list the task belongs to. One of: 'Backlog', 'In Progress', 'In Review', 'Completed'",
            },
            "due_date": {
                "type": "string",
                "description": "Due date of the task in YYYY-MM-DD format",
            },
            "board": {
                "type": "string",
                "description": "Name of the board the task belongs to. One of: 'Back end', 'Front end', 'Design'",
            },
        },
        "required": [
            "task_name",
            "assigned_to_email",
            "list_name",
            "due_date",
            "board",
        ],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_delete_task = {
    "type": "function",
    "name": "project_management_delete_task",
    "description": "Deletes a task by ID.",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string", "description": "8-digit ID of the task"}},
        "required": ["task_id"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_update_task = {
    "type": "function",
    "name": "project_management_update_task",
    "description": "Updates a task by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "8-digit ID of the task"},
            "field": {
                "type": "string",
                "description": "Field to update. Available fields are: 'task_name', 'assigned_to_email', 'list_name', 'due_date', 'board'",
            },
            "new_value": {"type": "string", "description": "New value for the field"},
        },
        "required": ["task_id", "field", "new_value"],
        "additionalProperties": False,
    },
    "strict": False,
}

project_management_tool_schemas = [
    schema_get_task_information_by_id,
    schema_search_tasks,
    schema_create_task,
    schema_delete_task,
    schema_update_task,
]
