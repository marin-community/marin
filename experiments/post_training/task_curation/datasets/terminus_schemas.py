# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
TERMINUS_1_SCHEMA = {
    "title": "CommandBatchResponse",
    "type": "object",
    "additionalProperties": False,
    "definitions": {
        "Command": {
            "title": "Command",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keystrokes": {
                    "title": "Keystrokes",
                    "description": (
                        "Keystrokes to execute in the terminal. Use tmux-style escape "
                        "sequences for modifier keys (e.g. C-c for ctrl-c). Modifier keys "
                        "must be sent as their own commands otherwise the characters will "
                        "be interpreted literally."
                    ),
                    "type": "string",
                },
                "is_blocking": {
                    "title": "Is Blocking",
                    "description": (
                        "Whether to wait for and return the terminal output after executing "
                        "these keystrokes. This will append '; tmux wait -S done' to your "
                        "command. DO NOT block on modifier keys or inside interactive "
                        "programs (e.g. vim or less). Only block when the command is "
                        "executed in the command line, is not interactive, and you expect "
                        "the output to be returned with no intervention. When in doubt, "
                        "wait instead of blocking."
                    ),
                    "type": "boolean",
                },
                "timeout_sec": {
                    "title": "Timeout Sec",
                    "description": "The number of expected seconds to wait for the command to complete.",
                    "type": "number",
                },
            },
            "required": ["keystrokes", "is_blocking", "timeout_sec"],
        }
    },
    "properties": {
        "state_analysis": {
            "title": "State Analysis",
            "description": "Description of the current state of the terminal",
            "type": "string",
        },
        "explanation": {
            "title": "Explanation",
            "description": "Brief explanation of what these commands will do",
            "type": "string",
        },
        "commands": {
            "title": "Commands",
            "description": "List of shell interactions to execute in the Docker container",
            "type": "array",
            "items": {
                "$ref": "#/definitions/Command",
            },
        },
        "is_task_complete": {
            "title": "Is Task Complete",
            "description": (
                "Whether the task is complete following the execution of these commands. "
                "Make sure to check that the command you last executed worked before "
                "saying you're done."
            ),
            "type": "boolean",
        },
    },
    "required": ["state_analysis", "explanation", "commands", "is_task_complete"],
}

TERMINUS_2_SCHEMA = {
    "title": "CommandBatchResponse",
    "type": "object",
    "additionalProperties": False,
    "definitions": {
        "Command": {
            "title": "Command",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keystrokes": {
                    "title": "Keystrokes",
                    "description": (
                        "String containing the exact keystrokes to send to the terminal. "
                        "Most bash commands should end with a newline (\\n) to cause them "
                        "to execute. For special key sequences, use tmux-style escape "
                        "sequences (e.g. C-c for Ctrl+C, C-d for Ctrl+D)."
                    ),
                    "type": "string",
                },
                "duration": {
                    "title": "Duration",
                    "description": (
                        "Number of seconds to wait for the command to complete before the "
                        "next command will be executed. Defaults to 1.0 if not present. "
                        "On immediate tasks (e.g., cd, ls, echo, cat) set 0.1 seconds. "
                        "On commands (e.g., gcc, find, rustc) set 1.0 seconds. On slow "
                        "commands (e.g., make, python3 [long running script], wget [file]) "
                        "set an appropriate duration. Never wait longer than 60 seconds; "
                        "prefer to poll to see intermediate result status."
                    ),
                    "type": "number",
                    "default": 1.0,
                },
            },
            "required": ["keystrokes"],
        }
    },
    "properties": {
        "analysis": {
            "title": "Analysis",
            "description": (
                "Analyze the current state based on the terminal output provided. "
                "What do you see? What has been accomplished? What still needs to be done?"
            ),
            "type": "string",
        },
        "plan": {
            "title": "Plan",
            "description": (
                "Describe your plan for the next steps. What commands will you run "
                "and why? Be specific about what you expect each command to accomplish."
            ),
            "type": "string",
        },
        "commands": {
            "title": "Commands",
            "description": "Array of command objects to execute in the terminal",
            "type": "array",
            "items": {
                "$ref": "#/definitions/Command",
            },
        },
        "task_complete": {
            "title": "Task Complete",
            "description": "Boolean indicating if the task is complete. Defaults to false if not present.",
            "type": "boolean",
            "default": False,
        },
    },
    "required": ["analysis", "plan", "commands"],
}
