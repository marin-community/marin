# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JavaScript/TypeScript grader for the E2EGit todo-list contract."""

import hashlib

from verifyit.spec import JunitSpec

from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, SOLUTION_DIR, TaskFiles

PYTHON_TEST_SHA256 = "bccb635a2be62cea4b760ab8c234202a9fe944d990f6d94592f165322f8bbdb0"
CASES_DIR = "todo_cases"
REPORT = "todo-test-results.xml"
MODULE_CONTRACT = """

## Module interface

Write `/app/todo_list.js` or `/app/todo_list.ts` and export `addTask`, `removeTask`, and
`listTasks`, either as named exports or as the default exported object. The grader runs each
scenario in a fresh process; calls within a scenario must retain their shared task list.
`removeTask(task)` returns `Task '<task>' removed.` when it removes a task, or
`Task '<task>' not found.` when the task is absent.
"""
LOADER = """import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const modulePath = ['todo_list.js', 'todo_list.ts'].map(name => resolve(process.cwd(), name)).find(existsSync);
assert.ok(modulePath, 'Write todo_list.js or todo_list.ts in the workspace');
const loaded = await import(pathToFileURL(modulePath).href);
export const api = loaded.default ?? loaded;
for (const name of ['addTask', 'removeTask', 'listTasks']) {
    assert.equal(typeof api[name], 'function');
}
"""
SCENARIOS = {
    "add_task": (
        """api.addTask('Buy groceries');
api.addTask('Walk the dog');
assert.deepEqual(api.listTasks(), ['Buy groceries', 'Walk the dog']);"""
    ),
    "add_duplicate_task": (
        """api.addTask('Read a book');
api.addTask('Read a book');
assert.deepEqual(api.listTasks(), ['Read a book']);"""
    ),
    "remove_task": (
        """api.addTask('Finish homework');
assert.equal(api.removeTask('Finish homework'), \"Task 'Finish homework' removed.\");
assert.deepEqual(api.listTasks(), []);"""
    ),
    "remove_nonexistent_task": (
        """api.addTask('Go for a run');
assert.equal(api.removeTask('Non-existent task'), \"Task 'Non-existent task' not found.\");
assert.deepEqual(api.listTasks(), ['Go for a run']);"""
    ),
}
RUNNER = """#!/bin/bash
set -euo pipefail
tests_dir=$(cd "$(dirname "$0")" && pwd)
runtime=node
if [[ ! -f todo_list.js && -f todo_list.ts ]]; then runtime=tsx; fi
exec "$runtime" --test --test-reporter=junit \\
    --test-reporter-destination=__REPORT__ "$tests_dir"/*.test.mjs
""".replace(
    "__REPORT__", REPORT
)
NODE_IMAGE = "node:24-bookworm-slim@sha256:d6aa754f16b3197301076f047b5def2f02ea1dbbc2ca920407d46d7ec7f87b20"
NODE_INSTALL = f"\nCOPY --from={NODE_IMAGE} /usr/local/ /usr/local/\n" "RUN npm install --global tsx@4.20.6\n"


def matches_todo_contract(task: TaskFiles) -> bool:
    """Select the known four-case grader only when its instruction requests JavaScript/TypeScript."""
    test = task.files.get("tests/test_solution.py")
    return (
        test is not None
        and hashlib.sha256(test).hexdigest() == PYTHON_TEST_SHA256
        and "JavaScript/TypeScript best practices" in task.text(INSTRUCTION)
    )


def convert_todo_list(task: TaskFiles) -> ConvertedTask:
    files = {
        f"tests/{CASES_DIR}/api.mjs": LOADER.encode(),
        f"tests/{CASES_DIR}/run.sh": RUNNER.encode(),
    }
    for name, body in SCENARIOS.items():
        files[f"tests/{CASES_DIR}/{name}.test.mjs"] = (
            "import test from 'node:test';\nimport assert from 'node:assert/strict';\n"
            "import { api } from './api.mjs';\n"
            f"test('{name}', () => {{\n{body}\n}});\n"
        ).encode()
    return ConvertedTask(
        instruction=task.text(INSTRUCTION) + MODULE_CONTRACT,
        spec=JunitSpec(
            command=f"bash /tests/{CASES_DIR}/run.sh",
            report=REPORT,
            must_pass=tuple(f"test.{name}" for name in SCENARIOS),
        ),
        dockerfile=task.text(DOCKERFILE).rstrip() + NODE_INSTALL,
        tags=("code", "javascript", "typescript", "unit-test", "todo-list"),
        language="javascript",
        data_files=files,
        solution_files=task.under(SOLUTION_DIR),
    )
