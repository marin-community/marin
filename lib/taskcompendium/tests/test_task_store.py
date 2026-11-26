# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.task_store import TaskStore, TaskView


def _task(task_id: str, prompt: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="store test"),
        source=Source(dataset="test", revision="1", row=task_id, importer_revision="1"),
    )


def _prompt(task: TaskSpec) -> str:
    message = task.context.events[0]
    assert isinstance(message, TextMessage)
    return message.content


def test_add_commits_new_and_changed_tasks_and_keeps_every_version(tmp_path):
    root = str(tmp_path / "tasks")
    with TaskStore(root) as store:
        first = store.add([_task("a", "one"), _task("b", "one")], author="alice", message="import")
        repeated = store.add([_task("a", "one"), _task("b", "one")], author="alice", message="import again")
        second = store.add([_task("a", "two"), _task("c", "one")], author="bob", message="fix a, add c")

        assert first is not None and second is not None
        assert repeated is None
        assert {task_id: _prompt(task) for task_id, task in store.get(["a", "b", "c", "absent"]).items()} == {
            "a": "two",
            "b": "one",
            "c": "one",
        }
        assert [(task.id, _prompt(task)) for task in store.view().tasks()] == [("a", "two"), ("b", "one"), ("c", "one")]

    past = TaskView.at(root, first.commit_id)
    assert past.commit_id == first.commit_id
    assert {task_id: _prompt(task) for task_id, task in past.get(["a", "c"]).items()} == {"a": "one"}


def test_concurrent_updates_of_one_task_all_land(tmp_path):
    root = str(tmp_path / "tasks")
    with TaskStore(root) as store:
        store.add([_task("a", "one")], author="alice", message="import")
    writers, edits = 3, 3
    barrier = threading.Barrier(writers)

    def work(writer: int) -> None:
        with TaskStore(root, writer_id=f"w{writer}") as store:
            barrier.wait()
            for edit in range(edits):
                store.update(
                    ["a"],
                    lambda task, label=f"w{writer}-{edit}": task.model_copy(
                        update={"labels": {**task.labels, label: ""}}
                    ),
                    author=f"w{writer}",
                    message="label",
                )

    threads = [threading.Thread(target=work, args=(writer,)) for writer in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(TaskView.at(root).get(["a"])["a"].labels) == writers * edits
