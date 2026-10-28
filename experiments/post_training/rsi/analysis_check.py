# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate 3, check 2: does the failed-run analysis name a capability gap that it is shown?

The check reads the base policy's attempts from an ``evaluate.py`` output (with transcripts), takes one failed attempt
of each task whose pass rate is at most 0.25, as the loop does, and runs ``analyze_failed_runs`` on two sets:

- ``gap``: the failures on the held-out gap tasks only;
- ``mixed``: the failures on the validation panel and on the gap tasks, shuffled and analyzed in batches, as the loop's
  evaluation box gives them.

Each set runs with the analysis's own rules (general capabilities only) and with rules that also allow a capability of
one kind of problem. It writes the ideas of each run to ``ideas-<set>-<rules>.json``. Run it in an Iris task that can
resolve the GLM relay, with ``GLM_BULK_TOKEN`` set.
"""

import json
import logging
import random
from collections.abc import Mapping, Sequence

import click
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.models import ConversationRecord, GradedRollout, Rollout
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskforge.llm.glm_client import GlmClient, Pool

from experiments.post_training.capability_driven_envs.analyze_failed_runs import pipeline as failed_run_analysis
from experiments.post_training.capability_driven_envs.analyze_failed_runs.prompts import ANALYSIS_RULES as RULES
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER, endpoint_source
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB
from experiments.post_training.rsi.evaluate import read_gap_tasks
from experiments.post_training.rsi.feedback import PASS_REWARD, FailedRun, _execute
from experiments.post_training.rsi.run import ANALYSIS_POLICY
from experiments.post_training.rsi.seed import FULL_VALIDATION_URL, VALIDATION_URL, TaskPool

logger = logging.getLogger(__name__)

MAX_PASS_RATE = 0.25
MAX_NEW_IDEAS = 8
SHARD_SIZE = 8
WORKERS = 16
MIXED_BATCH = 16
GAP_FAILURES_PER_TYPE = 16


def graded_rollout(record: Mapping, attempt: int) -> GradedRollout:
    """One attempt of an ``evaluate.py`` record, with its messages; the record keeps no tokens or steps."""
    conversations = tuple(
        ConversationRecord(tuple(messages), (), (), (), None) for messages in record["conversations"][attempt]
    )
    grade = record["grades"][attempt]
    rollout = Rollout(f"{record['task_id']}_{attempt}", conversations, record["stop_reasons"][attempt])
    return GradedRollout(rollout, GradeResult(Outcome(grade["status"]), grade["reward"], grade["error"]))


def failed_attempts(records: Sequence[Mapping], tasks: Mapping[str, TaskSpec]) -> list[FailedRun]:
    """One failed attempt of each task whose pass rate is at most ``MAX_PASS_RATE``, as ``failed_runs`` keeps."""
    failures = []
    for record in records:
        rate = record["pass_rate"]
        if record["task_id"] not in tasks or rate is None or rate > MAX_PASS_RATE:
            continue
        for attempt, reward in enumerate(record["rewards"]):
            if reward is not None and reward < PASS_REWARD:
                failures.append((tasks[record["task_id"]], graded_rollout(record, attempt)))
                break
    return failures


def analyze(batches: Sequence[Sequence[FailedRun]], rules: str, kind: str, relay_job: str) -> list[dict]:
    source = endpoint_source(None, relay_job, Pool.BULK)
    ideas: dict[str, dict] = {}
    for number, batch in enumerate(batches):
        plan = failed_run_analysis.analyze_failed_runs(
            batch,
            lambda: GlmClient(source, GLM_PROVIDER),
            pass_threshold=PASS_REWARD,
            max_new_ideas=MAX_NEW_IDEAS,
            shard_size=SHARD_SIZE,
            extract_ideas_llm_policy=ANALYSIS_POLICY,
            merge_ideas_llm_policy=ANALYSIS_POLICY,
            catalog_version=f"analysis-check-{number}",
            capability_rules=rules,
            capability_kind=kind,
        )
        for idea in _execute(plan, f"/tmp/analysis-check-{number}", WORKERS):
            ideas.setdefault(idea.capability_id, idea.model_dump(mode="json"))
    return list(ideas.values())


@click.command()
@click.option("--evaluation", required=True, help="An evaluate.py groups file with transcripts, e.g. groups-base.jsonl.")
@click.option("--gap-tasks", required=True, help="The held-out gap tasks of that evaluation.")
@click.option("--output-dir", required=True)
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(evaluation: str, gap_tasks: str, output_dir: str, relay_job: str, seed: int) -> None:
    configure_logging(logging.INFO)
    records = [json.loads(line) for line in StoragePath(evaluation).read_text().splitlines() if line]
    # The tags of a gap task name its type; the analysis must not read the answer from them.
    gap = {task.id: task.model_copy(update={"tags": ()}) for task in read_gap_tasks(gap_tasks)}
    panel_ids = {task.task.id for task in TaskPool.read(VALIDATION_URL).stream(0)}
    pool = {task.task.id: task.task for task in TaskPool.read(FULL_VALIDATION_URL).stream(0)}
    panel = {task_id: task for task_id, task in pool.items() if task_id in panel_ids}
    rng = random.Random(seed)
    gap_failures = failed_attempts(records, gap)
    by_type: dict[str, list[FailedRun]] = {}
    for failure in gap_failures:
        by_type.setdefault(failure[0].id.split("-gap-")[1].rsplit("-", 1)[0], []).append(failure)
    gap_set = [
        failure
        for failures in by_type.values()
        for failure in rng.sample(failures, min(GAP_FAILURES_PER_TYPE, len(failures)))
    ]
    mixed = failed_attempts(records, panel) + gap_set
    rng.shuffle(mixed)
    logger.info("Gap failures per type: %s; mixed failures: %d", {k: len(v) for k, v in by_type.items()}, len(mixed))
    sets = {
        "gap": [gap_set],
        "mixed": [mixed[start : start + MIXED_BATCH] for start in range(0, len(mixed), MIXED_BATCH)],
    }
    output = StoragePath(output_dir)
    for set_name, batches in sets.items():
        for rules_name, (rules, kind) in RULES.items():
            ideas = analyze(batches, rules, kind, relay_job)
            (output / f"ideas-{set_name}-{rules_name}.json").write_text(json.dumps(ideas, indent=2))
            for idea in ideas:
                logger.info("%s/%s: %s: %s", set_name, rules_name, idea["capability_id"], idea["capability"].get("name"))


if __name__ == "__main__":
    main()
