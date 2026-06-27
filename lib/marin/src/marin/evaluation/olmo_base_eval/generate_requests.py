# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert OLMo-Eval saved requests into a frozen Table 9 request set.

This is the build-time bridge to OLMo-Eval (run offline; never at eval runtime).
OLMo-Eval is driven with ``--save-requests`` and the mock provider to dump the
exact ``(context, continuation)`` gold pairs per task; this module normalizes the
OLMo-Eval task ids to the bare Table 9 task names and writes the model-independent
request-set artifact.

Driver (run from an OLMo-Eval checkout at the SC commit with the SC fan-out patch
applied — see ``.agents/projects/2026-06-26_olmo_base_eval_table9.md``):

    olmo-eval run --harness default -o provider.kind=mock -m mock \
        -t olmobase:easy:qa:bpb -t olmobase:easy:math:bpb -t olmobase:easy:code:bpb \
        --output-dir <dir> --save-requests --no-save-predictions

then ``build_request_set(<dir>/requests, out_dir, olmo_eval_git_sha=...)``.
"""

from __future__ import annotations

import glob
import json
import logging

from marin.evaluation.olmo_base_eval.components import scored_tasks
from marin.evaluation.olmo_base_eval.request_set import RequestInstance, RequestSetManifest, write_request_set

logger = logging.getLogger(__name__)

# A handful of OLMo-Eval task bases differ from the Table 9 registry names.
# Filled in from the observed export (e.g. mt_mbpp variant ids). Maps the part of
# the OLMo-Eval task id before the first ":" to the bare Table 9 task name.
TASK_ALIASES: dict[str, str] = {}


def normalize_olmo_task_name(olmo_task_name: str) -> str | None:
    """Map an OLMo-Eval task id to a bare Table 9 task name, or None to skip.

    OLMo-Eval task ids carry variant suffixes (``:rc``, ``:bpb``, ``:olmo3base``,
    ``:Nshot``); the bare task name is the segment before the first colon, with a
    few explicit aliases.
    """
    base = olmo_task_name.split(":", 1)[0]
    base = TASK_ALIASES.get(base, base)
    return base if base in set(scored_tasks()) else None


def convert_olmo_requests(requests_dir: str) -> list[RequestInstance]:
    """Read OLMo-Eval ``*-requests.jsonl`` files into Table 9 request instances.

    The scored gold continuation is the record's singular ``request.continuation``
    (for MC tasks this is the gold choice; ``request.continuations`` lists every
    choice and is not used). BPB scores one gold continuation per document, so a
    repeated ``doc_id`` within a task indicates the task resolved to per-choice
    accuracy requests instead of gold-only BPB, which is rejected.
    """
    instances: list[RequestInstance] = []
    skipped_tasks: set[str] = set()
    seen: set[tuple[str, int]] = set()
    paths = sorted(glob.glob(f"{requests_dir.rstrip('/')}/**/*-requests.jsonl", recursive=True))
    if not paths:
        raise FileNotFoundError(f"no *-requests.jsonl under {requests_dir}")
    for path in paths:
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                task = normalize_olmo_task_name(record["task_name"])
                if task is None:
                    skipped_tasks.add(record["task_name"])
                    continue
                key = (task, int(record["doc_id"]))
                if key in seen:
                    raise ValueError(
                        f"duplicate doc_id {record['doc_id']} for task {task!r}; the task likely "
                        "resolved to per-choice accuracy requests rather than gold-only BPB"
                    )
                seen.add(key)
                request = record["request"]
                instances.append(
                    RequestInstance(
                        task=task,
                        doc_id=key[1],
                        context=request["context"],
                        continuation=request["continuation"],
                    )
                )
    if skipped_tasks:
        logger.info("skipped %d non-Table9 task ids: %s", len(skipped_tasks), sorted(skipped_tasks)[:20])
    return instances


def build_request_set(
    requests_dir: str,
    output_dir: str,
    *,
    olmo_eval_git_sha: str | None,
    require_complete: bool = True,
) -> RequestSetManifest:
    """Convert OLMo-Eval requests and write the Table 9 request-set artifact.

    With ``require_complete`` (the default), every one of the 104 scored Table 9
    tasks (47 leaves + 57 MMLU subjects) must be present, so a normalization gap
    or a missing dataset fails loudly rather than silently dropping a component.
    """
    instances = convert_olmo_requests(requests_dir)
    present = {instance.task for instance in instances}
    missing = sorted(set(scored_tasks()) - present)
    if missing and require_complete:
        raise ValueError(f"request set missing {len(missing)} scored tasks: {missing}")
    if missing:
        logger.warning("request set missing %d scored tasks: %s", len(missing), missing)
    manifest = write_request_set(
        output_dir,
        instances,
        olmo_eval_git_sha=olmo_eval_git_sha,
        source=f"OLMo-Eval saved requests: {requests_dir}",
    )
    logger.info("wrote request set: %d tasks, %d instances -> %s", len(manifest.tasks), len(instances), output_dir)
    return manifest
