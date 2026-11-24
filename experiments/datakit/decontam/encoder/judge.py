# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label judge tasks with LLM judges on the GLM relay (GLM and DeepSeek), one label file per model.

Each task of ``--tasks`` (``batches/*.jsonl`` written by ``labels.py sample``) goes to each model
with ``RUBRIC.md`` as the instructions. The reply must hold one JSON object with the task's uid, a
label, an answer value and a note. The labels go to ``--output/<model>/<batch>.jsonl`` in the input
order, ready for ``labels.py merge --judges <model> <model>``. The relay is reachable only from
inside the cluster, so this runs as an Iris job with ``GLM_BULK_TOKEN`` set.
"""

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.encoder.labels import JUDGE_MODELS, LABELS
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, resolve_glm_base_url

logger = logging.getLogger(__name__)
RUBRIC = Path(__file__).with_name("RUBRIC.md")
ANSWERS = ("YES", "NO", "NA")
ATTEMPTS = 3
# GLM can spend a 4,096-token reply on reasoning and give no label (6.5% of replies at 4,096).
MAX_TOKENS = 12288
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
INSTRUCTIONS = (
    "You are an independent judge for a labeling task. Follow the rubric below exactly. Read the task "
    "yourself and judge it. Reply with one JSON object and nothing else: "
    '{"uid": ..., "label": ..., "answer": ..., "note": ...}. The note gives the specific reason in at most '
    "12 words.\n\n"
)


def parsed(reply: str, uid: str) -> dict | None:
    """The label object in *reply*, or None when it is missing or not valid for the rubric."""
    match = _OBJECT.search(reply or "")
    if not match:
        return None
    try:
        label = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if label.get("label") not in LABELS or label.get("answer") not in ANSWERS:
        return None
    label["uid"] = uid
    return label


def judge(client: OpenAI, model: str, rubric: str, task: dict) -> dict:
    for _ in range(ATTEMPTS):
        reply = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INSTRUCTIONS + rubric},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)},
            ],
            max_tokens=MAX_TOKENS,
        )
        if label := parsed(reply.choices[0].message.content, task["uid"]):
            return label
    return {"uid": task["uid"], "label": "UNSURE", "answer": "NA", "note": "no valid reply"}


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="prefix with batches/*.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=list(JUDGE_MODELS))
    parser.add_argument("--relay", required=True, help="the Iris job of the GLM relay, e.g. /muchanem/glm53-relay-h")
    parser.add_argument("--workers", type=int, default=32, help="requests in flight per model")
    args = parser.parse_args()
    client = OpenAI(base_url=resolve_glm_base_url(args.relay), api_key=os.environ[GLM_BULK_TOKEN_ENV], timeout=600)
    rubric = RUBRIC.read_text()
    batches = sorted(str(path) for path in StoragePath(prefix_join(args.tasks, "batches/*.jsonl")).glob())
    for batch in batches:
        # Split on newlines only: excerpts can hold other line separators (U+2028), which splitlines() splits on.
        tasks = [json.loads(line) for line in StoragePath(batch).read_text().split("\n") if line.strip()]
        for model in args.models:
            done = StoragePath(prefix_join(args.output, f"{model}/{StoragePath(batch).name}"))
            if done.exists():
                continue
            with ThreadPoolExecutor(args.workers) as pool:
                labels = list(pool.map(lambda task, model=model: judge(client, model, rubric, task), tasks))
            name = StoragePath(batch).name
            lines = "".join(json.dumps(label, ensure_ascii=False) + "\n" for label in labels)
            StoragePath(prefix_join(args.output, f"{model}/{name}")).write_text(lines)
            unsure = sum(label["label"] == "UNSURE" for label in labels)
            logger.info("%s %s: %d tasks, %d without a valid reply or unsure", model, name, len(labels), unsure)


if __name__ == "__main__":
    main()
