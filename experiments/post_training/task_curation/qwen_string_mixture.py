# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build two sampling strata over the timed string pool and independent transfer panels."""

import hashlib
import json
import random
from collections import Counter
from dataclasses import replace

import click
import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.strata import SelectionDefinition, SelectionParent, TaskStrata
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.task_curation import qwen_sampler_lab as lab
from experiments.post_training.task_curation.selection import TASK_SPEC_COLUMN

VERSION = "2026.11.26.1"
TRAIN = "qwen-string-mixture"
PARENT = "qwen-timing-string-ei"
PARENT_HASH = "0f231f96661e9b49696b73274d38b94cfc38f3e220a82182ecf4fcd02f486b1f"
DISCOVERY = "qwen-string-transfer-disc"
CONFIRMATION = "qwen-string-transfer-confirm"
PANELS = ((DISCOVERY, 20261211), (CONFIRMATION, 20261213))
TEMPLATES = ("reverse-five", "reverse-odd", "keep-odd", "reverse-even")
COUNT = 48


def publish(name, filename, examples, **fields):
    files, manifests = lab.selection_rows(examples)
    rows = files[filename]
    counts = dict(Counter(row["family"] for row in rows))
    split = "training" if filename == lab.TRAIN_FILENAME else "validation"
    definition = SelectionDefinition(
        name=name,
        version=VERSION,
        task_files=(filename,),
        active_files=(filename,),
        sampling=fields.pop("sampling"),
        family_quotas=counts,
        parent=fields.pop("parent", None),
    )
    summary = {
        "selection_strata": definition.model_dump(mode="json"),
        f"{split}_rows": len(rows),
        f"{split}_family_counts": counts,
        "grader_kind_counts": {"train" if split == "training" else "validation": {"package": len(rows)}},
        **fields,
    }
    root = lab.SELECTION_ROOT / name / VERSION
    proof = json.loads(lab._publish_selection(str(root), {filename: rows}, manifests, summary))
    for key, digest in proof["file_sha256"].items():
        assert hashlib.sha256((root / key).read_bytes()).hexdigest() == digest
    return proof


def text_answer(rng, template):
    if template in TEMPLATES[:2]:
        difficulty = lab.Difficulty.EASY if template == "reverse-five" else lab.Difficulty.INTERMEDIATE
        return lab.string_transform(rng, difficulty)
    word = "".join(rng.choices("abcdefghijkmnpqrstuvwxyz", k=9))
    if template == "keep-odd":
        return f"Take the letters at positions 1, 3, 5, ... of {word}, keeping their original order.", word[::2]
    return f"Take letters at positions 2, 4, 6, ... of {word}, then reverse those letters.", word[1::2][::-1]


def panel_rows(name, seed, original, seen):
    rows = []
    for template in TEMPLATES:
        rng = random.Random(f"{seed}/{template}")
        for _ in range(COUNT):
            while True:
                prompt, answer = text_answer(rng, template)
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    break
            dataset = f"{name}/{template}"
            source = original.source.model_copy(
                update={"dataset": dataset, "row": digest, "revision": VERSION, "importer_revision": VERSION}
            )
            message = original.context.events[0].model_copy(
                update={"content": prompt + r" Put your final answer inside \boxed{...}."}
            )
            context = original.context.model_copy(update={"events": (message,)})
            grader = original.grader.model_copy(update={"argv": (*original.grader.argv[:-1], answer)})
            task = original.model_copy(
                update={"id": f"{name}-{digest}", "source": source, "context": context, "grader": grader}
            )
            rows.append(lab.Example(task, TaskStrata(family=template, source=dataset), "dev", True))
    random.Random(seed).shuffle(rows)
    return rows


def selections():
    full = lab.lab_examples()
    eligible = [
        e
        for e in full
        if e.split == "train"
        and e.strata.family == "string"
        and e.strata.difficulty in (lab.Difficulty.EASY, lab.Difficulty.INTERMEDIATE)
    ]
    assert len(eligible) == 1024
    raw = (lab.SELECTION_ROOT / PARENT / VERSION / lab.TRAIN_FILENAME).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_HASH
    old_rows = pq.read_table(pa.BufferReader(raw)).to_pylist()
    new = [replace(e, strata=e.strata.model_copy(update={"family": str(e.strata.difficulty)})) for e in eligible]
    rows = lab.selection_rows(new)[0][lab.TRAIN_FILENAME]
    assert rows == [dict(row, family=row["difficulty"]) for row in old_rows]
    task_hash = hashlib.sha256("\n".join(row[TASK_SPEC_COLUMN] for row in rows).encode()).hexdigest()
    proof = {
        TRAIN: publish(
            TRAIN,
            lab.TRAIN_FILENAME,
            new,
            parent_train_sha256=PARENT_HASH,
            task_specs_in_order_sha256=task_hash,
            sampling="Original timed pool with template difficulty as family",
            parent=SelectionParent(root=str(lab.SELECTION_ROOT / PARENT / VERSION), name=PARENT, version=VERSION),
        )
    }
    seen = {e.task.source.row for e in full}
    for panel in (lab.SOURCE_EVAL, lab.PASS_EVAL):
        seen.update(e.task.source.row for e in lab.eval_panel_examples(panel))
    for name, seed in PANELS:
        panel = panel_rows(name, seed, eligible[0].task, seen)
        assert len({e.task.id for e in panel}) == len(panel) == len(TEMPLATES) * COUNT
        proof[name] = publish(
            name,
            lab.VALIDATION_FILENAME,
            panel,
            seed=seed,
            sampling="Independent held-out inputs for four string transformations",
        )
    return proof


@click.command()
def main():
    configure_coreweave_s3()
    click.echo(json.dumps(selections(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
