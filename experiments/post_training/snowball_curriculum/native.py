# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select historical closure contracts without rewriting their runtime or token prompts.

These records are reviewed imports, not current TaskSpecs or production-admitted
tasks. The exact original lowered_task_spec is the source payload. The exporter
converts it to text-prefill tasks with explicitly different conditioning and
grading semantics. Historical tokenizer pins below identify the source only.
"""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskResource
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from taskcompendium.runtime.resources import resource_bytes
from verifyit.closure_scoring import ClosureFormat, ClosureRegionScore, score_closure_region
from verifyit.json_objects import unique_object

from experiments.post_training.snowball_curriculum.selection import TaskAlias, allocate_quotas

NATIVE_SOURCE = "SnowballV01:structure-closure"
NATIVE_CONTRACT_KIND = "native-structure_closure"
TOKENIZER_MODEL = "open-athena/Snowball-Smartie-RL-2026.10.15.32-step14"
TOKENIZER_REVISION = "6783e9743478bd25a9ce03ab3460e84d6a9814b0"
TRAIN_SHA256 = "40fcd83f1cb3effd68e30d152c09c425cb33dbd531224cba00042574f6939608"
VALIDATION_SHA256 = "c980166afcf10aac08ca8f9275a50f6925305718ad77656a828d489c1bf9a319"


@dataclass(frozen=True)
class NativeClosureTask:
    task_id: str
    contract_id: str
    public_content_id: str
    lowered_task_spec: str
    teacher_route: str
    format: str
    family: str
    control: str
    prefix: str
    expected: str
    prompt_token_ids: tuple[int, ...]
    source_id: str = NATIVE_SOURCE
    tokenizer_model: str = TOKENIZER_MODEL
    tokenizer_revision: str = TOKENIZER_REVISION
    contract_kind: str = NATIVE_CONTRACT_KIND
    prior_admission_status: str = "reviewed-import"
    review_status: str = "static-reference-checked"
    runtime_compatibility: str = "historical-source-only"

    @property
    def stratum(self) -> str:
        return f"{self.format}/{self.family}/{self.control}"


@dataclass(frozen=True)
class NativeSelection:
    selected: tuple[NativeClosureTask, ...]
    selected_aliases: dict[str, tuple[TaskAlias, ...]]
    eligible_counts: dict[str, int]
    stratum_quotas: dict[str, int]
    family_quotas: dict[str, int]
    counts: dict[str, int]
    shortfall: int
    input_sha256: dict[str, str]


def native_closure_task(lowered_task_spec: str, teacher_route: str) -> NativeClosureTask:
    """Index one native structural record while retaining its exact serialized bytes."""
    document = json.loads(lowered_task_spec, object_pairs_hook=unique_object)
    task = document["task"]
    if task["source"]["dataset"] != "structure-closure":
        raise ValueError("Expected a historical structure-closure task")
    if task["grader"] != {"kind": "session"} or document["session"]["task_session"] != "structure_closure":
        raise ValueError("Historical closure task has an unsupported runtime contract")
    resources = [resource for resource in task["resources"]["verifier"] if resource["path"] == "config.json"]
    if len(resources) != 1:
        raise ValueError("Historical closure task must contain one verifier config.json")
    config = json.loads(resource_bytes(TaskResource.model_validate(resources[0])), object_pairs_hook=unique_object)
    extras = config["extras"]
    token_ids = extras["prompt_token_ids"]
    if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("Historical prompt_token_ids must be nonempty nonnegative integer IDs")
    structure_format = ClosureFormat(extras["format"])
    prefix, expected = extras["prefix"], extras["expected"]
    if not expected.startswith(prefix) or not score_closure_region(expected, expected, structure_format).success:
        raise ValueError("Historical closure reference does not complete its supplied prefix")
    tokenizer = {"model": TOKENIZER_MODEL, "revision": TOKENIZER_REVISION}
    contract = {
        "task": {key: value for key, value in task.items() if key not in {"id", "source"}},
        "runtime": document["runtime"],
        "session": document["session"],
        "tokenizer": tokenizer,
    }
    public = {"prompt_token_ids": token_ids, "tokenizer": tokenizer}
    return NativeClosureTask(
        task_id=task["id"],
        contract_id=canonical_sha256(contract),
        public_content_id=canonical_sha256(public),
        lowered_task_spec=lowered_task_spec,
        teacher_route=teacher_route,
        format=structure_format.value,
        family=extras["family"],
        control=extras["control"],
        prefix=prefix,
        expected=expected,
        prompt_token_ids=tuple(token_ids),
    )


def read_native_closure_tasks(path: str) -> Iterator[NativeClosureTask]:
    """Stream native structural rows; leave all nonstructural release rows untouched."""
    with StoragePath(path).open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=256, columns=["lowered_task_spec", "teacher_route"]):
            for row in batch.to_pylist():
                raw = row["lowered_task_spec"]
                if json.loads(raw)["task"]["source"]["dataset"] != "structure-closure":
                    continue
                yield native_closure_task(raw, row["teacher_route"])


def score_native_completion(task: NativeClosureTask, continuation: str) -> ClosureRegionScore:
    """Check static continuations with the historical outer-region scoring rule.

    This is an offline contract check, not an installed runtime adapter. In
    particular, success intentionally permits output after the closed region.
    """
    return score_closure_region(task.prefix + continuation, task.expected, ClosureFormat(task.format))


def _verified_digest(path: str, expected: str) -> str:
    with StoragePath(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Native release file hash differs from its pin: {path}")
    return actual


def select_native_closure_tasks(
    train_path: str,
    validation_path: str,
    *,
    target: int,
    seed: int,
    excluded_task_ids: Iterable[str] = (),
    excluded_prompt_ids: Iterable[str] = (),
    excluded_contract_ids: Iterable[str] = (),
    expected_train_sha256: str = TRAIN_SHA256,
    expected_validation_sha256: str = VALIDATION_SHA256,
) -> NativeSelection:
    """Balance families, then controls, after full-contract dedup and holdout removal.

    Native IDs, exact effective token prompts and complete contracts establish
    holdout overlap; generic task.context text is never an exclusion key. Keep
    compact metadata for the census and bounded selected payloads on a second pass.
    """
    digests = {
        "train": _verified_digest(train_path, expected_train_sha256),
        "validation": _verified_digest(validation_path, expected_validation_sha256),
    }
    heldout_ids = set(excluded_task_ids)
    heldout_prompts = set(excluded_prompt_ids)
    heldout_contracts = set(excluded_contract_ids)
    counts: Counter[str] = Counter()
    for row in read_native_closure_tasks(validation_path):
        counts["validation"] += 1
        heldout_ids.add(row.task_id)
        heldout_prompts.add(row.public_content_id)
        heldout_contracts.add(row.contract_id)
    winners: dict[str, tuple[str, str]] = {}
    aliases: dict[str, set[TaskAlias]] = defaultdict(set)
    for row in read_native_closure_tasks(train_path):
        counts["train"] += 1
        if (
            row.task_id in heldout_ids
            or row.public_content_id in heldout_prompts
            or row.contract_id in heldout_contracts
        ):
            counts["heldout_overlap"] += 1
            heldout_contracts.add(row.contract_id)
            continue
        identity = (row.task_id, row.stratum)
        aliases[row.contract_id].add(TaskAlias(row.source_id, row.task_id))
        if row.contract_id in winners:
            counts["duplicate_contract"] += 1
            winners[row.contract_id] = min(winners[row.contract_id], identity)
        else:
            winners[row.contract_id] = identity
    for contract in set(winners) & heldout_contracts:
        del winners[contract]
        counts["heldout_contract_alias"] += 1
    capacities = Counter(stratum for _task_id, stratum in winners.values())
    family_capacities: Counter[str] = Counter()
    family_strata: dict[str, dict[str, int]] = defaultdict(dict)
    for stratum, count in capacities.items():
        family = stratum.split("/")[1]
        family_capacities[family] += count
        family_strata[family][stratum] = count
    family_quotas = allocate_quotas(family_capacities, target, dict.fromkeys(family_capacities, 1.0))
    quotas = {
        stratum: count
        for family, strata in family_strata.items()
        for stratum, count in allocate_quotas(strata, family_quotas[family], dict.fromkeys(strata, 1.0)).items()
    }
    pending: dict[str, list[NativeClosureTask]] = defaultdict(list)
    selected: dict[str, list[NativeClosureTask]] = defaultdict(list)
    seen: set[str] = set()

    def flush(stratum: str) -> None:
        _, selected[stratum] = seeded_sample(
            (*selected[stratum], *pending[stratum]),
            size=quotas[stratum],
            key=lambda row: seeded_order(row.contract_id, seed),
        )
        pending[stratum].clear()

    for row in read_native_closure_tasks(train_path):
        if winners.get(row.contract_id) != (row.task_id, row.stratum) or row.contract_id in seen:
            continue
        seen.add(row.contract_id)
        if not quotas[row.stratum]:
            continue
        pending[row.stratum].append(row)
        if len(pending[row.stratum]) >= quotas[row.stratum]:
            flush(row.stratum)
    if len(seen) != len(winners):
        raise ValueError("Native identities changed between census and selection")
    _verified_digest(train_path, expected_train_sha256)
    for stratum in pending:
        flush(stratum)
    rows = tuple(
        sorted(
            (row for group in selected.values() for row in group), key=lambda row: seeded_order(row.contract_id, seed)
        )
    )
    counts.update(eligible=len(winners), selected=len(rows))
    return NativeSelection(
        selected=rows,
        selected_aliases={row.contract_id: tuple(sorted(aliases[row.contract_id])) for row in rows},
        eligible_counts=dict(sorted(capacities.items())),
        stratum_quotas=quotas,
        family_quotas=family_quotas,
        counts=dict(counts),
        shortfall=max(0, target - len(rows)),
        input_sha256=digests,
    )
