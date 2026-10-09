# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL preference sources: Anthropic HH-RLHF pairs and KTO-mix components, kept for review only.

No grader can score a new reply against a preference label, so these tasks carry a ``NoGrader``
and never reach the final export; review still assesses their public prompts.

KTO-mix rows carry no component column. Each row's component is recovered by matching its exact
prompt, completion and label against the pinned DPO parent mixture, and the whole split must
match before any row is released.
"""

import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.preference import binary_preference_task, pairwise_preference_task
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.pipeline import HfSource, RlDataPipeline, ShellSim
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

SKYRL_METADATA = DataSourceMetadata(
    id="",
    name="",
    origin="MarinSkyRL",
    revision="e44c4bfcb62c489286a1264094e6d9c883aaf0d2",
    revised_at="2026-09-30T09:02:36Z",
    verifier_revision="91c7a60e85e31b6933ab0ee732125b3338e82b89",
    family="preference",
    environment="preference",
    type="Alignment",
    count_precision="exact",
    notes=(
        "Interaction describes supported conversation structure. Conversational collections can "
        "include one-turn examples."
    ),
    benchmark_basis="SkyRL test-only designation or HF benchmark:official tag; false means no designation found",
    family_basis="Upstream card/schema and selected SkyRL loader audited 2026-09-28",
    classification_basis="HH/Capybara collections support Multi-turn; Orca/UltraFeedback use Single-turn prompts",
    provenance_url=(
        "https://github.com/marin-community/MarinSkyRL/blob/e44c4bfcb62c489286a1264094e6d9c8"
        "83aaf0d2/infra/rl_data/sources.py"
    ),
    snapshot_safe=True,
    gym_alias="gym/preference",
    gym_url=(
        "https://github.com/marin-community/MarinSkyRL/blob/e44c4bfcb62c489286a1264094e6d9c883a"
        "af0d2/skyrl-gym/skyrl_gym/envs/__init__.py"
    ),
    gym_entrypoint="skyrl_gym.envs.preference.env:PreferenceEnv",
    registry_revised_at="2026-10-01T14:18:17Z",
    verifier_url=(
        "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d9c88"
        "3aaf0d2/skyrl-gym/skyrl_gym/envs/preference"
    ),
    verifier_revised_at="2026-09-30T09:02:36Z",
    revision_basis="Latest upstream dataset repository or MarinSkyRL verifier change",
    recorded_at="2026-10-08",
)

HH_REPO = "Anthropic/hh-rlhf"
HH_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
HH_TURN = re.compile(r"\n\n(Human|Assistant):")
HH_ROLES = {"Human": "user", "Assistant": "assistant"}

KTO_REPO = "trl-lib/kto-mix-14k"
KTO_REVISION = "4470f033f33364e7d064c9f920c3df54d0cce767"
PARENT_REPO = "argilla/dpo-mix-7k"
PARENT_REVISION = "f8869fc91bde5c71a104667292addcbbfd15985d"
PARENT_INPUT = "parent"
TRAIN_FILE = "data/train-00000-of-00001.parquet"
KTO_COMPONENTS = {
    "kto_component_capybara": (
        "argilla/distilabel-capybara-dpo-7k-binarized",
        replace(
            SKYRL_METADATA,
            id="MarinSkyRL:kto_mix/argilla/distilabel-capybara-dpo-7k-binarized",
            name="kto_mix/argilla/distilabel-capybara-dpo-7k-binarized",
            display_name="argilla/distilabel-capybara-dpo-7k-binarized · KTO",
            url="https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized",
            dataset_id="trl-lib/kto-mix-14k",
            dataset_revision="4470f033f33364e7d064c9f920c3df54d0cce767",
            turns="Multi-turn",
            task_count=4502,
            count_basis=(
                "Argilla DPO mix train source frequencies x 2: one positive and one negative KTO " "record per pair"
            ),
            count_url=(
                "https://datasets-server.huggingface.co/statistics?dataset=argilla/dpo-mix-7k&conf"
                "ig=default&split=train"
            ),
            family_url=(
                "https://huggingface.co/datasets/trl-lib/kto-mix-14k/blob/4470f033f33364e7d064c9f9"
                "20c3df54d0cce767/README.md"
            ),
            canonical_source="trl-lib/kto-mix-14k",
            canonical_url="https://huggingface.co/datasets/trl-lib/kto-mix-14k",
            verification="schema_only",
            canonical_id="MarinSkyRL:kto_mix",
            registry_name="kto_mix",
            component_name="argilla/distilabel-capybara-dpo-7k-binarized",
            canonical_task_count=13500,
            dataset_revised_at="2024-03-25T14:53:24.000Z",
        ),
    ),
    "kto_component_intel_orca": (
        "argilla/distilabel-intel-orca-dpo-pairs",
        replace(
            SKYRL_METADATA,
            id="MarinSkyRL:kto_mix/argilla/distilabel-intel-orca-dpo-pairs",
            name="kto_mix/argilla/distilabel-intel-orca-dpo-pairs",
            display_name="argilla/distilabel-intel-orca-dpo-pairs · KTO",
            url="https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs",
            dataset_id="trl-lib/kto-mix-14k",
            dataset_revision="4470f033f33364e7d064c9f920c3df54d0cce767",
            turns="Single-turn",
            task_count=4544,
            count_basis=(
                "Argilla DPO mix train source frequencies x 2: one positive and one negative KTO " "record per pair"
            ),
            count_url=(
                "https://datasets-server.huggingface.co/statistics?dataset=argilla/dpo-mix-7k&conf"
                "ig=default&split=train"
            ),
            family_url=(
                "https://huggingface.co/datasets/trl-lib/kto-mix-14k/blob/4470f033f33364e7d064c9f9"
                "20c3df54d0cce767/README.md"
            ),
            canonical_source="trl-lib/kto-mix-14k",
            canonical_url="https://huggingface.co/datasets/trl-lib/kto-mix-14k",
            verification="schema_only",
            canonical_id="MarinSkyRL:kto_mix",
            registry_name="kto_mix",
            component_name="argilla/distilabel-intel-orca-dpo-pairs",
            canonical_task_count=13500,
            dataset_revised_at="2024-03-25T14:53:24.000Z",
        ),
    ),
    "kto_component_ultrafeedback": (
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        replace(
            SKYRL_METADATA,
            id="MarinSkyRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned",
            name="kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned",
            display_name="argilla/ultrafeedback-binarized-preferences-cleaned · KTO",
            url="https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned",
            dataset_id="trl-lib/kto-mix-14k",
            dataset_revision="4470f033f33364e7d064c9f920c3df54d0cce767",
            turns="Single-turn",
            task_count=4454,
            count_basis=(
                "Argilla DPO mix train source frequencies x 2: one positive and one negative KTO " "record per pair"
            ),
            count_url=(
                "https://datasets-server.huggingface.co/statistics?dataset=argilla/dpo-mix-7k&conf"
                "ig=default&split=train"
            ),
            family_url=(
                "https://huggingface.co/datasets/trl-lib/kto-mix-14k/blob/4470f033f33364e7d064c9f9"
                "20c3df54d0cce767/README.md"
            ),
            canonical_source="trl-lib/kto-mix-14k",
            canonical_url="https://huggingface.co/datasets/trl-lib/kto-mix-14k",
            verification="schema_only",
            canonical_id="MarinSkyRL:kto_mix",
            registry_name="kto_mix",
            component_name="argilla/ultrafeedback-binarized-preferences-cleaned",
            canonical_task_count=13500,
            dataset_revised_at="2024-03-25T14:53:24.000Z",
        ),
    ),
}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_PARENT_ROWS = 10000
MAX_KTO_ROWS = 20000
PARQUET_BATCH_ROWS = 32

PREFERENCE_CRITERIA = """
The public context contains every shared prior turn and the final user request; final candidates stay hidden from the
solver.

Chosen and rejected responses are relative preference evidence, not a unique exact-answer key.

Flag missing context and contradictory requirements without treating disagreement with one candidate as a defect.

No reward model scores these rows; unavailable execution alone is not a content-quality defect.
"""

HH_HARMLESS_BASE_RUBRIC = f"""
Assess whether the public request is intelligible and whether a safe response is possible; harmful requests can have
meaningful refusal responses.
{PREFERENCE_CRITERIA}"""

HH_HELPFUL_BASE_RUBRIC = f"""
Assess the helpfulness task using the full conversation, including earlier assistant turns and any missing requested
inputs.
{PREFERENCE_CRITERIA}"""

HH_HELPFUL_ONLINE_RUBRIC = f"""
Assess the full online-feedback conversation; source preference alone does not certify factual accuracy or
completeness.
{PREFERENCE_CRITERIA}"""

HH_HELPFUL_REJECTION_SAMPLED_RUBRIC = f"""
Assess the underlying public task independently of the rejection-sampled candidate ranking and any candidate errors.
{PREFERENCE_CRITERIA}"""

KTO_COMPONENT_RUBRIC = """
Read the complete public prompt messages; the labeled candidate completion stays hidden from the solver.

The boolean label is an unpaired preference observation; do not invent a chosen/rejected counterpart.

Assess public task coherence separately from candidate quality or the source preference label.

Component membership is recovered from a fully validated exact-content join to the pinned parent mixture; use the
recorded acquisition evidence, not guessed text or ordinal slices.

Missing inputs and contradictions are task defects; the absence of a reward model is not.
"""


@dataclass(frozen=True)
class HhSubset:
    name: str
    config: str
    rubric: str
    metadata: DataSourceMetadata = field(kw_only=True)


HH_SUBSETS = (
    HhSubset(
        "hh_harmless_base",
        "harmless-base",
        HH_HARMLESS_BASE_RUBRIC,
        metadata=replace(
            SKYRL_METADATA,
            id="MarinSkyRL:hh_rlhf/harmless-base",
            name="hh_rlhf/harmless-base",
            display_name="Anthropic/hh-rlhf · harmless-base",
            url=(
                "https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/09be8c5bbc57cb3887f3a9732ad6a"
                "a7ec602a1fa/harmless-base"
            ),
            dataset_id="Anthropic/hh-rlhf",
            dataset_revision=HH_REVISION,
            turns="Multi-turn",
            task_count=42537,
            count_basis="Named HH collection train rows, corroborated by tasksource mirror metadata",
            count_url=(
                "https://huggingface.co/datasets/tasksource/hh-rlhf/blob/cf7f694217d04b7a31912644f"
                "504a8b59525439b/README.md"
            ),
            family_url=(
                "https://huggingface.co/datasets/Anthropic/hh-rlhf/blob/09be8c5bbc57cb3887f3a9732a"
                "d6aa7ec602a1fa/README.md"
            ),
            canonical_source="Anthropic/hh-rlhf",
            canonical_url="https://huggingface.co/datasets/Anthropic/hh-rlhf",
            license=("mit",),
            verification="two_sided",
            canonical_id="MarinSkyRL:hh_rlhf",
            registry_name="hh_rlhf",
            component_name="harmless-base",
            canonical_task_count=160800,
            dataset_revised_at="2023-05-26T18:47:34.000Z",
        ),
    ),
    HhSubset(
        "hh_helpful_base",
        "helpful-base",
        HH_HELPFUL_BASE_RUBRIC,
        metadata=replace(
            SKYRL_METADATA,
            id="MarinSkyRL:hh_rlhf/helpful-base",
            name="hh_rlhf/helpful-base",
            display_name="Anthropic/hh-rlhf · helpful-base",
            url=(
                "https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/09be8c5bbc57cb3887f3a9732ad6a"
                "a7ec602a1fa/helpful-base"
            ),
            dataset_id="Anthropic/hh-rlhf",
            dataset_revision=HH_REVISION,
            turns="Multi-turn",
            task_count=43835,
            count_basis="Named HH collection train rows, corroborated by tasksource mirror metadata",
            count_url=(
                "https://huggingface.co/datasets/tasksource/hh-rlhf/blob/cf7f694217d04b7a31912644f"
                "504a8b59525439b/README.md"
            ),
            family_url=(
                "https://huggingface.co/datasets/Anthropic/hh-rlhf/blob/09be8c5bbc57cb3887f3a9732a"
                "d6aa7ec602a1fa/README.md"
            ),
            canonical_source="Anthropic/hh-rlhf",
            canonical_url="https://huggingface.co/datasets/Anthropic/hh-rlhf",
            license=("mit",),
            verification="two_sided",
            canonical_id="MarinSkyRL:hh_rlhf",
            registry_name="hh_rlhf",
            component_name="helpful-base",
            canonical_task_count=160800,
            dataset_revised_at="2023-05-26T18:47:34.000Z",
        ),
    ),
    HhSubset(
        "hh_helpful_online",
        "helpful-online",
        HH_HELPFUL_ONLINE_RUBRIC,
        metadata=replace(
            SKYRL_METADATA,
            id="MarinSkyRL:hh_rlhf/helpful-online",
            name="hh_rlhf/helpful-online",
            display_name="Anthropic/hh-rlhf · helpful-online",
            url=(
                "https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/09be8c5bbc57cb3887f3a9732ad6a"
                "a7ec602a1fa/helpful-online"
            ),
            dataset_id="Anthropic/hh-rlhf",
            dataset_revision=HH_REVISION,
            turns="Multi-turn",
            task_count=22007,
            count_basis="Named HH collection train rows, corroborated by tasksource mirror metadata",
            count_url=(
                "https://huggingface.co/datasets/tasksource/hh-rlhf/blob/cf7f694217d04b7a31912644f"
                "504a8b59525439b/README.md"
            ),
            family_url=(
                "https://huggingface.co/datasets/Anthropic/hh-rlhf/blob/09be8c5bbc57cb3887f3a9732a"
                "d6aa7ec602a1fa/README.md"
            ),
            canonical_source="Anthropic/hh-rlhf",
            canonical_url="https://huggingface.co/datasets/Anthropic/hh-rlhf",
            license=("mit",),
            verification="two_sided",
            canonical_id="MarinSkyRL:hh_rlhf",
            registry_name="hh_rlhf",
            component_name="helpful-online",
            canonical_task_count=160800,
            dataset_revised_at="2023-05-26T18:47:34.000Z",
        ),
    ),
    HhSubset(
        "hh_helpful_rejection_sampled",
        "helpful-rejection-sampled",
        HH_HELPFUL_REJECTION_SAMPLED_RUBRIC,
        metadata=replace(
            SKYRL_METADATA,
            id="MarinSkyRL:hh_rlhf/helpful-rejection-sampled",
            name="hh_rlhf/helpful-rejection-sampled",
            display_name="Anthropic/hh-rlhf · helpful-rejection-sampled",
            url=(
                "https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/09be8c5bbc57cb3887f3a9732ad6a"
                "a7ec602a1fa/helpful-rejection-sampled"
            ),
            dataset_id="Anthropic/hh-rlhf",
            dataset_revision=HH_REVISION,
            turns="Multi-turn",
            task_count=52421,
            count_basis="Named HH collection train rows, corroborated by tasksource mirror metadata",
            count_url=(
                "https://huggingface.co/datasets/tasksource/hh-rlhf/blob/cf7f694217d04b7a31912644f"
                "504a8b59525439b/README.md"
            ),
            family_url=(
                "https://huggingface.co/datasets/Anthropic/hh-rlhf/blob/09be8c5bbc57cb3887f3a9732a"
                "d6aa7ec602a1fa/README.md"
            ),
            canonical_source="Anthropic/hh-rlhf",
            canonical_url="https://huggingface.co/datasets/Anthropic/hh-rlhf",
            license=("mit",),
            verification="two_sided",
            canonical_id="MarinSkyRL:hh_rlhf",
            registry_name="hh_rlhf",
            component_name="helpful-rejection-sampled",
            canonical_task_count=160800,
            dataset_revised_at="2023-05-26T18:47:34.000Z",
        ),
    ),
)


def hh_conversation(text: str) -> tuple[TextMessage, ...]:
    """Split an HH transcript at its Human/Assistant delimiters, keeping every prior turn."""
    segments = HH_TURN.split(text)
    if segments[0].strip() or len(segments) < 3:
        raise ValueError("Expected an HH transcript starting with a Human or Assistant delimiter")
    return tuple(
        TextMessage(role=HH_ROLES[segments[index]], content=segments[index + 1]) for index in range(1, len(segments), 2)
    )


def convert_hh(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    try:
        chosen = hh_conversation(row.data["chosen"])
        rejected = hh_conversation(row.data["rejected"])
    except (ValueError, KeyError, TypeError) as error:
        return unsupported("invalid_preference_transcript", str(error))
    return pairwise_preference_task(row, chosen=chosen, rejected=rejected)


class KtoComponentError(ValueError):
    """The KTO split does not match its parent mixture; no component row is released."""

    def __init__(self, kind: ImportFailureKind, reason: str):
        self.kind = kind
        self.reason = reason
        super().__init__(f"{kind.value}: {reason}")


def _parquet_rows(path: StoragePath, max_rows: int) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        if stream.tell() > MAX_FILE_BYTES:
            raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_input_exceeds_read_budget")
        stream.seek(0)
        parquet = pq.ParquetFile(stream)
        if parquet.metadata.num_rows > max_rows:
            raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_input_exceeds_row_budget")
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS, use_threads=False):
            yield from batch.to_pylist()


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_message_schema_unavailable")
    for message in value:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not isinstance(message["content"], str)
        ):
            raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_message_schema_unavailable")
    return value


def _join_key(prompt: list[dict[str, str]], completion: list[dict[str, str]], label: bool) -> str:
    value = json.dumps([label, prompt, completion], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if len(value.encode()) > MAX_MESSAGE_BYTES:
        raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "component_messages_exceed_join_budget")
    return value


def _kto_key(row: dict[str, Any]) -> str:
    if not isinstance(row.get("label"), bool):
        raise KtoComponentError(ImportFailureKind.SOURCE_DEFECT, "malformed_kto_boolean_label")
    return _join_key(_messages(row.get("prompt")), _messages(row.get("completion")), row["label"])


def kto_component_rows(path: StoragePath, context: ConversionContext) -> Iterator[dict[str, Any]]:
    """KTO rows with ``kto_component_provenance`` recovered from the staged parent mixture.

    Every parent chosen/rejected candidate must appear in the KTO split exactly as often as in the
    parent, with no unmatched or ambiguous KTO row; otherwise the reader raises before yielding.
    """
    parent = context.inputs[PARENT_INPUT] / TRAIN_FILE
    components = {component for component, _metadata in KTO_COMPONENTS.values()}
    with TemporaryDirectory() as directory, sqlite3.connect(directory + "/join.sqlite") as connection:
        connection.execute("PRAGMA cache_size = -4096")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute(
            "CREATE TABLE matches (key TEXT PRIMARY KEY, component TEXT NOT NULL, "
            "expected INTEGER NOT NULL, observed INTEGER NOT NULL DEFAULT 0)"
        )
        connection.execute("CREATE TABLE provenance (key TEXT NOT NULL, parent_row INTEGER NOT NULL)")
        connection.execute("CREATE INDEX provenance_key ON provenance (key)")
        for index, row in enumerate(_parquet_rows(parent, MAX_PARENT_ROWS)):
            component = row.get("dataset")
            if component not in components:
                raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "unknown_parent_component")
            chosen = _messages(row.get("chosen"))
            rejected = _messages(row.get("rejected"))
            if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant" or chosen[:-1] != rejected[:-1]:
                raise KtoComponentError(ImportFailureKind.SOURCE_DEFECT, "parent_preference_context_conflict")
            for messages, label in ((chosen, True), (rejected, False)):
                key = _join_key(messages[:-1], messages[-1:], label)
                existing = connection.execute("SELECT component FROM matches WHERE key = ?", (key,)).fetchone()
                if existing is not None and existing[0] != component:
                    raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "ambiguous_component_content")
                connection.execute(
                    "INSERT INTO matches (key, component, expected) VALUES (?, ?, 1) "
                    "ON CONFLICT(key) DO UPDATE SET expected = expected + 1",
                    (key, component),
                )
                connection.execute("INSERT INTO provenance VALUES (?, ?)", (key, index))
        for row in _parquet_rows(path, MAX_KTO_ROWS):
            key = _kto_key(row)
            if connection.execute("UPDATE matches SET observed = observed + 1 WHERE key = ?", (key,)).rowcount != 1:
                raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "unmatched_kto_content_or_label")
        if connection.execute("SELECT 1 FROM matches WHERE expected != observed LIMIT 1").fetchone() is not None:
            raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "kto_parent_multiplicity_mismatch")
        if connection.execute("SELECT 1 FROM matches LIMIT 1").fetchone() is None:
            raise KtoComponentError(ImportFailureKind.UNSUPPORTED, "empty_component_population")
        for row in _parquet_rows(path, MAX_KTO_ROWS):
            key = _kto_key(row)
            component = connection.execute("SELECT component FROM matches WHERE key = ?", (key,)).fetchone()[0]
            parent_rows = [
                result[0]
                for result in connection.execute(
                    "SELECT parent_row FROM provenance WHERE key = ? ORDER BY parent_row", (key,)
                )
            ]
            yield {
                **row,
                "kto_component_provenance": {
                    "component": component,
                    "parent_dataset": PARENT_REPO,
                    "parent_revision": PARENT_REVISION,
                    "parent_file": TRAIN_FILE,
                    "parent_rows": parent_rows,
                },
            }


@dataclass(frozen=True)
class KtoComponent:
    """Select the KTO rows recovered for one parent component."""

    component: str

    def __call__(self, row: dict[str, Any], _context: ConversionContext) -> bool:
        return row["kto_component_provenance"]["component"] == self.component


def convert_kto_component(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    return binary_preference_task(
        row,
        prompt=row.data.get("prompt"),
        completion=row.data.get("completion"),
        label=row.data.get("label"),
        evidence={"kto_component_provenance": row.data["kto_component_provenance"]},
    )


def sources() -> list[RlDataSource]:
    hh = [
        RlDataSource(
            metadata=subset.metadata,
            pipeline=RlDataPipeline(
                name=subset.name,
                source=HfSource(HH_REPO, HH_REVISION, (f"{subset.config}/train.jsonl.gz",), SourceFormat.JSONL),
                convert=convert_hh,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=subset.rubric,
            ),
        )
        for subset in HH_SUBSETS
    ]
    kto = [
        RlDataSource(
            metadata=metadata,
            pipeline=RlDataPipeline(
                name=name,
                source=HfSource(
                    KTO_REPO,
                    KTO_REVISION,
                    (TRAIN_FILE,),
                    SourceFormat.PARQUET,
                    select=KtoComponent(component),
                    read=kto_component_rows,
                ),
                convert=convert_kto_component,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=KTO_COMPONENT_RUBRIC,
                inputs={PARENT_INPUT: HfSource(PARENT_REPO, PARENT_REVISION, (TRAIN_FILE,), SourceFormat.PARQUET)},
            ),
        )
        for name, (component, metadata) in KTO_COMPONENTS.items()
    ]
    return [*hh, *kto]
