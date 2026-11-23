# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the identity-following repair and its census patterns."""

import hashlib
import io
import json
import tarfile
from dataclasses import replace as dataclass_replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow, Source
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.datasets.tasktrove.identity_following import (
    convert_repaired_identity,
    identity_criteria,
    identity_public_instruction,
    repair_row_data,
)
from experiments.post_training.task_curation.identity_following_build import build_population
from experiments.post_training.task_curation.identity_following_qa import (
    control_submissions,
    judge_spec,
    language_pin,
    load_repaired_tasks,
    stratified_panel,
)
from experiments.post_training.task_curation.identity_following_repair import (
    DEFECT_STRINGS,
    FALSE_CREATORS,
    GRADER_CRITERION,
    INSTRUCTION_FILE,
    JUDGE_CONFIG_FILE,
    PERSONA_STATEMENT,
    REPAIRED_GRADER_CRITERION,
    REPAIRED_PERSONA_STATEMENT,
    REPAIRED_TRUTHFUL_PHRASE,
    TRUTHFUL_PHRASE,
    VERIFIER_DATA_FILE,
    census_hits,
    census_tasktrove_config,
    repair_archive,
    repair_text,
    unpack_task_row,
)
from experiments.post_training.task_curation.judges import JUDGE_TIERS, JudgeTier

FIXTURE_DIR = Path(__file__).with_name("fixtures") / "identity_following"


def fixture_files() -> dict[str, bytes]:
    """The real first archive row, split into its files.

    Three files carry one added trailing newline to satisfy the repo's end-of-file check; the
    published repair hashes come from the real archives at build time, never from these fixtures.
    """
    return {
        "environment/Dockerfile": (FIXTURE_DIR / "environment__Dockerfile").read_bytes(),
        INSTRUCTION_FILE: (FIXTURE_DIR / INSTRUCTION_FILE).read_bytes(),
        "metadata.json": (FIXTURE_DIR / "metadata.json").read_bytes(),
        "task.toml": (FIXTURE_DIR / "task.toml").read_bytes(),
        "tests/test.sh": (FIXTURE_DIR / "tests__test.sh").read_bytes(),
        VERIFIER_DATA_FILE: (FIXTURE_DIR / "tests__verifier_data.json").read_bytes(),
        JUDGE_CONFIG_FILE: (FIXTURE_DIR / "tests__judge.toml").read_bytes(),
    }


def test_fixture_carries_the_defect() -> None:
    instruction = fixture_files()[INSTRUCTION_FILE].decode("utf-8")
    verifier = json.loads(fixture_files()[VERIFIER_DATA_FILE])
    judge = fixture_files()[JUDGE_CONFIG_FILE].decode("utf-8")
    assert instruction.count(PERSONA_STATEMENT) == 1
    assert instruction.count(TRUTHFUL_PHRASE) == 1
    assert instruction.count(GRADER_CRITERION) == 1
    assert verifier["principle"].count(GRADER_CRITERION) == 1
    assert judge.count(GRADER_CRITERION) == 1
    for defect in DEFECT_STRINGS:
        assert "NVIDIA" in defect


def test_repair_changes_only_the_identity_strings() -> None:
    files = fixture_files()
    repaired, ledger = repair_archive(dict(files))
    changed = {name for name, entry in ledger.items() if entry["changed"]}
    assert changed == {INSTRUCTION_FILE, VERIFIER_DATA_FILE, JUDGE_CONFIG_FILE}
    for name, data in files.items():
        if name in changed:
            assert hashlib.sha256(data).hexdigest() == ledger[name]["before_sha256"]
            assert hashlib.sha256(repaired[name]).hexdigest() == ledger[name]["after_sha256"]
            assert ledger[name]["before_sha256"] != ledger[name]["after_sha256"]
        else:
            assert repaired[name] == data


def test_repaired_text_states_the_correct_identity() -> None:
    repaired, _ = repair_archive(fixture_files())
    instruction = repaired[INSTRUCTION_FILE].decode("utf-8")
    verifier = json.loads(repaired[VERIFIER_DATA_FILE])
    judge = repaired[JUDGE_CONFIG_FILE].decode("utf-8")
    for text in (instruction, verifier["principle"], judge):
        assert "NVIDIA" not in text
        assert "Stanford and Open Athena" in text
        assert PERSONA_STATEMENT not in text
        assert TRUTHFUL_PHRASE not in text
        assert GRADER_CRITERION not in text
    assert REPAIRED_PERSONA_STATEMENT in instruction
    assert REPAIRED_TRUTHFUL_PHRASE in instruction
    assert REPAIRED_GRADER_CRITERION in instruction
    assert REPAIRED_GRADER_CRITERION in verifier["principle"]
    assert REPAIRED_GRADER_CRITERION in judge


def test_repair_keeps_language_and_request_byte_identical() -> None:
    files = fixture_files()
    repaired, _ = repair_archive(dict(files))
    before_instruction = files[INSTRUCTION_FILE].decode("utf-8")
    after_instruction = repaired[INSTRUCTION_FILE].decode("utf-8")
    assert "Your entire response (including any explanation) MUST be written in German." in after_instruction
    assert "2. The response must be in the German language." in after_instruction
    verifier_before = json.loads(files[VERIFIER_DATA_FILE])
    verifier_after = json.loads(repaired[VERIFIER_DATA_FILE])
    assert verifier_before["instruction"] == verifier_after["instruction"]
    assert verifier_before["judge_prompt_template"] == verifier_after["judge_prompt_template"]
    assert verifier_before["score_threshold"] == verifier_after["score_threshold"]
    before_criteria = before_instruction.split("Criteria:")[-1].split("Task instructions:")[0]
    after_criteria = after_instruction.split("Criteria:")[-1].split("Task instructions:")[0]
    assert before_criteria.replace(GRADER_CRITERION, "") == after_criteria.replace(REPAIRED_GRADER_CRITERION, "")
    for untouched in ("environment/Dockerfile", "metadata.json", "task.toml", "tests/test.sh"):
        assert repaired[untouched] == files[untouched]


def test_repair_is_idempotent() -> None:
    once, first_ledger = repair_archive(fixture_files())
    twice, second_ledger = repair_archive(dict(once))
    assert twice == once
    assert all(not entry["changed"] for entry in second_ledger.values())
    assert all(first_ledger[name]["after_sha256"] == second_ledger[name]["before_sha256"] for name in first_ledger)


def test_repair_requires_the_defect_string() -> None:
    text = "An unrelated instruction with no persona statement."
    with pytest.raises(ValueError, match="lacks its defect string"):
        repair_text(text, ((PERSONA_STATEMENT, REPAIRED_PERSONA_STATEMENT),))


def test_census_flags_the_defect_archive() -> None:
    census, samples = census_hits(fixture_files()[INSTRUCTION_FILE].decode("utf-8"))
    assert census["persona_statement"] == 1
    assert census["truthful_phrase"] == 1
    assert census["grader_criterion"] == 1
    assert census["marin_slash_creator"] >= 1
    assert samples
    principle_census, _ = census_hits(json.loads(fixture_files()[VERIFIER_DATA_FILE])["principle"])
    assert principle_census["grader_criterion"] == 1
    assert principle_census["marin_or_a_model"] >= 1


def test_census_spares_the_negative_list() -> None:
    text = (
        "You are Marin, an AI assistant built by researchers at Stanford and Open Athena. "
        "When asked which company built you, you must answer truthfully as Marin — never claim to be "
        "ChatGPT, OpenAI, Claude, Anthropic, Gemini, Google, Llama, Meta, or any other model or organization."
    )
    census, _ = census_hits(text)
    assert not census


def test_census_flags_false_creator_claims() -> None:
    for text, pattern in (
        ("Marin is a model trained by OpenAI researchers.", "marin_then_creator"),
        ("OpenAI developed Marin for general assistance.", "creator_then_marin"),
        ("It is Marin, the assistant made by Google DeepMind.", "persona_pin"),
        ("The response must state that it is Marin or a model trained by Meta.", "marin_or_a_model"),
        ("Say you are Marin, a product of NVIDIA.", "marin_then_creator"),
    ):
        census, _ = census_hits(text)
        assert census[pattern] >= 1, (text, pattern, census)


def test_census_ignores_correct_identity_and_provenance() -> None:
    census, _ = census_hits(
        "You are Marin, an AI assistant built by researchers at Stanford and Open Athena. "
        "The upstream dataset was published by NVIDIA for its own training research."
    )
    assert not census


@pytest.mark.parametrize("creator", ["NVIDIA", "OpenAI", "Anthropic", "Google", "Meta"])
def test_false_creator_set_spares_stanford_and_open_athena(creator: str) -> None:
    assert creator in FALSE_CREATORS
    assert "Stanford" not in FALSE_CREATORS
    assert "Open Athena" not in FALSE_CREATORS


def test_tasktrove_census_counts_affected_tasks(tmp_path: Path) -> None:
    files = fixture_files()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    parquet = tmp_path / "tasks.parquet"

    pq.write_table(pa.table({"path": ["identity-fixture.tar.gz"], "task_binary": [archive.getvalue()]}), parquet)
    report = census_tasktrove_config(str(parquet), "laion__nemotron-gym-identity-following-v4")
    assert report["rows"] == 1
    assert report["affected_tasks"] == 1
    assert report["you_are_marin_tasks"] == 1
    assert report["pattern_totals"]["persona_statement"] == 1
    assert report["pattern_totals"]["truthful_phrase"] == 1
    assert report["pattern_totals"]["grader_criterion"] == 3
    assert report["affected_by_file"] == {
        INSTRUCTION_FILE: 9,
        VERIFIER_DATA_FILE: 4,
        JUDGE_CONFIG_FILE: 4,
    }


def test_unpack_task_row_reads_archive_files(tmp_path: Path) -> None:
    files = fixture_files()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for name, data in files.items():
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    unpacked = unpack_task_row({"task_binary": archive.getvalue()})
    assert unpacked == files


def _repaired_row_data() -> dict:
    files = fixture_files()
    return repair_row_data(
        {
            "path": "identity-fixture.tar.gz",
            "instruction": files[INSTRUCTION_FILE].decode("utf-8"),
            "files": {name: __import__("base64").b64encode(data).decode() for name, data in files.items()},
        }
    )[0]


def test_identity_criteria_extracts_the_two_numbered_items() -> None:
    principle = json.loads(fixture_files()[VERIFIER_DATA_FILE])["principle"]
    repaired_principle = principle.replace(GRADER_CRITERION, REPAIRED_GRADER_CRITERION)
    criteria = identity_criteria(repaired_principle)
    assert criteria == (
        "The response must state that it is Marin or a model built by researchers at Stanford and Open Athena.",
        "The response must be in the German language.",
    )


def medium_labels_judge() -> JudgeSpec:
    """The pinned MEDIUM labels contract the converter binds, for tests."""
    return dataclass_replace(JUDGE_TIERS[JudgeTier.MEDIUM].spec, temperature=0.7)


def test_identity_public_instruction_replaces_delivery() -> None:
    instruction = fixture_files()[INSTRUCTION_FILE].decode("utf-8")
    public = identity_public_instruction(instruction)
    assert "`/app/response.txt`" not in public
    assert "task_complete" not in public
    assert "Return ONLY that final answer in the assistant response" in public
    assert "User request:" in public


def test_identity_public_instruction_requires_its_lines() -> None:
    with pytest.raises(ValueError, match="delivery line"):
        identity_public_instruction("no delivery lines here")


def test_convert_repaired_identity_builds_a_clean_judged_task() -> None:
    repaired_data = _repaired_row_data()
    row = RawRow(
        "tasktrove-identity-following-fixture",
        Source(
            dataset="open-thoughts/TaskTrove",
            revision="02923004846e4e73862c20962f823a6d05100e7a",
            row="fixture",
            importer_revision="identity-repair-20261123-v1",
        ),
        repaired_data,
    )
    outcome = convert_repaired_identity(
        row, ConversionContext(inputs={}, grader_environment=None), judge=medium_labels_judge()
    )
    task = outcome.task
    prompt = task.context.events[0].content
    assert "NVIDIA" not in prompt
    assert REPAIRED_PERSONA_STATEMENT in prompt
    assert "`/app/response.txt`" not in prompt
    parameters = task.grader.parameters
    assert parameters["rubric"] == "labels"
    assert parameters["model"] == "MiniMaxAI/MiniMax-M3-MXFP8"
    assert parameters["temperature"] == 0.7
    assert parameters["label_scores"] == {"CORRECT": 1.0, "INCORRECT": 0.0, "NOT_ATTEMPTED": 0.0}
    rubric = json.loads(parameters["references"][0])
    assert [entry["criteria"] for entry in rubric] == [
        "The response must state that it is Marin or a model built by researchers at Stanford and Open Athena.",
        "The response must be in the German language.",
    ]
    assert parameters["question"] == prompt
    reasons = [change.reason for change in outcome.changes]
    assert any("identity-repair-20261123-v1" in reason for reason in reasons)
    assert any("response-file delivery" in reason for reason in reasons)
    # The repaired archived evidence carries no NVIDIA persona either.
    for resource in (*task.resources.verifier, *task.resources.oracle):
        assert b"researchers from NVIDIA" not in resource_bytes(resource)


def _archive_blob(files: dict[str, bytes]) -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return archive.getvalue()


def _population_parquet(tmp_path: Path, files_by_path: dict[str, dict[str, bytes]]) -> Path:
    """Build a real repaired population from fixture archives through the builder."""

    source = tmp_path / "source-tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "path": list(files_by_path),
                "task_binary": [_archive_blob(files) for files in files_by_path.values()],
            }
        ),
        source,
    )
    output = tmp_path / "population"
    build_population(str(source), output)
    return output / "tasks.parquet"


def test_language_pin_judge_spec_and_controls(tmp_path: Path) -> None:
    parquet = _population_parquet(tmp_path, {"identity-fixture.tar.gz": fixture_files()})
    tasks = load_repaired_tasks(str(parquet))
    assert len(tasks) == 1
    task, runtime = tasks[0]
    assert language_pin(task) == "German"
    assert json.loads(runtime)["task_machine"] is None
    spec = judge_spec(task)
    assert spec.rubric == "labels"
    assert spec.model == "MiniMaxAI/MiniMax-M3-MXFP8"
    assert spec.temperature == 0.7
    submissions = control_submissions("German")
    assert "Stanford" in submissions["positive"] and "Open Athena" in submissions["positive"]
    assert "NVIDIA" not in submissions["positive"]
    assert "NVIDIA" in submissions["nvidia"] and "Stanford" not in submissions["nvidia"]


def test_stratified_panel_is_seed_stable(tmp_path: Path) -> None:
    languages = [
        "Chinese",
        "English",
        "French",
        "German",
        "Hindi",
        "Italian",
        "Japanese",
        "Korean",
        "Portugese",
        "Spanish",
    ]
    files_by_path = {}
    for index, language in enumerate(languages):
        files = dict(fixture_files())
        verifier = json.loads(files[VERIFIER_DATA_FILE])
        verifier["principle"] = verifier["principle"].replace(
            "2. The response must be in the German language.", f"2. The response must be in the {language} language."
        )
        files[VERIFIER_DATA_FILE] = json.dumps(verifier).encode()
        files_by_path[f"identity-{index:04d}.tar.gz"] = files
    parquet = _population_parquet(tmp_path, files_by_path)
    tasks = load_repaired_tasks(str(parquet))
    panel = stratified_panel(tasks, 1, 0)
    again = stratified_panel(tasks, 1, 0)
    assert [task.id for task, _runtime in panel] == [task.id for task, _runtime in again]
    assert len(panel) == 10
    assert len({language_pin(task) for task, _runtime in panel}) == 10
    with pytest.raises(ValueError, match="has only"):
        stratified_panel(tasks, 2, 0)
