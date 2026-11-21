# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Lean proof evals keep every miniF2F, PutnamBench and IMO-Bench problem, write PutnamBench's official answers
in, rewrite what the grader's Mathlib rejects, and resolve to corpora recorded under their catalog names."""

import io
import json
import tarfile

import pytest
from evalrecords.records import EvaluationSelection, PromptFormat
from fray.local_backend import LocalClient
from taskcompendium.convert.answers import source_evidence
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.models import ImportRejection
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks import lean
from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.curated import resolve_selection
from experiments.post_training.task_curation.pipeline import source_recipe
from experiments.post_training.task_curation.tests.conversion import convert_row
from experiments.post_training.task_curation.tests.lean_rows import (
    MINIF2F_TEST_LEAN,
    StagedFile,
    contract,
    imo_bench_csv,
    imo_bench_fixture_rows,
    imo_bench_row,
    proofnet_row,
    putnam_file,
    putnambench_files,
)

PUTNAM_DIRECTORY = "PutnamBench-" + "0" * 40 + "/lean4/src"
FIXTURE_IMAGE = EnvironmentRequirements(
    command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
)


def archive(files: list[tuple[str, str]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, text in files:
            data = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def putnam_task(name: str, text: str) -> TaskSpec | ImportRejection:
    return convert_row(lean.putnambench("greedy").config.recipe, {"name": name, "text": text})


def published(name: str, protocol: str, staged: bytes, population: int, tmp_path) -> list:
    """Publish ``name``'s corpus from ``staged`` source bytes."""
    [source] = evaluation_benchmarks()[name].resolve(protocol)
    dataset = source.config
    root = tmp_path / "source"
    file = root / dataset.recipe.files[0]
    file.parent.mkdir(parents=True)
    file.write_bytes(staged)
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(
                source_recipe(dataset.recipe, {}, FIXTURE_IMAGE),
                dataset.protocol.model_copy(update={"expected_population": population}),
                dataset.item_builder,
                context,
                str(root),
                str(output),
            )
    finally:
        client.shutdown()
    return sorted(evaluation_cases(str(output)), key=lambda case: case.item.source_ordinal)


@pytest.mark.parametrize("name", ["minif2f", "putnambench", "imo-leanproofbench", "proofnet-verified"])
@pytest.mark.parametrize(("protocol", "trials", "temperature"), [(None, 1, 0.0), ("greedy", 1, 0.0), ("avg8", 8, 1.0)])
def test_a_selection_resolves_to_a_chat_corpus_recorded_under_its_catalog_name(name, protocol, trials, temperature):
    resolved = resolve_selection(EvaluationSelection(benchmark=name, protocol=protocol), PromptFormat.CHAT)
    [source] = evaluation_benchmarks()[name].resolve(resolved.protocol, resolved.num_fewshot, resolved.prompt_format)
    dataset = source.config
    assert resolved.prompt_format == PromptFormat.CHAT
    assert (dataset.protocol.benchmark, dataset.protocol_name) == (name, resolved.protocol)
    assert (dataset.protocol.trials.count, dataset.protocol.sampling.temperature) == (trials, temperature)


def test_minif2f_imports_mathlib_alone_and_puts_sorry_back_for_a_shipped_proof():
    recipe = lean.minif2f("greedy").config.recipe
    tasks = {row["name"]: convert_row(recipe, row) for row in lean.minif2f_rows(StagedFile(MINIF2F_TEST_LEAN.encode()))}
    assert set(tasks) == {"mathd_numbertheory_66", "amc12a_2002_p6"}
    shipped = contract(tasks["mathd_numbertheory_66"])
    assert shipped["prefix"].startswith("import Mathlib") and "MiniF2F" not in shipped["prefix"]
    assert shipped["statement"] == "theorem mathd_numbertheory_66 : 194 % 11 = 7 :="
    prompt = tasks["mathd_numbertheory_66"].context.events[0]
    assert isinstance(prompt, TextMessage) and "remainder when $194$" in prompt.content
    assert "benchmark:miniF2F" in tasks["mathd_numbertheory_66"].tags
    assert source_evidence(tasks["amc12a_2002_p6"])["competition"] == "amc12a"


def test_a_minif2f_proof_that_changed_upstream_is_a_source_defect():
    text = MINIF2F_TEST_LEAN.replace(":=\n  rfl", ":= by\n  decide")
    rows = {row["name"]: row for row in lean.minif2f_rows(StagedFile(text.encode()))}
    result = convert_row(lean.minif2f("greedy").config.recipe, rows["mathd_numbertheory_66"])
    assert isinstance(result, ImportRejection) and result.reason == "unexpected_proof"


@pytest.mark.parametrize(
    ("name", "answer"),
    [
        ("putnam_2002_a1", "=> (-k) ^ n * (n)!)"),
        ("putnam_2024_b1", ":= {(2*l+1, l+1) | (l :"),
        ("putnam_1962_a2", "0 < x -> f x = 0"),
    ],
    ids=["typed_function", "comment_without_space", "long_set"],
)
def test_the_official_answer_of_a_real_putnambench_file_replaces_its_sorry(name, answer):
    text = putnambench_files()[name]
    task = putnam_task(name, text)
    assert isinstance(task, TaskSpec)
    prefix = contract(task)["prefix"]
    [declaration] = [line for line in prefix.splitlines() if f"{name}_solution" in line]
    assert "sorry" not in prefix
    assert answer.replace("->", "→") in declaration
    assert declaration.split(":=", 1)[1].strip() == text.split(f"{name}_solution", 1)[1].split("\n")[1][2:].strip()
    assert source_evidence(task)["setting"] == "answer given"


def test_a_real_valued_answer_is_marked_noncomputable():
    task = putnam_task("putnam_2004_b5", putnambench_files()["putnam_2004_b5"])
    assert "noncomputable abbrev putnam_2004_b5_solution" in contract(task)["prefix"]


def test_a_proof_only_problem_keeps_its_statement_and_informal_text():
    task = putnam_task("putnam_2001_b1", putnam_file("putnam_2001_b1"))
    assert source_evidence(task)["setting"] == "proof only"
    assert contract(task)["statement"] == "theorem putnam_2001_b1 : (2 : Nat) + 2 = 4 :="
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage) and "What is two plus two?" in prompt.content


@pytest.mark.parametrize(
    ("name", "text", "reason"),
    [
        ("putnam_2000_a2", putnam_file("putnam_2000_a2", "4").replace("-- 4\n", ""), "unfilled_answer"),
        ("putnam_2000_a3", putnam_file("putnam_2000_a1"), "ambiguous_target"),
        ("putnam_2021_a1", putnam_file("putnam_2021_a1"), "port_does_not_apply"),
    ],
    ids=["answer_without_comment", "file_named_after_another_theorem", "rewrite_that_no_longer_applies"],
)
def test_a_putnambench_problem_the_grader_cannot_freeze_is_a_source_defect(name, text, reason):
    result = putnam_task(name, text)
    assert isinstance(result, ImportRejection) and result.reason == reason


def test_the_putnambench_reader_takes_each_lean_file_once():
    files = [(f"{PUTNAM_DIRECTORY}/putnam_2000_a1.lean", putnam_file("putnam_2000_a1", "4")), ("README.md", "skip")]
    assert [row["name"] for row in lean.putnam_rows(StagedFile(archive(files)))] == ["putnam_2000_a1"]
    duplicated = archive([*files, (f"{PUTNAM_DIRECTORY}/putnam_2000_a1.lean", "again")])
    with pytest.raises(ValueError, match="twice"):
        list(lean.putnam_rows(StagedFile(duplicated)))


def test_putnambench_publishes_every_problem_without_rejections(tmp_path):
    files = [
        (f"{PUTNAM_DIRECTORY}/putnam_2000_a1.lean", putnam_file("putnam_2000_a1", "4")),
        (f"{PUTNAM_DIRECTORY}/putnam_2001_b1.lean", putnam_file("putnam_2001_b1")),
    ]
    cases = published("putnambench", "greedy", archive(files), 2, tmp_path)
    assert [case.item.groups for case in cases] == [{"setting": "answer given"}, {"setting": "proof only"}]
    [event] = cases[0].task.context.events
    assert "putnam_2000_a1_solution : Nat := 4" in event.content


REWRITTEN = {
    "PB-Basic-012": "points.IsChain (",
    "PB-Advanced-018": "c.IsChain (",
    "PB-Advanced-023": "P.IsChain IsAdjacentMove",
    "putnam_2021_a1": "l.IsChain fun",
    "putnam_2022_a5": "g.IsChain IsValidMove",
    "putnam_2025_a3": "(initialState n :: play).IsChain isValidMove",
    "putnam_1971_a4": "(P n \u03b5).coeff i",
    "putnam_1987_a4": "(P.coeff i).im",
    "putnam_1966_a4": "open Classical in\ntheorem putnam_1966_a4",
}
"""What each rewritten statement says in place of the construct the grader's Mathlib rejects."""


@pytest.mark.parametrize("name", sorted(lean.PORTS))
def test_a_statement_the_grader_rejects_is_rewritten_to_say_the_same_thing(name):
    if name.startswith("putnam_"):
        task = putnam_task(name, putnambench_files()[name])
    else:
        task = convert_row(lean.imo_lean_proof_bench("greedy").config.recipe, imo_bench_fixture_rows()[name])
    assert isinstance(task, TaskSpec)
    frozen = contract(task)
    source = frozen["prefix"] + "\n" + frozen["statement"]
    assert "Chain'" not in source and ".Chain " not in source and "MvPolynomial.coeff" not in source
    assert REWRITTEN[name] in source


@pytest.mark.parametrize("copies", [0, 2])
def test_a_rewrite_that_does_not_match_exactly_once_is_refused(copies):
    text = putnam_file("putnam_2021_a1") + "-- l.Chain' fun\n" * copies
    assert lean.ported("putnam_2021_a1", text) is None


def test_proofnet_leaves_out_its_repeated_and_answered_rows_and_never_shows_the_informal_proof(tmp_path):
    sentinel = "SENTINEL-INFORMAL-PROOF"
    rows = [proofnet_row(index=1, informal_proof=sentinel), proofnet_row(index=35), proofnet_row(index=351)]
    staged = "".join(json.dumps(row) + "\n" for row in rows).encode()
    [case] = published("proofnet-verified", "avg8", staged, 1, tmp_path)
    assert case.item.groups == {"textbook": "Artin"}
    [event] = case.task.context.events
    assert sentinel not in event.content and "Let $H$ be the subgroup" in event.content
    assert "benchmark:ProofNet-Verified" in case.task.tags


def test_an_imo_bench_task_shows_the_problem_and_never_its_written_solution():
    task = convert_row(lean.imo_lean_proof_bench("greedy").config.recipe, imo_bench_row())
    assert isinstance(task, TaskSpec)
    assert contract(task)["statement"] == "theorem PBBasic999 : (2 : Nat) + 2 = 4 :="
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage) and "Show that two plus two is four." in prompt.content
    assert "SENTINEL" not in task.model_dump_json()
    assert lean.grouped_item("level")(task).groups == {"level": "pre-IMO"}
    assert "benchmark:IMO-LeanProofBench" in task.tags


def test_imo_bench_publishes_every_row_with_its_level_and_no_written_solution(tmp_path):
    rows = [imo_bench_row("PB-Basic-998", "PBBasic998"), imo_bench_row("PB-Basic-999", "PBBasic999", Level="IMO-hard")]
    cases = published("imo-leanproofbench", "greedy", imo_bench_csv(rows), 2, tmp_path)
    assert [case.item.groups for case in cases] == [{"level": "pre-IMO"}, {"level": "IMO-hard"}]
    assert all("SENTINEL" not in case.task.model_dump_json() for case in cases)
