# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import cleaned TaskTrove Reasoning Gym tasks for direct chat."""

import hashlib
import json
import tomllib
from pathlib import Path

from tasktrove_verify.spec import ReasoningGymSpec, parse_spec

from taskcompendium.grading import exact_answer
from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.verifiers.script import ScriptVerifier, embedded_resource, script_verifier

FAMILY = "other"
CONVERTER = "nemotron_reasoning"
IMPORTER_REVISION = "taskcompendium-tasktrove-reasoning-gym-v0.3"
ANSWER_PATH = "/app/answer.txt"
REASONING_GYM_VERSION = "0.1.25"
_SCRIPT_ADAPTER = Path(__file__).with_name("reasoning_gym_adapter.py")
_CASE_SENSITIVE_EXACT = frozenset({"ab", "self_reference"})
_WHITESPACE_EXACT = frozenset({"path_star"})
# Audited against the pinned library; additions require evaluator review.
_SCRIPT_DATASETS = frozenset(
    {
        "acre",
        "advanced_geometry",
        "aiw",
        "arc_1d",
        "base_conversion",
        "basic_arithmetic",
        "bf",
        "binary_alternation",
        "binary_matrix",
        "bitwise_arithmetic",
        "boxnet",
        "caesar_cipher",
        "calendar_arithmetic",
        "chain_sum",
        "circuit_logic",
        "codeio",
        "coin_flip",
        "color_cube_rotation",
        "complex_arithmetic",
        "count_bits",
        "count_primes",
        "countdown",
        "course_schedule",
        "cryptarithm",
        "decimal_arithmetic",
        "decimal_chain_sum",
        "dice",
        "emoji_mystery",
        "family_relationships",
        "figlet_font",
        "fraction_simplification",
        "futoshiki",
        "game_of_life",
        "game_of_life_halting",
        "gcd",
        "graph_color",
        "group_anagrams",
        "gsm_symbolic",
        "intermediate_integration",
        "isomorphic_strings",
        "jugs",
        "kakurasu",
        "knight_swap",
        "knights_knaves",
        "largest_island",
        "lcm",
        "leg_counting",
        "letter_counting",
        "letter_jumble",
        "list_functions",
        "mahjong_puzzle",
        "manipulate_matrix",
        "maze",
        "mini_sudoku",
        "modulo_grid",
        "n_queens",
        "needle_haystack",
        "number_filtering",
        "number_format",
        "number_sequence",
        "number_sorting",
        "palindrome_generation",
        "palindrome_partitioning",
        "polynomial_equations",
        "polynomial_multiplication",
        "pool_matrix",
        "power_function",
        "prime_factorization",
        "products",
        "propositional_logic",
        "puzzle24",
        "quantum_lock",
        "ransom_note",
        "rectangle_count",
        "rotate_matrix",
        "rotten_oranges",
        "rubiks_cube",
        "rush_hour",
        "sentence_reordering",
        "shortest_path",
        "simple_equations",
        "simple_geometry",
        "simple_integration",
        "sokoban",
        "spell_backward",
        "spiral_matrix",
        "string_insertion",
        "string_manipulation",
        "string_splitting",
        "string_synthesis",
        "sudoku",
        "survo",
        "syllogism",
        "time_intervals",
        "tower_of_hanoi",
        "tsumego",
        "word_ladder",
        "word_sequence_reversal",
        "word_sorting",
        "zebra_puzzles",
    }
)
_UNSUPPORTED_DATASETS = {
    "arc_agi": "JSON entry output does not match the scorer's tuple board representation",
    "rearc": "JSON entry output does not match the scorer's tuple board representation",
    "composite": "The entry does not identify the component scorer configuration",
}
_INSTRUCTION_PREFIX = (
    "You are solving a procedurally-generated reasoning task from Reasoning Gym. Read the problem below and write "
    f"your final answer to `{ANSWER_PATH}`. The verifier will try the upstream Reasoning Gym scorer first, then fall "
    "back to normalized exact-match.\n\n---\n\n"
)


def _plain_text_instruction(instructions: str) -> str:
    """Remove the recognized source harness preamble and retain the problem."""
    if not instructions.startswith(_INSTRUCTION_PREFIX):
        raise ValueError("Unsupported Reasoning Gym instruction template")
    question = instructions.removeprefix(_INSTRUCTION_PREFIX).strip()
    if not question:
        raise ValueError("Reasoning Gym instruction has no question")
    return question


def import_task(
    archive: TaskArchive, *, runtime_image: str | None = None, timeout_seconds: float | None = None
) -> TaskSpec:
    """Import a Reasoning Gym archive with its generated entry kept in the verifier."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if (
            metadata.get("family") != FAMILY
            or metadata.get("converter") != CONVERTER
            or metadata.get("mode") != "reasoning-gym"
        ):
            raise ValueError("Unsupported TaskTrove Reasoning Gym source")
        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("TaskTrove tags must be an ordered list of strings")
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, ReasoningGymSpec) or contract.entry != "entry.json":
            raise ValueError("TaskTrove archive must declare a Reasoning Gym verifier with entry.json")
        if contract.output != ANSWER_PATH:
            raise ValueError("TaskTrove Reasoning Gym archive has an unsupported answer path")
        entry = json.loads(archive.files[f"tests/{contract.entry}"])
        if not isinstance(entry, dict):
            raise ValueError("Reasoning Gym entry must be an object")
        instructions = _plain_text_instruction(archive.files["instruction.md"].decode())
        if not isinstance(entry.get("question"), str) or instructions != entry["question"].strip():
            raise ValueError("Reasoning Gym instruction differs from its generated entry")
        entry_metadata = entry.get("metadata")
        if not isinstance(entry_metadata, dict) or entry_metadata.get("source_dataset") != contract.dataset:
            raise ValueError("Reasoning Gym entry dataset differs from its verifier")
        expected = entry.get("answer")
        if "answer" not in entry or (expected is not None and not isinstance(expected, str)):
            raise ValueError("Reasoning Gym entry requires a string or null answer field")
        if contract.dataset in _UNSUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported Reasoning Gym evaluator {contract.dataset!r}: {_UNSUPPORTED_DATASETS[contract.dataset]}"
            )
        if contract.dataset in _CASE_SENSITIVE_EXACT | _WHITESPACE_EXACT:
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError("Exact Reasoning Gym evaluator requires a nonempty string answer")
            if contract.dataset in _CASE_SENSITIVE_EXACT and expected != expected.strip():
                raise ValueError("Case-sensitive Reasoning Gym gold must not contain outer whitespace")
            verifier = exact_answer(
                expected, ignore_case=False, collapse_whitespace=contract.dataset in _WHITESPACE_EXACT
            )
        elif contract.dataset in _SCRIPT_DATASETS:
            if runtime_image is None or timeout_seconds is None:
                raise ValueError("This Reasoning Gym evaluator requires an explicit runtime_image and timeout_seconds")
            scoring = json.dumps({"dataset": contract.dataset, "reasoning_gym_version": REASONING_GYM_VERSION}).encode()
            verifier = script_verifier(
                ScriptVerifier(
                    entrypoint="grade.py",
                    args=("/tests", "/verifier"),
                    timeout_seconds=timeout_seconds,
                    runtime_image=runtime_image,
                    resources=(
                        embedded_resource("grade.py", _SCRIPT_ADAPTER.read_bytes(), executable=True),
                        embedded_resource("entry.json", archive.files[f"tests/{contract.entry}"]),
                        embedded_resource("scoring.json", scoring),
                    ),
                )
            )
        else:
            raise ValueError(f"Unaudited Reasoning Gym evaluator {contract.dataset!r}")
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove Reasoning Gym archive: {error}") from error
    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=instructions),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=verifier,
        source=archive.source.model_copy(update={"importer_revision": IMPORTER_REVISION}),
        tags=tuple(tags),
    )
