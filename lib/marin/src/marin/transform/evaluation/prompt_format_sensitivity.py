# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic prompt-format sensitivity records for supervised PPL evals."""

import csv
import html
import io
import json
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from rigging.filesystem import StoragePath, open_url
from zephyr.writers import atomic_rename

PROMPT_FORMAT_NUM_FEWSHOT = 5
DEFAULT_PROMPT_FORMAT_OUTPUT_FILENAME = "staged.jsonl.gz"
PROMPT_FORMAT_RENDERER_VERSION = "v2"

_INDENT_RE = re.compile(r"^", re.MULTILINE)


@dataclass(frozen=True)
class PromptFormatExample:
    """One semantic input/output pair used across prompt surface forms."""

    example_id: str
    input_text: str
    target: str


@dataclass(frozen=True)
class PromptFormatTask:
    """A fixed task with five support examples and held-out queries."""

    key: str
    title: str
    description: str
    support_examples: tuple[PromptFormatExample, ...]
    heldout_examples: tuple[PromptFormatExample, ...]


@dataclass(frozen=True)
class PromptFormatTemplate:
    """A prompt surface form that can render support rows and an unfinished query."""

    key: str
    family: str
    description: str
    renderer: Callable[[PromptFormatExample, bool], str]


@dataclass(frozen=True)
class PromptFormatSensitivityStagingConfig:
    """Configuration for staging one task/template slice."""

    output_path: str
    task_key: str
    template_key: str
    output_filename: str = DEFAULT_PROMPT_FORMAT_OUTPUT_FILENAME
    source_manifest: IngestionSourceManifest | None = None
    content_fingerprint: str = ""


def _indent(text: str, spaces: int = 4) -> str:
    return _INDENT_RE.sub(" " * spaces, text)


def _json_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def _csv_field(text: str) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="")
    writer.writerow([text])
    return stream.getvalue()


def _csv_pair(input_text: str, target: str) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="")
    writer.writerow([input_text, target])
    return stream.getvalue()


def _tsv_field(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def _markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _triple_quoted(text: str) -> str:
    return text.replace('"""', '\\"\\"\\"')


def _render_arrow(example: PromptFormatExample, include_target: bool) -> str:
    return f"{example.input_text}\n-> {example.target if include_target else ''}"


def _render_equals(example: PromptFormatExample, include_target: bool) -> str:
    return f"{example.input_text}\n= {example.target if include_target else ''}"


def _render_colon_mapping(example: PromptFormatExample, include_target: bool) -> str:
    return f"Input: {example.input_text}\nOutput: {example.target if include_target else ''}"


def _render_key_value_block(example: PromptFormatExample, include_target: bool) -> str:
    return f"input:\n{example.input_text}\n\noutput:\n{example.target if include_target else ''}"


def _render_qa(example: PromptFormatExample, include_target: bool) -> str:
    return f"Q: {example.input_text}\nA: {example.target if include_target else ''}"


def _render_faq(example: PromptFormatExample, include_target: bool) -> str:
    return f"FAQ entry\nQuestion: {example.input_text}\nAnswer: {example.target if include_target else ''}"


def _render_flashcard(example: PromptFormatExample, include_target: bool) -> str:
    return f"Front: {example.input_text}\nBack: {example.target if include_target else ''}"


def _render_problem_solution(example: PromptFormatExample, include_target: bool) -> str:
    return f"Problem:\n{example.input_text}\n\nSolution:\n{example.target if include_target else ''}"


def _render_labeled_example(example: PromptFormatExample, include_target: bool) -> str:
    return f"Example\nGiven:\n{example.input_text}\nResult:\n{example.target if include_target else ''}"


def _render_bullet_list(example: PromptFormatExample, include_target: bool) -> str:
    return f"- input: {example.input_text}\n  output: {example.target if include_target else ''}"


def _render_numbered(example: PromptFormatExample, include_target: bool) -> str:
    return f"1. Given {example.input_text}\n2. Return {example.target if include_target else ''}"


def _render_json_object(example: PromptFormatExample, include_target: bool) -> str:
    prefix = f'{{"input": {_json_string(example.input_text)}, "output": '
    return f"{prefix}{_json_string(example.target)}}}" if include_target else prefix


def _render_jsonl(example: PromptFormatExample, include_target: bool) -> str:
    return _render_json_object(example, include_target)


def _render_yaml(example: PromptFormatExample, include_target: bool) -> str:
    prefix = f"- input: |-\n{_indent(example.input_text)}\n  output: "
    return f"{prefix}{example.target}" if include_target else prefix


def _render_xml(example: PromptFormatExample, include_target: bool) -> str:
    prefix = f"<example><input>{html.escape(example.input_text)}</input><output>"
    suffix = "</output></example>"
    return f"{prefix}{html.escape(example.target)}{suffix}" if include_target else prefix


def _render_csv(example: PromptFormatExample, include_target: bool) -> str:
    if include_target:
        return _csv_pair(example.input_text, example.target)
    return f"{_csv_field(example.input_text)},"


def _render_tsv(example: PromptFormatExample, include_target: bool) -> str:
    output = _tsv_field(example.target) if include_target else ""
    return f"{_tsv_field(example.input_text)}\t{output}"


def _render_markdown_table(example: PromptFormatExample, include_target: bool) -> str:
    output = _markdown_cell(example.target) if include_target else ""
    return f"| {_markdown_cell(example.input_text)} | {output}"


def _render_python_doctest(example: PromptFormatExample, include_target: bool) -> str:
    return f'>>> solve("""{_triple_quoted(example.input_text)}""")\n{example.target if include_target else ""}'


def _render_python_repl(example: PromptFormatExample, include_target: bool) -> str:
    return f">>> solve({_json_string(example.input_text)})\n{example.target if include_target else ''}"


def _render_shell(example: PromptFormatExample, include_target: bool) -> str:
    return f"$ solve <<'INPUT'\n{example.input_text}\nINPUT\n{example.target if include_target else ''}"


def _render_sql(example: PromptFormatExample, include_target: bool) -> str:
    output = example.target if include_target else ""
    return f"SELECT solve($${example.input_text}$$) AS output;\noutput\n------\n{output}"


def _render_toml(example: PromptFormatExample, include_target: bool) -> str:
    prefix = f'[[example]]\ninput = """{_triple_quoted(example.input_text)}"""\noutput = """'
    return f'{prefix}{_triple_quoted(example.target)}"""' if include_target else prefix


def _render_ini(example: PromptFormatExample, include_target: bool) -> str:
    return f"[example]\ninput={example.input_text}\noutput={example.target if include_target else ''}"


def _render_s_expression(example: PromptFormatExample, include_target: bool) -> str:
    prefix = f"(example (input {_json_string(example.input_text)}) (output "
    return f"{prefix}{_json_string(example.target)}))" if include_target else prefix


def _render_html_dl(example: PromptFormatExample, include_target: bool) -> str:
    prefix = f"<dl><dt>{html.escape(example.input_text)}</dt><dd>"
    suffix = "</dd></dl>"
    return f"{prefix}{html.escape(example.target)}{suffix}" if include_target else prefix


PROMPT_FORMAT_TEMPLATES: tuple[PromptFormatTemplate, ...] = (
    PromptFormatTemplate("plain_arrow", "plain", "input arrow output", _render_arrow),
    PromptFormatTemplate("plain_equals", "plain", "input equals output", _render_equals),
    PromptFormatTemplate("colon_mapping", "plain", "Input/Output colon labels", _render_colon_mapping),
    PromptFormatTemplate("key_value_block", "plain", "lowercase key/value block", _render_key_value_block),
    PromptFormatTemplate("qa", "qa", "Q/A pairs", _render_qa),
    PromptFormatTemplate("faq", "qa", "FAQ entry", _render_faq),
    PromptFormatTemplate("flashcard", "qa", "front/back flashcard", _render_flashcard),
    PromptFormatTemplate("problem_solution", "qa", "problem/solution block", _render_problem_solution),
    PromptFormatTemplate("labeled_example", "plain", "given/result example", _render_labeled_example),
    PromptFormatTemplate("bullet_list", "plain", "bullet key/value list", _render_bullet_list),
    PromptFormatTemplate("numbered_steps", "plain", "numbered given/return steps", _render_numbered),
    PromptFormatTemplate("json_object", "structured", "JSON object", _render_json_object),
    PromptFormatTemplate("jsonl", "structured", "JSONL records", _render_jsonl),
    PromptFormatTemplate("yaml", "structured", "YAML sequence", _render_yaml),
    PromptFormatTemplate("xml", "structured", "XML example element", _render_xml),
    PromptFormatTemplate("csv", "table", "CSV row", _render_csv),
    PromptFormatTemplate("tsv", "table", "TSV row", _render_tsv),
    PromptFormatTemplate("markdown_table", "table", "Markdown table row", _render_markdown_table),
    PromptFormatTemplate("python_doctest", "code", "Python doctest", _render_python_doctest),
    PromptFormatTemplate("python_repl", "code", "Python REPL", _render_python_repl),
    PromptFormatTemplate("shell_transcript", "code", "shell transcript", _render_shell),
    PromptFormatTemplate("sql_result", "code", "SQL result transcript", _render_sql),
    PromptFormatTemplate("toml", "structured", "TOML record", _render_toml),
    PromptFormatTemplate("ini", "structured", "INI section", _render_ini),
    PromptFormatTemplate("s_expression", "structured", "S-expression", _render_s_expression),
    PromptFormatTemplate("html_dl", "structured", "HTML description list", _render_html_dl),
)
PROMPT_FORMAT_TEMPLATES_BY_KEY = {template.key: template for template in PROMPT_FORMAT_TEMPLATES}


def _examples(prefix: str, pairs: tuple[tuple[str, str], ...]) -> tuple[PromptFormatExample, ...]:
    return tuple(
        PromptFormatExample(example_id=f"{prefix}-{index:02d}", input_text=input_text, target=target)
        for index, (input_text, target) in enumerate(pairs, start=1)
    )


PROMPT_FORMAT_TASKS: tuple[PromptFormatTask, ...] = (
    PromptFormatTask(
        key="mcqa_science",
        title="Science multiple choice",
        description="Choose the correct answer text for a multiple-choice science question.",
        support_examples=_examples(
            "mcqa-support",
            (
                (
                    "Question: Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Neptune",
                    "Mars is known as the Red Planet.",
                ),
                (
                    "Question: What gas do plants absorb during photosynthesis?\n"
                    "A. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Helium",
                    "Plants absorb carbon dioxide during photosynthesis.",
                ),
                (
                    "Question: Which organ pumps blood through the human body?\nA. Lung\nB. Heart\nC. Kidney\nD. Liver",
                    "The heart pumps blood through the human body.",
                ),
                (
                    "Question: What force pulls objects toward Earth?\n"
                    "A. Magnetism\nB. Friction\nC. Gravity\nD. Evaporation",
                    "Gravity pulls objects toward Earth.",
                ),
                (
                    "Question: Water freezes at what temperature at sea level?\n"
                    "A. 0 degrees Celsius\nB. 10 degrees Celsius\n"
                    "C. 50 degrees Celsius\nD. 100 degrees Celsius",
                    "Water freezes at 0 degrees Celsius at sea level.",
                ),
            ),
        ),
        heldout_examples=_examples(
            "mcqa-heldout",
            (
                (
                    "Question: Which part of a plant takes in water from soil?\nA. Flower\nB. Root\nC. Petal\nD. Seed",
                    "The root takes in water from soil.",
                ),
                (
                    "Question: Which celestial body gives Earth most of its light?\n"
                    "A. Moon\nB. Sun\nC. Mars\nD. Polaris",
                    "The Sun gives Earth most of its light.",
                ),
                (
                    "Question: Which material is a good electrical conductor?\nA. Rubber\nB. Copper\nC. Glass\nD. Wood",
                    "Copper is a good electrical conductor.",
                ),
            ),
        ),
    ),
    PromptFormatTask(
        key="short_factual_qa",
        title="Short factual QA",
        description="Answer the factual question with a concise sentence.",
        support_examples=_examples(
            "qa-support",
            (
                ("What city is the capital of Japan?", "Tokyo is the capital of Japan."),
                ("Who wrote the play Hamlet?", "William Shakespeare wrote Hamlet."),
                ("What instrument measures atmospheric pressure?", "A barometer measures atmospheric pressure."),
                ("Which ocean borders California?", "The Pacific Ocean borders California."),
                ("What is the largest mammal?", "The blue whale is the largest mammal."),
            ),
        ),
        heldout_examples=_examples(
            "qa-heldout",
            (
                ("What city is the capital of Canada?", "Ottawa is the capital of Canada."),
                ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
                ("Which planet has the most prominent ring system?", "Saturn has the most prominent ring system."),
            ),
        ),
    ),
    PromptFormatTask(
        key="news_classification",
        title="News classification",
        description="Classify the news blurb into a stable topic label.",
        support_examples=_examples(
            "class-support",
            (
                (
                    "Stocks rose after the central bank signaled slower rate increases.",
                    "Business and finance news",
                ),
                ("The striker scored twice in the championship match.", "Sports competition report"),
                (
                    "Researchers announced a battery that charges in under ten minutes.",
                    "Science and technology news",
                ),
                ("The senate committee advanced the revised budget bill.", "Politics and government news"),
                ("A new museum exhibition opened downtown this weekend.", "Arts and culture news"),
            ),
        ),
        heldout_examples=_examples(
            "class-heldout",
            (
                (
                    "The software company released a smaller language model for phones.",
                    "Science and technology news",
                ),
                ("The mayor proposed new rules for short-term rentals.", "Politics and government news"),
                ("The tennis final was delayed by heavy rain.", "Sports competition report"),
            ),
        ),
    ),
    PromptFormatTask(
        key="string_transformation",
        title="String transformation",
        description="Normalize the record into REGION-ID-COLOR with a four digit id.",
        support_examples=_examples(
            "transform-support",
            (
                ("region=west; id=42; color=blue", "WEST-0042-BLUE"),
                ("region=north; id=7; color=green", "NORTH-0007-GREEN"),
                ("region=south; id=315; color=red", "SOUTH-0315-RED"),
                ("region=east; id=90; color=yellow", "EAST-0090-YELLOW"),
                ("region=central; id=5; color=purple", "CENTRAL-0005-PURPLE"),
            ),
        ),
        heldout_examples=_examples(
            "transform-heldout",
            (
                ("region=west; id=108; color=orange", "WEST-0108-ORANGE"),
                ("region=north; id=64; color=silver", "NORTH-0064-SILVER"),
                ("region=east; id=901; color=black", "EAST-0901-BLACK"),
            ),
        ),
    ),
    PromptFormatTask(
        key="record_extraction",
        title="Structured record extraction",
        description="Extract the requested fields as a compact field list.",
        support_examples=_examples(
            "extract-support",
            (
                (
                    "Order 1842 ships to Lima on 2026-03-14 with 12 crates of valves.",
                    "order_id=1842; city=Lima; date=2026-03-14; items=valves; quantity=12",
                ),
                (
                    "Ticket 9001 assigns Priya to audit two routers in Oslo before 2026-04-02.",
                    "ticket_id=9001; owner=Priya; city=Oslo; date=2026-04-02; assets=routers; quantity=2",
                ),
                (
                    "Batch 77 sends 5 microscopes and 3 lenses to Accra on 2026-01-09.",
                    "batch_id=77; city=Accra; date=2026-01-09; items=microscopes,lenses; quantity=8",
                ),
                (
                    "Case 312 lists Omar as reviewer for the Berlin archive due 2026-02-20.",
                    "case_id=312; owner=Omar; city=Berlin; date=2026-02-20; asset=archive",
                ),
                (
                    "Shipment 451 sends 6 pumps to Quito for Mina on 2026-05-18.",
                    "shipment_id=451; owner=Mina; city=Quito; date=2026-05-18; items=pumps; quantity=6",
                ),
            ),
        ),
        heldout_examples=_examples(
            "extract-heldout",
            (
                (
                    "Order 2880 ships 9 sensors to Tallinn for Jules on 2026-06-03.",
                    "order_id=2880; owner=Jules; city=Tallinn; date=2026-06-03; items=sensors; quantity=9",
                ),
                (
                    "Ticket 415 asks Nia to inspect 4 bridges in Porto by 2026-07-11.",
                    "ticket_id=415; owner=Nia; city=Porto; date=2026-07-11; assets=bridges; quantity=4",
                ),
                (
                    "Batch 63 delivers 2 cameras and 8 tripods to Busan on 2026-08-22.",
                    "batch_id=63; city=Busan; date=2026-08-22; items=cameras,tripods; quantity=10",
                ),
            ),
        ),
    ),
    PromptFormatTask(
        key="python_repl",
        title="Python REPL reasoning",
        description="Return the deterministic result of the small Python-like expression.",
        support_examples=_examples(
            "code-support",
            (
                ("len('marin') + 4", "9"),
                ("'-'.join(['red', 'blue']).upper()", "RED-BLUE"),
                ("sum([3, 5, 8])", "16"),
                ("sorted(['delta', 'alpha', 'beta'])[0]", "alpha"),
                ("{'a': 2, 'b': 5}['b'] * 3", "15"),
            ),
        ),
        heldout_examples=_examples(
            "code-heldout",
            (
                ("len('prompt') * 7", "42"),
                ("'-'.join(reversed(['north', 'east']))", "east-north"),
                ("sum(x * x for x in [2, 3, 4])", "29"),
            ),
        ),
    ),
)
PROMPT_FORMAT_TASKS_BY_KEY = {task.key: task for task in PROMPT_FORMAT_TASKS}


def render_prompt_format_input(
    *,
    task: PromptFormatTask,
    template: PromptFormatTemplate,
    heldout: PromptFormatExample,
) -> str:
    """Render five support examples and one unfinished held-out query."""

    if len(task.support_examples) != PROMPT_FORMAT_NUM_FEWSHOT:
        raise ValueError(f"{task.key} must have exactly {PROMPT_FORMAT_NUM_FEWSHOT} support examples")
    header = f"Task: {task.title}\nInstruction: {task.description}\nFormat: {template.description}"
    blocks = [header, *(template.renderer(example, True) for example in task.support_examples)]
    blocks.append(template.renderer(heldout, False))
    return "\n\n".join(blocks)


def render_prompt_format_target(*, template: PromptFormatTemplate, heldout: PromptFormatExample) -> str:
    """Render the scored continuation for an unfinished held-out query."""

    unfinished = template.renderer(heldout, False)
    finished = template.renderer(heldout, True)
    if not finished.startswith(unfinished):
        raise ValueError(f"{template.key} renderer must extend its unfinished held-out query")
    return finished[len(unfinished) :]


def prompt_format_record(task_key: str, template_key: str, heldout_index: int) -> dict[str, Any]:
    """Return one supervised target-only record for a task/template/held-out index."""

    task = PROMPT_FORMAT_TASKS_BY_KEY[task_key]
    template = PROMPT_FORMAT_TEMPLATES_BY_KEY[template_key]
    heldout = task.heldout_examples[heldout_index]
    return {
        "id": f"{task.key}/{template.key}/{heldout.example_id}",
        "input": render_prompt_format_input(task=task, template=template, heldout=heldout),
        "target": render_prompt_format_target(template=template, heldout=heldout),
        "source": "prompt_format_sensitivity_static",
        "provenance": {
            "task_key": task.key,
            "template_key": template.key,
            "template_family": template.family,
            "renderer_version": PROMPT_FORMAT_RENDERER_VERSION,
            "num_fewshot": PROMPT_FORMAT_NUM_FEWSHOT,
            "support_example_ids": [example.example_id for example in task.support_examples],
            "heldout_example_id": heldout.example_id,
            "semantic_target": heldout.target,
        },
    }


def stage_prompt_format_sensitivity_source(cfg: PromptFormatSensitivityStagingConfig) -> dict[str, Any]:
    """Stage one prompt-format sensitivity task/template slice as JSONL."""

    if cfg.source_manifest is not None and cfg.content_fingerprint:
        expected = cfg.source_manifest.fingerprint()
        if cfg.content_fingerprint != expected:
            raise ValueError(
                f"content_fingerprint mismatch: config has {cfg.content_fingerprint}, source manifest has {expected}"
            )

    task = PROMPT_FORMAT_TASKS_BY_KEY[cfg.task_key]
    if cfg.template_key not in PROMPT_FORMAT_TEMPLATES_BY_KEY:
        raise ValueError(f"Unknown prompt-format template: {cfg.template_key}")
    if len(task.support_examples) != PROMPT_FORMAT_NUM_FEWSHOT:
        raise ValueError(f"{cfg.task_key} must have exactly {PROMPT_FORMAT_NUM_FEWSHOT} support examples")

    StoragePath(cfg.output_path).mkdirs(exist_ok=True)
    out_file = posixpath.join(cfg.output_path, cfg.output_filename)
    compression = "gzip" if out_file.endswith(".gz") else None

    with atomic_rename(out_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression=compression) as outfile:
            for heldout_index in range(len(task.heldout_examples)):
                json.dump(
                    prompt_format_record(cfg.task_key, cfg.template_key, heldout_index), outfile, ensure_ascii=True
                )
                outfile.write("\n")

    output_size = StoragePath(out_file).size()
    result: dict[str, Any] = {
        "record_count": len(task.heldout_examples),
        "bytes_written": output_size,
        "output_file": out_file,
    }

    if cfg.source_manifest is not None:
        metadata_path = write_ingestion_metadata_json(
            manifest=cfg.source_manifest,
            materialized_output=MaterializedOutputMetadata(
                input_path="prompt_format_sensitivity_static",
                output_path=cfg.output_path,
                output_file=out_file,
                record_count=len(task.heldout_examples),
                bytes_written=output_size,
                metadata={
                    "task_key": cfg.task_key,
                    "template_key": cfg.template_key,
                    "renderer_version": PROMPT_FORMAT_RENDERER_VERSION,
                    "num_fewshot": PROMPT_FORMAT_NUM_FEWSHOT,
                    "heldout_examples": len(task.heldout_examples),
                },
            ),
        )
        result["metadata_file"] = metadata_path

    return result
