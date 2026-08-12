# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Artificial Analysis and lm-eval-harness text for decontamination.

The mandatory AA corpus and its manifests are written under a versioned path
in ``MARIN_PREFIX``:

- ``aa/<eval>/<split>.parquet`` -- all nine AA Intelligence Index v4.1.1
  benchmarks. These artifacts are mandatory.
- ``datakit/decontam/evals/lmh/<task>/<split>.parquet`` -- every unique task in
  ``experiments/evals/task_configs.py`` bundles, loaded via lm-eval-harness.
  These best-effort artifacts keep their extraction-version sidecars and are
  reused across AA corpus versions.

Each record has the form ``{id: str, text: str}``. AA extraction is pinned by
benchmark. AA preparation fails if a source cannot load or if its extracted
record count differs from the registered count. A complete manifest is written
only after all mandatory AA artifacts pass validation. Bloom creation requires
this manifest.

Test split is preferred; tasks without a test split fall back to
validation, then training. Tasks that fail to load (e.g. removed from
lm-eval since our pinned commit, gated HF datasets) are logged and skipped.

Submit on a CPU Iris cluster. The ``lm_eval`` extra supplies the optional
lm-eval-harness package:

    uv run iris --cluster=cw-rno2a job run --no-wait \\
        --extra=cpu --extra=lm_eval --priority interactive \\
        --memory 16GB --cpu 2 --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/prepare_eval_corpus.py

The Iris worker installs lm-eval through the Levanter ``lm_eval`` extra. The
script monkey-patches ``datasets.load_dataset`` to force
``trust_remote_code=True`` and sets ``HF_ALLOW_CODE_EVAL=1`` before
loading any task, so tasks shipping custom HF loading scripts (logiqa,
piqa, ethics_*, crows_pairs_*, ...) and humaneval load without per-task
plumbing.
"""

import dataclasses
import io as io_mod
import json
import logging
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image as DatasetsImage
from datasets import load_dataset
from rigging.filesystem import StoragePath, marin_prefix
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.lmh_loader import (
    flatten_task_dict,
    materialize_first_nonempty_split,
    trust_remote_code_for_hf,
)
from experiments.evals.task_configs import (
    ACTION_TASKS,
    BIAS_SAFETY_TASKS,
    CODE_TASKS,
    CORE_TASKS,
    EMOTIONAL_ETHICS_TASKS,
    KEY_GENERATION_TASKS,
    KEY_MULTIPLE_CHOICE_TASKS,
    KNOWLEDGE_TASKS,
    LANGUAGE_TASKS,
    MATH_TASKS,
    MEDICAL_TASKS,
    MGSM_MULTILINGUAL_TASKS,
    MMLU_TASKS,
    MULTILINGUAL_LM_EVAL_LOGPROB_TASKS,
    OPEN_LM_LEADERBOARD_GEN,
    OPEN_LM_LEADERBOARD_MCQ,
    REASONING_TASKS,
    SPECIALIZED_TASKS,
    TRUTHFULNESS_TASKS,
    XSTORYCLOZE_MULTILINGUAL_TASKS,
)

logger = logging.getLogger(__name__)

AA_INDEX_VERSION = "4.1.1"
EVAL_CORPUS_VERSION = "aa-index-v4.1.1-v2"
EVALS_RELATIVE = f"datakit/decontam/evals/{EVAL_CORPUS_VERSION}"
LMH_EVALS_RELATIVE = "datakit/decontam/evals/lmh"
AA_MANIFEST_RELATIVE = "aa/_manifest.json"
LMH_MANIFEST_RELATIVE = "lmh/_manifest.json"


def _output_root() -> str:
    """Eval-corpus write root, relative to the active ``MARIN_PREFIX`` (store-agnostic)."""
    return f"{marin_prefix()}/{EVALS_RELATIVE}"


def _lmh_output_root() -> str:
    """Stable best-effort lm-eval artifact root."""
    return f"{marin_prefix()}/{LMH_EVALS_RELATIVE}"


# Bump when the LMH text-extraction policy (`_lmh_doc_text`) changes. Written to a
# `.extraction_version` sidecar next to each `lmh/<task>/eval.parquet`; the prepare
# step rewrites (does not skip) any shard whose sidecar doesn't match. Without this,
# a policy change like the cluster-B passage drop (marin#6852) never reaches an
# already-staged corpus — the bloom keeps reading the old passage-bearing text.
_LMH_EXTRACTION_VERSION = "2-passage-drop"
_LMH_VERSION_SIDECAR = ".extraction_version"


def _staged_lmh_version(version_path: str) -> str | None:
    """Return the extraction version recorded next to a staged LMH shard, or None."""
    p = StoragePath(version_path)
    if not p.exists():
        return None
    with p.open("r") as f:
        return f.read().strip()


class AARecordType(StrEnum):
    """AA source-row conversion policy."""

    ROW = "row"
    SCICODE_SUBPROBLEM = "scicode_subproblem"
    TAU3_TASK = "tau3_task"
    TERMINAL_BENCH_TASK = "terminal_bench_task"


@dataclasses.dataclass(frozen=True)
class AAEvalConfig:
    name: str
    subdir: str
    source_revision: str
    expected_records: int
    official_records: int
    hf_id: str | None
    subset: str | None
    split: str
    source_url: str | None = None
    record_type: AARecordType = AARecordType.ROW
    text_fields: tuple[str, ...] = ()
    list_fields: tuple[str, ...] = ()
    skip_if: Callable[[dict], bool] | None = None
    coverage_note: str = "Full official task set."
    filter_note: str | None = None


AA_EVALS: tuple[AAEvalConfig, ...] = (
    AAEvalConfig(
        name="GDPval-AA v2",
        subdir="gdpval_aa_v2",
        source_revision="11e7900cdcac61bc4daf59e65feb238acda98fbf",
        expected_records=220,
        official_records=220,
        hf_id="openai/gdpval",
        subset=None,
        split="train",
        text_fields=("prompt", "rubric_pretty"),
        coverage_note="All public GDPval tasks; prompt and hidden grading rubric are indexed.",
    ),
    AAEvalConfig(
        name="tau3-Banking",
        subdir="tau3_banking",
        source_revision="fc0055dc4e0a316c3f83133267fbd6faaa770992",
        expected_records=97,
        official_records=97,
        hf_id=None,
        subset=None,
        split="test",
        source_url=(
            "https://raw.githubusercontent.com/sierra-research/tau2-bench/"
            "fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/domains/banking_knowledge/tasks.json"
        ),
        record_type=AARecordType.TAU3_TASK,
        coverage_note="Full upstream v1.0.1 task set; user scenarios and hidden evaluation criteria are indexed.",
    ),
    AAEvalConfig(
        name="Terminal-Bench v2.1",
        subdir="terminal_bench_2_1",
        source_revision="c5ee500c185224c97cd6caff7866a990a0057f41",
        expected_records=89,
        official_records=89,
        hf_id=None,
        subset=None,
        split="test",
        source_url=(
            "https://github.com/harbor-framework/terminal-bench-2-1/archive/"
            "c5ee500c185224c97cd6caff7866a990a0057f41.zip"
        ),
        record_type=AARecordType.TERMINAL_BENCH_TASK,
        text_fields=("instruction", "solution"),
        coverage_note="All 89 task instructions and oracle solutions from the pinned v2.1 task tree.",
    ),
    AAEvalConfig(
        name="SciCode",
        subdir="scicode",
        source_revision="4510f6a6aa27c43fad7b43da2c59602a86e88480",
        expected_records=291,
        official_records=288,
        hf_id="SciCode1/SciCode",
        subset=None,
        split="test",
        record_type=AARecordType.SCICODE_SUBPROBLEM,
        coverage_note=(
            "Pinned public test split. It contains 291 subproblems, three more than the 288 reported by AA; "
            "the conservative superset is indexed."
        ),
    ),
    AAEvalConfig(
        name="AA-LCR",
        subdir="aa_lcr",
        source_revision="bdae010bbce259820c0e34c1d7cce210d966fb75",
        expected_records=100,
        official_records=100,
        hf_id="ArtificialAnalysis/AA-LCR",
        subset=None,
        split="test",
        text_fields=("question", "answer"),
        coverage_note=(
            "All public questions and answers. Public source documents are prompt context and are not indexed."
        ),
    ),
    AAEvalConfig(
        name="AA-Omniscience",
        subdir="aa_omniscience",
        source_revision="4a8ffc87c4650054825fb767fe0da4a4fc97ff32",
        expected_records=600,
        official_records=6000,
        hf_id="ArtificialAnalysis/AA-Omniscience-Public",
        subset=None,
        split="train",
        text_fields=("question", "answer"),
        coverage_note="The mandatory public release contains 600 of the 6,000 private evaluation questions.",
    ),
    AAEvalConfig(
        name="Humanity's Last Exam",
        subdir="hle",
        source_revision="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
        expected_records=2500,
        official_records=2500,
        hf_id="cais/hle",
        subset=None,
        split="test",
        text_fields=("question", "answer"),
        coverage_note=(
            "All 2,500 questions from the May 2025 revision. For multimodal rows, the question and answer text "
            "are indexed; image bytes are not."
        ),
    ),
    AAEvalConfig(
        name="GPQA Diamond",
        subdir="gpqa_diamond",
        source_revision="633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        expected_records=198,
        official_records=198,
        hf_id="Idavidrein/gpqa",
        subset="gpqa_diamond",
        split="train",
        text_fields=(
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        ),
    ),
    AAEvalConfig(
        name="CritPt",
        subdir="critpt",
        source_revision="9b9fc8498596ec08ab5437a72f4aa18beef2b876",
        expected_records=70,
        official_records=70,
        hf_id="CritPt-Benchmark/CritPt",
        subset=None,
        split="train",
        text_fields=("problem_description", "code_template", "answer_code", "answer_only_code"),
        coverage_note=(
            "All 70 public composite challenges used by AA, with response templates and published answer code. "
            "CritPt reports 189 modular checkpoints for the 70 test challenges, but those checkpoint prompts are "
            "not part of the public dataset."
        ),
    ),
)

AA_BENCHMARK_NAMES: tuple[str, ...] = tuple(cfg.name for cfg in AA_EVALS)


def _extract_aa_text(row: dict[str, Any], cfg: AAEvalConfig) -> str:
    """Pin-named extraction first; fall back to generic concat when nothing matches."""
    parts: list[str] = []
    for field in cfg.text_fields:
        v = row.get(field)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    for field in cfg.list_fields:
        v = row.get(field)
        if isinstance(v, list):
            parts.extend(s for s in v if isinstance(s, str) and s.strip())
    if parts:
        return "\n\n".join(parts)
    return _concat_strings(row)


def _concat_strings(record: dict[str, Any], exclude: frozenset[str] = frozenset()) -> str:
    """Concat all string-typed fields in sorted key order; flatten list[str].

    Keys whose lowercase name is in *exclude* are skipped.
    """
    parts: list[str] = []
    for k in sorted(record.keys()):
        if k.lower() in exclude:
            continue
        v = record[k]
        if isinstance(v, str) and v.strip():
            parts.append(v)
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            parts.extend(s for s in v if s.strip())
    return "\n\n".join(parts)


# Reading-comprehension / QA eval docs carry a long, public PASSAGE (article,
# story, premise, context, …) alongside the distinctive question + answer. The
# passage is public text, so indexing it flags any corpus doc that merely quotes
# it (marin#6852 cluster B: anli_r3 news premises, race/coqa/squad passages). For
# a doc bearing any of these fields we drop the passage — both the raw field and
# doc_to_text, which renders it — and index only the answer + the remaining raw
# fields (question / options / hypothesis). This keeps genuine-leakage detection
# (question + answer) while removing the public-passage false positives.
# Corner (documented): a passage field named outside this set, or a question
# field mis-named like a passage, is mis-handled — rare in practice.
_PASSAGE_FIELDS: frozenset[str] = frozenset(
    {"passage", "context", "ctx", "article", "story", "premise", "background", "document", "paragraph", "support"}
)


def _lmh_doc_text(doc: Any, prompt_fn: Callable, target_fn: Callable) -> str:
    """Indexed eval text for one lm-eval-harness doc.

    Passage-bearing docs (a field in :data:`_PASSAGE_FIELDS`) index only
    question + answer (drop the passage field and ``doc_to_text``, which renders
    it). Non-passage docs are unchanged: ``doc_to_text`` (question) +
    ``doc_to_target`` (answer) + every raw string field.
    """
    has_passage = isinstance(doc, dict) and any(k.lower() in _PASSAGE_FIELDS for k in doc)
    parts: list[str] = []
    if not has_passage:
        try:
            prompt = prompt_fn(doc) or ""
        except Exception:
            prompt = ""
        if prompt:
            parts.append(str(prompt))
    try:
        target = target_fn(doc) or ""
    except Exception:
        target = ""
    if target:
        parts.append(str(target))
    if isinstance(doc, dict):
        parts.append(_concat_strings(doc, exclude=_PASSAGE_FIELDS if has_passage else frozenset()))
    return "\n\n".join(p for p in parts if p.strip())


_PARQUET_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])
_PARQUET_BATCH = 1000


def _write_parquet(path: str, records: Iterator[dict]) -> int:
    """Write ``records`` ({id, text}) to a single parquet file at ``path``.

    Streams in ``_PARQUET_BATCH``-row chunks so memory stays bounded for tasks
    with tens of thousands of docs (bbq=58k, swag=20k, babi=20k, ...).
    Compression: zstd. zephyr's parquet reader picks up the file regardless
    of the compression codec.
    """
    parent = StoragePath(path).parent
    if parent.key:
        parent.mkdirs()
    n = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    with StoragePath(path).open("wb") as raw:
        writer = pq.ParquetWriter(raw, _PARQUET_SCHEMA, compression="zstd")
        try:
            for rec in records:
                batch_ids.append(rec["id"])
                batch_texts.append(rec["text"])
                if len(batch_ids) >= _PARQUET_BATCH:
                    writer.write_table(pa.table({"id": batch_ids, "text": batch_texts}, schema=_PARQUET_SCHEMA))
                    n += len(batch_ids)
                    batch_ids, batch_texts = [], []
            if batch_ids:
                writer.write_table(pa.table({"id": batch_ids, "text": batch_texts}, schema=_PARQUET_SCHEMA))
                n += len(batch_ids)
        finally:
            writer.close()
    return n


def _iter_aa_rows(cfg: AAEvalConfig) -> Iterator[dict[str, Any]]:
    """Stream raw rows for one pinned AA source."""
    if cfg.record_type == AARecordType.TERMINAL_BENCH_TASK:
        if cfg.source_url is None:
            raise ValueError(f"{cfg.name}: source_url is required")
        with urllib.request.urlopen(cfg.source_url) as response:
            archive = response.read()
        with zipfile.ZipFile(io_mod.BytesIO(archive)) as source_zip:
            instructions: dict[str, str] = {}
            solutions: dict[str, str] = {}
            for path in source_zip.namelist():
                if "/tasks/" not in path:
                    continue
                task_path = path.split("/tasks/", 1)[1]
                task_name, separator, relative_path = task_path.partition("/")
                if not separator:
                    continue
                if relative_path == "instruction.md":
                    instructions[task_name] = source_zip.read(path).decode("utf-8")
                elif relative_path == "solution/solve.sh":
                    solutions[task_name] = source_zip.read(path).decode("utf-8")
        if instructions.keys() != solutions.keys():
            missing_instructions = sorted(solutions.keys() - instructions.keys())
            missing_solutions = sorted(instructions.keys() - solutions.keys())
            raise ValueError(
                f"{cfg.name}: task archive is incomplete; missing instructions={missing_instructions}, "
                f"missing solutions={missing_solutions}"
            )
        for task_name in sorted(instructions):
            yield {
                "task_name": task_name,
                "instruction": instructions[task_name],
                "solution": solutions[task_name],
            }
        return

    if cfg.source_url is not None:
        with urllib.request.urlopen(cfg.source_url) as response:
            rows = json.load(response)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{cfg.name}: expected a JSON list of objects")
        yield from rows
        return

    if cfg.hf_id is None:
        raise ValueError(f"{cfg.name}: hf_id or source_url is required")
    ds = load_dataset(cfg.hf_id, name=cfg.subset, split=cfg.split, revision=cfg.source_revision)
    # Disable Image-feature decoding so iteration does not decode HLE image
    # bytes. The row still supplies question and answer text for decontamination.
    features = getattr(ds, "features", None) or {}
    for col, ftype in features.items():
        if isinstance(ftype, DatasetsImage):
            ds = ds.cast_column(col, DatasetsImage(decode=False))
    for row in ds:
        yield dict(row)


def _row_identifier(row: dict[str, Any], index: int) -> str:
    for field in ("id", "question_id", "problem_id"):
        value = row.get(field)
        if isinstance(value, str | int):
            return str(value)
    return str(index)


def _scicode_text(row: dict[str, Any], subproblem: dict[str, Any]) -> str:
    parts = [
        row.get("problem_description_main"),
        row.get("problem_background_main"),
        row.get("required_dependencies"),
        subproblem.get("step_description_prompt"),
        subproblem.get("step_background"),
        subproblem.get("function_header"),
        *(subproblem.get("test_cases") or []),
        subproblem.get("return_line"),
        subproblem.get("ground_truth_code"),
    ]
    return "\n\n".join(part for part in parts if isinstance(part, str) and part.strip())


def _aa_records(raw_rows: Iterable[dict[str, Any]], cfg: AAEvalConfig) -> list[dict[str, str]]:
    """Convert one AA source to validated decontamination records."""
    records: list[dict[str, str]] = []
    for index, row in enumerate(raw_rows):
        if cfg.skip_if is not None and cfg.skip_if(row):
            continue

        if cfg.record_type == AARecordType.SCICODE_SUBPROBLEM:
            subproblems = row.get("sub_steps")
            if not isinstance(subproblems, list):
                raise ValueError(f"{cfg.name}: row {_row_identifier(row, index)} has no sub_steps list")
            for subproblem in subproblems:
                if not isinstance(subproblem, dict):
                    raise ValueError(f"{cfg.name}: row {_row_identifier(row, index)} has an invalid subproblem")
                subproblem_id = subproblem.get("step_number")
                if not isinstance(subproblem_id, str) or not subproblem_id:
                    raise ValueError(f"{cfg.name}: row {_row_identifier(row, index)} has a subproblem without an ID")
                text = _scicode_text(row, subproblem)
                if not text:
                    raise ValueError(f"{cfg.name}: subproblem {subproblem_id} has no indexed text")
                records.append({"id": f"{cfg.subdir}-{subproblem_id}", "text": text})
            continue

        if cfg.record_type == AARecordType.TAU3_TASK:
            task_id = row.get("id")
            scenario = row.get("user_scenario")
            criteria = row.get("evaluation_criteria")
            if not isinstance(task_id, str) or not isinstance(scenario, dict) or not isinstance(criteria, dict):
                raise ValueError(f"{cfg.name}: task {index} has an invalid schema")
            instructions = scenario.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError(f"{cfg.name}: task {task_id} has no user instructions")
            text = f"{instructions}\n\n{json.dumps(criteria, sort_keys=True, ensure_ascii=False)}"
            records.append({"id": f"{cfg.subdir}-{task_id}", "text": text})
            continue

        text = _extract_aa_text(row, cfg)
        if not text:
            raise ValueError(f"{cfg.name}: row {_row_identifier(row, index)} has no indexed text")
        if cfg.record_type == AARecordType.TERMINAL_BENCH_TASK:
            identifier = row.get("task_name")
            if not isinstance(identifier, str):
                raise ValueError(f"{cfg.name}: task {index} has no task_name")
        else:
            identifier = _row_identifier(row, index)
        records.append({"id": f"{cfg.subdir}-{identifier}", "text": text})

    if len(records) != cfg.expected_records:
        raise ValueError(f"{cfg.name}: expected {cfg.expected_records} records, extracted {len(records)}")
    return records


def _aa_manifest_entry(cfg: AAEvalConfig) -> dict[str, Any]:
    source = cfg.hf_id if cfg.hf_id is not None else cfg.source_url
    assert source is not None
    return {
        "name": cfg.name,
        "artifact": f"{cfg.subdir}/{cfg.split}.parquet",
        "source": source,
        "source_revision": cfg.source_revision,
        "subset": cfg.subset,
        "split": cfg.split,
        "record_type": cfg.record_type.value,
        "text_fields": list(cfg.text_fields),
        "list_fields": list(cfg.list_fields),
        "expected_records": cfg.expected_records,
        "official_records": cfg.official_records,
        "coverage_note": cfg.coverage_note,
        "filter_note": cfg.filter_note,
    }


def _write_json(path: str, value: dict[str, Any]) -> None:
    parent = StoragePath(path).parent
    if parent.key:
        parent.mkdirs()
    with StoragePath(path).open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def _staged_aa_is_current(out_path: str, sidecar_path: str, entry: dict[str, Any]) -> bool:
    if not StoragePath(out_path).exists() or not StoragePath(sidecar_path).exists():
        return False
    with StoragePath(sidecar_path).open("r") as source:
        staged_entry = json.load(source)
    if staged_entry != entry:
        return False
    with StoragePath(out_path).open("rb") as source:
        return pq.ParquetFile(source).metadata.num_rows == entry["expected_records"]


def _aa_manifest(status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus_version": EVAL_CORPUS_VERSION,
        "suite": f"Artificial Analysis Intelligence Index v{AA_INDEX_VERSION}",
        "required": True,
        "status": status,
        "benchmarks": [_aa_manifest_entry(cfg) for cfg in AA_EVALS],
    }


def _prepare_aa() -> dict[str, Any]:
    manifest_path = f"{_output_root()}/{AA_MANIFEST_RELATIVE}"
    _write_json(manifest_path, _aa_manifest("building"))

    for cfg in AA_EVALS:
        out_path = f"{_output_root()}/aa/{cfg.subdir}/{cfg.split}.parquet"
        sidecar_path = f"{_output_root()}/aa/{cfg.subdir}/{cfg.split}.source.json"
        entry = _aa_manifest_entry(cfg)
        if _staged_aa_is_current(out_path, sidecar_path, entry):
            logger.info("aa/%s: validated existing artifact", cfg.subdir)
            continue

        records = _aa_records(_iter_aa_rows(cfg), cfg)
        n = _write_parquet(out_path, iter(records))
        if n != cfg.expected_records:
            raise ValueError(f"{cfg.name}: wrote {n} records, expected {cfg.expected_records}")
        _write_json(sidecar_path, entry)
        logger.info("aa/%s: %d records -> %s", cfg.subdir, n, out_path)

    manifest = _aa_manifest("complete")
    _write_json(manifest_path, manifest)
    logger.info("aa: all %d mandatory benchmarks are complete", len(AA_EVALS))
    return manifest


# Eval tasks excluded from the *decontamination* bloom (they remain valid
# benchmarks for actual evaluation — this only affects what we treat as
# "eval content to scrub from training data"). Each of these has test
# "documents" that are ordinary, ubiquitous corpus material rather than a
# secret answer key, so matching against them flags large volumes of
# legitimate training data with no real test-answer leakage:
#   - code2text_* (CodeXGLUE): documents are plain public GitHub functions;
#     any code corpus containing those popular files matches verbatim.
#   - jsonschema_bench_*: documents are public JSON schemas from GitHub.
#   - swde: documents are raw scraped web pages (structured web extraction).
#   - realtoxicityprompts: documents are random spans of open web text.
# Confirmed empirically as dominant false-positive drivers on code/web
# corpora (see marin#6852).
DECON_EXCLUDED_EVAL_TASKS: frozenset[str] = frozenset(
    {
        "code2text_go",
        "code2text_java",
        "code2text_javascript",
        "code2text_php",
        "code2text_python",
        "code2text_ruby",
        "jsonschema_bench_easy",
        "jsonschema_bench_medium",
        "jsonschema_bench_hard",
        "swde",
        "realtoxicityprompts",
        # Perplexity / cloze evals over public text — the "document" is ordinary
        # public material with no answer to leak (marin#6852 cluster A):
        "wikitext",  # raw Wikipedia; every web/book corpus overlaps it
        "lambada_openai",  # last-word cloze over public book passages
        "lambada_standard",
        "lambada_openai_cloze_yaml",
        "lambada_standard_cloze_yaml",
    }
)


def _lmh_task_names() -> list[str]:
    bundles: tuple[Iterable, ...] = (
        CORE_TASKS,
        MMLU_TASKS,
        KEY_GENERATION_TASKS,
        KEY_MULTIPLE_CHOICE_TASKS,
        OPEN_LM_LEADERBOARD_MCQ,
        OPEN_LM_LEADERBOARD_GEN,
        REASONING_TASKS,
        MATH_TASKS,
        LANGUAGE_TASKS,
        CODE_TASKS,
        MEDICAL_TASKS,
        KNOWLEDGE_TASKS,
        EMOTIONAL_ETHICS_TASKS,
        BIAS_SAFETY_TASKS,
        ACTION_TASKS,
        TRUTHFULNESS_TASKS,
        SPECIALIZED_TASKS,
        MGSM_MULTILINGUAL_TASKS,
        XSTORYCLOZE_MULTILINGUAL_TASKS,
        MULTILINGUAL_LM_EVAL_LOGPROB_TASKS,
    )
    names: set[str] = set()
    for bundle in bundles:
        for cfg in bundle:
            names.add(cfg.name)
    return sorted(names - DECON_EXCLUDED_EVAL_TASKS)


def _prepare_lmh() -> dict[str, Any]:
    trust_remote_code_for_hf()
    names = _lmh_task_names()
    logger.info("lmh: %d unique task names from task_configs.py", len(names))

    try:
        from lm_eval.tasks import get_task_dict  # noqa: PLC0415  # optional dep: lm_eval
    except Exception as exc:
        logger.warning("lmh: optional loader is unavailable: %s", exc)
        manifest = {
            "schema_version": 1,
            "corpus_version": EVAL_CORPUS_VERSION,
            "required": False,
            "status": "complete_with_failures",
            "artifact_root": LMH_EVALS_RELATIVE,
            "configured_tasks": names,
            "included_leaf_tasks": [],
            "artifacts": [],
            "failed": [{"task": "*", "reason": f"lm_eval import: {exc}"}],
        }
        _write_json(f"{_output_root()}/{LMH_MANIFEST_RELATIVE}", manifest)
        return manifest

    succeeded: list[str] = []
    skipped_existing: list[str] = []
    failed: list[tuple[str, str]] = []
    for name in names:
        direct_out_path = f"{_lmh_output_root()}/{name}/eval.parquet"
        direct_version_path = f"{_lmh_output_root()}/{name}/{_LMH_VERSION_SIDECAR}"
        if StoragePath(direct_out_path).exists() and _staged_lmh_version(direct_version_path) == _LMH_EXTRACTION_VERSION:
            logger.info("lmh/%s: exists (extraction %s), skipping", name, _LMH_EXTRACTION_VERSION)
            skipped_existing.append(name)
            continue

        try:
            task_dict = get_task_dict([name])
        except Exception as exc:
            logger.warning("lmh/%s: load failed: %s", name, exc)
            failed.append((name, f"load: {exc}"))
            continue

        leaves = list(flatten_task_dict(task_dict))
        if not leaves:
            logger.warning("lmh/%s: no leaf tasks after flatten", name)
            failed.append((name, "no leaf tasks"))
            continue
        if len(leaves) > 1:
            logger.info("lmh/%s: group expanded to %d leaf tasks", name, len(leaves))

        for child_name, task in leaves:
            out_path = f"{_lmh_output_root()}/{child_name}/eval.parquet"
            version_path = f"{_lmh_output_root()}/{child_name}/{_LMH_VERSION_SIDECAR}"
            # Skip only if the shard exists AND was built with the current extraction
            # policy; a version bump forces a rewrite so policy changes reach the corpus.
            if StoragePath(out_path).exists() and _staged_lmh_version(version_path) == _LMH_EXTRACTION_VERSION:
                logger.info("lmh/%s: exists (extraction %s), skipping", child_name, _LMH_EXTRACTION_VERSION)
                skipped_existing.append(child_name)
                continue

            chosen = materialize_first_nonempty_split(task)
            if chosen is None:
                logger.warning("lmh/%s: no docs in any split", child_name)
                failed.append((child_name, "no docs"))
                continue
            split, docs = chosen

            def rows(task=task, docs=docs, split=split, name=child_name) -> Iterator[dict]:
                for i, doc in enumerate(docs):
                    text = _lmh_doc_text(doc, task.doc_to_text, task.doc_to_target)
                    if not text:
                        continue
                    yield {"id": f"{name}-{split}-{i}", "text": text}

            try:
                n = _write_parquet(out_path, rows())
                with StoragePath(version_path).open("w") as vf:
                    vf.write(_LMH_EXTRACTION_VERSION)
                logger.info("lmh/%s: %d records (%s split) -> %s", child_name, n, split, out_path)
                succeeded.append(child_name)
            except Exception as exc:
                logger.warning("lmh/%s: write failed: %s", child_name, exc)
                failed.append((child_name, f"write: {exc}"))

    logger.info(
        "lmh summary: %d succeeded, %d skipped (existing), %d failed",
        len(succeeded),
        len(skipped_existing),
        len(failed),
    )
    if failed:
        for n, reason in failed:
            logger.info("  FAIL lmh/%s: %s", n, reason)

    included_leaf_tasks = sorted(set(succeeded + skipped_existing))
    artifacts: list[dict[str, Any]] = []
    for task_name in included_leaf_tasks:
        artifact = f"{task_name}/eval.parquet"
        artifact_path = f"{_lmh_output_root()}/{artifact}"
        with StoragePath(artifact_path).open("rb") as source:
            records = pq.ParquetFile(source).metadata.num_rows
        artifacts.append({"task": task_name, "artifact": artifact, "records": records})

    manifest = {
        "schema_version": 1,
        "corpus_version": EVAL_CORPUS_VERSION,
        "required": False,
        "status": "complete_with_failures" if failed else "complete",
        "artifact_root": LMH_EVALS_RELATIVE,
        "configured_tasks": names,
        "included_leaf_tasks": included_leaf_tasks,
        "artifacts": artifacts,
        "failed": [{"task": name, "reason": reason} for name, reason in failed],
    }
    _write_json(f"{_output_root()}/{LMH_MANIFEST_RELATIVE}", manifest)
    return manifest


def main() -> None:
    configure_logging(logging.INFO)
    aa_manifest = _prepare_aa()
    lmh_manifest = _prepare_lmh()
    _write_json(
        f"{_output_root()}/_manifest.json",
        {
            "schema_version": 1,
            "corpus_version": EVAL_CORPUS_VERSION,
            "artificial_analysis": aa_manifest,
            "lm_eval_harness": lmh_manifest,
        },
    )


if __name__ == "__main__":
    main()
