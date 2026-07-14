# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare AA + lm-eval-harness eval text as decon bloom input.

Two sub-corpora written under ``{MARIN_PREFIX}/datakit/decontam/evals/`` (the
iris ``--region`` selects the store MARIN_PREFIX resolves to):

- ``aa/<eval>/<split>.parquet`` -- AA Intelligence Index v4.0 core 8.
- ``lmh/<task>/<split>.parquet`` -- every unique task in
  ``experiments/evals/task_configs.py`` bundles, loaded via lm-eval-harness.
  Group names (``mmlu``, ``agieval``, ``bbh_zeroshot``, ``leaderboard_bbh``,
  ...) are expanded to their leaf tasks; one file per leaf.

Each record: ``{id: str, text: str}`` where ``text`` concatenates every
string-typed field of the source row in deterministic key order. Generic
extraction (no per-eval schema config) trades a bit of noise for uniform
treatment of arbitrary HF / lm-eval schemas.

Test split is preferred; tasks without a test split fall back to
validation, then training. Tasks that fail to load (e.g. removed from
lm-eval since our pinned commit, gated HF datasets) are logged and skipped.

Submit on iris (eu-west4, CPU-only, has HF access). The ``eval`` extra is
needed for ``lm-eval``; AA prep accumulates rows in memory before writing
so bump RAM (1GB default OOMs on livecodebench):

    uv run iris --cluster=marin job run --region europe-west4 \\
        --extra=cpu --extra=eval --priority interactive \\
        --memory 16GB --cpu 2 --enable-extra-resources \\
        -- python experiments/datakit/decontam/prepare_eval_corpus.py

The iris worker pulls lm-eval via the marin image's ``eval`` extras. To
include ifeval / leaderboard_ifeval we depend on ``lm-eval[ifeval]`` so
``langdetect`` is available; see ``lib/marin/pyproject.toml`` (the ``eval``
extra). The script monkey-patches ``datasets.load_dataset`` to force
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
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image as DatasetsImage
from datasets import load_dataset
from huggingface_hub import hf_hub_download
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

_EVALS_RELATIVE = "datakit/decontam/evals"


def _output_root() -> str:
    """Eval-corpus write root, relative to the active ``MARIN_PREFIX`` (store-agnostic)."""
    return f"{marin_prefix()}/{_EVALS_RELATIVE}"


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


# AA Intelligence Index v4.0 core (8 text benchmarks). Each entry pins
# the HF source + the canonical "eval content" fields. ``text_fields`` are
# string columns concatenated in order; ``list_fields`` are list<string>
# columns flattened in order. When both are empty the loop falls back to
# the generic _concat_strings extractor (for schemas not yet pinned).
@dataclasses.dataclass(frozen=True)
class AAEvalConfig:
    subdir: str
    hf_id: str
    subset: str | None
    split: str
    text_fields: tuple[str, ...] = ()
    list_fields: tuple[str, ...] = ()
    skip_if: Callable[[dict], bool] | None = None
    # When set, bypass `datasets.load_dataset` and pull raw jsonl files via
    # ``huggingface_hub.hf_hub_download``. Needed for repos that ship a
    # Python loading script (deprecated in datasets>=4) but commit usable
    # jsonl alongside it (e.g. livecodebench).
    hf_jsonl_files: tuple[str, ...] = ()
    # When set, download a zip from an HTTP URL and stream a jsonl member
    # inside it. Used for evals not on the HF Hub (e.g. scicode lives on
    # the project's GitHub Pages repo).
    download_zip_url: str | None = None
    zip_jsonl_member: str | None = None


AA_EVALS: tuple[AAEvalConfig, ...] = (
    # Humanity's Last Exam: text-only subset; skip multimodal rows.
    AAEvalConfig(
        subdir="hle",
        hf_id="cais/hle",
        subset=None,
        split="test",
        text_fields=("question", "answer"),
        skip_if=lambda r: bool(r.get("image")) or bool(r.get("image_url")),
    ),
    AAEvalConfig(
        subdir="aa_omniscience",
        hf_id="ArtificialAnalysis/AA-Omniscience-Public",
        subset=None,
        split="train",
        text_fields=("question", "answer"),
    ),
    # IFBench is instruction-following; the prompt IS the eval content.
    # The dataset is named IFBench_test so the only split is "train".
    AAEvalConfig(
        subdir="ifbench",
        hf_id="allenai/IFBench_test",
        subset=None,
        split="train",
        text_fields=("prompt",),
    ),
    # GPQA: HF schema uses Title-Case field names; filter to the diamond subset.
    AAEvalConfig(
        subdir="gpqa_diamond",
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
        subdir="mmlu_pro",
        hf_id="TIGER-Lab/MMLU-Pro",
        subset=None,
        split="test",
        text_fields=("question",),
        list_fields=("options",),
    ),
    # SciCode lives on GitHub Pages, not the HF Hub. The data zip ships
    # ``data/problems_all.jsonl`` with one record per problem (fields:
    # problem_name, problem_id, problem_description_main, sub_steps, ...).
    AAEvalConfig(
        subdir="scicode",
        hf_id="scicode-bench/SciCode",  # informational; real source is download_zip_url
        subset=None,
        split="test",
        text_fields=("problem_description_main",),
        download_zip_url="https://raw.githubusercontent.com/scicode-bench/scicode-bench.github.io/main/data/data.zip",
        zip_jsonl_member="data/problems_all.jsonl",
    ),
    # GDPval rows are ~20KB each: large rubric JSON + URLs to deliverable files.
    # Pin to ``prompt`` (the actual task description) to avoid polluting the bloom.
    AAEvalConfig(
        subdir="gdpval",
        hf_id="openai/gdpval",
        subset=None,
        split="train",
        text_fields=("prompt",),
    ),
    # LiveCodeBench commits its eval items as plain jsonl at repo root; the
    # `code_generation_lite.py` loader is deprecated. test6.jsonl is the
    # latest version per the dataset README.
    AAEvalConfig(
        subdir="livecodebench",
        hf_id="livecodebench/code_generation_lite",
        subset=None,
        split="test",  # informational; real source is hf_jsonl_files
        text_fields=("question_content", "starter_code"),
        hf_jsonl_files=("test6.jsonl",),
    ),
)


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
    """Stream raw rows for one AA eval. Three loaders, picked in order:

    1. ``download_zip_url`` + ``zip_jsonl_member`` -- HTTP zip with a jsonl
       member (e.g. scicode on GitHub Pages).
    2. ``hf_jsonl_files`` -- raw jsonl files via huggingface_hub
       (e.g. livecodebench, which ships a deprecated loading script).
    3. ``datasets.load_dataset`` (default) -- the normal HF Hub path.
    """
    if cfg.download_zip_url:
        if not cfg.zip_jsonl_member:
            raise ValueError(f"aa/{cfg.subdir}: zip_jsonl_member must be set with download_zip_url")
        with urllib.request.urlopen(cfg.download_zip_url) as resp:
            data = resp.read()
        with zipfile.ZipFile(io_mod.BytesIO(data)) as zf, zf.open(cfg.zip_jsonl_member) as jf:
            for line in jf:
                if line.strip():
                    yield json.loads(line)
        return

    if cfg.hf_jsonl_files:
        for fname in cfg.hf_jsonl_files:
            local = hf_hub_download(repo_id=cfg.hf_id, filename=fname, repo_type="dataset")
            with open(local, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        return

    ds = load_dataset(cfg.hf_id, name=cfg.subset, split=cfg.split)
    # Disable Image-feature decoding so iteration doesn't pull in Pillow on
    # multimodal datasets (e.g. HLE) -- we filter out those rows via skip_if
    # without ever touching the bytes.
    features = getattr(ds, "features", None) or {}
    for col, ftype in features.items():
        if isinstance(ftype, DatasetsImage):
            ds = ds.cast_column(col, DatasetsImage(decode=False))
    for row in ds:
        yield dict(row)


def _prepare_aa() -> None:
    for cfg in AA_EVALS:
        out_path = f"{_output_root()}/aa/{cfg.subdir}/{cfg.split}.parquet"
        if StoragePath(out_path).exists():
            logger.info("aa/%s: exists, skipping", cfg.subdir)
            continue
        try:
            raw_rows = list(_iter_aa_rows(cfg))
        except Exception as exc:
            logger.warning(
                "aa/%s: load(%s subset=%s split=%s) failed: %s",
                cfg.subdir,
                cfg.hf_id,
                cfg.subset,
                cfg.split,
                exc,
            )
            continue

        def rows(raw_rows=raw_rows, cfg=cfg) -> Iterator[dict]:
            n_skipped = 0
            for i, row_dict in enumerate(raw_rows):
                if cfg.skip_if is not None and cfg.skip_if(row_dict):
                    n_skipped += 1
                    continue
                text = _extract_aa_text(row_dict, cfg)
                if not text:
                    n_skipped += 1
                    continue
                yield {"id": f"{cfg.hf_id}-{cfg.split}-{i}", "text": text}
            if n_skipped:
                logger.info("aa/%s: skipped %d rows", cfg.subdir, n_skipped)

        n = _write_parquet(out_path, rows())
        logger.info("aa/%s: %d records -> %s", cfg.subdir, n, out_path)


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


def _prepare_lmh() -> None:
    trust_remote_code_for_hf()
    from lm_eval.tasks import get_task_dict  # noqa: PLC0415  # optional dep: lm_eval

    names = _lmh_task_names()
    logger.info("lmh: %d unique task names from task_configs.py", len(names))

    succeeded = 0
    skipped_existing = 0
    failed: list[tuple[str, str]] = []
    for name in names:
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
            out_path = f"{_output_root()}/lmh/{child_name}/eval.parquet"
            version_path = f"{_output_root()}/lmh/{child_name}/{_LMH_VERSION_SIDECAR}"
            # Skip only if the shard exists AND was built with the current extraction
            # policy; a version bump forces a rewrite so policy changes reach the corpus.
            if StoragePath(out_path).exists() and _staged_lmh_version(version_path) == _LMH_EXTRACTION_VERSION:
                logger.info("lmh/%s: exists (extraction %s), skipping", child_name, _LMH_EXTRACTION_VERSION)
                skipped_existing += 1
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
                succeeded += 1
            except Exception as exc:
                logger.warning("lmh/%s: write failed: %s", child_name, exc)
                failed.append((child_name, f"write: {exc}"))

    logger.info(
        "lmh summary: %d succeeded, %d skipped (existing), %d failed",
        succeeded,
        skipped_existing,
        len(failed),
    )
    if failed:
        for n, reason in failed:
            logger.info("  FAIL lmh/%s: %s", n, reason)


def main() -> None:
    configure_logging(logging.INFO)
    _prepare_aa()
    _prepare_lmh()


if __name__ == "__main__":
    main()
