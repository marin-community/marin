# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare AA + lm-eval-harness eval text as decon bloom input.

Two sub-corpora written under ``gs://marin-eu-west4/datakit/decontam/evals/``:

- ``aa/<eval>/<split>.jsonl.gz`` -- AA Intelligence Index v4.0 core 8.
- ``lmh/<task>/<split>.jsonl.gz`` -- every unique task in
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

Submit on iris (eu-west4, CPU-only, has HF access):

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive \\
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
import gzip
import json
import logging
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

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

# TODO (rav): don't hardcode
OUTPUT_ROOT = "gs://marin-eu-west4/datakit/decontam/evals"


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


def _concat_strings(record: dict[str, Any]) -> str:
    """Concat all string-typed fields in sorted key order; flatten list[str]."""
    parts: list[str] = []
    for k in sorted(record.keys()):
        v = record[k]
        if isinstance(v, str) and v.strip():
            parts.append(v)
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            parts.extend(s for s in v if s.strip())
    return "\n\n".join(parts)


def _write_jsonl_gz(path: str, records: Iterator[dict]) -> int:
    fs_, resolved = url_to_fs(path)
    parent = "/".join(resolved.split("/")[:-1])
    if parent:
        fs_.makedirs(parent, exist_ok=True)
    n = 0
    with fs_.open(resolved, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as gz:
        for rec in records:
            gz.write((json.dumps(rec) + "\n").encode("utf-8"))
            n += 1
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
        import io as io_mod
        import urllib.request
        import zipfile

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
        from huggingface_hub import hf_hub_download

        for fname in cfg.hf_jsonl_files:
            local = hf_hub_download(repo_id=cfg.hf_id, filename=fname, repo_type="dataset")
            with open(local, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        return

    from datasets import Image as DatasetsImage
    from datasets import load_dataset

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
        out_path = f"{OUTPUT_ROOT}/aa/{cfg.subdir}/{cfg.split}.jsonl.gz"
        fs_, resolved = url_to_fs(out_path)
        if fs_.exists(resolved):
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

        n = _write_jsonl_gz(out_path, rows())
        logger.info("aa/%s: %d records -> %s", cfg.subdir, n, out_path)


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
    return sorted(names)


def _materialize_first_nonempty_split(task) -> tuple[str, list] | None:
    """Return (split_name, docs) for the first non-empty split, or None."""
    for split_name, getter in (
        ("test", task.test_docs),
        ("validation", task.validation_docs),
        ("training", task.training_docs),
    ):
        try:
            docs = list(getter())
        except Exception:
            docs = []
        if docs:
            return (split_name, docs)
    return None


def _flatten_task_dict(d: dict) -> Iterator[tuple[str, Any]]:
    """Yield ``(task_name, task_obj)`` pairs from a (possibly nested) ``get_task_dict`` result.

    lm-eval-harness returns groups as nested dicts keyed by ``ConfigurableGroup``
    objects (``mmlu -> mmlu_stem -> {str_name: ConfigurableTask}``); only leaf
    values are real tasks. Walk top-down and yield ``(str, ConfigurableTask)``
    pairs only. ``ConfigurableGroup`` keys at the leaf level (single-task
    groups) fall back to their ``.group`` attribute.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten_task_dict(v)
            continue
        name = k if isinstance(k, str) else getattr(k, "group", str(k))
        yield (name, v)


def _trust_remote_code_for_hf() -> None:
    """Force ``trust_remote_code=True`` on every ``datasets.load_dataset`` call.

    In ``datasets`` 4.x the ``HF_DATASETS_TRUST_REMOTE_CODE`` env var and
    ``datasets.config.HF_DATASETS_TRUST_REMOTE_CODE`` flag were removed; only
    the per-call kwarg is honored. lm-eval task configs don't pass it, so
    tasks shipping custom HF loading scripts (piqa, logiqa*, ethics_*,
    crows_pairs_*, social_iqa, ...) fail to load. Wrap ``load_dataset`` /
    ``load_dataset_builder`` to always pass it. We're only ingesting public
    eval text into a decon bloom — no model code is executed downstream.

    Patches both the ``datasets`` module attrs AND any already-imported
    ``from datasets import load_dataset`` bindings (lm-eval pulls
    ``datasets`` transitively when this module is imported).

    Also sets ``HF_ALLOW_CODE_EVAL=1`` so humaneval's ``code_eval`` metric
    initializes without an interactive disclaimer.
    """
    import sys

    import datasets

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    original_load = datasets.load_dataset
    original_builder = datasets.load_dataset_builder

    def patched_load_dataset(*args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return original_load(*args, **kwargs)

    def patched_builder(*args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return original_builder(*args, **kwargs)

    datasets.load_dataset = patched_load_dataset  # type: ignore[assignment]
    datasets.load_dataset_builder = patched_builder  # type: ignore[assignment]
    for mod in list(sys.modules.values()):
        if mod is None or mod is datasets:
            continue
        if getattr(mod, "load_dataset", None) is original_load:
            mod.load_dataset = patched_load_dataset  # type: ignore[attr-defined]
        if getattr(mod, "load_dataset_builder", None) is original_builder:
            mod.load_dataset_builder = patched_builder  # type: ignore[attr-defined]


def _prepare_lmh() -> None:
    _trust_remote_code_for_hf()
    from lm_eval.tasks import get_task_dict

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

        leaves = list(_flatten_task_dict(task_dict))
        if not leaves:
            logger.warning("lmh/%s: no leaf tasks after flatten", name)
            failed.append((name, "no leaf tasks"))
            continue
        if len(leaves) > 1:
            logger.info("lmh/%s: group expanded to %d leaf tasks", name, len(leaves))

        for child_name, task in leaves:
            out_path = f"{OUTPUT_ROOT}/lmh/{child_name}/eval.jsonl.gz"
            fs_, resolved = url_to_fs(out_path)
            if fs_.exists(resolved):
                logger.info("lmh/%s: exists, skipping", child_name)
                skipped_existing += 1
                continue

            chosen = _materialize_first_nonempty_split(task)
            if chosen is None:
                logger.warning("lmh/%s: no docs in any split", child_name)
                failed.append((child_name, "no docs"))
                continue
            split, docs = chosen

            def rows(task=task, docs=docs, split=split, name=child_name) -> Iterator[dict]:
                for i, doc in enumerate(docs):
                    try:
                        prompt = task.doc_to_text(doc) or ""
                    except Exception:
                        prompt = ""
                    try:
                        target = task.doc_to_target(doc) or ""
                    except Exception:
                        target = ""
                    parts: list[str] = []
                    if prompt:
                        parts.append(str(prompt))
                    if target:
                        parts.append(str(target))
                    if isinstance(doc, dict):
                        parts.append(_concat_strings(doc))
                    text = "\n\n".join(p for p in parts if p.strip())
                    if not text:
                        continue
                    yield {"id": f"{name}-{split}-{i}", "text": text}

            try:
                n = _write_jsonl_gz(out_path, rows())
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
