# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared lm-eval-harness loading helpers for the decontam corpus prep.

Three pieces of glue used by both ``prepare_eval_corpus.py`` (writes the
corpus) and ``verify_lmh_tasks.py`` (previews the fail set):

* ``trust_remote_code_for_hf`` -- monkey-patch ``datasets.load_dataset`` /
  ``load_dataset_builder`` to inject ``trust_remote_code=True``. Necessary
  on ``datasets`` 4.x where the env var and config flag were removed.
* ``flatten_task_dict`` -- walk the (possibly nested) result of
  ``lm_eval.tasks.get_task_dict`` and yield only ``(str_name, Task)`` leaf
  pairs. lm-eval-harness returns groups as nested dicts keyed by
  ``ConfigurableGroup`` objects (``mmlu -> mmlu_stem -> ...``).
* ``materialize_first_nonempty_split`` -- pick the first split that yields
  any docs, preferring ``test`` then ``validation`` then ``training``.

Importing this module pulls only ``datasets`` + stdlib, NOT
``experiments.evals.task_configs`` (which transitively imports torch).
That keeps both callers cheap to import in a torch-less environment.
"""

import logging
import os
import sys
from collections.abc import Iterator
from typing import Any

import datasets

logger = logging.getLogger(__name__)


def trust_remote_code_for_hf() -> None:
    """Try ``trust_remote_code=True`` on every ``datasets.load_dataset`` call, fall back without on kwarg rejection.

    In ``datasets`` 4.x the ``HF_DATASETS_TRUST_REMOTE_CODE`` env var and
    ``datasets.config.HF_DATASETS_TRUST_REMOTE_CODE`` flag were removed; only
    the per-call kwarg is honored. lm-eval task configs don't pass it, so
    custom-loader datasets (piqa, logiqa*, ethics_*, crows_pairs_*, ...)
    fail without it. But many other lm-eval tasks resolve to built-in
    csv / parquet / json builders that DON'T accept ``trust_remote_code``
    -- forcing the kwarg unconditionally breaks them with
    ``BuilderConfig X doesn't have a 'trust_remote_code' key``.

    Strategy: attempt with ``trust_remote_code=True`` first; on the specific
    "doesn't have a 'trust_remote_code' key" error, retry without. Any other
    error propagates as-is. We're only ingesting public eval text into a
    decon bloom -- no model code is executed downstream.

    Patches both the ``datasets`` module attrs AND any already-imported
    ``from datasets import load_dataset`` bindings (lm-eval pulls
    ``datasets`` transitively).

    Also sets ``HF_ALLOW_CODE_EVAL=1`` so humaneval's ``code_eval`` metric
    initializes without an interactive disclaimer.
    """
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # transformers>=5 removed ``AutoModelForVision2Seq``, but the pinned lm-eval
    # fork still does ``from transformers import AutoModelForVision2Seq`` at module
    # load. We only use lm-eval's task loading (not its model classes), so alias a
    # placeholder so the import resolves — it is imported but never instantiated.
    import transformers  # noqa: PLC0415

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = transformers.AutoModel  # type: ignore[attr-defined]

    original_load = datasets.load_dataset
    original_builder = datasets.load_dataset_builder

    def _is_trc_rejection(exc: BaseException) -> bool:
        # Builders that don't accept the kwarg raise either TypeError or
        # ValueError; the message reliably mentions ``'trust_remote_code'``.
        return isinstance(exc, (TypeError, ValueError)) and "trust_remote_code" in str(exc)

    def patched_load_dataset(*args, **kwargs):
        if "trust_remote_code" in kwargs:
            return original_load(*args, **kwargs)
        try:
            return original_load(*args, trust_remote_code=True, **kwargs)
        except Exception as exc:
            if _is_trc_rejection(exc):
                return original_load(*args, **kwargs)
            raise

    def patched_builder(*args, **kwargs):
        if "trust_remote_code" in kwargs:
            return original_builder(*args, **kwargs)
        try:
            return original_builder(*args, trust_remote_code=True, **kwargs)
        except Exception as exc:
            if _is_trc_rejection(exc):
                return original_builder(*args, **kwargs)
            raise

    datasets.load_dataset = patched_load_dataset  # type: ignore[assignment]
    datasets.load_dataset_builder = patched_builder  # type: ignore[assignment]
    for mod in list(sys.modules.values()):
        if mod is None or mod is datasets:
            continue
        if getattr(mod, "load_dataset", None) is original_load:
            mod.load_dataset = patched_load_dataset  # type: ignore[attr-defined]
        if getattr(mod, "load_dataset_builder", None) is original_builder:
            mod.load_dataset_builder = patched_builder  # type: ignore[attr-defined]


def flatten_task_dict(d: dict) -> Iterator[tuple[str, Any]]:
    """Yield ``(task_name, task_obj)`` pairs from a (possibly nested) ``get_task_dict`` result.

    lm-eval-harness returns groups as nested dicts keyed by ``ConfigurableGroup``
    objects (``mmlu -> mmlu_stem -> {str_name: ConfigurableTask}``); only leaf
    values are real tasks. Walk top-down and yield ``(str, ConfigurableTask)``
    pairs only. ``ConfigurableGroup`` keys at the leaf level (single-task
    groups) fall back to their ``.group`` attribute.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten_task_dict(v)
            continue
        name = k if isinstance(k, str) else getattr(k, "group", str(k))
        yield (name, v)


def materialize_first_nonempty_split(task) -> tuple[str, list] | None:
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
