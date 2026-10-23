# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual pinned FineStore manager, without an inference allocation."""

import ast
import copy
import hashlib
import importlib.util
import logging
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest

from experiments.weight_merging.rsi.levanter_267.checkpoint_entrypoint import (
    PINNED_SOURCES,
    checkpoint_client_source,
    patched_resume_source,
    selected_prefix_results,
)


def _method(source):
    namespace = dict(List=list, Optional=Optional, LM=object, Instance=object, SampleEntry=object)
    exec(compile(source, "<pinned-resume-test>", "exec"), namespace)
    return namespace["_generate_with_resume"]


def test_checkpoint_commits_eight_then_restores_all_500_in_original_order(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("eval")
    if spec is None:
        pytest.skip("Install the pinned Evalchemy package without dependencies for this experiment regression")
    root = Path(spec.origin).parent
    task_source = (root / "task.py").read_text()
    if hashlib.sha256(task_source.encode()).hexdigest() not in PINNED_SOURCES.values():
        pytest.skip("This regression requires one of the two exact Evalchemy pins")

    # Only the unused result-JSON fallback needs lm-eval. Assert it stays unused;
    # persistence, fingerprints, serialization and reads use the real package.
    if importlib.util.find_spec("lm_eval") is None:
        utils = ModuleType("lm_eval.utils")

        def unused_formatter(value):
            raise AssertionError(f"Unexpected result formatter call: {type(value)}")

        utils.handle_non_serializable = unused_formatter
        monkeypatch.setitem(sys.modules, "lm_eval", ModuleType("lm_eval"))
        monkeypatch.setitem(sys.modules, "lm_eval.utils", utils)
    resume = pytest.importorskip("eval.contracts.finestore_resume")
    fingerprints = pytest.importorskip("eval.resume.fingerprint")
    completions = pytest.importorskip("eval.completion_response")
    module = ast.parse(task_source)
    benchmark = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "BaseBenchmark")
    method = next(
        node for node in benchmark.body if isinstance(node, ast.FunctionDef) and node.name == "_generate_with_resume"
    )
    source = textwrap.dedent("\n".join(task_source.splitlines()[method.lineno - 1 : method.end_lineno]))

    class Entry:
        def __init__(self, idx):
            self.idx = idx

        def resume_unit(self, task):
            return dict(task=task, sample_id=f"sample-{self.idx}")

        def to_dict(self):
            return dict(sample_id=f"sample-{self.idx}", source_id=self.idx, ordinal=self.idx)

    class Manifest:
        def validate_prior_entries(self, payloads, entries):
            allowed = {entry.idx for entry in entries}
            assert all(payload["sample"]["source_id"] in allowed for payload in payloads)

        def validate_output_count(self, count, outputs):
            assert len(outputs) == count

        def mark_generated(self, entries, outputs):
            assert len(entries) == len(outputs) == 500

    class Model:
        rank = 0

        def __init__(self, fail_after=None):
            self.fail_after = fail_after
            self.emitted = 0
            self.calls = []

        def generate_until(self, instances):
            self.calls.append(list(instances))
            outputs = []
            for instance in instances:
                if self.emitted == self.fail_after:
                    raise RuntimeError("forced interruption after eight completed responses")
                text = f"answer-{instance.idx}"
                response = completions.CompletionResponse(
                    content=text,
                    reasoning_content=None,
                    finish_reason="stop",
                    usage=dict(completion_tokens=17),
                    provider_metadata={},
                    raw_choice={},
                )
                outputs.append(completions.CompletionText(text, response, "combine"))
                self.emitted += 1
            return outputs

    entries = [Entry(idx) for idx in range(500)]
    instances = [
        SimpleNamespace(idx=idx, args=(f"prompt-{idx}", dict(max_tokens=31574, temperature=1, seed=None)))
        for idx in range(500)
    ]
    original_args = copy.deepcopy([instance.args for instance in instances])
    fingerprint = fingerprints.RunFingerprint(inputs=dict(task_name="MATH500", model_repo="test", seed_set=[19311] * 4))

    def manager(path):
        return resume.FineStoreResumeManager(str(path), "math500", "MATH500", fingerprint)

    def run(method_source, store, model):
        owner = SimpleNamespace(
            _resume_manager=store,
            _suspend_resume=False,
            sample_manifest=Manifest(),
            logger=logging.getLogger(__name__),
            _unit_key=lambda task, instance: entries[instance.idx].resume_unit(task),
        )
        return _method(method_source)(owner, model, "MATH500", instances, entries)

    # The original implementation loses all eight completed responses.
    unpatched = manager(tmp_path / "unpatched")
    with pytest.raises(RuntimeError, match="forced interruption"):
        run(source, unpatched, Model(fail_after=8))
    unpatched.finalize()
    assert manager(tmp_path / "unpatched").restore() == {}

    patched = patched_resume_source(source, 8)
    store = manager(tmp_path / "patched")
    with pytest.raises(RuntimeError, match="forced interruption"):
        run(patched, store, Model(fail_after=8))
    # A fresh manager sees the committed chunk without finalizing the failed run.
    reopened = manager(tmp_path / "patched")
    assert len(reopened.restore()) == 8
    remaining = Model()
    outputs = run(patched, reopened, remaining)
    assert [instance.idx for call in remaining.calls for instance in call] == list(range(8, 500))
    assert outputs == [f"answer-{idx}" for idx in range(500)]
    assert all(output.response.finish_reason == "stop" for output in outputs)
    assert all(output.response.usage["completion_tokens"] == 17 for output in outputs)
    final = manager(tmp_path / "patched").restore()
    assert len(final) == 500
    assert sorted(payload["sample"]["source_id"] for payload in final.values()) == list(range(500))
    assert [instance.args for instance in instances] == original_args


def test_checkpoint_patch_refuses_an_unrecognized_resume_function():
    with pytest.raises(ValueError, match="Unexpected pinned"):
        patched_resume_source("def changed(): pass", 8)


def test_client_wrapper_preserves_all_evaluator_arguments():
    source = 'cmd = [str(Path(python).with_name("evalchemy")), "--model", "local-chat-completions", "--seed", "19311"]'
    namespace = dict(
        Path=Path,
        python="/pinned/bin/python",
        os=SimpleNamespace(environ={"LEVANTER267_EVAL_ENTRYPOINT": "/job/wrapper.py"}),
    )
    original = dict(namespace)
    exec(source, original)
    exec(checkpoint_client_source(source), namespace)
    assert namespace["cmd"][:2] == ["/pinned/bin/python", "/job/wrapper.py"]
    assert namespace["cmd"][2:] == original["cmd"][1:]


def test_selected_prefix_bookkeeping_avoids_the_known_limit_grader_keyerror():
    examples = [dict(id=idx, answer="correct") for idx in range(500)]
    for example in examples[:8]:
        example["model_answer"] = "correct"
    results = dict(examples=examples)

    def score(unchanged_examples):
        return sum(example["model_answer"] == example["answer"] for example in unchanged_examples)

    with pytest.raises(KeyError, match="model_answer"):
        score(results["examples"])
    selected = selected_prefix_results(results, 8.0)
    assert [example["id"] for example in selected["examples"]] == list(range(8))
    assert score(selected["examples"]) == 8
    assert selected_prefix_results(results, None) is results
