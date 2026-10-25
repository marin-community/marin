# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded-run patches for the two explicitly pinned Evalchemy revisions.

Already materialized Instances and the unchanged grader retain their original
order. Request scheduling has an eight-item barrier; unseeded samples need not
match an unpatched run byte-for-byte. MATH500's generated prefix is passed to
its unchanged grader when the owner selects a dataset prefix with --limit.
"""

import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import numbers
import os
import sys
import textwrap
from pathlib import Path

PINNED_SOURCES = {
    "3bb8278078e7ad4b5a6b1f81e8a08db3dd206ae6": "82f8c88a8d2d680f946afded81c3f495883c7af1b7336c227aa89ea55ac42da4",
    "c1d5c62cdc3939b42f64411af44fd50716b53454": "709a4c766f2c3136f26f8eb06758202e841a472820592a361c9f38e650851257",
}
MATH500_SOURCES = {
    "3bb8278078e7ad4b5a6b1f81e8a08db3dd206ae6": "ea71e09800dd3b801a8298bea4ae573e0a232e2ae753f144d73938b63278cd7e",
    "c1d5c62cdc3939b42f64411af44fd50716b53454": "5d0561bb9d8e243416b1dd353fc4a139d9643fc88a1a13e77a3c0a0b73ae8612",
}

ORIGINAL = """    if remaining_instances:
        new_outputs = model.generate_until(remaining_instances)
        self.sample_manifest.validate_output_count(len(remaining_instances), new_outputs)
        for pos, inst, output in zip(remaining_positions, remaining_instances, new_outputs):
            results[pos] = output
            manager.record(
                keys[pos],
                {"output": output, "sample": entries[pos].to_dict()},
            )
"""


def checkpoint_client_source(source):
    marker = 'str(Path(python).with_name("evalchemy")),'
    if source.count(marker) != 1:
        raise ValueError("Unexpected historical evaluator command; refusing checkpoint wrapper")
    return source.replace(marker, 'str(python), os.environ["LEVANTER267_EVAL_ENTRYPOINT"],')


def patched_resume_source(source, chunk_size):
    source = textwrap.dedent(source)
    if chunk_size < 1 or source.count(ORIGINAL) != 1:
        raise ValueError("Unexpected pinned resume function or invalid checkpoint chunk size")
    replacement = f"""    if remaining_instances:
        for chunk_start in range(0, len(remaining_instances), {chunk_size}):
            chunk_instances = remaining_instances[chunk_start:chunk_start + {chunk_size}]
            chunk_positions = remaining_positions[chunk_start:chunk_start + {chunk_size}]
            new_outputs = model.generate_until(chunk_instances)
            for output in new_outputs:
                if (not isinstance(output, str) or not output.strip()
                    or getattr(output, "failure_category", None)
                    or getattr(getattr(output, "response", None), "failure_category", None)):
                    raise RuntimeError("Empty or failed generation; refusing checkpoint and scoring")
            self.sample_manifest.validate_output_count(len(chunk_instances), new_outputs)
            for pos, inst, output in zip(chunk_positions, chunk_instances, new_outputs):
                results[pos] = output
                manager.record(
                    keys[pos],
                    {{"output": output, "sample": entries[pos].to_dict()}},
                )
            # record() otherwise buffers32 units. Closing commits this chunk;
            # the manager reopens a writer while retaining its namespace/keys.
            manager.finalize()
"""
    patched = source.replace(ORIGINAL, replacement)
    validation = """if (not isinstance(output, str) or not output.strip()
    or getattr(output, "failure_category", None)
    or getattr(getattr(output, "response", None), "failure_category", None)):
    raise RuntimeError("Empty or failed restored generation; refusing scoring")
"""
    restored_assignment = '            results[pos] = payload["output"]'
    if patched.count(restored_assignment) != 1:
        raise ValueError("Unexpected pinned restore boundary")
    patched = patched.replace(
        restored_assignment,
        restored_assignment + "\n" + textwrap.indent("output = results[pos]\n" + validation, " " * 12),
    )
    before_scoring = "    self.sample_manifest.mark_generated(entries, results)"
    if patched.count(before_scoring) != 1:
        raise ValueError("Unexpected pinned pre-scoring boundary")
    return patched.replace(
        before_scoring,
        "    for output in results:\n" + textwrap.indent(validation, " " * 8) + before_scoring,
    )


def selected_prefix_results(results, limit):
    """Match the scorer's examples to the prefix already selected by --limit."""
    if results is None or limit is None:
        return results
    if results.get("pass_at_k"):
        raise ValueError("Selected-prefix bookkeeping is validated for single-sample MATH500 only")
    if int(limit) != limit or limit < 1:
        raise ValueError("Selected-prefix limit must be a positive integer")
    limit = int(limit)
    examples = results["examples"]
    selected = examples[:limit]
    if any("model_answer" not in example for example in selected) or any(
        "model_answer" in example for example in examples[limit:]
    ):
        raise ValueError("Generated MATH500 examples do not match the requested dataset prefix")
    return dict(results, examples=selected)


NLL_NATIVE_SHA256 = "683151acfdfef0991224081eb4cb1c1f666d0789cabcb69403441c8c937c046a"


def valid_nll_output(output):
    """Reject transport failures and nonfinite scores before commit or restore."""
    if (
        not isinstance(output, (list, tuple))
        or len(output) != 2
        or not isinstance(output[0], numbers.Real)
        or isinstance(output[0], bool)
        or not math.isfinite(output[0])
        or not isinstance(output[1], bool)
    ):
        raise RuntimeError("Invalid loglikelihood result; refusing checkpoint and scoring")
    return float(output[0]), output[1]


def checkpoint_loglikelihood(owner, requests, plan_batch, find_restored, *args, chunk_size=256, **kwargs):
    """Commit choice-level NLL requests without changing requests, order or grading."""
    manager = owner._manager
    if manager is None:
        raise RuntimeError("NLL checkpointing requires the pinned FineStore manager")
    entries, unique = plan_batch(owner._sample_manifest, requests)
    restored = manager.restore()
    owner._sample_manifest.validate_prior_entries(list(restored.values()), unique)
    keys = []
    for request, entry in zip(requests, entries, strict=True):
        if len(request.args) != 2 or not all(isinstance(arg, str) for arg in request.args):
            raise ValueError("Only materialized context/continuation loglikelihood requests are supported")
        identity = json.dumps(dict(args=request.args, choice_index=request.idx), sort_keys=True, allow_nan=False)
        key = dict(
            task=owner._task_name,
            request_type="loglikelihood",
            request_sha256=hashlib.sha256(identity.encode()).hexdigest(),
            sample_id=entry.unit_id if entry is not None else None,
        )
        keys.append(key)
    results = [None] * len(requests)
    remaining = []
    matched_keys = set()
    for pos, key in enumerate(keys):
        match = find_restored(restored, [key])
        if match is not None:
            matched_keys.add(tuple(sorted(match[0].items())))
            results[pos] = valid_nll_output(match[1]["output"])
        else:
            remaining.append(pos)
    if set(restored) != matched_keys:
        raise ValueError("Persisted NLL request identities differ from the materialized request batch")
    for start in range(0, len(remaining), chunk_size):
        positions = remaining[start : start + chunk_size]
        outputs = owner.lm.loglikelihood([requests[pos] for pos in positions], *args, **kwargs)
        owner._sample_manifest.validate_output_count(len(positions), outputs)
        outputs = [valid_nll_output(output) for output in outputs]
        for pos, output in zip(positions, outputs, strict=True):
            results[pos] = output
            if entries[pos] is not None:
                manager.record(keys[pos], dict(output=output, sample=entries[pos].to_dict()))
        manager.finalize()
        print(
            "NLL_CHECKPOINT_COMMITTED",
            json.dumps(
                dict(
                    new_requests=len(positions),
                    restored_requests=len(requests) - len(remaining),
                    completed_requests=len(requests) - len(remaining) + start + len(positions),
                    total_requests=len(requests),
                )
            ),
            flush=True,
        )
    owner._sample_manifest.mark_generated(unique, [None] * len(unique))
    return results


def install_nll_checkpointing():
    native = importlib.import_module("eval.resume.lm_eval_native")
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != NLL_NATIVE_SHA256:
        raise ValueError("Pinned native lm-eval resume source differs; refusing NLL patch")
    original = native._track_request_method

    @functools.wraps(original)
    def track(owner, method_name, requests, *args, **kwargs):
        if method_name != "loglikelihood":
            return original(owner, method_name, requests, *args, **kwargs)
        return checkpoint_loglikelihood(
            owner, requests, native._plan_request_batch, native.find_restored_payload, *args, **kwargs
        )

    native._track_request_method = track


def install_checkpointing(commit, chunk_size=8, math_prefix=False):
    module = importlib.import_module("eval.task")
    source_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if PINNED_SOURCES.get(commit) != source_hash:
        raise ValueError("Evalchemy source differs from the explicitly pinned checkpoint patch")
    install_nll_checkpointing()
    source = patched_resume_source(inspect.getsource(module.BaseBenchmark._generate_with_resume), chunk_size)
    namespace = {}
    exec(compile(source, "<levanter267-checkpoint-only>", "exec"), module.__dict__, namespace)
    module.BaseBenchmark._generate_with_resume = namespace["_generate_with_resume"]
    if math_prefix:
        find_classes = module.find_custom_benchmark_classes

        @functools.wraps(find_classes)
        def find_selected_classes(benchmark_module, base):
            # The registry executes a fresh module rather than using sys.modules.
            if benchmark_module.__name__ == "eval.chat_benchmarks.MATH500.eval_instruct":
                if hashlib.sha256(Path(benchmark_module.__file__).read_bytes()).hexdigest() != MATH500_SOURCES[commit]:
                    raise ValueError("MATH500 source differs from the explicitly pinned prefix bookkeeping patch")
                generate = benchmark_module.MATH500Benchmark.generate_responses

                @functools.wraps(generate)
                def generate_selected(self, model):
                    return selected_prefix_results(generate(self, model), self.evaluation_limit)

                benchmark_module.MATH500Benchmark.generate_responses = generate_selected
            return find_classes(benchmark_module, base)

        module.find_custom_benchmark_classes = find_selected_classes
    print(
        "BOUNDED_EVALUATOR_PATCH",
        json.dumps(
            dict(
                commit=commit,
                module_sha256=source_hash,
                chunk_size=chunk_size,
                nll_chunk_size=256,
                math_prefix=math_prefix,
            )
        ),
        flush=True,
    )


def main():
    tasks = sys.argv[sys.argv.index("--tasks") + 1].split(",")
    install_checkpointing(os.environ["LEVANTER267_EVALCHEMY_COMMIT"], math_prefix="MATH500" in tasks)
    command = next(ep for ep in importlib.metadata.entry_points(group="console_scripts") if ep.name == "evalchemy")
    return command.load()()


if __name__ == "__main__":
    raise SystemExit(main())
