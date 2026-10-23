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
    return source.replace(ORIGINAL, replacement)


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


def install_checkpointing(commit, chunk_size=8, math_prefix=False):
    module = importlib.import_module("eval.task")
    source_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if PINNED_SOURCES.get(commit) != source_hash:
        raise ValueError("Evalchemy source differs from the explicitly pinned checkpoint patch")
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
        json.dumps(dict(commit=commit, module_sha256=source_hash, chunk_size=chunk_size, math_prefix=math_prefix)),
        flush=True,
    )


def main():
    tasks = sys.argv[sys.argv.index("--tasks") + 1].split(",")
    install_checkpointing(os.environ["LEVANTER267_EVALCHEMY_COMMIT"], math_prefix="MATH500" in tasks)
    command = next(ep for ep in importlib.metadata.entry_points(group="console_scripts") if ep.name == "evalchemy")
    return command.load()()


if __name__ == "__main__":
    raise SystemExit(main())
