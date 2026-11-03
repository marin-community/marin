# borrowed and extended from
# https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py

import os
import threading

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
from tqdm import tqdm

from eval.graders import livecodebench as _lcb_grader
from eval.runtime.bounded_process import bounded_process
from eval.chat_benchmarks.LiveBench.livebench.lcb_runner.evaluation.testing_util import run_test
from eval.chat_benchmarks.LiveBench.livebench.lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results

_logger = logging.getLogger(__name__)

# Wall-clock bound per generation, shared with the bounded grader and independent of the
# test count (the earlier wrapper scaled it with the test count instead).
DEFAULT_DEADLINE = _lcb_grader.DEFAULT_DEADLINE


def _worker_run(sample, generation, debug, connection, timeout):
    """Grade one generation and report its test outcomes and metadata."""
    _lcb_grader.apply_memory_cap(_lcb_grader.DEFAULT_MAX_MEMORY_BYTES)
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    try:
        connection.send((res, metadata))
    finally:
        connection.close()


def check_correctness(sample, generation, timeout, debug=True, deadline=DEFAULT_DEADLINE):
    """Check correctness of code generation with a wall-clock deadline independent of the test count.

    Replaces the issue #147 unbounded Manager+Process pattern: a pipe carries the
    (res, metadata) result, the child's rlimit is applied best-effort, and the parent
    watchdogs the child to a fixed deadline (not ``(timeout + 1) * N + 5``).
    """
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = bounded_process(
        target=_worker_run,
        args=(sample, generation, debug, sender, timeout),
    )
    process.start()
    sender.close()
    pid = process.pid
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("lcb_runner check pid=%s deadline=%.2f", pid, deadline)

    watchdog = threading.Timer(deadline, _lcb_grader.terminate, args=(pid,))
    watchdog.daemon = True
    watchdog.start()

    process.join(timeout=deadline + 1.0)
    watchdog.cancel()
    if process.is_alive():
        _lcb_grader.terminate(pid)
        process.join(timeout=1.0)

    if receiver.poll():
        try:
            res, metadata = receiver.recv()
        except (EOFError, OSError):
            res, metadata = None, {}
    else:
        res, metadata = None, {}
    receiver.close()

    if not res:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        res = [-1 for _ in in_outs["inputs"]]
        metadata = metadata or {}
        if debug:
            print("global timeout")
    return res, metadata


def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness(sample, o, timeout=timeout, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases {curr_res=}\n")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            # break
            curr_metadata = {}
        finally:
            assert isinstance(curr_res, list)
            assert isinstance(curr_metadata, dict)
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            print("Sample\n")
            print(r)
            print("\n")
            print("Result\n")
            print(res[i])
            print("*" * 30 + "\n\n")
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset

    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
            futures = {executor.submit(evaluate_generations_by_problem, arg): index for arg, index in inputs}

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    assert len(results) == len(inputs), f"results = {len(results)} inputs = {len(inputs)} {results=}"
    # results = {i: r for r, (_, i) in zip(results, inputs)}

    return results, metadata


def codegen_metrics(
    samples,
    generations,
    k_list=[1, 5],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    results, metadata = evaluate_generations(
        samples,
        generations,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadata.keys())):
        final_metadata.append(metadata[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(generations[0]), f"{len(final_metadata[i])=}"

    return metrics, results, final_metadata
