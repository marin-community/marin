# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU execution functions with gang scheduling and retry logic."""

import logging
from collections.abc import Callable, Sequence

import ray
from ray.dag import FunctionNode
from ray.exceptions import RayError
from ray.remote_function import RemoteFunction

from fray.cluster.ray.tpu.orchestration import (
    SlicePoolManager,
    _cancel_tasks_and_wait,
    _start_fn_on_slice,
    _validate_num_slices,
)
from fray.cluster.ray.tpu.types import (
    SliceResource,
    TpuCancelled,
    TpuFailed,
    TpuPreempted,
    TpuRunError,
    TpuSuccess,
    _TpuRunResult,
)
from fray.cluster.ray.tpu.utils import (
    HEALTH_CHECK_TIMEOUT,
    _handle_ray_error,
    _multislice_info_from_head,
    _multislice_info_to_env_vars,
)

logger = logging.getLogger(__name__)


def run_on_pod(
    remote_fn: RemoteFunction | Callable,
    tpu_type: str,
    *,
    num_slices: int | Sequence[int] = 1,
    max_retries_preemption=10000,
    max_retries_failure=10,
):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Note: This function will block until the function completes or fails too many times.
    If you want to run it asynchronously, use `run_on_pod_ray` instead.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails

    Returns:
        The result of the function (not an ObjectRef)
    """

    _validate_num_slices(num_slices)

    return ray.get(run_on_pod_ray.remote(remote_fn, tpu_type, num_slices, max_retries_preemption, max_retries_failure))


@ray.remote(num_cpus=0.01)
def run_on_pod_ray(
    remote_fn: RemoteFunction,
    tpu_type: str,
    num_slices: int | Sequence[int] = 1,
    max_retries_preemption: int = 10000,
    max_retries_failure: int = 10,
):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    This function is a Ray remote function that can be called from anywhere in the Ray cluster.
    """
    _validate_num_slices(num_slices)

    # failures here means the job failed due to an error in the remote function, not a preemption
    num_failures = 0
    # we include any kind of non-`remote_fn` failure in this count, including preemptions
    num_preemptions = 0

    slice_pool: list[SliceResource] = []
    problems: list[Exception] = []
    problem: Exception | None

    if isinstance(remote_fn, FunctionNode):
        raise ValueError(
            "Remote function must be a Ray remote function or a plain function, not a FunctionNode. Don't use bind."
        )
    elif not isinstance(remote_fn, RemoteFunction):
        remote_fn = ray.remote(max_calls=1)(remote_fn)
    elif remote_fn._default_options.get("max_calls") is None:
        raise ValueError("Remote function must have max_calls set to 1 for TPU workloads.")

    slice_pool_manager = SlicePoolManager(tpu_type)

    try:
        while num_failures <= max_retries_failure and num_preemptions <= max_retries_preemption:
            logger.info(
                f"Running on {num_slices} x TPU {tpu_type}. "
                f"Previous failures: {num_failures}. Previous pre-emptions: {num_preemptions}."
            )
            problems.clear()

            # prune all bad actors from pool
            try:
                slice_pool_manager.scale_multislice(num_slices)
                slice_pool = slice_pool_manager.get_all_pool_members()
            except Exception as e:
                logger.exception("Failed to prune dead slices or create new actors", exc_info=e)
                problems.append(e)
                num_preemptions += 1
                continue

            # If we're doing multislice, we need to get the slice info from the first actor
            head_slice_info = slice_pool[0].actor_info if len(slice_pool) > 1 else None

            # Ok finally time to run the remote function on all slices
            futures: list[ray.ObjectRef] = []  # one per host in each slice
            future_to_index: dict[ray.ObjectRef, int] = {}  # maps futures to their index in the results list
            global_index = 0  # index into results list

            for i, tpu_slice in enumerate(slice_pool):
                if head_slice_info is not None:
                    multislice_info = _multislice_info_from_head(head_slice_info, i, len(slice_pool))
                    mxla_env = _multislice_info_to_env_vars(multislice_info)
                else:
                    mxla_env = {}

                futures_for_slice = _start_fn_on_slice(tpu_slice.actor, remote_fn, mxla_env)
                logger.info(f"Futures for slice {tpu_slice.actor_info.slice_name}: {futures_for_slice}")

                futures.extend(futures_for_slice)
                for future in futures_for_slice:
                    future_to_index[future] = global_index
                    global_index += 1

            if not futures:
                error = "Failed to schedule any futures"
                exception = RuntimeError(error)
                logger.exception(error, exc_info=exception)
                problems.append(exception)
                break

            tpu_results: list[_TpuRunResult | None] = [None] * len(futures)

            # We wait for jobs to finish one at a time. If a preemption or failure occurs, we cancel all
            pending_futures = list(futures)
            had_a_failure = False
            should_scale_up_multislice = False

            # check health of actors once
            # TODO: Check health repeatedly given some interval
            actor_health_futures = [tpu_slice.actor.healthy.remote() for tpu_slice in slice_pool]

            while pending_futures:
                finished, pending_futures = ray.wait(pending_futures, num_returns=1, timeout=10.0)

                for f in finished:
                    try:
                        tpu_results[future_to_index[f]] = TpuSuccess(ray.get(f))
                    except RayError as e:
                        had_a_failure = True
                        problems.append(e)
                        tpu_results[future_to_index[f]] = _handle_ray_error(e)
                    except Exception as e:
                        logger.warning(f"Task {f} failed with unexpected error {e}. Will retry.")
                        had_a_failure = True
                        tpu_results[future_to_index[f]] = TpuRunError(e)

                if had_a_failure:
                    # skip health checks if we already had a failure
                    break

                # Check if any actors are unhealthy. We hit this if it's been 10 seconds or we got a result
                try:
                    actor_healths = ray.get(actor_health_futures, timeout=HEALTH_CHECK_TIMEOUT)
                except RayError as e:
                    logger.warning("Failed to get actor healths", exc_info=e)
                    # assume things are bad
                    had_a_failure = True
                else:
                    for i, healthy in enumerate(actor_healths):
                        if not healthy:
                            logger.warning(f"Actor {slice_pool[i]} is unhealthy. Will retry.")
                            had_a_failure = True

                if had_a_failure:
                    break

                # It is safe to call this check_should_scale_up_multislice() frequently in a loop
                # because it is debounced internally and most calls will finish quickly.
                # It will only do the actual slow check once in a while.
                should_scale_up_multislice = slice_pool_manager.check_should_scale_up_multislice(num_slices)
                if should_scale_up_multislice:
                    break

            # Proactively cancel jobs if one fails.
            if pending_futures:
                logger.info(f"Failure detected. Cancelling {len(pending_futures)} pending futures.")
                try:
                    _cancel_tasks_and_wait(pending_futures)
                except Exception as e:
                    logger.error(f"Could not cancel all pending futures: {e}")
                # Now, fill in the cancellations
                for f in pending_futures:
                    index = future_to_index.get(f)
                    if index is not None:
                        tpu_results[index] = TpuCancelled(
                            RuntimeError("Task was cancelled due to a failure in another task")
                        )
                    else:
                        logger.warning(f"Future {f} was not found in future_to_index. Skipping.")

            # Process results, figure out if we succeeded or failed or preempted
            out_results: list = []
            any_preempted = False
            any_failed = False
            any_cancelled = False

            for result in tpu_results:
                if isinstance(result, TpuSuccess):
                    out_results.append(result.result)
                elif isinstance(result, TpuPreempted):
                    problems.append(result.error)
                    any_preempted = True
                elif isinstance(result, TpuFailed):
                    any_preempted = True
                    problems.append(result.error)
                    logger.warning(f"TPU node failure. Treating as preempted: {num_preemptions} times")
                elif isinstance(result, TpuRunError):
                    problems.append(result.error)
                    any_failed = True
                elif isinstance(result, TpuCancelled):
                    logger.info("TPU job was cancelled, probably because something else failed.")
                    any_cancelled = True
                elif result is None:
                    raise AssertionError("We should never have None results here.")
                else:
                    raise RuntimeError(f"Unexpected result: {result}")

            if any_preempted:
                problem = problems[0] if problems else RuntimeError("TPU job was preempted")
                num_preemptions += 1
                if any_failed:
                    logger.exception(
                        f"Preempted {num_preemptions} times. "
                        "Got some failures, but assuming they are due to preemption.",
                        exc_info=problem,
                    )
                else:
                    logger.warning(f"Preempted {num_preemptions} times. Continuing to retry.", exc_info=problem)
                continue
            elif any_failed:
                problem = problems[0] if problems else RuntimeError("TPU job failed")
                num_failures += 1
                logger.warning(f"Failed {num_failures} times. Continuing to retry.", exc_info=problem)
                continue
            elif any_cancelled:
                logger.info("A slice's task was cancelled, probably due to another slice's failure. Retrying.")
                continue
            elif should_scale_up_multislice:
                logger.info("Additional slices are available. Increasing the number of slices and retrying.")
                num_preemptions += 1
                continue
            else:
                logger.info("All slices succeeded. Returning results.")
                return out_results
    except Exception as e:
        logger.exception("Unexpected error. This is a bug in fray. Please report it.", exc_info=e)
        raise
    finally:
        # Cleanup actors
        logger.info("Cleaning up actors")
        slice_pool_manager.drain_actor_pool()

    # Note: PyCharm flags this as unreachable code, but it is reachable if the loop exits without returning.
    problem = problems[0] if problems else None

    if num_preemptions > max_retries_preemption:
        logger.exception("Preempted too many times", exc_info=problem)
        raise RuntimeError("TPU job was preempted too many times") from problem
    elif num_failures >= max_retries_failure:
        logger.exception("Failed too many times", exc_info=problem)
        raise problem or RuntimeError("TPU job failed too many times")
    else:
        raise RuntimeError("Unknown error occurred during TPU job") from problem


def run_on_pod_multislice(
    remote_fn: RemoteFunction | Callable, tpu_type: str, num_slices: Sequence[int]
) -> list[ray.ObjectRef]:
    """
    Run a remote function on multiple TPU slices.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        num_slices: The number of slices to run

    Returns:
        A Ray ObjectRef that represents the result of the function
    """
    return ray.get(
        run_on_pod(
            remote_fn,
            tpu_type,
            num_slices=num_slices,
            max_retries_failure=0,
            max_retries_preemption=0,
        )
    )


def run_on_pod_resumable(
    remote_fn: RemoteFunction | Callable,
    tpu_type: str,
    max_retries_preemption: int = 1_000_000,
    max_retries_failure: int = 10,
):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails

    Returns:
        The result of the function (not an ObjectRef)

    """
    return run_on_pod(
        remote_fn,
        tpu_type,
        num_slices=1,
        max_retries_preemption=max_retries_preemption,
        max_retries_failure=max_retries_failure,
    )


def run_on_pod_multislice_resumable(
    remote_fn: RemoteFunction | Callable,
    tpu_type: str,
    num_slices: int | Sequence[int],
    max_retries_preemption: int = 1_000_000,
    max_retries_failure: int = 10,
):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        num_slices: The number of slices to run
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails

    Returns:
        The result of the function (not an ObjectRef)

    """
    return run_on_pod(
        remote_fn,
        tpu_type,
        num_slices=num_slices,
        max_retries_preemption=max_retries_preemption,
        max_retries_failure=max_retries_failure,
    )
