import functools
import logging
import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass

import ray
from levanter.main import train_lm
from levanter.utils.ray_utils import ser_exc_info
from ray._private.accelerators import TPUAcceleratorManager
from ray.exceptions import NodeDiedError, RayError, RaySystemError, RayTaskError, WorkerCrashedError
from ray.remote_function import RemoteFunction

from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


@dataclass
class _TpuInfo:
    """Internal class to hold information about a TPU pod."""

    name: str
    state: str
    kind: str


# My kingdom for ADTs
@dataclass
class _TpuRunResult:
    """Internal class to hold the result of a TPU job."""

    info: _TpuInfo


@dataclass
class TpuSuccess(_TpuRunResult):
    result: object


@dataclass
class TpuPreempted(_TpuRunResult):
    error: Exception


@dataclass
class TpuFailed(_TpuRunResult):
    error: Exception


@dataclass
class TpuRunError(_TpuRunResult):
    error: Exception


def _forkify_remote_fn(remote_fn: RemoteFunction | Callable):
    # This is a bit of a hacky way to force a remote function to run in its own
    if isinstance(remote_fn, RemoteFunction):
        fn = remote_fn._function

        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            return _separate_process_fn(fn, args, kwargs)

        # We need these arguments to be able to reconstruct the remote function
        # def __init__(
        #         self,
        #         language,
        #         function,
        #         function_descriptor,
        #         task_options,
        # ):
        remote_fn = RemoteFunction(
            language=remote_fn._language,
            function=wrapped_fn,
            function_descriptor=remote_fn._function_descriptor,
            task_options=remote_fn._default_options,
        )
        return remote_fn
    else:
        return functools.partial(_separate_process_fn, remote_fn)


def _separate_process_fn(underlying_function, args, kwargs):
    @remove_tpu_lockfile_on_exit
    def target_fn(queue, args, kwargs):
        try:
            # Call the original function
            result = underlying_function(*args, **kwargs)
            queue.put((True, result))  # Success, put the result
        except Exception as e:
            # Capture and return the full traceback in case of an exception
            info = ser_exc_info(e)
            queue.put((False, info))

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target_fn, args=(queue, args, kwargs))
    process.start()
    process.join()

    # Retrieve the result or error from the queue
    success, value = queue.get()

    if success:
        return value
    else:
        value.reraise()


def run_on_pod(remote_fn: RemoteFunction | Callable, tpu_type: str):
    """
    Run a remote function on a TPU pod.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
    """

    # jax.distributed doesn't like being invoked twice

    @ray.remote(resources={f"TPU-{tpu_type}-head": 1})
    def do_run(remote_fn) -> _TpuRunResult:
        num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()  # -> 4

        remote_fn, tpu_name = _redecorate_remote_fn_for_tpu(remote_fn, num_hosts)

        info = _TpuInfo(tpu_name, "ACTIVE", "TPU")
        futures = [remote_fn.remote() for _ in range(num_hosts)]
        try:
            out = ray.get(futures)
            logger.info("TPU job finished?!?")
            return TpuSuccess(info, out)
        except RayError as e:
            for f in futures:
                try:
                    ray.cancel(f)
                except Exception:
                    logger.exception("Failed to kill job after primary failure")
            return _handle_ray_error(info, e)

    return do_run.remote(remote_fn)


def _redecorate_remote_fn_for_tpu(remote_fn, num_hosts):
    remote_fn = _forkify_remote_fn(remote_fn)
    if not isinstance(remote_fn, RemoteFunction):
        remote_fn = ray.remote(remote_fn)

    tpu_name = ray.util.accelerators.tpu.get_current_pod_name()  # -> my-tpu
    num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()  # -> 8
    remote_fn = remote_fn.options(resources={tpu_name: 1, "TPU": num_tpus_per_host})
    logger.info(f"Running on TPU {tpu_name} with {num_hosts} hosts and {num_tpus_per_host} TPUs per host")
    return remote_fn, tpu_name


def run_on_pod_resumable(remote_fn, tpu_type, max_retries_preemption=1e6, max_retries_failure=10):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails
    """
    num_failures = 0
    num_preemptions = 0
    attempt = 0
    problem: Exception | None = None

    while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
        logger.info(f"Running on TPU {tpu_type}. Attempt {attempt}")
        attempt += 1
        problem = None
        try:
            out = ray.get(run_on_pod(remote_fn, tpu_type))
        except ray.exceptions.RayTaskError as e:
            problem = e
            if "preempted" in str(e):
                num_preemptions += 1
                logger.warning(f"Preempted {num_preemptions} times, {e}")
            else:
                num_failures += 1
                logger.warning(f"Failed {num_failures} times")
            continue
        except Exception as e:
            problem = e
            num_failures += 1
            if num_failures >= max_retries_failure:
                logger.exception("Failed too many times", exc_info=e)
                raise e
            else:
                logger.warning(f"Failed {num_failures} times", exc_info=e)
                continue

        if isinstance(out, TpuSuccess):
            result = out.result
            logger.info("Success")
            return result
        elif isinstance(out, TpuPreempted):
            problem = out.error
            num_preemptions += 1
            logger.warning(f"Preempted {num_preemptions} times. {problem}", exc_info=problem)
        elif isinstance(out, TpuFailed):
            num_preemptions += 1
            logger.warning(f"TPU node failure. Treating as preempted: {num_preemptions} times")
        elif isinstance(out, TpuRunError):
            problem = out.error
            num_failures += 1
            logger.warning(f"Failed {num_failures} times", exc_info=problem)
        else:
            raise RuntimeError(f"Unexpected result: {out}")

    if num_preemptions >= max_retries_preemption:
        raise RuntimeError("Preempted too many times") from problem
    elif num_failures >= max_retries_failure:
        raise RuntimeError("Failed too many times") from problem


def _handle_ray_error(tpu_info: _TpuInfo, e: RayError):
    """
    Handle a Ray error that occurred on a TPU pod. Tries to determine if the error was due to a
    node failure or preemption or just an application error.
    """
    # treat node failures as preemptions
    if isinstance(e, NodeDiedError):
        logger.exception("Node died", exc_info=e)
        return TpuPreempted(tpu_info, e)
    elif isinstance(e, WorkerCrashedError):
        logger.exception("Worker crashed", exc_info=e)
        return TpuPreempted(tpu_info, e)
    elif isinstance(e, RaySystemError):
        logger.exception("System error", exc_info=e)
        return TpuRunError(tpu_info, e)
    elif isinstance(e, RayTaskError):
        # node preemptions don't always show up as one of the above errors and can just be a RayTaskError. We have
        # to try to sniff out the TPU's status.
        from levanter.infra.tpus import get_current_tpu_is_preempted

        if get_current_tpu_is_preempted():
            print("Preempted")
            logger.exception("Preempted", exc_info=e)
            return TpuPreempted(tpu_info, e)

        logger.exception(f"Task error {e}", exc_info=e)
        return TpuRunError(tpu_info, e)

    else:
        logger.exception("Unknown error", exc_info=e)
        return TpuRunError(tpu_info, e)


@dataclass
class TrainLmOnPodConfig(train_lm.TrainLmConfig):
    tpu_type: str = "v4-64"  # have to specify defaults b/c dataclasses


@ray.remote(num_cpus=0.1)
def run_levanter_train_lm(config: TrainLmOnPodConfig):
    _suppress_ray_config(config)

    @ray.remote
    def run_on_pod_resumable_fn():
        train_lm.main(config)

    return run_on_pod_resumable(run_on_pod_resumable_fn, config.tpu_type)


def _suppress_ray_config(config):
    """
    Levanter wants to auto-start the Ray cluster, but we're already in a Ray cluster. Disable that.
    """
    if config.trainer.ray.auto_start_cluster:
        logger.info("Ray cluster is set to auto-start, but that's not what we want for Marin. Disabling.")
        # TODO: hacky mutation, but there are no lenses in python i think
        config.trainer.ray.auto_start_cluster = False
        config.trainer.ray.start_workers = False
    elif config.trainer.ray.start_workers:
        logger.info("Ray cluster is set to start workers, but that's not what we want for Marin. Disabling.")
        config.trainer.ray.start_workers = False


if __name__ == "__main__":
    ray.init()
    default_config = TrainLmOnPodConfig(tpu_type="v4-16")
    default_config.data.cache_dir = "gs://marin-us-central2/scratch/dlwh/wikitext_103_detokenized"
    default_config.data.id = "dlwh/wikitext_103_detokenized"
    default_config.trainer.tracker = ()
    default_config.trainer.require_accelerator = True
    from levanter.models import gpt2

    default_config.model = gpt2.Gpt2Config(num_heads=4, hidden_dim=128)
    default_config.trainer.num_train_steps = 2000
    ray.get(run_levanter_train_lm.remote(default_config))
    ray.shutdown()
