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

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor classes for TPU slice pool management and execution."""

import logging
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence

import mergedeep
import ray
from ray.actor import ActorHandle
from ray.exceptions import ActorDiedError, ActorUnavailableError, GetTimeoutError, RayActorError, RayTaskError
from ray.remote_function import RemoteFunction
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from fray.cluster.ray.tpu.config import _get_current_node_tpu_worker_id, get_tpu_config
from fray.cluster.ray.tpu.types import ActorPoolMember, SliceInfo, TPUHostInfo
from fray.cluster.ray.tpu.utils import (
    CANCEL_TASK_TIMEOUT,
    HEALTH_CHECK_TIMEOUT,
    SCALE_UP_MULTISLICE_CHECK_INTERVAL,
    SCALE_UP_MULTISLICE_INTERVAL,
    START_ACTOR_TIMEOUT,
    TEARDOWN_ACTOR_TIMEOUT,
    TERMINATE_ACTOR_TIMEOUT,
    _hacky_remove_tpu_lockfile,
    get_current_tpu_is_preempted,
)

logger = logging.getLogger(__name__)


def _validate_num_slices(num_slices: int | Sequence[int]):
    if isinstance(num_slices, int):
        is_valid = num_slices > 0
    elif isinstance(num_slices, list):
        is_valid = len(num_slices) > 0 and all(isinstance(n, int) and n > 0 for n in num_slices)
    else:
        is_valid = False
    if not is_valid:
        raise Exception(
            f"num_slices must be a positive integer or non-empty list of positive integers, but instead it was {num_slices}"
        )


def _stop_actor(actor: ActorHandle) -> None:
    try:
        # This is recommended by https://docs.ray.io/en/latest/ray-core/api/doc/ray.kill.html
        #
        # > If you want to kill the actor but let pending tasks finish, you can call actor.__ray_terminate__.remote()
        # > instead to queue a termination task. Any atexit handlers installed in the actor will be run in this case.
        #
        # NOTE: Not sure if this always returns an exception (because the actor will terminate before finishing)
        # but it doesn't really matter
        ray.get(actor.teardown.remote(), timeout=TEARDOWN_ACTOR_TIMEOUT)
        ray.get(actor.__ray_terminate__.remote(), timeout=TERMINATE_ACTOR_TIMEOUT)
    except (ActorDiedError, ActorUnavailableError):
        # This is expected because the actor will terminate within  __ray_terminate__() task,
        # so the task will never succeed.
        pass
    except GetTimeoutError as e:
        logger.warning(
            f"Failed to gracefully shut down actor in {TERMINATE_ACTOR_TIMEOUT} seconds; killing it instead: {e}"
        )
    finally:
        ray.kill(actor)


def _cancel_tasks_and_wait(tasks: list[ray.ObjectRef]) -> None:
    _, tasks = ray.wait(tasks, timeout=0)
    if not tasks:
        return
    logger.info(f"Cancelling {len(tasks)} tasks")
    try:
        for task in tasks:
            ray.cancel(task, force=True, recursive=True)
    except Exception as exc:
        message = f"Failed to cancel {len(tasks)} tasks"
        logger.error(message)
        raise Exception(message) from exc
    logger.info(f"Waiting for {len(tasks)} tasks to be cancelled.")
    cancel_ready, cancel_unready = ray.wait(tasks, num_returns=len(tasks), timeout=CANCEL_TASK_TIMEOUT)
    if cancel_unready:
        message = f"Cancelled {len(cancel_ready)} tasks; could not cancel {len(cancel_unready)} tasks"
        logger.error(message)
        raise Exception(message)
    else:
        logger.info(f"Cancelled {len(cancel_ready)} tasks")


def _start_fn_on_slice(
    slice_actor: ActorHandle, remote_fn: RemoteFunction, mxla_env: dict | None
) -> list[ray.ObjectRef]:
    """
    Start the remote function on a slice of the TPU pod.
    """
    runtime_env = remote_fn._runtime_env or {}
    if mxla_env is not None:
        mxla_env = dict(env_vars=mxla_env)
        runtime_env = mergedeep.merge({}, runtime_env, mxla_env, strategy=mergedeep.Strategy.ADDITIVE)
    futures_for_slice = ray.get(slice_actor.run_remote_fn.remote(remote_fn, runtime_env))
    return futures_for_slice


class ResourcePoolManager(ABC):
    def __init__(self):
        self._actor_pool: list[ActorPoolMember] = []

    @abstractmethod
    def get_actor_pool_name(self) -> str:
        return str(self)

    @abstractmethod
    def get_actor_name_from_actor_info(self, actor_info) -> str:
        raise NotImplementedError()

    @abstractmethod
    def create_actor(self) -> ActorHandle:
        raise NotImplementedError()

    def get_all_actors_in_pool(self) -> list[ActorHandle]:
        return [member.actor for member in self._actor_pool]

    def get_all_pool_members(self) -> list[ActorPoolMember]:
        return self._actor_pool.copy()

    def _remove_unhealthy_members_from_actor_pool(self) -> None:
        logger.info(
            f"{self.get_actor_pool_name()} actor pool members before removing unhealthy members: {[self.get_actor_name_from_actor_info(member.actor_info) for member in self._actor_pool]}"
        )
        members_and_health = [(member, member.actor.healthy.remote()) for member in self._actor_pool]
        healthy_members: list[ActorPoolMember] = []
        unhealthy_members: list[ActorPoolMember] = []
        ray.wait(
            [health for _, health in members_and_health],
            num_returns=len(members_and_health),
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        for member, health in members_and_health:
            try:
                if ray.get(health, timeout=0):
                    healthy_members.append(member)
                else:
                    logger.warning(
                        f"{self.get_actor_pool_name()} actor pool member {self.get_actor_name_from_actor_info(member.actor_info)} is unhealthy. Removing from actor pool."
                    )
                    unhealthy_members.append(member)
            except (
                RayActorError,
                RayTaskError,
                ActorDiedError,
                ActorUnavailableError,
                GetTimeoutError,
            ) as e:
                logger.warning(
                    f"{self.get_actor_pool_name()} actor pool member {self.get_actor_name_from_actor_info(member.actor_info)} is dead or unavailable. Removing from actor pool. Error: {e}"
                )
                unhealthy_members.append(member)

        # NOTE: For simplicity, we serially process the unhealthy actors, rather than doing it in parallel.
        for unhealthy_member in unhealthy_members:
            # This is a synchronous blocking call.
            _stop_actor(unhealthy_member.actor)

        self._actor_pool = healthy_members
        logger.info(
            f"{self.get_actor_pool_name()} actor pool members after unhealthy members: {[self.get_actor_name_from_actor_info(member.actor_info) for member in self._actor_pool]}"
        )

    def _add_members_to_actor_pool(self, desired_num_actors: int) -> None:
        logger.info(
            f"{self.get_actor_pool_name()} actor pool has {len(self._actor_pool)} members, scaling up to {desired_num_actors} members; current members: {[self.get_actor_name_from_actor_info(member.actor_info) for member in self._actor_pool]}"
        )
        actors = [self.create_actor() for _ in range(desired_num_actors - len(self._actor_pool))]
        actors_and_actor_info_awaitables = [(actor, actor.get_info.remote()) for actor in actors]
        logger.info(f"{self.get_actor_pool_name()} actor pool waiting for {len(actors)} new actors to start...")
        ray.wait(
            [actor_info_awaitable for _, actor_info_awaitable in actors_and_actor_info_awaitables],
            num_returns=len(actors_and_actor_info_awaitables),
            timeout=START_ACTOR_TIMEOUT,
        )
        for actor, actor_info_awaitable in actors_and_actor_info_awaitables:
            try:
                actor_info = ray.get(actor_info_awaitable, timeout=0)
            except Exception as e:
                logger.exception(f"{self.get_actor_pool_name()} actor pool actor {actor} failed to start: {e}")
                _stop_actor(actor)
                continue
            logger.info(
                f"{self.get_actor_pool_name()} actor pool member {self.get_actor_name_from_actor_info(actor_info)} started."
            )
            self._actor_pool.append(ActorPoolMember(actor, actor_info))
        logger.info(
            f"{self.get_actor_pool_name()} actor pool scaled up to {len(self._actor_pool)} members: {[self.get_actor_name_from_actor_info(member.actor_info) for member in self._actor_pool]}"
        )

    def _remove_members_from_actor_pool(self, desired_num_actors: int) -> None:
        logger.info(
            f"{self.get_actor_pool_name()} actor pool has {len(self._actor_pool)} members; scaling down to {desired_num_actors} members; current members: {[self.get_actor_name_from_actor_info(member.actor_info) for member in self._actor_pool]}"
        )
        members_to_remove = self._actor_pool[desired_num_actors:]
        self._actor_pool = self._actor_pool[:desired_num_actors]
        for member in members_to_remove:
            logger.info(
                f"{self.get_actor_pool_name()} actor pool member {self.get_actor_name_from_actor_info(member.actor_info)} stopping."
            )
            _stop_actor(member.actor)
        logger.info(
            f"{self.get_actor_pool_name()} actor pool scaled down to {len(self._actor_pool)} members: {[self.get_actor_name_from_actor_info(member.actor_info) for member in self._actor_pool]}"
        )

    def _scale_actor_pool(self, desired_num_actors: int) -> None:
        # NOTE: There is no retry loop in this function.
        # You should wrap this in an external retry loop.
        if self._actor_pool:
            self._remove_unhealthy_members_from_actor_pool()
        if len(self._actor_pool) < desired_num_actors:
            self._add_members_to_actor_pool(desired_num_actors)
        elif len(self._actor_pool) > desired_num_actors:
            self._remove_members_from_actor_pool(desired_num_actors)
        else:
            logger.info(
                f"{self.get_actor_pool_name()} actor pool already has {len(self._actor_pool)} members, and we wanted {desired_num_actors}. Skipping scaling."
            )
            return
        if len(self._actor_pool) != desired_num_actors:
            raise Exception(
                f"{self.get_actor_pool_name()} actor pool wanted to scale to {desired_num_actors} actors, but scaled to {len(self._actor_pool)} actors instead"
            )

    def drain_actor_pool(self) -> None:
        logger.info(f"{self.get_actor_pool_name()} actor pool members draining.")
        self._remove_members_from_actor_pool(0)
        logger.info(f"{self.get_actor_pool_name()} actor pool drained.")


class SlicePoolManager(ResourcePoolManager):
    def __init__(self, tpu_type: str):
        super().__init__()
        self._tpu_type = tpu_type
        self._last_scale_multislice_time: float | None = None
        self._last_check_should_scale_up_multislice_time: float | None = None

    # CHAOS MONKEY: Simulate randomly not getting enough slices
    # def _add_members_to_actor_pool(self, desired_num_actors: int) -> None:
    #     import random
    #     prev_desired_num_actors = desired_num_actors
    #     desired_num_actors = random.randint(len(self._actor_pool) + 1, desired_num_actors)
    #     logger.info(f"CHAOS MONKEY: wanted {prev_desired_num_actors} slices but getting {desired_num_actors} slices")
    #     super()._add_members_to_actor_pool(desired_num_actors)

    def get_actor_pool_name(self) -> str:
        return f"{self._tpu_type} slices"

    def get_actor_name_from_actor_info(self, actor_info: SliceInfo) -> str:
        return str(actor_info.slice_name)

    def create_actor(self) -> ActorHandle:
        return SliceActor.options(resources={f"TPU-{self._tpu_type}-head": 1}).remote()  # type: ignore

    def scale_multislice(self, num_slices: int | Sequence[int]) -> None:
        self._last_scale_multislice_time = time.time()

        if isinstance(num_slices, int):
            self._scale_actor_pool(num_slices)
            return

        sorted_valid_sizes = sorted(num_slices)
        max_valid_size = sorted_valid_sizes[-1]

        # CHAOS MONKEY: Simulate initial scaling not getting enough slices:
        # if len(self._actor_pool) == 0:
        #     max_valid_size -= 1

        logger.info(
            f"Attempting to scale to {max_valid_size} slices based on the maximum of valid sizes: {sorted_valid_sizes}"
        )
        try:
            self._scale_actor_pool(max_valid_size)
            # self.scale_actor_pool(max_valid_size)
        except Exception as e:
            logger.warning(f"Error when scaling to {max_valid_size} slices: {e}")

        if len(self._actor_pool) in num_slices:
            return

        current_size = len(self._actor_pool)
        feasible_sizes = [size for size in sorted_valid_sizes if size <= current_size]

        if not feasible_sizes:
            raise Exception(
                f"Only got {current_size} slices, which is not enough for minimum of valid sizes: {sorted_valid_sizes}"
            )

        max_feasible_size = feasible_sizes[-1]
        logger.warning(
            f"Attempting to scale to {max_feasible_size} slices based on a feasible size in valid sizes: {sorted_valid_sizes}"
        )
        try:
            self._scale_actor_pool(max_feasible_size)
        except Exception as e:
            logger.warning(f"Error when scaling to {max_feasible_size} slices: {e}")

        if len(self._actor_pool) not in num_slices:
            raise Exception(f"Could not scale to {max_feasible_size} slices")

    def check_should_scale_up_multislice(self, num_slices: int | Sequence[int]) -> bool:
        if isinstance(num_slices, int):
            return False

        # Don't check for rescaling if:
        # - Not enough time has passed since the the last scale up check
        # - Not enough time has passed since the the last actual scale up
        current_time = time.time()
        last_check_time = self._last_check_should_scale_up_multislice_time
        self._last_check_should_scale_up_multislice_time = current_time
        if last_check_time is None or current_time - last_check_time < SCALE_UP_MULTISLICE_CHECK_INTERVAL:
            return False

        if (
            self._last_scale_multislice_time is None
            or current_time - self._last_scale_multislice_time < SCALE_UP_MULTISLICE_INTERVAL
        ):
            return False

        # Want to scale to the next largest desired size that is bigger than the current size.
        sorted_valid_sizes = sorted(num_slices)
        current_size = len(self._actor_pool)
        larger_sizes = [size for size in sorted_valid_sizes if size > current_size]
        if not larger_sizes:
            return False
        next_larger_size = larger_sizes[0]

        # Attempt to start enough slice actors to go up to the next largest desired size
        # If we don't succeed, release the new slice actors
        previous_size = len(self._actor_pool)
        logger.info(
            f"Currently have {previous_size} slices; next larger size is {next_larger_size} (valid sizes: {num_slices}). Trying to acquire more slices."
        )
        self._add_members_to_actor_pool(next_larger_size)
        if len(self._actor_pool) == next_larger_size:
            logger.info(f"Successfully acquired more slices; now have {next_larger_size} slices.")
            return True
        else:
            logger.info(
                f"Wanted {next_larger_size} slices but could only get {len(self._actor_pool)} slices; scaling slices back down to previous size {previous_size}."
            )
            self._remove_members_from_actor_pool(previous_size)
            return False


@ray.remote
class SliceActor(ResourcePoolManager):
    """
    Actor that manages a single TPU slice.
    """

    def __init__(self):
        super().__init__()
        self._failed = False
        self._slice_info: SliceInfo | None = None

    def healthy(self) -> bool:
        return not self._failed and not self.is_being_preempted()

    def is_being_preempted(self) -> bool:
        """
        Check if the TPU slice is being preempted.
        This is a workaround for the fact that Ray doesn't expose this information directly.
        """
        return get_current_tpu_is_preempted()

    def get_actor_pool_name(self) -> str:
        assert self._slice_info
        return f"slice {self._slice_info.slice_name}"

    def get_actor_name_from_actor_info(self, actor_info: TPUHostInfo) -> str:
        return str(actor_info.worker_index)

    def create_actor(self) -> ActorHandle:
        assert self._slice_info
        slice_name = self._slice_info.slice_name
        return TPUHostActor.options(resources={slice_name: 1}, num_cpus=0.0).remote(self._slice_info)  # type: ignore

    def get_info(self) -> SliceInfo:
        from fray.cluster.ray.tpu.config import _get_current_tpu_pod_type

        pod_name = ray.util.accelerators.tpu.get_current_pod_name()
        tpe = _get_current_tpu_pod_type()

        config = get_tpu_config(tpe)

        ip_address = socket.gethostbyname(socket.gethostname())

        logger.info(f"TPU type: {tpe}, {config}")

        self._slice_info = SliceInfo(
            slice_name=pod_name,
            num_vms=config.vm_count,
            num_tpus_per_vm=config.chips_per_vm,
            ip_address=ip_address,
        )
        self._scale_actor_pool(config.vm_count)
        return self._slice_info

    def run_remote_fn(self, remote_fn: RemoteFunction, runtime_env: dict) -> list[ray.ObjectRef]:
        """Run the remote function on this slice.

        NOTE: This runs the remote function in a different task. It does not block on the remote function call.
        NOTE: This returns a list of Ray futures. If calling this method on a remote Actor, you will get a future of a list of futures.
        """
        actors = self.get_all_actors_in_pool()
        if not self._slice_info or len(actors) < self._slice_info.num_vms:
            raise Exception("Insufficient host actors; call setup() before calling run_remote_fn()")
        futures_of_futures: list[ray.ObjectRef] = [
            actor.run_remote_fn.remote(remote_fn, runtime_env) for actor in actors
        ]
        return [ray.get(future_of_future) for future_of_future in futures_of_futures]

    def teardown(self):
        self.drain_actor_pool()
        self._slice_info = None


@ray.remote
class TPUHostActor:
    """
    Actor that manages a single TPU host.
    """

    def __init__(self, slice_info: SliceInfo):
        self._awaitable: ray.ObjectRef | None = None
        self._host_info: TPUHostInfo | None = None
        self._slice_info = slice_info

    def healthy(self) -> bool:
        return not self.is_being_preempted()

    def is_being_preempted(self) -> bool:
        return get_current_tpu_is_preempted()

    def get_info(self) -> TPUHostInfo:
        if self._host_info:
            return self._host_info

        worker_id = _get_current_node_tpu_worker_id()
        if worker_id is None:
            raise Exception("Could not get TPU worker ID. This should never happen.")
        self._host_info = TPUHostInfo(
            slice_name=self._slice_info.slice_name,
            worker_index=worker_id,
            node_id=ray.get_runtime_context().get_node_id(),
            num_tpus=self._slice_info.num_tpus_per_vm,
        )
        return self._host_info

    def run_remote_fn(self, remote_fn: RemoteFunction, runtime_env: dict) -> ray.ObjectRef:
        """Run the remote function on this host.

        NOTE: This runs the remote function in a different task. It does not block on the remote function call.
        NOTE: This returns a Ray future. If calling this method on a remote Actor, you will get a future of a future.
        """
        if not self._host_info:
            raise Exception("Call setup() before calling run_remote_fn()")

        if self._awaitable:
            _cancel_tasks_and_wait([self._awaitable])
        _hacky_remove_tpu_lockfile()

        self._awaitable = remote_fn.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(self._host_info.node_id, soft=False),
            resources={
                "TPU": self._host_info.num_tpus,
            },
            num_cpus=8,
            num_gpus=0,
            memory=20e9,
            runtime_env=runtime_env,
        ).remote()
        return self._awaitable

    def teardown(self) -> None:
        if self._awaitable:
            _cancel_tasks_and_wait([self._awaitable])
        self._awaitable = None
        self._host_info = None
