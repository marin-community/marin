# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import atexit
import itertools
import logging
import os
import re
import socket
from dataclasses import dataclass
from typing import List, Optional, Union

import jax
import ray
from jax._src import clusters, distributed

from levanter.utils.py_utils import logical_cpu_core_count


logger = logging.getLogger(__name__)


_JOBID_PARAM = "SLURM_JOB_ID"
_NODE_LIST_CHOICES = ["SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST"]
_PROCESS_COUNT = "SLURM_NTASKS"
_PROCESS_ID = "SLURM_PROCID"
_LOCAL_PROCESS_ID = "SLURM_LOCALID"
_NUM_NODES = "SLURM_STEP_NUM_NODES"
_TASKS_PER_NODE = "SLURM_STEP_TASKS_PER_NODE"
_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
_NODE_NAME = "SLURMD_NODENAME"


class LevanterSlurmCluster(clusters.SlurmCluster):
    """
    This class is a copy-paste and modification of the original SlurmCluster class in jax, with a few differences:
    - It uses the SLURM_LOCAL_PROCESS_COUNT to determine how many devices to use
    - It looks in a few places for the node list, since the environment variable is set differently
    depending on how you run
    # TODO: upstream this
    """

    # this is mostly copy paste, but it looks at range of different env variables that slurm sometimes sets
    @classmethod
    def get_coordinator_address(cls, timeout_secs: int | None = None) -> str:
        # Pick port in ephemeral range [(65535 - 2^12 + 1), 65535]
        id = os.environ[_JOBID_PARAM]
        port = _choose_port(id)

        # Parse the first hostname of the job
        # If we are looking for 'node001',
        # node_list potential formats are 'node001', 'node001,host2',
        # 'node[001-0015],host2', and 'node[001,007-015],host2'.
        node_list = LevanterSlurmCluster._node_list()
        if node_list is None:
            raise ValueError(
                "Could not find node list in environment variables. You must set coordinator_address manually."
            )
        delims = {",", "["}
        ind = next((i for i, ch in enumerate(node_list) if ch in delims), len(node_list))
        if ind == len(node_list) or node_list[ind] == ",":  # Formats: 'node001' or 'node001,host2'
            return f"{node_list[:ind]}:{port}"
        else:  # Formats: 'node[001-0015],host2' or 'node[001,007-015],host2'
            prefix = node_list[:ind]
            suffix = node_list[ind + 1 :]
            delims2 = {",", "-"}
            ind2 = next((i for i, ch in enumerate(suffix) if ch in delims2), None)
            return f"{prefix}{suffix[:ind2]}:{port}"

    @classmethod
    def _node_list(cls):
        return next((os.environ[o] for o in _NODE_LIST_CHOICES if o in os.environ), None)

    @classmethod
    def get_process_count(cls) -> int:  # type: ignore[override]
        if _PROCESS_COUNT in os.environ:
            return int(os.environ[_PROCESS_COUNT])

        if cls.is_env_present():
            num_nodes = next(
                (os.environ[o] for o in ["SLURM_JOB_NUM_NODES", _NUM_NODES, "SLURM_NNODES"] if o in os.environ),
                None,
            )
            if num_nodes == "1":
                logger.info("%s not set; assuming single-process job", _PROCESS_COUNT)
                return 1

        return super().get_process_count()

    @classmethod
    def get_local_device_ids_for_process(cls) -> Optional[List[int]]:
        local_process_id = cls.get_local_process_id()

        if local_process_id is None:
            return None

        if _VISIBLE_DEVICES not in os.environ:
            # if we don't expose CUDA_VISIBLE_DEVICES, we use JAX's default behavior, which is assuming 1 per process.
            # this happens typically if CUDA_VISIBLE_DEVICES isn't forwarded to the docker container
            return None

        local_process_count = cls._infer_local_process_count()

        all_visible_devices = [int(x) for x in os.environ[_VISIBLE_DEVICES].split(",")]

        if len(all_visible_devices) % local_process_count != 0:
            raise ValueError(
                f"Number of visible devices ({len(all_visible_devices)}) is not divisible by the number "
                f"of local tasks ({local_process_count})"
            )
            return None

        num_devices_per_local_process = len(all_visible_devices) // local_process_count

        # select contiguous devices for this process
        begin = local_process_id * num_devices_per_local_process
        return all_visible_devices[begin : begin + num_devices_per_local_process]

    @classmethod
    def _infer_local_process_count(cls):
        # Figure out which node we're on. This is also annoying because the node list
        # is a comma separated list of nodes, but they collapse the list if there are multiple nodes
        # with the same name e.g. node001,node002,node003,node004,node007 -> node[001-004,007]
        #  slurm exposes a command to expand this list for us, but it's not always available
        node_list = LevanterSlurmCluster._node_list()
        if node_list is None:
            raise ValueError(
                "Could not find node list in environment variables. You must set coordinator_address manually."
            )

        node_list = _square_brace_expand(node_list)
        local_node = os.environ[_NODE_NAME]
        local_node_index = node_list.index(local_node)

        # We want to figure out how many tasks are running on this node
        # the only env variable that is reliably set here is SLURM_STEP_TASKS_PER_NODE
        # which is a comma separated list of the number of tasks per node, except they "helpfully"
        # collapse the list if there are multiple nodes with the same number of tasks e.g.
        # 1(x2),3,4(x3) -> 1,1,3,4,4,4
        # So we have to do some parsing to figure out how many tasks are on each node
        # and then figure out which node we are on
        # first replace the repeated values with the number of times they are repeated
        if _TASKS_PER_NODE not in os.environ:
            logger.warning("%s not set in environment, assuming a single task per node", _TASKS_PER_NODE)
            return 1

        unrolled_tasks_per_node = []
        multi_match = re.compile(r"(\d+)\(x(\d+)\)")
        for x in os.environ[_TASKS_PER_NODE].split(","):
            match = multi_match.match(x)
            if match:
                unrolled_tasks_per_node.extend([int(match.group(1))] * int(match.group(2)))
            else:
                unrolled_tasks_per_node.append(int(x))

        tasks_on_local_node = unrolled_tasks_per_node[local_node_index]
        return tasks_on_local_node


def _square_brace_expand(node_list):
    # Find all parts of the sequence including text and number ranges
    parts = re.findall(r"(\[.*?\]|[^\[\]]+)", node_list)

    # This function will generate numbers from a range or a single number string
    def generate_numbers(number_string):
        if "-" in number_string:  # it's a range
            start, end = map(int, number_string.split("-"))
            return [str(i).zfill(len(number_string.split("-")[0])) for i in range(start, end + 1)]
        else:  # it's a single number
            return [number_string]

    # This function will process each part and return a list of strings or a list of lists of strings
    # Process each part to create lists of possible variations
    processed_parts = []
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            # Extract the number sequences and expand each one
            number_sequences = part.strip("[]").split(",")
            processed_parts.append(
                list(itertools.chain.from_iterable(generate_numbers(seq) for seq in number_sequences))
            )
        else:
            processed_parts.append([part])

    # Compute the Cartesian product of all parts to generate all combinations
    expanded_nodes = ["".join(combination) for combination in itertools.product(*processed_parts)]

    # Join the nodes with commas
    return expanded_nodes


def _choose_port(id):
    port = int(id) % 2**12 + (65535 - 2**12 + 1)
    return port


_already_initialized = False


def auto_ray_cluster(
    address: Optional[str] = None,
    namespace: Optional[str] = "levanter",
    start_workers: bool = True,
    fail_if_cluster_already_initialized: bool = False,
    **kwargs,
):
    """Initializes ray, automatically discovering the address if it is not provided.
    Currently supports slurm and TPU.

    NB that Ray has Slurm support: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html

    We don't use that because it's more geared towards submitting jobs to a ray cluster that is backed by slurm.
    Instead, we have our machines already.
    """
    global _already_initialized

    if _already_initialized:
        logger.warning("auto_ray_cluster has already been called. Ignoring subsequent calls.")
        return

    def _munge_address_port(address: str):
        # the coordinator address typically includes a port that jax wants to use. we want to use our own port
        # we add a deterministic number to the chosen port and then cycle through the ephemeral range
        # this is a hack, but it works
        host, port_str = address.split(":")
        port = int(port_str)
        return host, port

    if address is None:
        # Ray automatically looks at RAY_ADDRESS. We don't want to use our defaulting logic if that is set
        if os.getenv("RAY_ADDRESS") is not None:
            address = os.getenv("RAY_ADDRESS")
            logger.info("Auto-discovered ray address using RAY_ADDRESS: %s", address)
        else:
            coord_address = getattr(distributed.global_state, "coordinator_address", None)

            if coord_address is None:
                logger.info("No auto-discovered ray address found. Using ray.init('local').")
                address = "local"
            else:
                logger.info(f"Auto-discovered ray address using JAX coordinator address: {coord_address}")
                host, port = _munge_address_port(coord_address)

                ray_port = _choose_port(port + 240)
                address = f"{host}:{ray_port}"

                # Explicitly setting the number of CPUs on ray init stops init errors
                num_cpus = logical_cpu_core_count()

                if _is_local_leader():
                    # it used to be that if we were coordinator, we were also process 0
                    # this is no longer the case, so instead we need to check if we are the coordinator
                    # and if so, start the head
                    if _is_this_machine(host):
                        logger.info(f"Starting ray head on port {ray_port}. We are process the coordinator {host}.")
                        logger.info(f"Starting ray head with num_cpus set to {num_cpus}.")
                        ret = os.system(
                            f"ray start --head --port {ray_port} --num-cpus {num_cpus} --dashboard-host=0.0.0.0"
                        )
                        if ret != 0:
                            if not fail_if_cluster_already_initialized:
                                # see if we can connect to the head
                                logger.warning(
                                    f"Failed to start ray head with exit code {ret}. Checking if we can connect to"
                                    " the head..."
                                )
                                ret = os.system("ray status")
                                if ret != 0:
                                    raise RuntimeError(f"Failed to start ray head with exit code {ret}")
                                else:
                                    logger.info(f"Ray head already running on port {ray_port}. Connecting to it.")
                            else:
                                raise RuntimeError(f"Failed to start ray head with exit code {ret}")
                        else:
                            logger.info(f"Successfully started ray head on port {ray_port}.")

                        # install an atexit handler to kill the head when we exit
                        def kill_ray():
                            # silence spam from ray stop
                            os.system("bash -c 'ray stop -g 10 --force &> /dev/null'")

                        atexit.register(kill_ray)
                    elif start_workers:
                        logger.info(
                            f"Starting ray worker and connecting to {address}. We are process {jax.process_index()}."
                        )
                        logger.info(f"Starting ray worker with num_cpus set to {num_cpus}.")
                        ret = os.system(f"ray start --address {address} --num-cpus {num_cpus}")
                        if ret != 0:
                            raise RuntimeError(f"Failed to start ray head with exit code {ret}")
                        else:
                            logger.info(f"Successfully started ray worker and connected to {address}.")

    logger.info(f"ray.init(address={repr(address)}, namespace={repr(namespace)}, **{repr(kwargs)})")
    # Ray has retry logic, but it doesn't seem to work super well, so we retry manually
    for i in range(0, 5):
        try:
            ray.init(address=address, namespace=namespace, **kwargs)
            break
        except Exception as e:
            if i == 4:
                raise e
            else:
                logger.warning(f"Failed to initialize ray with address {address}. Retrying...")
                continue

    def do_shutdown():
        logger.info("Shutting down ray...")
        ray.shutdown()

    atexit.register(do_shutdown)
    _already_initialized = True


@dataclass(frozen=True)
class DistributedConfig:
    coordinator_address: Optional[str] = None  # if None, we'll use the default coordinator address (for TPU or GPU)
    num_processes: Optional[int] = None
    process_id: Optional[int] = None
    local_device_ids: Optional[Union[int, List[int]]] = None

    def _is_distributed(self):
        if (
            (self.coordinator_address is not None)
            or (self.num_processes is not None)
            or (self.process_id is not None)
            or (self.local_device_ids is not None)
        ):
            return True

        # jax will automatically detect slurm or tpu, so we check those too. This is a bit fragile
        # since it depends on the jax internals, but it's the best we can do
        if any(env.is_env_present() for env in clusters.ClusterEnv._cluster_types):
            return True

        return False

    def initialize(self):
        if self._is_distributed():
            device_ids = self.local_device_ids
            coordinator_address = self.coordinator_address

            if LevanterSlurmCluster.is_env_present():
                if device_ids is None:
                    device_ids = LevanterSlurmCluster.get_local_device_ids_for_process()

                if coordinator_address is None:
                    coordinator_address = LevanterSlurmCluster.get_coordinator_address(300.0)

                if self.num_processes is None:
                    self_num_processes = LevanterSlurmCluster.get_process_count()
                else:
                    self_num_processes = self.num_processes
            else:
                self_num_processes = self.num_processes

            jax.distributed.initialize(
                coordinator_address,
                self_num_processes,
                self.process_id,
                device_ids,
                initialization_timeout=30 * 60,
            )
            logger.info(
                f"Initialized jax.distributed with {jax.device_count()} devices, {jax.process_count()} processes,"
                f" coordinator_address={coordinator_address}, process_id={self.process_id}, my"
                f" device_ids={device_ids}."
            )
        else:
            logger.info(
                "Not initializing jax.distributed because no distributed config "
                "was provided, and no cluster was detected."
            )


@dataclass(frozen=True)
class RayConfig:
    address: Optional[str] = None
    start_workers: bool = True
    auto_start_cluster: bool = True

    def initialize(self):
        if self.auto_start_cluster:
            auto_ray_cluster(address=self.address, start_workers=self.start_workers)


def _is_this_machine(host):
    """
    Checks if the given host identifies this machine.
    """
    if host == "localhost" or host == "0.0.0.0":
        return True

    try:
        # Get IP addresses of all interfaces
        machine_ips = [addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None)]

        # Get the IP address of the host
        host_ip = socket.gethostbyname(host)
    except socket.gaierror:
        # Host or interface could not be resolved
        return False

    # Check if the host IP matches any of the machine IPs
    return any(host_ip == machine_ip for machine_ip in machine_ips)


def _remove_if_possible(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _touch(file_path):
    with open(file_path, "a"):
        os.utime(file_path, None)


def _is_local_leader():
    import atexit

    import filelock
    from jax.experimental.multihost_utils import broadcast_one_to_all

    if jax.process_count() == 1:
        return True

    import random

    random_id = random.randint(0, 1000000)
    random_id = broadcast_one_to_all(random_id)

    lock = filelock.FileLock(f"/tmp/levanter_local_process_zero_lock.{random_id}")
    action_performed_file = f"/tmp/levanter_local_process_zero_action_performed.{random_id}"

    try:
        with lock.acquire(timeout=0.1):
            if not os.path.exists(action_performed_file):
                _touch(action_performed_file)
                return True  # Action needs to be performed
            else:
                return False  # Action already performed
            atexit.register(_remove_if_possible, lock.lock_file)
            atexit.register(_remove_if_possible, action_performed_file)
    except filelock.Timeout:
        return False
