import datetime
from datetime import timezone # Explicitly import timezone
import threading
import time
import math
import re
import logging
import requests
from google.api_core import exceptions as google_exceptions
from google.cloud import tpu_v2alpha1 # For Node.State and TpuClient
from pathlib import Path
import ray # For @ray.remote decorator and potential future use in actual_hosts

# Logger Setup
logger = logging.getLogger(__name__)
# Basic configuration for the logger if this file is used as a module.
# The importing application can override this.
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Or logging.DEBUG for more verbosity

# Constants
BAD_STATES = [tpu_v2alpha1.Node.State.PREEMPTED, tpu_v2alpha1.Node.State.TERMINATED, tpu_v2alpha1.Node.State.STOPPED]
GOOD_STATES = [tpu_v2alpha1.Node.State.READY]
MIN_WAIT_FOR_INCOMPLETE_TPUS = 15 # Global default for actor config (in minutes)
CHECK_INTERVAL = 60 * 10  # Default check interval for actor (10 minutes, in seconds)

# Copied 'configs' dictionary for get_hosts_for_tpu_spec
configs = {
    "v2-": (2, 4, True),
    "v3-": (2, 4, True),
    "v4-": (2, 2, True), 
    "v5litepod-": (1, 1, False),
    "v5p-": (1, 4, False), 
}

def get_hosts_for_tpu_spec(spec_string: str) -> int | None:
    """
    Calculates the number of hosts required for a given TPU specification string.

    The function parses TPU specification strings (e.g., "v4-32", "v5litepod-4") 
    to determine the expected number of host machines. It uses a predefined 
    `configs` dictionary that stores parameters for different TPU types, including 
    cores per chip, standard chips per host, and whether the numeric part of the 
    spec string ('N' in "prefix-N") represents cores or chips.

    For TPU types like v2, v3, and v4, 'N' typically represents the total number of cores,
    and the function calculates total chips by dividing 'N' by `cores_per_chip`.
    For types like v5litepod and v5p, 'N' directly represents the total number of chips.
    v5litepod configurations are special as they are single-host setups.

    If the spec string is invalid, doesn't match a known prefix, has a non-numeric 
    or non-positive numeric part, or implies an invalid chip configuration (e.g., 
    core count not divisible by cores_per_chip when N represents cores), the function 
    logs an error/warning and returns `None`.
    If the total chips are not perfectly divisible by `chips_per_host_standard`
    (for multi-host types), it calculates hosts using `math.ceil` to ensure all
    chips are accommodated and logs this as an INFO.

    Args:
        spec_string: The TPU specification string (e.g., "v4-32", "v5litepod-4").
                     It should consist of a known prefix from `configs` and a number.

    Returns:
        The calculated number of hosts as an integer, or `None` if the spec_string
        is invalid, unsupported, or leads to an ambiguous/invalid configuration.
    """
    for prefix, (cores_per_chip, chips_per_host_standard, n_represents_cores) in configs.items():
        if spec_string.startswith(prefix):
            n_str = spec_string[len(prefix):]
            if not n_str.isdigit(): # Attempt to extract leading digits if followed by non-digits
                match = re.match(r"(\d+)", n_str)
                if not match:
                    logger.error(f"Invalid or non-numeric part '{n_str}' in spec string: {spec_string}")
                    return None
                n_val_str = match.group(1)
            else:
                n_val_str = n_str
            try:
                n = int(n_val_str)
            except ValueError:
                logger.error(f"Could not convert '{n_val_str}' to an integer for spec string: {spec_string}")
                return None
            if n <= 0: # Numeric part must be positive
                logger.error(f"Numeric part '{n}' must be positive for spec string: {spec_string}")
                return None
            
            if n_represents_cores:
                if cores_per_chip == 0: # Avoid division by zero with bad config
                     logger.error(f"Configuration error: cores_per_chip is zero for {prefix}")
                     return None
                if n % cores_per_chip != 0:
                    logger.warning(
                        f"Core count {n} is not divisible by cores_per_chip {cores_per_chip} for {spec_string}. "
                        "This is likely an invalid or unsupported configuration."
                    )
                    return None
                total_chips = n // cores_per_chip
            else:
                total_chips = n
            
            if "v5litepod-" in prefix: # Special handling for v5litepod single-host types
                if total_chips not in [1, 4, 8]: # v5litepod specific chip counts
                    logger.warning(
                        f"Unsupported chip count {total_chips} for v5litepod spec {spec_string}. "
                        "Valid counts for v5litepod are 1, 4, or 8."
                    )
                    return None
                return 1 # v5litepod are always single host

            if chips_per_host_standard <= 0: # Avoid division by zero with bad config
                logger.error(
                    f"Configuration error: chips_per_host_standard is {chips_per_host_standard} (must be >0) for {prefix}"
                )
                return None
            
            # Standard calculation for number of hosts
            if total_chips % chips_per_host_standard == 0:
                num_hosts = total_chips // chips_per_host_standard
            else:
                # If not perfectly divisible, use ceiling to ensure enough hosts
                num_hosts = math.ceil(total_chips / chips_per_host_standard)
                logger.info(
                    f"Total chips {total_chips} for {spec_string} is not a multiple of "
                    f"chips_per_host_standard {chips_per_host_standard}. "
                    f"Calculated hosts: {num_hosts} (using ceil division)."
                )
            return int(num_hosts) # Ensure integer result
            
    logger.warning(f"Unknown or invalid TPU spec string: {spec_string}. Prefix not found in configs.")
    return None

@ray.remote
class TPUMonitorActor:
    """
    A Ray actor for monitoring and managing the lifecycle of TPUs in a specific GCP project and location.

    The actor periodically checks the status of TPUs, identifies problematic ones
    (e.g., stuck in intermediate states, bad states, or host/Ray resource mismatches),
    and schedules them for deletion if they persist in a problematic state beyond a configurable timeout.
    """
    def __init__(self, 
                 project_name: str | None = None, 
                 location: str | None = None, 
                 check_interval_seconds: int = CHECK_INTERVAL,
                 min_wait_incomplete_tpus_minutes: int = MIN_WAIT_FOR_INCOMPLETE_TPUS,
                 incomplete_log_path_str: str = "logs/incomplete_tpus.log"):
        """
        Initializes the TPUMonitorActor.

        The constructor sets up configuration for monitoring, including GCP project details,
        monitoring intervals, and logging paths. It attempts to fetch project_name and
        location from the GCP metadata server if not explicitly provided.

        Args:
            project_name: GCP project ID. If None, attempts to fetch from metadata server.
            location: GCP location (zone or region) to monitor. If None, attempts to derive
                      the region from the metadata server's zone.
            check_interval_seconds: Interval in seconds at which the actor checks TPU statuses.
                                     Defaults to `CHECK_INTERVAL` (module-level constant).
            min_wait_incomplete_tpus_minutes: Minimum time in minutes to wait before an "incomplete"
                                               or problematic TPU is scheduled for deletion.
                                               Defaults to `MIN_WAIT_FOR_INCOMPLETE_TPUS` (module-level constant).
            incomplete_log_path_str: Path string for the log file where details of TPUs
                                     marked for deletion due to timeout are recorded.
                                     Defaults to "logs/incomplete_tpus.log".

        Raises:
            RuntimeError: If project_name or location cannot be determined (neither
                          provided nor successfully fetched from metadata server), or if
                          the TpuClient fails to initialize.
        """
        self.logger = logging.getLogger(__name__) # Actor-specific logger instance
        self.project_name = project_name
        self.location = location 
        
        self.check_interval = check_interval_seconds
        self.min_wait_incomplete_tpus = min_wait_incomplete_tpus_minutes 
        self.incomplete_log_path = Path(incomplete_log_path_str)
        
        self.incomplete_tpu_first_seen_times: dict[str, tuple[datetime.datetime, str]] = {} 

        self.log_prefix = f"[{self.location or 'unknown_location'}]" 
        if not self.project_name or not self.location:
            try:
                project_id_url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
                zone_url = "http://metadata.google.internal/computeMetadata/v1/instance/zone" 
                headers = {"Metadata-Flavor": "Google"}
                
                fetched_project_id = None
                fetched_region_from_zone = None 

                try:
                    response = requests.get(project_id_url, headers=headers, timeout=5)
                    response.raise_for_status() 
                    fetched_project_id = response.text
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"{self.log_prefix} Failed to fetch project_id from metadata: {e}.")

                try:
                    response = requests.get(zone_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    fetched_zone_full = response.text 
                    fetched_zone = fetched_zone_full.split('/')[-1]
                    if '-' in fetched_zone: 
                        fetched_region_from_zone = '-'.join(fetched_zone.split('-')[:-1])
                    else: 
                        fetched_region_from_zone = fetched_zone 
                    self.logger.info(f"{self.log_prefix} Fetched zone '{fetched_zone}', derived region '{fetched_region_from_zone}' from metadata server.")
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"{self.log_prefix} Failed to fetch zone/region from metadata: {e}.")
                except Exception as e: 
                    self.logger.warning(f"{self.log_prefix} Error processing fetched zone/region: {e}")

                if not self.project_name and fetched_project_id:
                    self.project_name = fetched_project_id
                
                if not self.location and fetched_region_from_zone:
                    self.location = fetched_region_from_zone 
                    self.log_prefix = f"[{self.location}]" 

            except Exception as e: 
                self.logger.warning(f"{self.log_prefix} Generic error during metadata server request: {e}.")
            
            if not self.project_name or not self.location:
                final_proj = self.project_name or "Unknown"
                final_loc = self.location or "Unknown"
                raise RuntimeError(
                    f"Could not determine GCP project/location for TPUMonitorActor (Project: {final_proj}, Location: {final_loc}). "
                    "Please provide them manually or ensure metadata server is accessible and instance has correct scopes."
                )
        
        self.log_prefix = f"[{self.location}]" 
        self.logger.info(f"{self.log_prefix} TPUMonitorActor initialized. Project: {self.project_name}, Location: {self.location}, "
                    f"CheckInterval: {self.check_interval}s, MinWaitIncomplete: {self.min_wait_incomplete_tpus}min, "
                    f"LogPath: '{self.incomplete_log_path}'")
        
        try:
            self.tpu_client = tpu_v2alpha1.TpuClient()
        except Exception as e: 
            self.logger.error(f"{self.log_prefix} Failed to initialize TpuClient: {e}", exc_info=True)
            raise RuntimeError(f"{self.log_prefix} Could not initialize TpuClient.") from e
            
        self.vms_to_delete: set[str] = set() 
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    def check_tpus(selfself):
        """
        Checks the status of TPUs in the configured project and location.
        (Full docstring from previous turn - retained for brevity in this prompt)
        """
        
        self.logger.debug(f"{self.log_prefix} Starting TPU check cycle.")
        parent = f"projects/{self.project_name}/locations/{self.location}"
        
        try:
            if not hasattr(self, 'tpu_client') or self.tpu_client is None:
                self.logger.error(f"{self.log_prefix} TpuClient not available for check_tpus.")
                return
            nodes_iterable = self.tpu_client.list_nodes(parent=parent)
            current_nodes_from_api = list(nodes_iterable) 
        except google_exceptions.GoogleAPIError as e:
            self.logger.error(f"{self.log_prefix} Failed to list TPU nodes due to API error: {e}", exc_info=True)
            return
        except Exception as e: 
            self.logger.error(f"{self.log_prefix} An unexpected error occurred while listing TPU nodes: {e}", exc_info=True)
            return

        current_api_node_names = {node.name for node in current_nodes_from_api}
        newly_marked_for_deletion_count = 0

        for tracked_node_name in list(self.incomplete_tpu_first_seen_times.keys()): 
            if tracked_node_name not in current_api_node_names:
                _first_seen_time, initial_reason = self.incomplete_tpu_first_seen_times.get(tracked_node_name, (None, "unknown_reason"))
                self.logger.debug(f"{self.log_prefix} Node {tracked_node_name.split('/')[-1]} (was tracked for: {initial_reason}) "
                             "is no longer reported by API (reason: disappeared_from_api). Removing from problematic tracking.")
                del self.incomplete_tpu_first_seen_times[tracked_node_name]
        
        if not current_nodes_from_api:
            self.logger.info(f"{self.log_prefix} No TPU nodes found in {parent} after API list and potential cleanup.")
            if self.incomplete_tpu_first_seen_times: 
                 self.logger.info(f"{self.log_prefix} Clearing {len(self.incomplete_tpu_first_seen_times)} tracked incomplete TPUs as no nodes were returned from API.")
                 self.incomplete_tpu_first_seen_times.clear()
            return

        cluster_resources = {} 
        try:
            if ray.is_initialized(): 
                 cluster_resources = ray.cluster_resources()
            else:
                 self.logger.info(f"{self.log_prefix} Ray is not initialized locally. Host count checks against local Ray resources will be skipped or show 0 actual hosts.")
        except Exception as e: 
            self.logger.error(f"{self.log_prefix} Failed to get ray.cluster_resources(): {e}. Host count checks will be impacted.", exc_info=True)

        current_time = datetime.datetime.now(timezone.utc) 

        for node in current_nodes_from_api:
            node_name = node.name 
            node_short_name = node.name.split('/')[-1]
            node_state = node.state 
            tpu_config = node.accelerator_type 
            
            self.logger.debug(f"{self.log_prefix} Processing node: {node_short_name} (State: {node_state.name}, Type: {tpu_config})")
            
            if node_state in BAD_STATES:
                self.logger.warning(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) is in bad state {node_state.name}. Scheduling for deletion.")
                if node_name not in self.vms_to_delete: 
                     self.vms_to_delete.add(node_name)
                     newly_marked_for_deletion_count +=1
                if node_name in self.incomplete_tpu_first_seen_times: 
                    del self.incomplete_tpu_first_seen_times[node_name]
                continue 

            is_problematic = False
            problem_reason_detail = "" 
            expected_hosts = None
            actual_hosts = 0 

            if node_state in GOOD_STATES: 
                expected_hosts = get_hosts_for_tpu_spec(tpu_config)
                if expected_hosts is None:
                    self.logger.warning(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) is {node_state.name}, but could not determine expected host count for its type. Skipping host count check.")
                    if node_name in self.incomplete_tpu_first_seen_times:
                        _, initial_reason = self.incomplete_tpu_first_seen_times[node_name]
                        self.logger.info(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) (previously {initial_reason}, state {node_state.name}) is uncheckable for hosts. Removing from problematic tracking as it's in a good state.")
                        del self.incomplete_tpu_first_seen_times[node_name]
                    continue 
                else: 
                    # actual_hosts = int(cluster_resources.get(tpu_config, 0)) # TODO: Integrate with Ray cluster_resources
                    pass 

                    if actual_hosts != 0 and actual_hosts != expected_hosts: 
                        is_problematic = True
                        problem_reason_detail = f"host_mismatch (state: {node_state.name}, expected: {expected_hosts}, actual: {actual_hosts} in this Ray cluster)"
                        self.logger.debug(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) is {node_state.name} but has host mismatch: expected {expected_hosts}, found {actual_hosts}.")
                    else: 
                        if node_name in self.incomplete_tpu_first_seen_times:
                            log_msg_reason = "host count correct" if actual_hosts == expected_hosts and actual_hosts != 0 else "not actively part of this Ray cluster's resources or host check dormant"
                            _, initial_reason = self.incomplete_tpu_first_seen_times[node_name]
                            self.logger.info(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) (previously {initial_reason}) in state {node_state.name} and {log_msg_reason} (expected {expected_hosts}, actual {actual_hosts}). Removing from problematic tracking.")
                            del self.incomplete_tpu_first_seen_times[node_name]
                        if not is_problematic: continue  
            else: 
                is_problematic = True
                problem_reason_detail = f"stuck_in_state_{node_state.name}" 
                self.logger.debug(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) is in intermediate state {node_state.name}.")

            if is_problematic:
                wait_minutes = self.min_wait_incomplete_tpus 

                if node_name not in self.incomplete_tpu_first_seen_times:
                    self.incomplete_tpu_first_seen_times[node_name] = (current_time, problem_reason_detail)
                    self.logger.info(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) detected as problematic: {problem_reason_detail}. Starting monitoring period of {wait_minutes} minutes.")
                else: 
                    first_seen_time, initial_reason = self.incomplete_tpu_first_seen_times[node_name]
                    if initial_reason != problem_reason_detail:
                         self.logger.info(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) problem reason updated from '{initial_reason}' to '{problem_reason_detail}'. Monitoring continues with original start time {first_seen_time.isoformat()}.")
                         self.incomplete_tpu_first_seen_times[node_name] = (first_seen_time, problem_reason_detail) 

                    if (current_time - first_seen_time).total_seconds() > wait_minutes * 60:
                        final_reason_for_deletion = self.incomplete_tpu_first_seen_times[node_name][1] 
                        self.logger.warning(f"{self.log_prefix} Node {node_short_name} ({tpu_config}) has been problematic for too long (since {first_seen_time.isoformat()}; initial reason: {initial_reason}, current: {final_reason_for_deletion}). Scheduling for deletion.")
                        if node_name not in self.vms_to_delete: 
                            self.vms_to_delete.add(node_name)
                            newly_marked_for_deletion_count +=1
                        
                        try:
                            self.incomplete_log_path.parent.mkdir(parents=True, exist_ok=True) 
                            with open(self.incomplete_log_path, "a") as f:
                                current_expected_h_log = get_hosts_for_tpu_spec(tpu_config) 
                                current_actual_h_log = 0 
                                f.write(f"{current_time.isoformat()},{node_name},{tpu_config},{self.location},"
                                        f"{current_expected_h_log if current_expected_h_log is not None else 'N/A'},"
                                        f"{current_actual_h_log if current_actual_h_log != 0 else 'N/A_or_DormantCheck'}," 
                                        f"{initial_reason},timeout_exceeded\n")
                        except Exception as e: 
                            self.logger.error(f"{self.log_prefix} Failed to write to incomplete TPU log {self.incomplete_log_path} for node {node_short_name}: {e}", exc_info=True)
                        
                        del self.incomplete_tpu_first_seen_times[node_name] 
        
        if newly_marked_for_deletion_count > 0:
             self.logger.info(f"{self.log_prefix} Marked {newly_marked_for_deletion_count} new TPUs for deletion in this cycle.")
        self.logger.info(f"{self.log_prefix} Finished TPU check cycle. {len(self.vms_to_delete)} total VMs marked for deletion. {len(self.incomplete_tpu_first_seen_times)} TPUs currently being tracked as problematic.")


    def delete_stale_vms(self):
        """
        Attempts to delete all TPU nodes currently listed in `self.vms_to_delete`.
        (Full docstring from previous turn - retained for brevity in this prompt)
        """
        # Use self.logger
        if not self.vms_to_delete:
            self.logger.info(f"{self.log_prefix} No VMs to delete.") 
            return

        self.logger.info(f"{self.log_prefix} Attempting to delete {len(self.vms_to_delete)} VMs.") 
        
        successfully_deleted_this_run = set()
        for vm_full_name in list(self.vms_to_delete): 
            vm_short_name = vm_full_name.split('/')[-1]
            self.logger.info(f"{self.log_prefix} Attempting deletion of VM: {vm_short_name} (Path: {vm_full_name})")
            try:
                self.tpu_client.delete_node(name=vm_full_name)
                self.logger.info(f"{self.log_prefix} Successfully deleted VM: {vm_short_name}.")
                successfully_deleted_this_run.add(vm_full_name)
            except google_exceptions.NotFound:
                 self.logger.warning(f"{self.log_prefix} VM {vm_short_name} not found during deletion. Already deleted?")
                 successfully_deleted_this_run.add(vm_full_name) 
            except google_exceptions.GoogleAPIError as e: 
                self.logger.error(f"{self.log_prefix} Failed to delete VM {vm_short_name} due to API error: {e}", exc_info=True)
            except Exception as e: 
                self.logger.error(f"{self.log_prefix} Unexpected error deleting VM {vm_short_name}: {e}", exc_info=True)
        
        self.vms_to_delete.difference_update(successfully_deleted_this_run) 
        if self.vms_to_delete: 
            self.logger.warning(f"{self.log_prefix} {len(self.vms_to_delete)} VMs remain in deletion queue after this attempt.")

    def _monitor_loop(self):
        """
        The main background monitoring loop for the actor.
        (Full docstring from previous turn - retained for brevity in this prompt)
        """
        # Use self.logger
        self.logger.info(f"{self.log_prefix} TPU monitor loop started.") 
        while not self._stop_event.is_set():
            try:
                self.check_tpus()
                if self.vms_to_delete:
                    self.logger.info(f"{self.log_prefix} {len(self.vms_to_delete)} VMs in deletion queue. Proceeding to delete.") 
                    self.delete_stale_vms()
                else:
                    self.logger.info(f"{self.log_prefix} No VMs in deletion queue after check.") 
            except Exception as e: 
                self.logger.error(f"{self.log_prefix} Unhandled error in monitor loop: {e}", exc_info=True) 
            
            self.logger.info(f"{self.log_prefix} Monitor loop iteration complete. Waiting for {self.check_interval}s or stop signal.") 
            self._stop_event.wait(self.check_interval) 
        
        self.logger.info(f"{self.log_prefix} TPU monitor loop stopped.") 

    def start(self):
        """
        Starts the background monitoring thread if not already running.
        (Full docstring from previous turn - retained for brevity in this prompt)
        """
        # Use self.logger
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.warning(f"{self.log_prefix} Monitor thread already running.") 
            return
        
        self._stop_event.clear() 
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.name = f"TPUMonitorThread-{self.location}" 
        self._monitor_thread.start()
        self.logger.info(f"{self.log_prefix} TPU monitor thread started.") 

    def stop(self):
        """
        Signals the background monitoring thread to stop and waits for it to join.
        (Full docstring from previous turn - retained for brevity in this prompt)
        """
        # Use self.logger
        if not self._monitor_thread or not self._monitor_thread.is_alive():
            self.logger.info(f"{self.log_prefix} Monitor thread is not running or already stopped.") 
            return

        self.logger.info(f"{self.log_prefix} Stopping TPU monitor thread...") 
        self._stop_event.set()
        join_timeout = self.check_interval + 10 
        self._monitor_thread.join(timeout=join_timeout) 
        if self._monitor_thread.is_alive(): 
            self.logger.warning(f"{self.log_prefix} Monitor thread did not stop in time ({join_timeout}s).") 
        else:
            self.logger.info(f"{self.log_prefix} TPU monitor thread stopped successfully.") 
        self._monitor_thread = None

if __name__ == '__main__':
    # This block provides an example of how TPUMonitorActor might be used.
    # It's intended for direct testing of this file, not for typical Ray application deployment.
    logging.basicConfig(level=logging.DEBUG, # Use DEBUG for more verbose output during testing
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s')

    logger.info("Starting TPUMonitorActor example...")

    # --- Configuration ---
    # For testing, explicitly set project and location.
    # Ensure the location is a specific ZONE where you have TPUs or expect to test.
    # Replace with your actual GCP Project ID and a specific Zone.
    TEST_PROJECT_NAME = "your-gcp-project-id"  # <--- REPLACE THIS
    TEST_LOCATION_ZONE = "us-central1-a"     # <--- REPLACE THIS (must be a zone)
    
    # Shorter intervals for testing purposes
    TEST_CHECK_INTERVAL_SECONDS = 60  # Check every 60 seconds
    TEST_MIN_WAIT_MINUTES = 1       # Wait only 1 minute for problematic TPUs
    TEST_LOG_PATH = "logs/test_actor_incomplete_tpus.log"
    
    # Ensure the logs directory exists if using the default path or a custom one
    try:
        Path(TEST_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create log directory for {TEST_LOG_PATH}: {e}")
        # Depending on desired behavior, you might exit or continue without file logging for this test.

    logger.info(f"Attempting to create TPUMonitorActor for project='{TEST_PROJECT_NAME}', location='{TEST_LOCATION_ZONE}'")
    
    # Example of running the actor directly using its methods (not as a Ray remote actor)
    # This is for simple, non-distributed testing of the actor's logic.
    try:
        # To test as a Ray actor, you would need to:
        # 1. Ensure Ray is running (e.g., `ray start --head`).
        # 2. Instantiate with `TPUMonitorActor.remote(...)`.
        # monitor_actor_handle = TPUMonitorActor.remote(
        # For this example, we'll create a local instance to test its logic directly.
        monitor_actor = TPUMonitorActor(
            project_name=TEST_PROJECT_NAME,
            location=TEST_LOCATION_ZONE,
            check_interval_seconds=TEST_CHECK_INTERVAL_SECONDS,
            min_wait_incomplete_tpus_minutes=TEST_MIN_WAIT_MINUTES,
            incomplete_log_path_str=TEST_LOG_PATH
        )
        
        logger.info("TPUMonitorActor instance created. Starting monitor loop in a thread...")
        monitor_actor.start() # This starts the internal thread for _monitor_loop

        run_duration_seconds = 300 # e.g., 5 minutes
        logger.info(f"Example will run for approximately {run_duration_seconds} seconds. Check logs for activity.")
        logger.info("TPUs in BAD_STATES will be added to a deletion queue by the actor.")
        logger.info("TPUs stuck in intermediate states will be tracked and logged to "
                    f"'{TEST_LOG_PATH}' if they exceed the timeout ({TEST_MIN_WAIT_MINUTES} min).")
        
        time.sleep(run_duration_seconds) # Let the actor run for a bit

        logger.info("Example run duration elapsed. Stopping TPUMonitorActor...")
        monitor_actor.stop() # Stop the monitoring loop
        logger.info("TPUMonitorActor example finished.")

    except RuntimeError as e:
        logger.error(f"Error during TPUMonitorActor example setup or execution: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the example: {e}", exc_info=True)
    finally:
        logger.info("Exiting example script for TPUMonitorActor.")
        # if ray.is_initialized(): # If you had called ray.init() for remote actor testing
        #     ray.shutdown()
# Ensure a newline at the end of the file
