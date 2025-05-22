import datetime
import subprocess
import sys
import threading # Used by get_ray_tpu_usage
import time
# import math # No longer needed after removing get_hosts_for_tpu_spec
import re # Used by scrape_ray_tpu_usage and parse_head_resource
import logging
# import requests # No longer needed after removing TPUMonitorActor
from google.api_core import exceptions as google_exceptions # Used by gather_tpu_info_from_vms and delete_stale_vms_global
from collections import Counter, defaultdict
from pathlib import Path # Used for CONFIG_DIR, YAML_FILES, DEFAULT_INCOMPLETE_LOG_PATH
from datetime import timezone # Used by gather_tpu_info_from_vms

import ray # Used by gather_ray_cluster_info, get_ray_tpu_usage
import wandb # Used in __main__ and make_report
import yaml as pyyaml # Used for YAML_FILES processing
from bs4 import BeautifulSoup # Used by scrape_ray_tpu_usage
from google.cloud import compute_v1, tpu_v2alpha1 # Used by several global functions
from selenium import webdriver # Used by scrape_ray_tpu_usage
from selenium.webdriver.common.by import By # Used by scrape_ray_tpu_usage
from selenium.webdriver.support import expected_conditions as EC # Used by scrape_ray_tpu_usage
from selenium.webdriver.support.ui import WebDriverWait # Used by scrape_ray_tpu_usage

sys.path.append("../..")

PROJECT_NAME = "hai-gcp-models"
WANDB_PROJECT = "marin-monitoring"
WANDB_ID = "tpu-monitoring-v3-testing"

BAD_STATES = [tpu_v2alpha1.Node.State.PREEMPTED, tpu_v2alpha1.Node.State.TERMINATED, tpu_v2alpha1.Node.State.STOPPED]
GOOD_STATES = [tpu_v2alpha1.Node.State.READY]
MIN_WAIT_FOR_INCOMPLETE_TPUS = 15 # Retained as it's used by gather_all_incomplete_tpus
DEFAULT_INCOMPLETE_LOG_PATH = Path("logs/global_incomplete_tpus.log") # Added for gather_tpu_info_from_vms

CONFIG_DIR = Path(__file__).parent

YAML_FILES = [
    "marin-asia-northeast1.yaml",
    "marin-eu-west4-a.yaml",
    "marin-eu-west4.yaml",
    "marin-us-central1.yaml",
    "marin-us-central2.yaml",
    "marin-us-east1.yaml",
    "marin-us-east5.yaml",
    "marin-us-west4.yaml",
    "marin-big-run.yaml",
]

if not YAML_FILES:
    YAML_FILES = [f.resolve() for f in CONFIG_DIR.glob("*.yaml")]
else:
    YAML_FILES = [(CONFIG_DIR / f).resolve() for f in YAML_FILES]

LOCATION_TO_CLI_FILE = defaultdict(list)
for f in YAML_FILES:
    try:
        config_data = pyyaml.safe_load(f.read_text())
        region = config_data.get("provider", {}).get("availability_zone")
        if region:
            LOCATION_TO_CLI_FILE[region].append(str(f))
    except Exception as e:
        print(f"Failed to parse {f.name}: {e}") # Keep print for now, as it's in global script

LOCATIONS = list(LOCATION_TO_CLI_FILE.keys())

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# TPUMonitorActor class, get_hosts_for_tpu_spec, configs dictionary, and CHECK_INTERVAL constant are removed.

def gather_incomplete_tpus(location):
    global logger, LOCATIONS, LOCATION_TO_CLI_FILE
    incomplete_usage = []
    if location in LOCATIONS:
        for cli_file in LOCATION_TO_CLI_FILE[location]:
            logger.debug(f"Getting Ray TPU usage for {location} using CLI file: {cli_file}")
            ray_usage = get_ray_tpu_usage(cli_file)
            if ray_usage:
                for tpu_type, (used, total) in ray_usage.items():
                    total_val = int(total)
                    if total_val > 0 and (total_val & (total_val - 1) != 0):
                        logger.warning(f"Location {location}, TPU type {tpu_type} has non-power-of-2 total usage: {total_val} (Used: {used})")
                        incomplete_usage.append((tpu_type, used, total_val))
                    elif total_val == 0:
                         logger.info(f"Location {location}, TPU type {tpu_type} has zero total usage reported by Ray dashboard.")
    return incomplete_usage

def gather_tpu_info_from_vms(location, incomplete_tpus_for_loc):
    global PROJECT_NAME, BAD_STATES, GOOD_STATES, logger, DEFAULT_INCOMPLETE_LOG_PATH
    tpu_client = tpu_v2alpha1.TpuClient()
    
    effective_location = location
    if location == "big-run":
        effective_location = "us-central2-b"
        logger.info(f"Alias 'big-run' mapped to location '{effective_location}' for API calls.")

    parent = f"projects/{PROJECT_NAME}/locations/{effective_location}"
    try:
        nodes = list(tpu_client.list_nodes(parent=parent))
    except Exception as e:
        logger.error(f"Failed to list nodes for {parent}: {e}", exc_info=True)
        return Counter(), [], Counter()

    total_devices_zone = 0
    total_preemptible_devices_zone = 0
    nodes_types_zone = Counter()
    tpu_by_generation = Counter()
    vms_to_delete_loc = []

    DEFAULT_INCOMPLETE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    for node in nodes:
        node_short_name = node.name.split('/')[-1]
        for tpu_type_incomplete, used_incomplete, total_incomplete in incomplete_tpus_for_loc:
            if tpu_type_incomplete.lower() in node.accelerator_type.lower() or tpu_type_incomplete.lower() in node.name.lower():
                logger.warning(f"Node {node_short_name} ({node.accelerator_type}) in {location} matches incomplete usage criteria "
                               f"from Ray dashboard ({tpu_type_incomplete}, used: {used_incomplete}, total: {total_incomplete}). Marking for deletion by global script logic.")
                try:
                    with open(DEFAULT_INCOMPLETE_LOG_PATH, "a") as f: 
                        timestamp = datetime.datetime.now(timezone.utc).isoformat()
                        f.write(f"{timestamp},{node.name},{node.accelerator_type},{location},{used_incomplete},{total_incomplete},global_script_incomplete_usage_check\n")
                except IOError as e:
                     logger.error(f"Failed to write to global incomplete TPU log {DEFAULT_INCOMPLETE_LOG_PATH}: {e}")

                if node.name not in vms_to_delete_loc:
                    vms_to_delete_loc.append(node.name)
                break 
                
        if node.name in vms_to_delete_loc: 
            if node.state not in BAD_STATES: 
                 logger.info(f"Node {node_short_name} in {location} was marked for deletion due to incomplete Ray usage, current state: {node.state.name}")

        if node.state in BAD_STATES:
            logger.warning(f"Node {node_short_name} ({node.accelerator_type}) in {location} is in bad state {node.state.name}. Marking for deletion by global script logic.")
            if node.name not in vms_to_delete_loc:
                 vms_to_delete_loc.append(node.name)
            continue 
        
        if node.state not in GOOD_STATES:
            logger.info(f"Node {node_short_name} ({node.accelerator_type}) in {location} is in state {node.state.name} (not GOOD), skipping metrics for global script.")
            continue

        tpu_config_str = node.accelerator_type 
        devices_in_node = 0
        try:
            parts = tpu_config_str.split('-')
            generation_prefix = parts[0].lower()
            
            if len(parts) > 1 and parts[-1].isdigit():
                n_val = int(parts[-1])
                if generation_prefix in ["v2", "v3", "v4"]:
                    devices_in_node = n_val // 2
                elif generation_prefix in ["v5litepod", "v5p"]:
                    devices_in_node = n_val
                else:
                    logger.warning(f"Metrics: Unknown generation prefix '{generation_prefix}' for device calculation from {tpu_config_str}. Assuming N ({n_val}) is chip count.")
                    devices_in_node = n_val
            else:
                 logger.warning(f"Metrics: Could not parse N from tpu_config_str: {tpu_config_str} for node {node_short_name}. Assuming 1 device for metrics.")
                 devices_in_node = 1
        except ValueError:
            logger.warning(f"Metrics: ValueError parsing device count from tpu_config_str: {tpu_config_str}. Assuming 1 device.")
            devices_in_node = 1

        total_devices_zone += devices_in_node
        generation = tpu_config_str.split("-")[0] 

        if node.scheduling_config and node.scheduling_config.preemptible:
            total_preemptible_devices_zone += devices_in_node

        nodes_types_zone[tpu_config_str] += 1
        tpu_by_generation[generation] += devices_in_node

    metrics_for_location = Counter()
    for tpu_type, count_nodes in nodes_types_zone.items():
        metrics_for_location[f"{location}/nodes/{tpu_type}"] = count_nodes
    for generation, sum_devices in tpu_by_generation.items():
        metrics_for_location[f"{location}/devices_gen/{generation}"] = sum_devices
    metrics_for_location[f"{location}/devices/total"] = total_devices_zone
    metrics_for_location[f"{location}/devices/total_preemptible"] = total_preemptible_devices_zone

    global_metrics_contribution = Counter()
    for tpu_type, count_nodes in nodes_types_zone.items():
        global_metrics_contribution[f"nodes/{tpu_type}"] += count_nodes 
    for generation, sum_devices in tpu_by_generation.items():
        global_metrics_contribution[f"devices_gen/{generation}"] += sum_devices
    global_metrics_contribution["devices/total"] += total_devices_zone
    global_metrics_contribution["devices/total_preemptible"] += total_preemptible_devices_zone
    
    return metrics_for_location, vms_to_delete_loc, global_metrics_contribution


def gather_ray_cluster_info(location_zone: str):
    global PROJECT_NAME, logger
    compute_client = compute_v1.InstancesClient()
    
    logger.info(f"Gathering Ray cluster info for zone: {location_zone}")
    try:
        instances = list(compute_client.list(project=PROJECT_NAME, zone=location_zone))
    except Exception as e:
        logger.error(f"Could not list instances for zone {location_zone}: {e}. Ensure it's a specific zone (e.g. us-central1-a).", exc_info=True)
        return {} 

    ray_resources_output = Counter() 

    for instance in instances:
        if is_ray_head_node(instance):
            try:
                instance_ip = get_ip(instance)
                logger.info(f"Connecting to Ray cluster on head node {instance.name} ({instance_ip}) in zone {location_zone}")
                
                if ray.is_initialized(): ray.shutdown() 
                
                ray_namespace = f"monitor_{instance.name.replace('-', '_')}"
                ray.init(address=f"{instance_ip}:6379", namespace=ray_namespace, ignore_reinit_error=True, logging_level=logging.ERROR) 
                
                available_resources_ray = ray.available_resources() 
                logger.debug(f"Available Ray resources on {instance.name}: {available_resources_ray}")

                for resource_name, available_count_nodes in available_resources_ray.items():
                    if is_tpu_head_resource(resource_name):
                        parsed_tpu_generation, devices_per_node_in_name = parse_head_resource(resource_name)
                        
                        if parsed_tpu_generation and devices_per_node_in_name > 0:
                            ray_resources_output[f"{location_zone}/ray_devices_gen/{parsed_tpu_generation}"] += devices_per_node_in_name * int(available_count_nodes)
                            ray_resources_output[f"{location_zone}/ray_nodes_type/{parsed_tpu_generation}-{devices_per_node_in_name}"] += int(available_count_nodes)
                            
                            ray_resources_output[f"ray_devices_gen/{parsed_tpu_generation}"] += devices_per_node_in_name * int(available_count_nodes)
                            ray_resources_output[f"ray_nodes_type/{parsed_tpu_generation}-{devices_per_node_in_name}"] += int(available_count_nodes)
            except Exception as e:
                logger.error(f"Failed to connect or query Ray cluster on {instance.name} in {location_zone}: {e}", exc_info=True)
            finally:
                if ray.is_initialized(): 
                    ray.shutdown()
    
    if not ray_resources_output:
        logger.info(f"No Ray head nodes with TPU resources found or processed in zone {location_zone}.")
    else:
        logger.info(f"Ray cluster resources gathered for {location_zone}: {dict(ray_resources_output)}")
        
    return ray_resources_output


def get_ip(instance: compute_v1.types.Instance) -> str:
    if instance.network_interfaces:
        for interface in instance.network_interfaces:
            if interface.access_configs:
                for config in interface.access_configs:
                    if hasattr(config, "nat_i_p") and config.nat_i_p:
                        return config.nat_i_p
    if instance.network_interfaces:
        for interface in instance.network_interfaces:
            if hasattr(interface, "network_i_p") and interface.network_i_p:
                logger.warning(f"Instance {instance.name} has no external IP, using internal IP: {interface.network_i_p}")
                return interface.network_i_p
    raise ValueError(f"Could not find any IP (external or internal) for instance {instance.name}")

def is_ray_head_node(instance: compute_v1.types.Instance) -> bool:
    return "ray-" in instance.name and "-head-" in instance.name and "-compute" in instance.name

def is_tpu_head_resource(resource_name: str) -> bool:
    return "TPU-" in resource_name.upper() and "-HEAD" in resource_name.upper()

def parse_head_resource(resource_name: str) -> tuple[str | None, int]:
    parts = resource_name.split("-")
    if len(parts) >= 3:
        tpu_type_from_name = parts[1]
        try:
            devices_count_from_name = int(parts[2])
            return tpu_type_from_name, devices_count_from_name
        except ValueError:
            logger.error(f"Could not parse device count from Ray resource: {resource_name}")
            return None, 0
    else:
        logger.error(f"Unexpected Ray resource name format: {resource_name}")
        return None, 0

def make_report():
    global logger, all_metrics_global_script, LOCATIONS, WANDB_PROJECT

    if not WANDB_PROJECT:
        logger.warning("WANDB_PROJECT not set. Skipping report generation.")
        return
    try:
        import wandb_workspaces.reports.v2 as wr # type: ignore
    except ImportError:
        logger.error("wandb_workspaces.reports.v2 not found. Cannot generate report. Please install wandb.")
        return

    logger.info("Generating W&B Report...")
    report = wr.Report(project=WANDB_PROJECT, title="TPU Monitoring Report", description="Aggregated TPU Monitoring Metrics")
    
    runset = wr.Runset() 
    
    layout = wr.Layout(w=24, h=9) 
    
    main_metrics_panels = []
    if "devices/total" in all_metrics_global_script:
        main_metrics_panels.append(wr.LinePlot(y=["devices/total"], title="Global: Total Devices (All Locations)", x="_timestamp", layout=layout, runsets=[runset]))
    if "devices/total_preemptible" in all_metrics_global_script:
        main_metrics_panels.append(wr.LinePlot(y=["devices/total_preemptible"], title="Global: Total Preemptible Devices (All Locations)", x="_timestamp", layout=layout, runsets=[runset]))
    
    global_device_generations = sorted([k for k in all_metrics_global_script.keys() if k.startswith("devices_gen/") and not any(loc_prefix + "/" in k for loc_prefix in LOCATIONS)])
    if global_device_generations:
         main_metrics_panels.append(wr.LinePlot(y=global_device_generations, title="Global Devices by Generation", x="_timestamp", layout=layout, runsets=[runset]))

    global_ray_devices_gen = sorted([k for k in all_metrics_global_script.keys() if k.startswith("ray_devices_gen/") and not any(loc_prefix + "/" in k for loc_prefix in LOCATIONS)])
    if global_ray_devices_gen:
        main_metrics_panels.append(wr.LinePlot(y=global_ray_devices_gen, title="Global Ray TPU Devices by Generation", x="_timestamp", layout=layout, runsets=[runset]))
    
    global_ray_nodes_type = sorted([k for k in all_metrics_global_script.keys() if k.startswith("ray_nodes_type/") and not any(loc_prefix + "/" in k for loc_prefix in LOCATIONS)])
    if global_ray_nodes_type:
        main_metrics_panels.append(wr.LinePlot(y=global_ray_nodes_type, title="Global Ray TPU Nodes by Type", x="_timestamp", layout=layout, runsets=[runset]))


    blocks = [wr.H1("TPU Fleet Monitoring")]
    if main_metrics_panels:
        blocks.append(wr.PanelGrid(panels=main_metrics_panels))
    else:
        blocks.append(wr.P("No global aggregated data available to display."))

    for loc_item in sorted(list(LOCATIONS)):
        blocks.append(wr.H2(f"Status for {loc_item}"))
        loc_specific_panels = []
        
        loc_api_metrics_nodes = sorted([k for k in all_metrics_global_script.keys() if k.startswith(f"{loc_item}/nodes/")])
        if loc_api_metrics_nodes:
             loc_specific_panels.append(wr.LinePlot(y=loc_api_metrics_nodes, title=f"API Nodes by Type - {loc_item}", x="_timestamp", layout=layout, runsets=[runset]))
        
        loc_api_metrics_devgen = sorted([k for k in all_metrics_global_script.keys() if k.startswith(f"{loc_item}/devices_gen/")])
        if loc_api_metrics_devgen:
            loc_specific_panels.append(wr.LinePlot(y=loc_api_metrics_devgen, title=f"API Devices by Generation - {loc_item}", x="_timestamp", layout=layout, runsets=[runset]))

        loc_api_metrics_total = sorted([k for k in all_metrics_global_script.keys() if k.startswith(f"{loc_item}/devices/total")])
        if loc_api_metrics_total:
             loc_specific_panels.append(wr.LinePlot(y=loc_api_metrics_total, title=f"API Total Devices - {loc_item}", x="_timestamp", layout=layout, runsets=[runset]))
            
        loc_ray_devices_gen = sorted([k for k in all_metrics_global_script.keys() if k.startswith(f"{loc_item}/ray_devices_gen/")])
        if loc_ray_devices_gen:
            loc_specific_panels.append(wr.LinePlot(y=loc_ray_devices_gen, title=f"Ray TPU Devices by Generation - {loc_item}", x="_timestamp", layout=layout, runsets=[runset]))
        
        loc_ray_nodes_type = sorted([k for k in all_metrics_global_script.keys() if k.startswith(f"{loc_item}/ray_nodes_type/")])
        if loc_ray_nodes_type:
             loc_specific_panels.append(wr.LinePlot(y=loc_ray_nodes_type, title=f"Ray TPU Nodes by Type - {loc_item}", x="_timestamp", layout=layout, runsets=[runset]))


        if loc_specific_panels:
            pg_loc = wr.PanelGrid(panels=loc_specific_panels)
            blocks.append(pg_loc)
        else:
            blocks.append(wr.P(f"No specific metrics available to display for {loc_item}."))
            
    report.blocks = blocks
    try:
        report.save()
        logger.info(f"W&B Report saved: {report.url if hasattr(report, 'url') else 'URL not available'}")
    except Exception as e:
        logger.error(f"Failed to save W&B report: {e}", exc_info=True)


def scrape_ray_tpu_usage():
    global logger
    driver = None 
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox") 
        options.add_argument("--disable-dev-shm-usage") 
        options.add_argument("--enable-javascript") 
        driver = webdriver.Chrome(options=options)
        
        dashboard_url = "http://localhost:8265" 
        logger.info(f"Attempting to scrape Ray dashboard at {dashboard_url}")
        driver.get(dashboard_url)
        
        wait = WebDriverWait(driver, 20) 
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Resource Status') or contains(text(), 'Node Resources')]")))

        soup = BeautifulSoup(driver.page_source, "html.parser")
        resource_section_options = ["Resource Status", "Node Resources"] 
        resource_section = None
        for text_option in resource_section_options:
            resource_section = soup.find(lambda tag: tag.name in ["h3", "h4", "div"] and re.search(text_option, tag.get_text(strip=True), re.IGNORECASE))
            if resource_section: break
        
        if resource_section:
            parent_container = resource_section.find_parent() 
            for _ in range(5): 
                if parent_container and (parent_container.find_all("div", string=re.compile(r'\d+\s*/\s*\d+')) or parent_container.find_all("div", string=re.compile(r'TPU', re.IGNORECASE))):
                    break
                if parent_container: parent_container = parent_container.find_parent()
                else: break 
            
            if not parent_container: parent_container = soup 

            resource_data_divs = parent_container.find_all("div")
            tpu_usage = {}
            for div in resource_data_divs:
                text = div.get_text(separator=" ", strip=True)
                match = re.search(r'(\d+)\s*/\s*(\d+)\s+([\w.-]+tpu[\w.-]*)', text, re.IGNORECASE)
                if not match: 
                     match = re.search(r'([\w.-]+tpu[\w.-]*):\s*(\d+)\s*/\s*(\d+)', text, re.IGNORECASE)
                     if match: 
                          type_key, used_val_str, total_val_str = match.groups()
                          try:
                            used_val, total_val = int(used_val_str), int(total_val_str)
                            tpu_usage[type_key.strip()] = (used_val, total_val)
                            logger.debug(f"Parsed Ray usage (P2): {type_key.strip()} -> ({used_val}/{total_val})")
                          except ValueError:
                            logger.warning(f"Could not parse Ray resource (pattern 2): '{text}'")
                          continue 
                if match:
                    try:
                        if len(match.groups()) == 3 and match.group(3).lower().count("tpu"):
                             used_val_str, total_val_str, type_key = match.groups()
                        else:
                             logger.warning(f"Unexpected match group for Ray resource: '{text}'")
                             continue
                        
                        used_val, total_val = int(used_val_str.strip()), int(total_val_str.strip())
                        tpu_usage[type_key.strip()] = (used_val, total_val)
                        logger.debug(f"Parsed Ray usage (P1): {type_key.strip()} -> ({used_val}/{total_val})")
                    except ValueError:
                        logger.warning(f"Could not parse Ray resource (pattern 1): '{text}'")
                        continue
            if not tpu_usage: logger.warning("Ray dashboard scraped, but no TPU usage data found with known patterns.")
            return tpu_usage
        else:
            logger.warning("Could not find 'Resource Status' or 'Node Resources' section in Ray dashboard.")
            return {}

    except Exception as e:
        logger.error(f"Error scraping Ray dashboard: {e}", exc_info=True)
        return {}
    finally:
        if driver:
            driver.quit()


def get_ray_tpu_usage(cli_file: str):
    global logger
    dashboard_process = None

    def run_ray_dashboard_proc_inner(config_path: str) -> subprocess.Popen | None:
        nonlocal dashboard_process
        try:
            logger.info(f"Attempting to start Ray dashboard with config: {config_path}")
            dashboard_process = subprocess.Popen(
                ["ray", "dashboard", config_path], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(20)
            if dashboard_process.poll() is None:
                logger.info(f"Ray dashboard process started for {config_path} (PID: {dashboard_process.pid}). Assuming it's healthy after 20s.")
                return dashboard_process
            else:
                stdout, stderr = dashboard_process.communicate()
                logger.error(f"Ray dashboard process for {config_path} failed to start or terminated early. Return code: {dashboard_process.returncode}. "
                             f"Stdout: {stdout.strip()}. Stderr: {stderr.strip()}")
                return None
        except FileNotFoundError:
            logger.error("Ray CLI ('ray') not found. Please ensure Ray is installed and in PATH.")
            return None
        except Exception as e:
            logger.error(f"Failed to start Ray dashboard process for {config_path}: {e}", exc_info=True)
            return None

    dashboard_process = run_ray_dashboard_proc_inner(cli_file)
    if not dashboard_process:
        return {}
    
    tpu_usage_data = scrape_ray_tpu_usage()

    logger.info(f"Terminating Ray dashboard process (PID: {dashboard_process.pid}) for {cli_file}.")
    dashboard_process.terminate()
    try:
        dashboard_process.wait(timeout=10)
        logger.info(f"Ray dashboard process for {cli_file} terminated gracefully.")
    except subprocess.TimeoutExpired:
        logger.warning(f"Ray dashboard process for {cli_file} did not terminate gracefully after SIGTERM. Sending SIGKILL.")
        dashboard_process.kill()
        try:
            dashboard_process.wait(timeout=5)
            logger.info(f"Ray dashboard process for {cli_file} killed.")
        except subprocess.TimeoutExpired:
            logger.error(f"Ray dashboard process for {cli_file} could not be killed. It may become a zombie process.")
    except Exception as e:
        logger.error(f"Error during Ray dashboard process termination for {cli_file}: {e}", exc_info=True)
        
    return tpu_usage_data


def gather_all_incomplete_tpus():
    global logger, LOCATIONS, MIN_WAIT_FOR_INCOMPLETE_TPUS
    incomplete_tpus_initial_pass = {}

    logger.info("Starting initial pass to gather incomplete TPUs across all locations...")
    for location_item in LOCATIONS:
        logger.debug(f"Gathering incomplete TPU usage for location: {location_item}")
        tpus_in_loc = gather_incomplete_tpus(location_item)
        if tpus_in_loc:
            incomplete_tpus_initial_pass[location_item] = tpus_in_loc
            logger.info(f"Found incomplete TPUs in {location_item}: {tpus_in_loc}")
    
    if not incomplete_tpus_initial_pass:
        logger.info("No incomplete TPUs found in any location during initial pass.")
        return {}

    logger.info(f"Found incomplete TPUs in some locations: {list(incomplete_tpus_initial_pass.keys())}. "
                f"Waiting {MIN_WAIT_FOR_INCOMPLETE_TPUS} minutes before re-checking these locations.")
    time.sleep(MIN_WAIT_FOR_INCOMPLETE_TPUS * 60)
    
    logger.info("Starting second pass to re-check incomplete TPUs after wait period...")
    incomplete_tpus_after_wait = {}
    for location_item, initially_found_tpus in incomplete_tpus_initial_pass.items():
        if initially_found_tpus:
            logger.debug(f"Re-checking incomplete TPU usage for location: {location_item}")
            current_tpus_for_loc_after_wait = gather_incomplete_tpus(location_item)
            if current_tpus_for_loc_after_wait:
                logger.warning(f"Location {location_item} still has incomplete TPUs after wait: {current_tpus_for_loc_after_wait}")
                incomplete_tpus_after_wait[location_item] = current_tpus_for_loc_after_wait
            else:
                logger.info(f"Location {location_item} no longer reports incomplete TPUs after wait period (resolved).")
                
    if not incomplete_tpus_after_wait:
        logger.info("All previously incomplete TPUs appear resolved after wait period.")
    else:
        logger.warning(f"Persistent incomplete TPUs found after wait: {incomplete_tpus_after_wait}")
        
    return incomplete_tpus_after_wait


all_vms_to_delete_globally = defaultdict(list)

def delete_stale_vms_global():
    global logger, all_vms_to_delete_globally
    
    if not any(all_vms_to_delete_globally.values()):
        logger.info("Global stale VM deletion: No VMs queued for deletion.")
        return

    logger.info(f"Global stale VM deletion: Attempting to delete VMs from queue: {dict(all_vms_to_delete_globally)}")
    tpu_client = tpu_v2alpha1.TpuClient()
    vms_actually_deleted_count = 0
    
    for location_key in list(all_vms_to_delete_globally.keys()):
        vm_list_for_loc = all_vms_to_delete_globally[location_key]
        if not vm_list_for_loc: continue

        logger.info(f"Global stale VM deletion: Processing {len(vm_list_for_loc)} VMs for location key '{location_key}'.")
        
        processed_vms_in_this_batch = []
        
        for vm_name_full_path in vm_list_for_loc:
            vm_short_name = vm_name_full_path.split('/')[-1]
            try:
                if not (vm_name_full_path.startswith("projects/") and "/locations/" in vm_name_full_path and "/nodes/" in vm_name_full_path):
                    logger.error(f"Global stale VM deletion: Invalid VM name format '{vm_name_full_path}'. Skipping.")
                    processed_vms_in_this_batch.append(vm_name_full_path)
                    continue

                logger.info(f"Global stale VM deletion: Attempting to delete VM {vm_short_name} ({vm_name_full_path})")
                tpu_client.delete_node(name=vm_name_full_path)
                logger.info(f"Global stale VM deletion: Successfully deleted VM {vm_short_name} from {location_key}.")
                processed_vms_in_this_batch.append(vm_name_full_path)
                vms_actually_deleted_count +=1
            except google_exceptions.NotFound:
                logger.warning(f"Global stale VM deletion: VM {vm_short_name} in {location_key} not found, likely already deleted.")
                processed_vms_in_this_batch.append(vm_name_full_path)
            except Exception as e:
                logger.error(f"Global stale VM deletion: Failed to delete VM {vm_short_name} in {location_key}: {e}", exc_info=True)
        
        all_vms_to_delete_globally[location_key] = [
            vm for vm in vm_list_for_loc if vm not in processed_vms_in_this_batch
        ]
        if not all_vms_to_delete_globally[location_key]:
            if location_key in all_vms_to_delete_globally:
                 del all_vms_to_delete_globally[location_key] 

    if vms_actually_deleted_count > 0:
        logger.info(f"Global stale VM deletion: Deleted {vms_actually_deleted_count} VMs in this run.")
    else:
        logger.info("Global stale VM deletion: No VMs were successfully deleted in this run (either none to delete, none found, or all attempts failed).")
    
    if any(all_vms_to_delete_globally.values()):
        logger.warning(f"Global stale VM deletion: Some VMs remain in queue after deletion attempts: {dict(all_vms_to_delete_globally)}")


all_metrics_global_script = Counter()

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.info("Starting __main__ execution for global TPU monitoring script (TPUMonitorActor is NOT started by this block).")
    Path("logs").mkdir(exist_ok=True)
    
    incomplete_tpus_for_deletion_check = gather_all_incomplete_tpus()

    aggregated_global_metrics_contributions = Counter()

    for main_location_item in LOCATIONS:
        logger.info(f"Processing location: {main_location_item} for global monitoring metrics.")
        
        loc_specific_incomplete_tpus = incomplete_tpus_for_deletion_check.get(main_location_item, [])
        
        metrics_loc, vms_to_del_loc, global_contrib = gather_tpu_info_from_vms(main_location_item, loc_specific_incomplete_tpus)
        
        all_metrics_global_script.update(metrics_loc)
        aggregated_global_metrics_contributions.update(global_contrib)

        if vms_to_del_loc:
            all_vms_to_delete_globally[main_location_item].extend(vms_to_del_loc)
            logger.info(f"VMs marked for deletion in {main_location_item} by global script: {len(vms_to_del_loc)}")
        
        if len(main_location_item.split('-')) >= 2 and main_location_item.split('-')[-1].isalpha(): 
            try:
                logger.info(f"Gathering Ray cluster info for zone: {main_location_item}")
                ray_info = gather_ray_cluster_info(main_location_item)
                if ray_info : all_metrics_global_script.update(ray_info)
            except Exception as e:
                logger.error(f"Failed to gather Ray cluster info for {main_location_item}: {e}", exc_info=True)
        elif main_location_item != "big-run":
            logger.info(f"Skipping Ray cluster info for {main_location_item} as it's likely a region or not a specific zone suitable for Ray head node listing.")


    all_metrics_global_script.update(aggregated_global_metrics_contributions)

    if WANDB_PROJECT and WANDB_ID:
        try:
            logger.info(f"Initializing W&B run (Project: {WANDB_PROJECT}, ID: {WANDB_ID})")
            run = wandb.init(project=WANDB_PROJECT, id=WANDB_ID, resume="allow")
            if run:
                wandb.log(dict(all_metrics_global_script), step=int(time.time())) 
                logger.info(f"Successfully logged metrics to W&B: {len(all_metrics_global_script)} items.")
                run.finish()
                logger.info("W&B run finished.")
            else:
                logger.error("Failed to initialize W&B run (wandb.init returned None).")
        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}", exc_info=True)
    else:
        logger.info("W&B project and/or ID not configured. Skipping W&B logging.")
        logger.info(f"Metrics that would be logged by global script: {dict(all_metrics_global_script)}")

    if all_metrics_global_script and WANDB_PROJECT and WANDB_ID:
       try:
           logger.info("Attempting to generate W&B report...")
           make_report()
       except Exception as e:
           logger.error(f"Failed to generate W&B report: {e}", exc_info=True)
    else:
       logger.info("No metrics collected or W&B not configured, skipping W&B report generation.")

    if any(all_vms_to_delete_globally.values()): 
        logger.info("Proceeding to delete VMs marked by global script logic...")
        delete_stale_vms_global()
    else:
        logger.info("No VMs marked for deletion by global script logic in this run.")

    logger.info("Finished __main__ execution for global TPU monitoring script.")

# Ensure a newline at the end of the file

[end of infra/tpu_monitor.py]
