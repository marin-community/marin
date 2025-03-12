import datetime
import subprocess
import sys
import threading
import time
from collections import Counter

import ray
import wandb
from bs4 import BeautifulSoup
from google.cloud import compute_v1, tpu_v2alpha1
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

sys.path.append("../..")

PROJECT_NAME = "hai-gcp-models"
WANDB_PROJECT = "marin-monitoring"
WANDB_ID = "tpu-monitoring-v3-testing"

BAD_STATES = [tpu_v2alpha1.Node.State.PREEMPTED, tpu_v2alpha1.Node.State.TERMINATED]
GOOD_STATES = [tpu_v2alpha1.Node.State.READY]
MIN_WAIT_FOR_INCOMPLETE_TPUS = 15

LOCATIONS = [
    # "asia-northeast1-b",
    # "europe-west4-a",
    # "europe-west4-b",
    "us-central2-b",
    "us-east1-d",
    "us-east5-a",
    "us-east5-b",
    "us-west4-a",
]

LOCATION_TO_CLI_FILE = {
    "asia-northeast1-b": "/home/abhinavg/marin/infra/marin-asia-northeast1.yaml",
    "europe-west4-a": "/home/abhinavg/marin/infra/marin-eu-west4-a.yaml",
    "europe-west4-b": "/home/abhinavg/marin/infra/marin-eu-west4.yaml",
    "us-central2-b": "/home/abhinavg/marin/infra/marin-us-central2.yaml",
    "us-east1-d": "/home/abhinavg/marin/infra/marin-us-east1.yaml",
    "us-east5-a": "/home/abhinavg/marin/infra/marin-us-east5.yaml",
    "us-east5-b": "/home/abhinavg/marin/infra/marin-us-east5-b-vllm.yaml",
    "us-west4-a": "/home/abhinavg/marin/infra/marin-us-west4.yaml",
}


def gather_incomplete_tpus(location):
    """Gather names of TPUs that do not have a power of 2 usage."""
    incomplete_usage = []
    if location in LOCATIONS:
        ray_usage = get_ray_tpu_usage(LOCATION_TO_CLI_FILE[location])
        if ray_usage:
            for tpu_type, (used, total) in ray_usage.items():
                total = int(total)
                if total & (total - 1) != 0:  # Bitwise check for power of 2
                    incomplete_usage.append((tpu_type, used, total))
    return incomplete_usage


def gather_tpu_info_from_vms(location, incomplete_tpus):
    """Gather TPU information from the TPU API and log metrics."""
    tpu_client = tpu_v2alpha1.TpuClient()
    parent = f"projects/{PROJECT_NAME}/locations/{location}"
    nodes = tpu_client.list_nodes(parent=parent)

    total_devices_zone = 0
    total_preemptible_devices_zone = 0

    nodes_types_zone = Counter()
    tpu_by_generation = Counter()
    vms_to_delete = []
    for node in nodes:
        for (incomplete_tpu, used, total) in incomplete_tpus:
            if incomplete_tpu in node.name:
                print(f"Node {node.name} of type {node.accelerator_type} does not have a power of 2 usage, deleting")
                with open("incomplete_tpus.log", "a") as f:
                    timestamp = datetime.datetime.now().isoformat()
                    f.write(f"{timestamp},{node.name},{node.accelerator_type},{location},{used},{total}\n")
                vms_to_delete.append(node.name)
                continue
        if node.state in BAD_STATES:
            print(f"Node {node.name} is in state {node.state}, deleting")
            vms_to_delete.append(node.name)
            continue
        if node.state not in GOOD_STATES:
            print(f"Node {node.name} is in state {node.state}, skipping")
            continue

        # Get TPU configuration (e.g., v4-64)
        tpu_config = node.accelerator_type
        # TODO: log health?
        # Get the number of chips for this TPU type
        # Log metrics to Cloud Monitoring
        total_devices_this_tpu = int(tpu_config.split("-")[-1])
        total_devices_zone += total_devices_this_tpu
        generation = tpu_config.split("-")[0]

        is_preemptible = node.scheduling_config.preemptible
        if is_preemptible:
            total_preemptible_devices_zone += total_devices_this_tpu

        nodes_types_zone[tpu_config] += 1
        tpu_by_generation[generation] += int(tpu_config.split("-")[-1])

    to_log = Counter()

    for tpu_type, count in nodes_types_zone.items():
        to_log[f"{location}/devices/{tpu_type}"] = count
        to_log[f"devices/{tpu_type}"] = count

    for generation, count in tpu_by_generation.items():
        to_log[f"{location}/devices/{generation}"] = count
        to_log[f"devices/{generation}"] = count

    to_log[f"{location}/devices/total"] = total_devices_zone
    to_log[f"{location}/devices/total_preemptible"] = total_preemptible_devices_zone

    to_log["devices/total"] = total_devices_zone
    to_log["devices/total_preemptible"] = total_preemptible_devices_zone

    return to_log, vms_to_delete


def gather_ray_cluster_info(location):
    """Gather Ray cluster information and log metrics."""
    compute_client = compute_v1.InstancesClient()
    instances = compute_client.list(project=PROJECT_NAME, zone=location)
    ray_resources_per_cluster = Counter()

    for instance in instances:

        # Identify Ray head nodes (replace with your logic)
        if is_ray_head_node(instance):
            try:
                # Connect to the Ray cluster
                print(f"Connecting to Ray cluster on {instance.name}")
                ray.init(address=f"{get_ip(instance)}:6379")
                available_resources = ray.available_resources()
                # cluster_resources = ray.cluster_resources()

                # Log metrics for available resources
                for resource_name, available in available_resources.items():
                    if is_tpu_head_resource(resource_name):
                        tpu_type, devices = parse_head_resource(resource_name)
                        ray_resources_per_cluster[tpu_type] += devices * available
                        ray_resources_per_cluster[f"{tpu_type}-{devices}"] += available

            except Exception as e:
                print(f"Failed to connect to Ray cluster on {instance.name}: {e}")
            finally:
                ray.shutdown()

    return {f"{location}/{k}": v for k, v in ray_resources_per_cluster.items()}


def get_ip(instance):
    """
    Returns the external IP of the instance.
    """
    for interface in instance.network_interfaces:
        for config in interface.access_configs:
            if hasattr(config, "nat_i_p"):
                return config.nat_i_p
    raise ValueError(f"Could not find external IP for instance {instance.name}")


def is_ray_head_node(instance):
    """
    Returns whether the instance is a Ray head node.
    """
    # looks like: ray-marin-eu-west4-a-head-9060241a-compute
    return "ray-" in instance.name and "-head-" in instance.name and "-compute" in instance.name


def is_tpu_head_resource(resource_name):
    """
    Returns whether the resource is a TPU resource.
    """
    # looks like TPU-v4-64-head
    return "TPU-" in resource_name and "-head" in resource_name


def parse_head_resource(resource_name):
    """
    Parses the resource name and returns the TPU type and number of devices.
    """
    # looks like TPU-v4-64-head
    _, tpu_type, devices, _ = resource_name.split("-")
    return tpu_type, int(devices)


def make_report():
    global blocks, location, report
    import wandb_workspaces.reports.v2 as wr

    report = wr.Report(project=WANDB_PROJECT, title="TPU Monitoring", description="TPU Monitoring")
    # juts this run id
    # TODO: how to runset?
    runset = wr.Runset()
    layout = wr.Layout(w=24, h=9)
    # first make the overall panel grid
    blocks = []
    blocks.append(wr.H1("TPU Monitoring"))
    panels = []
    walltime = "WallTime"
    panels.append(wr.LinePlot(y=["devices/total"], title="Total Devices", x=walltime, layout=layout))
    panels.append(
        wr.LinePlot(y=["devices/total_preemptible"], title="Total Preemptible Devices", x=walltime, layout=layout)
    )
    device_gen = [k for k in all_metrics.keys() if k.startswith("devices/v")]
    panels.append(wr.LinePlot(y=device_gen, title="Devices", x=walltime, layout=layout))
    pg = wr.PanelGrid(runsets=[runset], panels=panels)
    blocks.append(pg)
    # Make 1 panel grid per location
    for location in LOCATIONS:
        blocks.append(wr.H2(location))
        device_gen_loc = [k for k in all_metrics.keys() if k.startswith(location)]
        panels = []
        panels.append(wr.LinePlot(y=device_gen_loc, title="Devices", x=walltime, layout=layout))
        pg = wr.PanelGrid(runsets=[runset], panels=panels)
        blocks.append(pg)
    report.blocks = blocks
    report.save()


def scrape_ray_tpu_usage():
    """Scrape TPU usage data from Ray dashboard."""
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--enable-javascript")
        driver = webdriver.Chrome(options=options)

        driver.get("http://localhost:8265")
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        wait.until(EC.presence_of_element_located((By.XPATH, "//h3[contains(text(), 'Resource Status')]")))

        soup = BeautifulSoup(driver.page_source, "html.parser")
        resource_section = soup.find("h3", string="Resource Status")

        if resource_section:
            resource_container = resource_section.find_parent("div").find_parent("div")
            resource_data = resource_container.find_all("div")

            tpu_usage = {}
            for div in resource_data:
                text = div.get_text().strip()
                if "/" in text and "tpu" in text:
                    try:
                        used, total = text.split("/")
                        used = int(used.strip())
                        total = int(total.split()[0].strip())
                        tpu_usage[text.split()[-1]] = (used, total)
                    except ValueError:
                        continue

            return tpu_usage
        return {}

    except Exception as e:
        print(f"Error scraping Ray dashboard: {e}")
        return {}
    finally:
        if "driver" in locals():
            driver.quit()


def get_ray_tpu_usage(cli_file):
    """Get TPU usage from Ray dashboard."""

    def run_ray_dashboard(config_path):
        try:
            print(f"Starting Ray dashboard with config {config_path}")
            process = subprocess.Popen(["ray", "dashboard", config_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(10)
            print(f"Started Ray dashboard with config {config_path}")
            return process
        except FileNotFoundError:
            print("Ray CLI not found.")
            return None

    dashboard_process = run_ray_dashboard(cli_file)
    if not dashboard_process:
        return {}

    dashboard_thread = threading.Thread(target=lambda: dashboard_process.wait())
    dashboard_thread.daemon = True
    dashboard_thread.start()

    tpu_usage = scrape_ray_tpu_usage()

    dashboard_process.terminate()
    dashboard_process.wait()

    return tpu_usage


def gather_all_incomplete_tpus():
    """Check for incomplete TPUs across all locations."""
    incomplete_tpus = {location: gather_incomplete_tpus(location) for location in LOCATIONS}

    if any(incomplete_tpus.values()):
        print(f"Found incomplete TPUs, waiting {MIN_WAIT_FOR_INCOMPLETE_TPUS} minutes to check again...")

        time.sleep(MIN_WAIT_FOR_INCOMPLETE_TPUS * 60)

        incomplete_tpus_after = {}
        for location, tpus in incomplete_tpus.items():
            if tpus:
                incomplete_tpus_after[location] = gather_incomplete_tpus(location)


        return incomplete_tpus_after

    return incomplete_tpus


def delete_stale_vms():
    """Delete VMs marked for deletion."""
    compute_client = tpu_v2alpha1.TpuClient()
    for vms in all_vms_to_delete.values():
        for vm in vms:
            try:
                compute_client.delete_node(name=vm)
                print(f"Deleted VM {vm}")
            except Exception as e:
                print(f"Failed to delete VM {vm}: {e}")


if __name__ == "__main__":
    incomplete_tpus = gather_all_incomplete_tpus()

    all_metrics = Counter()
    all_vms_to_delete = {}

    for location in LOCATIONS:
        metrics, vms_to_delete = gather_tpu_info_from_vms(location, incomplete_tpus[location])
        all_metrics.update(metrics)
        if vms_to_delete:
            all_vms_to_delete[location] = vms_to_delete
        # this is broken right now
        # try:
        #     all_metrics.update(gather_ray_cluster_info(location))
        # except Exception as e:
        #     print(f"Failed to gather Ray cluster info for {location}: {e}")

    run = wandb.init(project=WANDB_PROJECT, id=WANDB_ID, resume="allow")
    wandb.log(all_metrics)
    print(all_metrics)
    wandb.finish()

    # make_report()

    if all_vms_to_delete:
        print("Deleting VMs")
        delete_stale_vms()
