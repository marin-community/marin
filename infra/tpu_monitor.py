from collections import Counter

import ray
from google.cloud import compute_v1, tpu_v2alpha1

PROJECT_NAME = "hai-gcp-models"

BAD_STATES = [tpu_v2alpha1.Node.State.PREEMPTED, tpu_v2alpha1.Node.State.TERMINATED]
GOOD_STATES = [tpu_v2alpha1.Node.State.READY]


def gather_tpu_info_from_vms(location):
    """
    Gathers TPU information from the TPU API and logs metrics.
    """
    tpu_client = tpu_v2alpha1.TpuClient()
    # Get all TPU nodes
    parent = f"projects/{PROJECT_NAME}/locations/{location}"
    nodes = tpu_client.list_nodes(parent=parent)

    total_devices_zone = 0
    total_preeemptible_devices_zone = 0

    nodes_types_zone = Counter()
    tpu_by_generation = Counter()

    vms_to_delete = []
    for node in nodes:
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
            total_preeemptible_devices_zone += total_devices_this_tpu

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
    to_log[f"{location}/devices/total_preemptible"] = total_preeemptible_devices_zone

    to_log["devices/total"] = total_devices_zone
    to_log["devices/total_preemptible"] = total_preeemptible_devices_zone

    return to_log, vms_to_delete


def gather_ray_cluster_info(location):
    """
    Gathers Ray cluster information and logs metrics.
    """
    compute_client = compute_v1.InstancesClient()

    # Get all instances in the location
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

    to_log = {}
    for resource_name, available in ray_resources_per_cluster.items():
        to_log[f"{location}/{resource_name}"] = available

    return to_log


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


WANDB_PROJECT = "marin-monitoring"
WANDB_ID = "tpu-monitoring-v3-testing"


def delete_stale_vms():
    compute_client = tpu_v2alpha1.TpuClient()
    for _, vms in all_vms_to_delete.items():
        for vm in vms:
            try:
                compute_client.delete_node(name=vm)
                print(f"Deleted VM {vm}")
            except Exception as e:
                print(f"Failed to delete VM {vm}: {e}")
                continue


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
    for location in locations:
        blocks.append(wr.H2(location))
        device_gen_loc = [k for k in all_metrics.keys() if k.startswith(location)]
        panels = []
        panels.append(wr.LinePlot(y=device_gen_loc, title="Devices", x=walltime, layout=layout))
        pg = wr.PanelGrid(runsets=[runset], panels=panels)
        blocks.append(pg)
    report.blocks = blocks
    report.save()


if __name__ == "__main__":
    import wandb

    locations = [
        "asia-northeast1-b",
        "europe-west4-a",
        "europe-west4-b",
        "us-central2-b",
        "us-east1-d",
        "us-east5-a",
        "us-west4-a",
    ]
    all_metrics = Counter()
    all_vms_to_delete = {}
    for location in locations:
        metrics, vms_to_delete = gather_tpu_info_from_vms(location)
        all_metrics.update(metrics)
        if vms_to_delete:
            all_vms_to_delete[location] = vms_to_delete
        # this is broken right now
        # try:
        #     all_metrics.update(gather_ray_cluster_info(location))
        # except Exception as e:
        #     print(f"Failed to gather Ray cluster info for {location}: {e}")

    r = wandb.init(project=WANDB_PROJECT, id=WANDB_ID, resume="allow")
    wandb.log(all_metrics)
    print(all_metrics)

    wandb.finish()

    # make_report()

    if all_vms_to_delete:
        print("Deleting VMs")
        delete_stale_vms()
