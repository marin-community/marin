# Uses the marin-cluster-template.yaml file to create the three cluster configuration files.
import os

import jinja2
import yaml

this_path = os.path.dirname(os.path.abspath(__file__))

cluster_template_path = os.path.join(this_path, "marin-cluster-template.yaml")
vllm_template_path = os.path.join(this_path, "marin-vllm-template.yaml")

# LAtest tahs ABh:  latest 4a47ffc0 20250305

DOCKER_TAGS = {
    "us-central2": "4a47ffc0",
    "big-run": "6da1c9ed",
    "us-west4": "89b461b3",
    "europe-west4": "89b461b3",
    "us-east1": "b63a1e90",
    "us-east5": "6da1c9ed",
    # NB: different naming convention because we have two zones in europe-west4
    "europe-west4-a": "6da1c9ed",
    "asia-northeast1": "6da1c9ed",
    "marin-us-east5-b-vllm": "296d2ef0",
}

configs = {
    "marin-us-central2": {
        "NAME": "marin-us-central2",
        "REGION": "us-central2",
        "ZONE": "us-central2-b",
        "BUCKET": "marin-us-central2",
        "DOCKER_TAG": DOCKER_TAGS["us-central2"],
        "tpu_generation": "v4",
        "min_workers": 4,
    },
    "marin-big-run": {
        "NAME": "marin-big-run",
        "REGION": "us-central2",
        "ZONE": "us-central2-b",
        "BUCKET": "marin-us-central2",
        "DOCKER_TAG": DOCKER_TAGS["big-run"],
        "tpu_generation": "v4",
        "min_workers": 0,
    },
    "marin-eu-west4": {
        "NAME": "marin-eu-west4",
        "REGION": "europe-west4",
        "ZONE": "europe-west4-b",
        "BUCKET": "marin-eu-west4",
        "DOCKER_TAG": DOCKER_TAGS["europe-west4"],
        "tpu_generation": "v5e",
        "min_workers": 0,
    },
    "marin-us-west4": {
        "NAME": "marin-us-west4",
        "REGION": "us-west4",
        "ZONE": "us-west4-a",
        "BUCKET": "marin-us-west4",
        "DOCKER_TAG": DOCKER_TAGS["us-west4"],
        "tpu_generation": "v5e",
        "min_workers": 0,
    },
    "marin-us-east1": {
        "NAME": "marin-us-east1-d",
        "REGION": "us-east1",
        "ZONE": "us-east1-d",
        "BUCKET": "marin-us-east1",
        "DOCKER_TAG": DOCKER_TAGS["us-east1"],
        "tpu_generation": "v6e",
        "min_workers": 0,
    },
    "marin-us-east5": {
        "NAME": "marin-us-east5",
        "REGION": "us-east5",
        "ZONE": "us-east5-b",
        "BUCKET": "marin-us-east5",
        "DOCKER_TAG": DOCKER_TAGS["us-east5"],
        "tpu_generation": "v6e",
        "min_workers": 0,
    },
    "marin-eu-west4-a": {
        "NAME": "marin-eu-west4-a",
        "REGION": "europe-west4",
        "ZONE": "europe-west4-a",
        "BUCKET": "marin-eu-west4",
        "DOCKER_TAG": DOCKER_TAGS["europe-west4-a"],
        "tpu_generation": "v6e",
        "min_workers": 0,
    },
    "marin-asia-northeast1": {
        "NAME": "marin-asia-northeast1",
        "REGION": "asia-northeast1",
        "ZONE": "asia-northeast1-b",
        "BUCKET": "marin-asia-northeast1",
        "DOCKER_TAG": DOCKER_TAGS["asia-northeast1"],
        "tpu_generation": "v6e",
        "min_workers": 0,
    },
    "marin-us-east5-b-vllm": {
        "NAME": "marin-us-east5-b-vllm",
        "REGION": "us-east5",
        "ZONE": "us-east5-b",
        "BUCKET": "marin-us-east5",
        "DOCKER_TAG": DOCKER_TAGS["marin-us-east5-b-vllm"],
        "tpu_generation": "v6e-serve",
        "min_workers": 2,
        "VLLM": True,
    },
}

generation_configs = {
    "v4": {
        "runtime_version": "tpu-ubuntu2204-base",
        "base_worker": "8",
        "slices": [16, 32, 64, 128, 256],
        "num_tpus": 4,
        "tpus_worker": 4,
    },
    "v5e": {
        "runtime_version": "v2-alpha-tpuv5-lite",
        "base_worker": "4",
        "slices": [8, 16, 32, 64, 128, 256],
        "num_tpus": 4,
        "tpus_worker": 1,
    },
    "v6e": {
        "runtime_version": "v2-alpha-tpuv6e",
        "base_worker": "4",
        "slices": [8, 16, 32, 64, 128, 256],
        "num_tpus": 4,
    },
    "v6e-serve": {
        "runtime_version": "v2-alpha-tpuv6e",
        "base_worker": "8",
        "slices": [],
        "num_tpus": 8,
    },
}


def make_tpu_slice_config(generation, count) -> dict[str, dict]:
    slice_gen_name = "v5litepod" if generation == "v5e" else generation
    slice_gen_name = "v6e" if generation == "v6e-serve" else slice_gen_name
    name = f"tpu_slice_{generation}_{count}"
    return {
        name: {
            "min_workers": 0,
            "max_workers": 1024,
            "resources": {"CPU": 120, "TPU": generation_configs[generation]["num_tpus"]},
            "node_config": {
                "acceleratorType": f"{slice_gen_name}-{count}",
                "runtimeVersion": generation_configs[generation]["runtime_version"],
                "schedulingConfig": {"preemptible": True},
            },
        }
    }


def get_template_path(config_name):
    if configs[config_name].get("VLLM", False):
        return vllm_template_path
    return cluster_template_path


def make_tpu_worker_config(generation, count, min_workers=4):
    _, config = next(iter(make_tpu_slice_config(generation, count).items()))
    config["min_workers"] = min_workers
    return {"tpu_worker": config}


if __name__ == "__main__":
    for config_name, config in configs.items():
        with open(os.path.join(this_path, f"{config_name}.yaml"), "w") as f:
            with open(get_template_path(config_name)) as f_template:
                template = jinja2.Template(f_template.read())

            yaml_string = template.render(**config)

            # pyyaml strips comments, which i'd like to keep
            # so instead of using yaml.dump, we'll write the string directly after appending worker types
            # (we need to indent it by 2 spaces)
            # available_node_types:
            generation = config["tpu_generation"]
            generation_config = generation_configs[generation]
            worker_config = make_tpu_worker_config(generation, generation_config["base_worker"], config["min_workers"])
            base_string = yaml.dump(worker_config, default_flow_style=False, indent=2)
            base_string = "\n  " + base_string.replace("\n", "\n  ")
            yaml_string += base_string

            for tpu_type in generation_config["slices"]:
                base_string = yaml.dump(make_tpu_slice_config(generation, tpu_type), default_flow_style=False, indent=2)
                base_string = "\n  " + base_string.replace("\n", "\n  ")
                yaml_string += base_string

            # Remove trailing whitespace from each line:
            lines = yaml_string.splitlines()
            lines = [line.rstrip() for line in lines]
            yaml_string = "\n".join(lines)

            f.write("#####################################################\n")
            f.write("#           THIS FILE IS AUTOGENERATED              #\n")
            f.write("# Update the template or the script, not this file! #\n")
            f.write("#####################################################\n")
            f.write(yaml_string)

            print(f"Generated {config_name} config")
