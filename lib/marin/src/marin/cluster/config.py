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

"""Configuration management for cluster operations."""

import os
from dataclasses import dataclass
from pathlib import Path

import jinja2
import yaml

# Cluster configuration constants and templates
LATEST = "4554f82"  # The latest docker tag used for the clusters
LATEST_VLLM = "20251209"


@dataclass
class RayClusterConfig:
    """Type-safe representation of Ray cluster configuration from YAML."""

    # Core fields
    cluster_name: str
    config_file: str  # Path to the YAML file

    # Provider fields
    region: str
    zone: str
    project_id: str

    # Docker fields
    docker_image: str
    docker_container_name: str

    @classmethod
    def from_yaml(cls, config_path: str) -> "RayClusterConfig":
        """Load cluster configuration from Ray YAML file."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        provider = data.get("provider", {})
        docker = data.get("docker", {})

        return cls(
            cluster_name=data["cluster_name"],
            config_file=config_path,
            region=provider["region"],
            zone=provider["availability_zone"],
            project_id=provider["project_id"],
            docker_image=docker["image"],
            docker_container_name=docker.get("container_name", "ray_docker"),
        )


def get_default_config_path(region: str) -> str:
    """Get default config path for a region."""
    return f"infra/marin-{region}.yaml"


def find_config_by_region(region: str) -> str:
    """Find cluster config file by region."""
    config_path = get_default_config_path(region)
    if os.path.exists(config_path):
        return config_path

    if region.startswith("marin-"):
        region = region[len("marin-") :]

    # Try with common region variations
    variations = [
        f"infra/marin-{region}.yaml",
        f"infra/marin-{region}-a.yaml",
        f"infra/marin-{region}-b.yaml",
        f"infra/marin-{region}-vllm.yaml",
    ]

    for path in variations:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No cluster config found for region {region}")


def list_available_configs() -> list[str]:
    """List all available cluster configurations."""
    infra_dir = Path("infra")
    if not infra_dir.exists():
        return []

    configs = []
    for yaml_file in infra_dir.glob("marin-*.yaml"):
        # Skip template file
        if yaml_file.name == "marin-cluster-template.yaml":
            continue
        configs.append(str(yaml_file))

    return sorted(configs)


CONFIGS = {
    "marin-us-central2": {
        "NAME": "marin-us-central2",
        "REGION": "us-central2",
        "ZONE": "us-central2-b",
        "BUCKET": "marin-us-central2",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v4",
        "min_workers": 4,
    },
    "marin-us-central1": {
        "NAME": "marin-us-central1",
        "REGION": "us-central1",
        "ZONE": "us-central1-a",
        "BUCKET": "marin-us-central1",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v5p",
        "min_workers": 1,
        "worker_targets": {
            "v5p-8": 12,
            "v5p-16": 1,
            "v5p-32": 1,
            "v5p-64": 1,
            "v5p-128": 0,
            "v5p-256": 0,
            "v5p-512": 0,
        },
    },
    "marin-us-central1-vllm": {
        "NAME": "marin-us-central1-vllm",
        "REGION": "us-central1",
        "ZONE": "us-central1-a",
        "BUCKET": "marin-us-central1",
        "DOCKER_TAG": LATEST_VLLM,
        "tpu_generation": "v5p",
        "min_workers": 1,
        "worker_targets": {
            "v5p-8": 2,
            "v5p-16": 0,
            "v5p-32": 0,
            "v5p-64": 0,
            "v5p-128": 0,
            "v5p-256": 0,
            "v5p-512": 0,
        },
        "VLLM": True,
    },
    "marin-big-run": {
        "NAME": "marin-big-run",
        "REGION": "us-central2",
        "ZONE": "us-central2-b",
        "BUCKET": "marin-us-central2",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v4",
        "min_workers": 0,
    },
    "marin-eu-west4": {
        "NAME": "marin-eu-west4",
        "REGION": "europe-west4",
        "ZONE": "europe-west4-b",
        "BUCKET": "marin-eu-west4",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v5e",
        "min_workers": 4,
        "worker_targets": {
            "v5e-128": 1,
        },
    },
    "marin-us-west4": {
        "NAME": "marin-us-west4",
        "REGION": "us-west4",
        "ZONE": "us-west4-a",
        "BUCKET": "marin-us-west4",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v5e",
        "min_workers": 0,
    },
    "marin-us-east1": {
        "NAME": "marin-us-east1-d",
        "REGION": "us-east1",
        "ZONE": "us-east1-d",
        "BUCKET": "marin-us-east1",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v6e",
        "min_workers": 0,
        "worker_targets": {
            "v6e-128": 8,
        },
    },
    "marin-us-east5": {
        "NAME": "marin-us-east5",
        "REGION": "us-east5",
        "ZONE": "us-east5-b",
        "BUCKET": "marin-us-east5",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v6e",
        "min_workers": 0,
        "worker_targets": {
            "v6e-128": 8,
        },
    },
    "marin-us-east5-a": {
        "NAME": "marin-us-east5-a",
        "REGION": "us-east5",
        "ZONE": "us-east5-a",
        "BUCKET": "marin-us-east5",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v5p",
        "min_workers": 8,
        "worker_targets": {
            "v5p-64": 4,
            "v5p-2048": 0,
        },
    },
    "marin-us-east5-a-vllm": {
        "NAME": "marin-us-east5-a-vllm",
        "REGION": "us-east5",
        "ZONE": "us-east5-a",
        "BUCKET": "marin-us-east5",
        "DOCKER_TAG": LATEST_VLLM,
        "tpu_generation": "v5p",
        "min_workers": 1,
        "worker_targets": {
            "v5p-8": 2,
            "v5p-16": 0,
            "v5p-32": 0,
            "v5p-64": 0,
            "v5p-128": 0,
            "v5p-256": 0,
            "v5p-512": 0,
        },
        "VLLM": True,
    },
    "marin-eu-west4-a": {
        "NAME": "marin-eu-west4-a",
        "REGION": "europe-west4",
        "ZONE": "europe-west4-a",
        "BUCKET": "marin-eu-west4",
        "DOCKER_TAG": LATEST,
        "tpu_generation": "v6e",
        "min_workers": 0,
        "worker_targets": {
            "v6e-128": 2,
        },
    },
    "marin-us-east5-b-vllm": {
        "NAME": "marin-us-east5-b-vllm",
        "REGION": "us-east5",
        "ZONE": "us-east5-b",
        "BUCKET": "marin-us-east5",
        "DOCKER_TAG": "6e804a10",
        "tpu_generation": "v6e-serve",
        "min_workers": 2,
        "VLLM": True,
    },
    "marin-eu-west4-vllm": {
        "NAME": "marin-eu-west4-vllm",
        "REGION": "europe-west4",
        "ZONE": "europe-west4-b",
        "BUCKET": "marin-eu-west4",
        "DOCKER_TAG": "7fab502e",
        "tpu_generation": "v5e",
        "min_workers": 2,
        "VLLM": True,
    },
    "marin-us-central2-vllm": {
        "NAME": "marin-us-central2-vllm",
        "REGION": "us-central2",
        "ZONE": "us-central2-b",
        "BUCKET": "marin-us-central2",
        "DOCKER_TAG": "1bc975e12",
        "tpu_generation": "v4",
        "min_workers": 2,
        "VLLM": True,
    },
    "marin-us-east1-d-vllm": {
        "NAME": "marin-us-east1-d-vllm",
        "REGION": "us-east1",
        "ZONE": "us-east1-d",
        "BUCKET": "marin-us-east1",
        "DOCKER_TAG": "1bc975e12",
        "tpu_generation": "v6e-serve",
        "min_workers": 2,
        "VLLM": True,
    },
}

GENERATION_CONFIGS = {
    "v4": {
        "runtime_version": "tpu-ubuntu2204-base",
        "base_worker": "8",
        "slices": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
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
    "v5p": {
        "runtime_version": "v2-alpha-tpuv5",
        "base_worker": "8",
        "slices": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "num_tpus": 4,
        "tpus_worker": 8,
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
    "v4-serve": {
        "runtime_version": "tpu-ubuntu2204-base",
        "base_worker": "16",
        "slices": [],
        "num_tpus": 4,
    },
}


def make_tpu_slice_config(generation: str, count: int, target_count: int) -> dict[str, dict]:
    """Create TPU slice configuration for worker nodes."""
    slice_gen_name = "v5litepod" if generation == "v5e" else generation

    if "serve" in generation:
        slice_gen_name = generation.replace("-serve", "")
    name = f"tpu_slice_{generation}_{count}"
    return {
        name: {
            "min_workers": target_count,
            "max_workers": 1024,
            "resources": {"CPU": 120, "TPU": GENERATION_CONFIGS[generation]["num_tpus"]},
            "node_config": {
                "acceleratorType": f"{slice_gen_name}-{count}",
                "runtimeVersion": GENERATION_CONFIGS[generation]["runtime_version"],
                "schedulingConfig": {"preemptible": True},
            },
        }
    }


def get_template_path(config_name: str, infra_path: str = "infra") -> str:
    """Get the template path for a given config."""
    cluster_template_path = os.path.join(infra_path, "marin-cluster-template.yaml")
    vllm_template_path = os.path.join(infra_path, "marin-vllm-template.yaml")

    if CONFIGS[config_name].get("VLLM", False):
        return vllm_template_path
    return cluster_template_path


def make_tpu_worker_config(generation: str, count: int, min_workers: int = 4) -> dict:
    """Create TPU worker configuration."""
    _, config = next(iter(make_tpu_slice_config(generation, count, min_workers).items()))
    return {"tpu_worker": config}


def update_cluster_configs(infra_path: str = "infra") -> None:
    """Generate all cluster configuration files from templates."""
    for config_name, config in CONFIGS.items():
        config_file_path = os.path.join(infra_path, f"{config_name}.yaml")

        with open(config_file_path, "w") as f:
            template_path = get_template_path(config_name, infra_path)
            with open(template_path) as f_template:
                template = jinja2.Template(f_template.read())

            yaml_string = template.render(**config)

            # pyyaml strips comments, which we'd like to keep
            # so instead of using yaml.dump, we'll write the string directly after
            # appending worker types
            # (we need to indent it by 2 spaces)
            # available_node_types:
            generation = config["tpu_generation"]
            generation_config = GENERATION_CONFIGS[generation]
            worker_config = make_tpu_worker_config(generation, generation_config["base_worker"], config["min_workers"])
            base_string = yaml.dump(worker_config, default_flow_style=False, indent=2)
            base_string = "\n  " + base_string.replace("\n", "\n  ")
            yaml_string += base_string

            for tpu_type in generation_config["slices"]:
                target_worker_count = config.get("worker_targets", {}).get(f"{generation}-{tpu_type}", 0)
                base_string = yaml.dump(
                    make_tpu_slice_config(generation, tpu_type, target_worker_count),
                    default_flow_style=False,
                    indent=2,
                )
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
