# Ray Cluster Management

This directory contains scripts for managing Ray clusters.

## `manage_cluster.py`

This script is the main entry point for managing Ray clusters. It provides
commands to start, stop, restart, and get the status of a cluster.

### Usage

- **Start a cluster:**
  ```
  python -m scripts.ray.manage_cluster start --config <path_to_cluster_config>
  ```
  This command starts the cluster and restores any previously backed-up jobs.

- **Stop a cluster:**
  ```
  python -m scripts.ray.manage_cluster stop --config <path_to_cluster_config>
  ```

- **Restart a cluster:**
  ```
  python -m scripts.ray.manage_cluster restart --config <path_to_cluster_config>
  ```
  This command backs up all running jobs, restarts the cluster, and then
  restores the jobs.

- **Get cluster status:**
  ```
  python -m scripts.ray.manage_cluster status --config <path_to_cluster_config>
  ```
