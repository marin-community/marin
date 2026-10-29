# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Promote each persisted primary and its old companions into one service list.

Inline files remain in job_workdir_files. Existing flat columns are retained as
the primary worker projection; resource columns continue to hold the task total.
Rows already converted are untouched, including after an interrupted migration.
"""

import json


def migrate(raw_conn) -> None:
    rows = raw_conn.execute(
        "SELECT job_id, environment_json, entrypoint_json, task_image, "
        "res_cpu_millicores, res_memory_bytes, res_disk_bytes, res_device_json FROM job_config"
    ).fetchall()
    for job_id, environment_json, entrypoint_json, image, cpu, memory, disk, device_json in rows:
        environment = json.loads(environment_json)
        if environment.get("services"):
            continue
        companions = environment.pop("sandbox_services", [])
        primary_name = environment.pop("primary_container_name", "") or "task"
        resources = {
            "cpu_millicores": int(cpu),
            "memory_bytes": int(memory),
            "disk_bytes": int(disk),
        }
        for companion in companions:
            for name in ("cpu_millicores", "memory_bytes", "disk_bytes"):
                resources[name] = int(resources[name]) - int(companion.get("resources", {}).get(name, 0))
                if resources[name] < 0:
                    raise ValueError(f"Stored task {job_id} has inconsistent service resource reservations")
        if device_json:
            resources["device"] = json.loads(device_json)
        primary = {
            "name": primary_name,
            "image": image or "",
            "workdir": "/app",
            "resources": resources,
            "env_vars": dict(environment.get("env_vars", {})),
            "secrets": dict(environment.get("secrets", {})),
            "setup_scripts": list(environment.get("setup_scripts", [])),
            "runtime_entrypoint": json.loads(entrypoint_json),
        }
        services = [primary]
        for companion in companions:
            service = dict(companion)
            service["image_command"] = {
                name: service.pop(name) for name in ("entrypoint", "arguments") if name in service
            }
            services.append(service)
        environment["services"] = services
        environment["primary_service"] = primary_name
        raw_conn.execute(
            "UPDATE job_config SET environment_json = ? WHERE job_id = ?",
            (json.dumps(environment), job_id),
        )
