# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed backend censuses for installed-app evidence journeys."""

from taskcompendium.backend_requirements import BackendSourceMap


def backend_source_maps() -> list[BackendSourceMap]:
    common = {
        "basis": "converted_spec_census",
        "task_count": 3,
        "inputs": {"s3://test/task-specs.parquet": "d" * 64},
        "demands": {},
        "resource_ranges": {},
        "role_backend_assignments": {},
        "grading_environment": None,
    }
    records = [
        {
            "source_id": "Marin:clean",
            "demands": {"agent": {"docker_image": 3}, "grader": {"packages_lock": 3}},
            "role_backend_assignments": {"agent": {"daytona": 3}, "grader": {"local": 3}},
            "backends": {
                "docker": {
                    "support": "supported",
                    "task_counts": {"supported": 3},
                    "roles": {"agent": {"supported": 3}, "grader": {"supported": 3}},
                    "blockers": {},
                    "validation_errors": {},
                },
                "local": {
                    "support": "unsupported",
                    "task_counts": {"unsupported": 3},
                    "roles": {"agent": {"unsupported": 3}, "grader": {"supported": 3}},
                    "blockers": {"agent:docker_image": 3},
                    "validation_errors": {},
                },
                "daytona": {
                    "support": "unsupported",
                    "task_counts": {"unsupported": 3},
                    "roles": {"agent": {"supported": 3}, "grader": {"unsupported": 3}},
                    "blockers": {"grader:packages_lock": 3},
                    "validation_errors": {},
                },
            },
        },
        {
            "source_id": "Marin:eval",
            "demands": {"grader": {"host_image": 3}},
            "role_backend_assignments": {"grader": {"local": 3}},
            "backends": {
                "local": {
                    "support": "supported",
                    "task_counts": {"supported": 3},
                    "roles": {"agent": {"role_absent": 3}, "grader": {"supported": 3}},
                    "not_applicable": {},
                    "blockers": {},
                    "validation_errors": {},
                },
                "shellsim": {
                    "support": "not_applicable",
                    "task_counts": {"not_applicable": 3},
                    "roles": {"agent": {"role_absent": 3}, "grader": {"not_applicable": 3}},
                    "not_applicable": {"grader:host_image": 3},
                    "blockers": {},
                    "validation_errors": {},
                },
            },
        },
        {
            "source_id": "Marin:stale-clean",
            "backends": {
                "qemu": {
                    "support": "unsupported",
                    "task_counts": {"unsupported": 2, "supported": 1},
                    # A missing legacy role remains unknown, unlike an explicit role_absent census.
                    "roles": {"agent": {"unsupported": 2, "supported": 1}},
                    "blockers": {"agent:packages_lock": 2},
                    "validation_errors": {"agent:packages_lock": 1},
                }
            },
        },
        {
            "source_id": "Marin:clean-subset",
            "backends": {
                "local": {
                    "support": "role_absent",
                    "task_counts": {"role_absent": 3},
                    "roles": {"agent": {"role_absent": 3}, "grader": {"role_absent": 3}},
                    "not_applicable": {},
                    "blockers": {},
                    "validation_errors": {},
                }
            },
        },
    ]
    return [BackendSourceMap.model_validate({**common, **record}) for record in records]
