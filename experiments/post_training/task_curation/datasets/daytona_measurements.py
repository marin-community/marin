# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete actor-only censuses at exact input pins; source artifacts remain immutable."""

from finestore.rl.curation import DaytonaMeasurement

DAYTONA_MEASUREMENTS = {
    source_id: DaytonaMeasurement.model_validate(record)
    for source_id, record in {
        "Campaign Release:smartie-calibforge-rebuilt-v1": {
            "dataset_id": "open-athena/calibforge-v1-rebuilt-taskcompendium-20261111",
            "dataset_revision": "8f00c12884d6a68107b245e7c98711b6bd64f4af",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; "
                "production Harbor machine_spec -> "
                "shellbox snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native task.toml or "
                "published runtime, absent values "
                "null; no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 1357,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 1357}},
                "grader": {"daytona": {"failures": {}, "machine_slots": 1357}},
            },
            "daytona_snapshot_count": 7,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:swesmith-verifier-repaired-v5": {
            "dataset_id": "open-athena/swesmith-verifier-repaired-v5-20261112",
            "dataset_revision": "26bb8c8ffba53adecd1c2a5e9d64b47b9f7e52fd",
            "daytona_count_basis": (
                "Native Harbor production machine_spec "
                "-> snapshot_identity; "
                "all7016pinnedpackages; noAPIcalls; "
                "https://huggingface.co/datasets/open-athena/swesmith-verifier-repaired-v5-20261112/blob/"
                "26bb8c8ffba53adecd1c2a5e9d64b47b9f7e52fd/daytona-identity.json; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 7016,
            "daytona_roles": {"agent": {"daytona": {"failures": {}, "machine_slots": 7016}}},
            "daytona_snapshot_count": 5,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-code-contests-built-image": {
            "dataset_id": "open-athena/tasktrove-code-contests-taskcompendium-20261112",
            "dataset_revision": "5b43c6f4c2322a5808e4dbdfaf4f0052037d3607",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native "
                "task.toml or published runtime, "
                "absent values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 8222,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 8222}},
                "grader": {"daytona": {"failures": {}, "machine_slots": 8222}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-codeforces-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-codeforces-v1-taskcompendium-20261112",
            "dataset_revision": "05167566303b67a1adea08743d26c39101c5c30a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native task.toml "
                "or published runtime, absent values "
                "null; no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 9745,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 9745}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 9745},
                        "machine_slots": 9745,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-competitive-coding-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-competitive-coding-v1-taskcompendium-20261112",
            "dataset_revision": "5a7b7a75eb32901947157e39516a15d894db332b",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock "
                "grader on LOCAL; production "
                "Harbor machine_spec -> "
                "shellbox snapshot_identity; "
                "converter checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version "
                "2026.11.12.1; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 13973,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 13973}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 13973},
                        "machine_slots": 13973,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-curriculum-easy-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-curriculum-easy-v1-taskcompendium-20261112",
            "dataset_revision": "4151b071dbb38df532b38c7de1becb12ca1e0f73",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor "
                "on Daytona, package-lock "
                "grader on LOCAL; production "
                "Harbor machine_spec -> "
                "shellbox snapshot_identity; "
                "converter checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native "
                "task.toml or published "
                "runtime, absent values null; "
                "no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 505,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 505}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 505},
                        "machine_slots": 505,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-curriculum-medium-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-curriculum-medium-v1-taskcompendium-20261112",
            "dataset_revision": "db17ab9b01d960d01f7a75c3d1d896504ceaf9fa",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock grader "
                "on LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native "
                "task.toml or published "
                "runtime, absent values null; "
                "no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 488,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 488}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 488},
                        "machine_slots": 488,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-e2egit-large-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-e2egit-large-v1-taskcompendium-20261112",
            "dataset_revision": "4bdaeaaf069eb6ad7b0a716d445e9e789f37a758",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native task.toml "
                "or published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4993,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4993}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 4993},
                        "machine_slots": 4993,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-e2egit-v2-rebuilt": {
            "dataset_id": "open-athena/tasktrove-e2egit-v2-taskcompendium-20261112",
            "dataset_revision": "dd34672091d917c96c922fd2ebf0d583a1dfc5b9",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; "
                "production Harbor machine_spec -> "
                "shellbox snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native task.toml or "
                "published runtime, absent values null; "
                "no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 477,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 477}},
                "grader": {"local": {"failures": {}, "machine_slots": 477}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-multifile-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-multifile-v1-taskcompendium-20261112",
            "dataset_revision": "5accbb5851dc5774a05424aa7d22f995cbfd7e34",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native task.toml or "
                "published runtime, absent values "
                "null; no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4842,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4842}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 4842},
                        "machine_slots": 4842,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-nl2bash-rebuilt-v1": {
            "dataset_id": "open-athena/nl2bash-v1-rebuilt-taskcompendium-20261111",
            "dataset_revision": "1be6be13a89ab3ef30c7390ca2f29d7b9eeb40dd",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; "
                "production Harbor machine_spec -> "
                "shellbox snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native task.toml or "
                "published runtime, absent values null; "
                "no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 1498,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 1498}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 1498},
                        "machine_slots": 1498,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-pymethods2test-large-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-pymethods2test-large-v1-taskcompendium-20261112",
            "dataset_revision": "a9ed5c72ade6670853fc28af8bf1b59541d3827c",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock "
                "grader on LOCAL; "
                "production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; "
                "converter checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version "
                "2026.11.12.1; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4990,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4990}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 4990},
                        "machine_slots": 4990,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-pymethods2test-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-pymethods2test-v1-taskcompendium-20261112",
            "dataset_revision": "f6a20442763e152ec662483dc0ee1476caa6d541",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native "
                "task.toml or published runtime, "
                "absent values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 500,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 500}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 500},
                        "machine_slots": 500,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-stack-pytest-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-stack-pytest-v1-taskcompendium-20261112",
            "dataset_revision": "f6909b5b5fe8d207fb442f24d4140556431d02a3",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native task.toml "
                "or published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 492,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 492}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 492},
                        "machine_slots": 492,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-taco-built-image": {
            "dataset_id": "open-athena/tasktrove-taco-taskcompendium-20261112",
            "dataset_revision": "c96a95bc94e4824fae425811f246501f64abfadd",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; production "
                "Harbor machine_spec -> shellbox "
                "snapshot_identity; converter checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; allocations "
                "from native task.toml or published "
                "runtime, absent values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 5116,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 5116}},
                "grader": {"daytona": {"failures": {}, "machine_slots": 5116}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-unitsyn-python-large-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-unitsyn-python-large-v1-taskcompendium-20261112",
            "dataset_revision": "3a64ac3ed258bd184c24c24e7c7578f64142c180",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock "
                "grader on LOCAL; "
                "production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; "
                "converter checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version "
                "2026.11.12.1; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4991,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4991}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 4991},
                        "machine_slots": 4991,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Campaign Release:tasktrove-unitsyn-python-rebuilt-v1": {
            "dataset_id": "open-athena/tasktrove-unitsyn-python-v1-taskcompendium-20261112",
            "dataset_revision": "3ed40d26a51c8ca0cf26c15cc84316409534a937",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 2026.11.12.1; "
                "allocations from native "
                "task.toml or published runtime, "
                "absent values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4991,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4991}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 4991},
                        "machine_slots": 4991,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 3; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 1494,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 1494}},
                "grader": {"local": {"failures": {}, "machine_slots": 1494}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__code-contests-noblock": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; production "
                "Harbor machine_spec -> shellbox "
                "snapshot_identity; converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 4; allocations from native "
                "task.toml or published runtime, absent "
                "values null; no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 8222,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 8222}},
                "grader": {"daytona": {"failures": {}, "machine_slots": 8222}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_curriculum-easy": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; production "
                "Harbor machine_spec -> shellbox "
                "snapshot_identity; converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 5; allocations from "
                "native task.toml or published runtime, "
                "absent values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 505,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 505}},
                "grader": {"local": {"failures": {}, "machine_slots": 505}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_curriculum-medium-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 6; allocations from "
                "native task.toml or published "
                "runtime, absent values null; no "
                "Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 488,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 488}},
                "grader": {"local": {"failures": {}, "machine_slots": 488}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_e2egit-large": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, package-lock "
                "grader on LOCAL; production Harbor "
                "machine_spec -> shellbox snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 5; allocations from native "
                "task.toml or published runtime, absent "
                "values null; no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4993,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4993}},
                "grader": {"local": {"failures": {}, "machine_slots": 4993}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_e2egit-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs before "
                "admission gates (post-18a population); actor "
                "on Daytona, package-lock grader on LOCAL; "
                "production Harbor machine_spec -> shellbox "
                "snapshot_identity; converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 8; allocations from native "
                "task.toml or published runtime, absent values "
                "null; no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 477,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 477}},
                "grader": {"local": {"failures": {}, "machine_slots": 477}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_multifile-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, package-lock "
                "grader on LOCAL; production Harbor "
                "machine_spec -> shellbox snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 5; allocations from native "
                "task.toml or published runtime, absent "
                "values null; no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4842,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4842}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 4842},
                        "machine_slots": 4842,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_pymethods2test-large-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 5; allocations "
                "from native task.toml or "
                "published runtime, absent values "
                "null; no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4990,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4990}},
                "grader": {"local": {"failures": {}, "machine_slots": 4990}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_pymethods2test-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; "
                "production Harbor machine_spec -> "
                "shellbox snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 5; allocations from "
                "native task.toml or published runtime, "
                "absent values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 500,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 500}},
                "grader": {"local": {"failures": {}, "machine_slots": 500}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_stack-pytest-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; production "
                "Harbor machine_spec -> shellbox "
                "snapshot_identity; converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 5; allocations from "
                "native task.toml or published runtime, "
                "absent values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 492,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 492}},
                "grader": {"local": {"failures": {}, "machine_slots": 492}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_unitsyn-python-large-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 5; allocations "
                "from native task.toml or "
                "published runtime, absent values "
                "null; no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 4991,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 4991}},
                "grader": {"local": {"failures": {}, "machine_slots": 4991}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__exp_rpt_unitsyn-python-v4": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; "
                "production Harbor machine_spec -> "
                "shellbox snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 5; allocations from "
                "native task.toml or published runtime, "
                "absent values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 491,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 491}},
                "grader": {"local": {"failures": {}, "machine_slots": 491}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:DCAgent__swe_rebench_v2_patched_oracle-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 3; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 13562,
            "daytona_machine_slots": 13562,
            "daytona_roles": {
                "agent": {
                    "daytona": {
                        "failures": {
                            "agent: Unresolved Docker build context must be built and pinned before execution": 13562
                        },
                        "machine_slots": 13562,
                    }
                },
                "grader": {
                    "daytona": {
                        "failures": {
                            "grader: Unresolved Docker build context must be built and pinned before execution": 13562
                        },
                        "machine_slots": 13562,
                    }
                },
            },
            "daytona_snapshot_count": 14,
            "daytona_unsafe_counts": {"too_many_snapshots": 13562, "unbuilt_image": 13562},
            "daytona_unsafe_details": {
                "too_many_snapshots": "14 " "actor " "snapshot " "identities " "exceed " "the " "10-snapshot " "budget",
                "unbuilt_image": (
                    "agent: "
                    "Unresolved "
                    "Docker "
                    "build "
                    "context "
                    "must be "
                    "built and "
                    "pinned "
                    "before "
                    "execution"
                ),
            },
            "daytona_unsafe_reasons": ["too_many_snapshots", "unbuilt_image"],
        },
        "Task Trove:SankalpKJ__nemotron-math-oracle-filtered-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission "
                "gates (post-18a population); "
                "actor on Daytona, "
                "package-lock grader on LOCAL; "
                "production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {"agent": {}, "grader": {"local": {"failures": {}, "machine_slots": 57777}}},
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__all-puzzles-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs before "
                "admission gates (post-18a population); actor on "
                "Daytona, package-lock grader on LOCAL; production "
                "Harbor machine_spec -> shellbox snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; recipe "
                "version 2; allocations from native task.toml or "
                "published runtime, absent values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {"agent": {}, "grader": {}},
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__codeforces-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs before "
                "admission gates (post-18a population); actor on "
                "Daytona, package-lock grader on LOCAL; production "
                "Harbor machine_spec -> shellbox snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; recipe "
                "version 1; allocations from native task.toml or "
                "published runtime, absent values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 9745,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 9745}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 9745},
                        "machine_slots": 9745,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__exp_rpt_taco-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs before "
                "admission gates (post-18a population); actor on "
                "Daytona, package-lock grader on LOCAL; production "
                "Harbor machine_spec -> shellbox snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; recipe "
                "version 5; allocations from native task.toml or "
                "published runtime, absent values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 5114,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 5114}},
                "grader": {"daytona": {"failures": {}, "machine_slots": 5114}},
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__glaive-code-assistant-sandboxes-verified-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs "
                "before admission gates "
                "(post-18a population); "
                "actor on Daytona, "
                "package-lock grader on "
                "LOCAL; production "
                "Harbor machine_spec -> "
                "shellbox "
                "snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; "
                "allocations from "
                "native task.toml or "
                "published runtime, "
                "absent values null; no "
                "Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 10000},
                        "machine_slots": 10000,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemo-prism-math-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs before "
                "admission gates (post-18a population); actor on "
                "Daytona, package-lock grader on LOCAL; "
                "production Harbor machine_spec -> shellbox "
                "snapshot_identity; converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; allocations from native "
                "task.toml or published runtime, absent values "
                "null; no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {"agent": {}, "grader": {"local": {"failures": {}, "machine_slots": 2404}}},
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-agent-calendar-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; allocations from "
                "native task.toml or published "
                "runtime, absent values null; no "
                "Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 2699},
                        "machine_slots": 2699,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-arc-agi-python-inductive-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock "
                "grader on LOCAL; "
                "production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; "
                "converter checkout "
                "34753b0e74e73d36dc0aec25c3e3f09574c86fd7; "
                "recipe version 2; "
                "allocations from native "
                "task.toml or published "
                "runtime, absent values "
                "null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 10000,
            "daytona_machine_slots": 10000,
            "daytona_roles": {
                "agent": {
                    "daytona": {
                        "failures": {"agent: Daytona requires a registry image or Docker build context": 10000},
                        "machine_slots": 10000,
                    }
                },
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 10000},
                        "machine_slots": 10000,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {"host_image_required": 10000},
            "daytona_unsafe_details": {
                "host_image_required": (
                    "agent: " "Daytona " "requires " "a " "registry " "image " "or " "Docker " "build " "context"
                )
            },
            "daytona_unsafe_reasons": ["host_image_required"],
        },
        "Task Trove:laion__nemotron-gym-arc-agi-transductive-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission "
                "gates (post-18a population); "
                "actor on Daytona, "
                "package-lock grader on LOCAL; "
                "production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "34753b0e74e73d36dc0aec25c3e3f09574c86fd7; "
                "recipe version 2; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 10000,
            "daytona_machine_slots": 10000,
            "daytona_roles": {
                "agent": {
                    "daytona": {
                        "failures": {"agent: Daytona requires a registry image or Docker build context": 10000},
                        "machine_slots": 10000,
                    }
                },
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 10000},
                        "machine_slots": 10000,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {"host_image_required": 10000},
            "daytona_unsafe_details": {
                "host_image_required": (
                    "agent: " "Daytona " "requires " "a " "registry " "image " "or " "Docker " "build " "context"
                )
            },
            "daytona_unsafe_reasons": ["host_image_required"],
        },
        "Task Trove:laion__nemotron-gym-competitive-coding-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 13973,
            "daytona_roles": {
                "agent": {"daytona": {"failures": {}, "machine_slots": 13973}},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 13973},
                        "machine_slots": 13973,
                    }
                },
            },
            "daytona_snapshot_count": 1,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-instruction-following-calendar-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered "
                "converter TaskSpec "
                "outputs before "
                "admission gates "
                "(post-18a "
                "population); actor "
                "on Daytona, "
                "package-lock grader "
                "on LOCAL; "
                "production Harbor "
                "machine_spec -> "
                "shellbox "
                "snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; "
                "allocations from "
                "native task.toml or "
                "published runtime, "
                "absent values null; "
                "no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 5673},
                        "machine_slots": 5673,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-instruction-following-structured-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered "
                "converter "
                "TaskSpec outputs "
                "before admission "
                "gates (post-18a "
                "population); "
                "actor on Daytona, "
                "package-lock "
                "grader on LOCAL; "
                "production Harbor "
                "machine_spec -> "
                "shellbox "
                "snapshot_identity; "
                "converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; "
                "allocations from "
                "native task.toml "
                "or published "
                "runtime, absent "
                "values null; no "
                "Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {"agent": {}, "grader": {}},
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-instruction-following-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock grader "
                "on LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; "
                "allocations from native "
                "task.toml or published "
                "runtime, absent values null; "
                "no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {"agent": {}, "grader": {}},
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-knowledge-mcqa-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 6; allocations from "
                "native task.toml or published "
                "runtime, absent values null; no "
                "Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {"agent": {}, "grader": {}},
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-knowledge-openqa-v4": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 4; allocations "
                "from native task.toml or "
                "published runtime, absent values "
                "null; no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 122357},
                        "machine_slots": 122357,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-math-openmathreasoning-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock "
                "grader on LOCAL; production "
                "Harbor machine_spec -> "
                "shellbox snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; "
                "allocations from native "
                "task.toml or published "
                "runtime, absent values "
                "null; no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 42636},
                        "machine_slots": 42636,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-math-stack-overflow-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor "
                "on Daytona, package-lock "
                "grader on LOCAL; production "
                "Harbor machine_spec -> "
                "shellbox snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 110729},
                        "machine_slots": 110729,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-math-v5": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs before "
                "admission gates (post-18a population); actor "
                "on Daytona, package-lock grader on LOCAL; "
                "production Harbor machine_spec -> shellbox "
                "snapshot_identity; converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; allocations from native "
                "task.toml or published runtime, absent values "
                "null; no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 4096},
                        "machine_slots": 4096,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-multichallenge-advanced-v4": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock "
                "grader on LOCAL; "
                "production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; "
                "converter checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 2; "
                "allocations from native "
                "task.toml or published "
                "runtime, absent values "
                "null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 1068},
                        "machine_slots": 1068,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-reasoning-gym-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "518dd8c50508c1ec02e52d64261a36fdca6c5ac3; "
                "recipe version 4; allocations from "
                "native task.toml or published "
                "runtime, absent values null; no "
                "Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 14259},
                        "machine_slots": 14259,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-safety-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, package-lock "
                "grader on LOCAL; production Harbor "
                "machine_spec -> shellbox snapshot_identity; "
                "converter checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 2; allocations from native "
                "task.toml or published runtime, absent "
                "values null; no Daytona API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 89066},
                        "machine_slots": 89066,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-science-so-openq-v3": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 4; allocations "
                "from native task.toml or "
                "published runtime, absent values "
                "null; no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 150644},
                        "machine_slots": 150644,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__nemotron-gym-structured-outputs-v4": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec "
                "outputs before admission gates "
                "(post-18a population); actor on "
                "Daytona, package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; converter "
                "checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 2; allocations "
                "from native task.toml or "
                "published runtime, absent "
                "values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 50606},
                        "machine_slots": 50606,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__stackexchange-codereview-sandboxes-verified-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered "
                "converter TaskSpec "
                "outputs before "
                "admission gates "
                "(post-18a "
                "population); actor "
                "on Daytona, "
                "package-lock grader "
                "on LOCAL; "
                "production Harbor "
                "machine_spec -> "
                "shellbox "
                "snapshot_identity; "
                "converter checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 2; "
                "allocations from "
                "native task.toml or "
                "published runtime, "
                "absent values null; "
                "no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 10000},
                        "machine_slots": 10000,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__stackexchange-overflow-sandboxes-verified-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs "
                "before admission "
                "gates (post-18a "
                "population); actor on "
                "Daytona, package-lock "
                "grader on LOCAL; "
                "production Harbor "
                "machine_spec -> "
                "shellbox "
                "snapshot_identity; "
                "converter checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 2; "
                "allocations from "
                "native task.toml or "
                "published runtime, "
                "absent values null; "
                "no Daytona API calls; "
                "evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 10000},
                        "machine_slots": 10000,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__stackexchange-superuser-sandboxes-verified-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs "
                "before admission "
                "gates (post-18a "
                "population); actor "
                "on Daytona, "
                "package-lock grader "
                "on LOCAL; production "
                "Harbor machine_spec "
                "-> shellbox "
                "snapshot_identity; "
                "converter checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 2; "
                "allocations from "
                "native task.toml or "
                "published runtime, "
                "absent values null; "
                "no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 10000},
                        "machine_slots": 10000,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__stackexchange-tezos-sandboxes-verified-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates "
                "(post-18a population); "
                "actor on Daytona, "
                "package-lock grader on "
                "LOCAL; production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; "
                "converter checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 2; "
                "allocations from native "
                "task.toml or published "
                "runtime, absent values "
                "null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 10000},
                        "machine_slots": 10000,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__stackexchange-unix-sandboxes-verified-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter "
                "TaskSpec outputs before "
                "admission gates (post-18a "
                "population); actor on "
                "Daytona, package-lock "
                "grader on LOCAL; "
                "production Harbor "
                "machine_spec -> shellbox "
                "snapshot_identity; "
                "converter checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; "
                "recipe version 2; "
                "allocations from native "
                "task.toml or published "
                "runtime, absent values "
                "null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 10000},
                        "machine_slots": 10000,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
        "Task Trove:laion__swesmith-oracle-filtered-v2": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs "
                "before admission gates (post-18a "
                "population); actor on Daytona, "
                "package-lock grader on LOCAL; "
                "production Harbor machine_spec -> "
                "shellbox snapshot_identity; converter "
                "checkout "
                "93040a060d7bb0c719321af505ce510e79e0e2fc; "
                "recipe version 4; allocations from "
                "native task.toml or published runtime, "
                "absent values null; no Daytona API "
                "calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 12720,
            "daytona_machine_slots": 12720,
            "daytona_roles": {
                "agent": {
                    "daytona": {
                        "failures": {
                            "agent: Unresolved Docker build context must be built and pinned before execution": 12720
                        },
                        "machine_slots": 12720,
                    }
                },
                "grader": {
                    "daytona": {
                        "failures": {
                            "grader: Unresolved Docker build context must be built and pinned before execution": 12720
                        },
                        "machine_slots": 12720,
                    }
                },
            },
            "daytona_snapshot_count": 5,
            "daytona_unsafe_counts": {"unbuilt_image": 12720},
            "daytona_unsafe_details": {
                "unbuilt_image": (
                    "agent: Unresolved " "Docker build " "context must be " "built and pinned " "before execution"
                )
            },
            "daytona_unsafe_reasons": ["unbuilt_image"],
        },
        "Task Trove:laion__wizardlm-orca-v4": {
            "dataset_id": "open-thoughts/TaskTrove",
            "dataset_revision": "02923004846e4e73862c20962f823a6d05100e7a",
            "daytona_count_basis": (
                "Registered converter TaskSpec outputs before "
                "admission gates (post-18a population); actor on "
                "Daytona, package-lock grader on LOCAL; production "
                "Harbor machine_spec -> shellbox "
                "snapshot_identity; converter checkout "
                "2d71bf7d25f7b2166fdee9999bf0c121148264c9; recipe "
                "version 2; allocations from native task.toml or "
                "published runtime, absent values null; no Daytona "
                "API calls; evidence "
                "s3://marin-us-east-02a/marin/rl-data/atlas/daytona-census/20261112/daytona-measurements-post18a-9a7ab5539ebb5557.json"
            ),
            "daytona_execution_failures": 0,
            "daytona_machine_slots": 0,
            "daytona_roles": {
                "agent": {},
                "grader": {
                    "local": {
                        "failures": {"The local backend does not provide disk quotas or GPU allocations": 9999},
                        "machine_slots": 9999,
                    }
                },
            },
            "daytona_snapshot_count": 0,
            "daytona_unsafe_counts": {},
            "daytona_unsafe_details": {},
            "daytona_unsafe_reasons": [],
        },
    }.items()
}
