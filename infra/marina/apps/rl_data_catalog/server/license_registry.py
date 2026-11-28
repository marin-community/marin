# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned user decisions; Python packaging preserves this registry in hosted backends."""

LICENSE_BANS = r"""{
  "version": "2026.11.26.b-license-exceptions.2",
  "bans": [
    {
      "source_id": "FWeindel:validated-tasks/tmax11-v1.1",
      "upstream_id": "FWeindel/validated-tasks/tmax11-v1.1",
      "upstream_revision": "317659ac88c8b706d318020c5fc3813a931fb6ec",
      "classification": "unknown",
      "license_ids": [
        "other"
      ],
      "reason": "User ruling: per-folder ban under the doubt rule; the upstream card is license: other, no terms.",
      "evidence_urls": [
        "https://huggingface.co/datasets/TMaxxx/tmax-13k-v1.1/tree/8d47de78dc85ae7f8cf4853b6777a0917e49fad9",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-16T20:26:00-05:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "FWeindel:validated-tasks/tmax11_v1.1",
      "upstream_id": "FWeindel/validated-tasks/tmax11_v1.1",
      "upstream_revision": "317659ac88c8b706d318020c5fc3813a931fb6ec",
      "classification": "unknown",
      "license_ids": [
        "other"
      ],
      "reason": "User ruling: per-folder ban under the doubt rule; the upstream card is license: other, no terms.",
      "evidence_urls": [
        "https://huggingface.co/datasets/TMaxxx/tmax-13k-v1.1/tree/8d47de78dc85ae7f8cf4853b6777a0917e49fad9",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-16T20:26:00-05:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "FWeindel:validated-tasks/termigen-v3",
      "upstream_id": "FWeindel/validated-tasks/termigen-v3",
      "upstream_revision": "317659ac88c8b706d318020c5fc3813a931fb6ec",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "Per-folder lineage decision: no license at the pinned upstream (F2); the doubt rule is binding.",
      "evidence_urls": [
        "https://github.com/ucsb-mlsec/terminal-bench-env/tree/03bddac74ada2fa344df0e93b4a3ace2034294d1",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-16T21:10:00-05:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "FWeindel:validated-tasks/termigen-v2",
      "upstream_id": "FWeindel/validated-tasks/termigen-v2",
      "upstream_revision": "317659ac88c8b706d318020c5fc3813a931fb6ec",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "Per-folder lineage decision: no license at the pinned upstream (F2); the doubt rule is binding.",
      "evidence_urls": [
        "https://github.com/ucsb-mlsec/terminal-bench-env/tree/03bddac74ada2fa344df0e93b4a3ace2034294d1",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-16T21:10:00-05:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "FWeindel:validated-tasks/termigen-v1",
      "upstream_id": "FWeindel/validated-tasks/termigen-v1",
      "upstream_revision": "317659ac88c8b706d318020c5fc3813a931fb6ec",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "Per-folder lineage decision: no license at the pinned upstream (F2); the doubt rule is binding.",
      "evidence_urls": [
        "https://github.com/ucsb-mlsec/terminal-bench-env/tree/03bddac74ada2fa344df0e93b4a3ace2034294d1",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-16T21:10:00-05:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "FWeindel:validated-tasks/bugsinpy-v3",
      "upstream_id": "FWeindel/validated-tasks/bugsinpy-v3",
      "upstream_revision": "317659ac88c8b706d318020c5fc3813a931fb6ec",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "Per-folder lineage decision: no license at the pinned upstream (F2); the doubt rule is binding.",
      "evidence_urls": [
        "https://github.com/soarsmu/BugsInPy/tree/11c5f1eea954a42132cfd06bf257766a7963e0fd",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-16T21:10:00-05:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "FWeindel:validated-tasks/bugsinpy-v2",
      "upstream_id": "FWeindel/validated-tasks/bugsinpy-v2",
      "upstream_revision": "317659ac88c8b706d318020c5fc3813a931fb6ec",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "Per-folder lineage decision: no license at the pinned upstream (F2); the doubt rule is binding.",
      "evidence_urls": [
        "https://github.com/soarsmu/BugsInPy/tree/11c5f1eea954a42132cfd06bf257766a7963e0fd",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-16T21:10:00-05:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "FWeindel:validated-tasks/bugsinpy-v1",
      "upstream_id": "FWeindel/validated-tasks/bugsinpy-v1",
      "upstream_revision": "317659ac88c8b706d318020c5fc3813a931fb6ec",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "Per-folder lineage decision: no license at the pinned upstream (F2); the doubt rule is binding.",
      "evidence_urls": [
        "https://github.com/soarsmu/BugsInPy/tree/11c5f1eea954a42132cfd06bf257766a7963e0fd",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-16T21:10:00-05:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "ASDiv:asdiv",
      "upstream_id": "chaochun/nlu-asdiv-dataset",
      "upstream_revision": "883f90a9a65bf00304ba8f37423910fe743abc47",
      "classification": "restricted",
      "license_ids": [
        "cc-by-nc-4.0"
      ],
      "reason": "Pinned upstream dataset README explicitly declares CC BY-NC 4.0.",
      "evidence_urls": [
        "https://github.com/chaochun/nlu-asdiv-dataset/blob/883f90a9a65bf00304ba8f37423910fe743abc47/README.md"
      ],
      "decided_at": "2026-11-15T17:27:51.574598+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "AllenAI:rlvr_math",
      "upstream_id": "allenai/RLVR-MATH",
      "upstream_revision": "bd2a93551b503a395fadd1a740d957559cfe6f3c",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/allenai/RLVR-MATH/resolve/bd2a93551b503a395fadd1a740d957559cfe6f3c/README.md"
      ],
      "decided_at": "2026-11-15T17:27:51.574606+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "Evalchemy:aime24",
      "upstream_id": "aime24",
      "upstream_revision": "2dfe4f134aee7989cd9978218ca914b3c3db4685ea06d841560ec1391ba0a409",
      "classification": "unknown",
      "reason": "No declared primary license or verified upstream origin pin for the bundled benchmark.",
      "evidence_urls": [
        "https://raw.githubusercontent.com/marin-community/marin/623134de90723e7f189036da005067f36817b6e7/vendor/evalchemy/eval/chat_benchmarks/AIME24/data/aime24.json"
      ],
      "decided_at": "2026-11-15T17:12:29.218480+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "unknown"
      ]
    },
    {
      "source_id": "Evalchemy:aime25",
      "upstream_id": "aime25",
      "upstream_revision": "9affac318c4dd399482a67226936e229f954628567c7d7b7ee8f1360c15d6f7c",
      "classification": "unknown",
      "reason": "No declared primary license or verified upstream origin pin for the bundled benchmark.",
      "evidence_urls": [
        "https://raw.githubusercontent.com/marin-community/marin/623134de90723e7f189036da005067f36817b6e7/vendor/evalchemy/eval/chat_benchmarks/AIME25/data/aime25.json"
      ],
      "decided_at": "2026-11-15T17:12:29.218481+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "unknown"
      ]
    },
    {
      "source_id": "Evalchemy:amc23",
      "upstream_id": "amc23",
      "upstream_revision": "02ae54425741562778916b5fa8f6ff07749689cfcd7a53e65e1d88a89e5065f9",
      "classification": "unknown",
      "reason": "No declared primary license or verified upstream origin pin for the bundled benchmark.",
      "evidence_urls": [
        "https://raw.githubusercontent.com/marin-community/marin/623134de90723e7f189036da005067f36817b6e7/vendor/evalchemy/eval/chat_benchmarks/AMC23/data/amc23.json"
      ],
      "decided_at": "2026-11-15T17:12:29.218481+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "unknown"
      ]
    },
    {
      "source_id": "Evalchemy:boolq",
      "upstream_id": "aps/super_glue",
      "upstream_revision": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
      "classification": "unknown",
      "reason": "Pinned upstream license is custom or unknown; no permissive grant established.",
      "evidence_urls": [
        "https://huggingface.co/datasets/aps/super_glue/resolve/3de24cf8022e94f4ee4b9d55a6f539891524d646/README.md"
      ],
      "decided_at": "2026-11-15T17:12:29.218478+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "other"
      ]
    },
    {
      "source_id": "Evalchemy:financebench",
      "upstream_id": "PatronusAI/financebench",
      "upstream_revision": "e04404e3a97f69f79c14d42f24981a1c9c3bcd18",
      "classification": "restricted",
      "reason": "Publisher dataset card declares CC-BY-NC-4.0.",
      "evidence_urls": [
        "https://huggingface.co/datasets/PatronusAI/financebench/blob/e04404e3a97f69f79c14d42f24981a1c9c3bcd18/README.md"
      ],
      "decided_at": "2026-11-15T17:12:29.218368+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "cc-by-nc-4.0"
      ]
    },
    {
      "source_id": "Evalchemy:hellaswag",
      "upstream_id": "Rowan/hellaswag",
      "upstream_revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
      "classification": "missing",
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/Rowan/hellaswag/resolve/218ec52e09a7e7462a5400043bb9a69a41d06b76/README.md"
      ],
      "decided_at": "2026-11-15T17:12:29.218475+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "missing"
      ]
    },
    {
      "source_id": "Evalchemy:math500",
      "upstream_id": "HuggingFaceH4/MATH-500",
      "upstream_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/README.md"
      ],
      "decided_at": "2026-11-15T17:27:51.574582+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "Evalchemy:olympiadbench",
      "upstream_id": "olympiadbench",
      "upstream_revision": "bcafaabc0128adfeef69c830d1682254428887d4e1b96b5b1b03fb5d3818f958",
      "classification": "unknown",
      "reason": "No declared primary license or verified upstream origin pin for the bundled benchmark.",
      "evidence_urls": [
        "https://raw.githubusercontent.com/marin-community/marin/623134de90723e7f189036da005067f36817b6e7/vendor/evalchemy/eval/chat_benchmarks/OlympiadBench/data/olympiadbench.jsonl"
      ],
      "decided_at": "2026-11-15T17:12:29.218482+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "unknown"
      ]
    },
    {
      "source_id": "Evalchemy:olympiadbenchdeterministic",
      "upstream_id": "olympiadbenchdeterministic",
      "upstream_revision": "bcafaabc0128adfeef69c830d1682254428887d4e1b96b5b1b03fb5d3818f958",
      "classification": "unknown",
      "reason": "No declared primary license or verified upstream origin pin for the bundled benchmark.",
      "evidence_urls": [
        "https://raw.githubusercontent.com/marin-community/marin/623134de90723e7f189036da005067f36817b6e7/vendor/evalchemy/eval/chat_benchmarks/OlympiadBench/data/olympiadbench.jsonl"
      ],
      "decided_at": "2026-11-15T17:12:29.218482+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "unknown"
      ]
    },
    {
      "source_id": "Evalchemy:openbookqa",
      "upstream_id": "allenai/openbookqa",
      "upstream_revision": "388097ea7776314e93a529163e0fea805b8a6454",
      "classification": "unknown",
      "reason": "Pinned upstream license is custom or unknown; no permissive grant established.",
      "evidence_urls": [
        "https://huggingface.co/datasets/allenai/openbookqa/resolve/388097ea7776314e93a529163e0fea805b8a6454/README.md"
      ],
      "decided_at": "2026-11-15T17:12:29.218479+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "unknown"
      ]
    },
    {
      "source_id": "Evalchemy:piqa",
      "upstream_id": "baber/piqa",
      "upstream_revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/baber/piqa/tree/142f6d7367fd9877f0fb3b5734ea6a545f54cdd1?recursive=true"
      ],
      "decided_at": "2026-11-15T17:27:51.574383+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "Evalchemy:triviaqa",
      "upstream_id": "mandarjoshi/trivia_qa",
      "upstream_revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
      "classification": "unknown",
      "reason": "Pinned upstream license is custom or unknown; no permissive grant established.",
      "evidence_urls": [
        "https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/README.md"
      ],
      "decided_at": "2026-11-15T17:12:29.218479+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "unknown"
      ]
    },
    {
      "source_id": "Evalchemy:winogrande",
      "upstream_id": "allenai/winogrande",
      "upstream_revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
      "classification": "missing",
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/allenai/winogrande/resolve/01e74176c63542e6b0bcb004dcdea22d94fb67b5/README.md"
      ],
      "decided_at": "2026-11-15T17:12:29.218477+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "license_ids": [
        "missing"
      ]
    },
    {
      "source_id": "Harbor Hub:skill2env/skill2env",
      "upstream_id": "unverified-origin:587210ec6d25",
      "upstream_revision": "6198459210ab98e0c9c4e021b20f50e93c41b9d4",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "Applicable permissive origin grant and exact release lineage have not been established.",
      "evidence_urls": [
        "https://github.com/marin-community/marin/blob/b35a72d3335dcd03e5f711cb09c2436ebf374bfe/experiments/post_training/task_curation/sources.py"
      ],
      "decided_at": "2026-11-15T17:27:51.574669+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "HuggingFaceH4:aime24",
      "upstream_id": "HuggingFaceH4/aime_2024",
      "upstream_revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/HuggingFaceH4/aime_2024/resolve/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/README.md"
      ],
      "decided_at": "2026-11-15T17:27:51.574595+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "HuggingFaceH4:math500",
      "upstream_id": "HuggingFaceH4/MATH-500",
      "upstream_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/README.md"
      ],
      "decided_at": "2026-11-15T17:27:51.574604+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "Open-R1:verifiable_code",
      "upstream_id": "open-r1/verifiable-coding-problems-python",
      "upstream_revision": "b761a24a95fa03289a231d2d31c183636ffb9833",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python/resolve/b761a24a95fa03289a231d2d31c183636ffb9833/README.md"
      ],
      "decided_at": "2026-11-15T17:27:51.574608+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "TaskTrove:GAIR__OpenSWE__openswe_oss",
      "upstream_id": "unverified-origin:2ac88a442692",
      "upstream_revision": "dc6df1f3f949439ad1fdd27ffa034a913184405c",
      "classification": "unknown",
      "license_ids": [
        "apache-2.0",
        "cc-by-4.0",
        "mit",
        "other"
      ],
      "reason": "Applicable permissive origin grant and exact release lineage have not been established.",
      "evidence_urls": [
        "https://github.com/marin-community/marin/blob/b35a72d3335dcd03e5f711cb09c2436ebf374bfe/experiments/post_training/task_curation/sources.py"
      ],
      "decided_at": "2026-11-15T17:27:51.574674+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "TaskTrove:GAIR__OpenSWE__openswe_other",
      "upstream_id": "unverified-origin:7ec063219ea6",
      "upstream_revision": "aec964ce07a928fbd7c6ba21e5c1dfbd54e38285",
      "classification": "unknown",
      "license_ids": [
        "apache-2.0",
        "cc-by-4.0",
        "mit",
        "other"
      ],
      "reason": "Applicable permissive origin grant and exact release lineage have not been established.",
      "evidence_urls": [
        "https://github.com/marin-community/marin/blob/b35a72d3335dcd03e5f711cb09c2436ebf374bfe/experiments/post_training/task_curation/sources.py"
      ],
      "decided_at": "2026-11-15T17:27:51.574676+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "pafitis:hardmath",
      "upstream_id": "pafitis/HARDMath_processed_training",
      "upstream_revision": "937e9f10356e31e854f6efb9a2507f1e200c8b25",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/pafitis/HARDMath_processed_training/resolve/937e9f10356e31e854f6efb9a2507f1e200c8b25/README.md"
      ],
      "decided_at": "2026-11-15T17:27:51.574601+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "exception": {
        "upstream_revision": "937e9f10356e31e854f6efb9a2507f1e200c8b25",
        "use": "train",
        "receipt_sha256": "fe67508421dd34ee42d5dbb92926320697d211a173981fa98c331c043565b2ef",
        "rationale": "Reversing user ruling: repackager ruling plus verified MIT lineage clear train use at this pin.",
        "evidence_urls": [
          "https://github.com/sarahmart/HARDMath/blob/9ade062a80b7d475666b11ae13a2fb0d7b7f0341/LICENSE",
          "https://huggingface.co/datasets/open-athena/atlas-source-pins/blob/eec74e0a1f6e7a8c81f382d193a82ae00ced069e/hardmath_lineage.json",
          "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.25.1/inputs/hardmath-license-receipt-v1.json",
          "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.25.1/inputs/hardmath-folder-lineage-receipt-v1.json",
          "file:///Users/benfeuer/Documents/experiments/active/atlas-rulings/repackager-license-follows-upstream-20261116.md"
        ],
        "decided_at": "2026-11-26T07:45:00-05:00",
        "decision_by": "user via coordinator (Claude session b610073d)"
      }
    },
    {
      "source_id": "pafitis:hardmath@rhs",
      "upstream_id": "pafitis/HARDMath_processed_training",
      "upstream_revision": "937e9f10356e31e854f6efb9a2507f1e200c8b25",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "No license declaration in pinned upstream primary evidence.",
      "evidence_urls": [
        "https://huggingface.co/datasets/pafitis/HARDMath_processed_training/resolve/937e9f10356e31e854f6efb9a2507f1e200c8b25/README.md"
      ],
      "decided_at": "2026-11-15T17:27:51.574601+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "exception": {
        "upstream_revision": "937e9f10356e31e854f6efb9a2507f1e200c8b25",
        "use": "train",
        "receipt_sha256": "eb4f7527271cd23953b93e0b3c8466e91ebf4f1853f098135e1849e6e0942800",
        "rationale": "Variant exception per option a (2026-11-26): base lineage, train scope, release pin b87667c4.",
        "evidence_urls": [
          "https://github.com/sarahmart/HARDMath/blob/9ade062a80b7d475666b11ae13a2fb0d7b7f0341/LICENSE",
          "https://huggingface.co/datasets/open-athena/atlas-source-pins/blob/eec74e0a1f6e7a8c81f382d193a82ae00ced069e/hardmath_lineage.json",
          "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.25.1/inputs/hardmath-license-receipt-v1.json",
          "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.25.1/inputs/hardmath-folder-lineage-receipt-v1.json",
          "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.25.1/inputs/hardmath-rhs-variant-receipt-v1.json",
          "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.25.1/hardmath-repair-republish-v2/result.json",
          "https://huggingface.co/datasets/open-athena/pafitis-hardmath-processed-training-repaired-20261126-v1/tree/b87667c4bf6ac31027e439e46dde77399af15f26",
          "file:///Users/benfeuer/Documents/experiments/active/atlas-rulings/repackager-license-follows-upstream-20261116.md"
        ],
        "decided_at": "2026-11-26T12:45:00-05:00",
        "decision_by": "user via coordinator (Claude session b610073d)"
      }
    },
    {
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "decided_at": "2026-11-15",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "source_id": "MAI-Gym:tasks",
      "upstream_id": "Tongyi-MAI/MAI-Gym",
      "upstream_revision": "dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9",
      "reason": "No license grant for the apps or task data at the pinned repository.",
      "evidence_urls": [
        "https://github.com/Tongyi-MAI/MAI-Gym/tree/dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9",
        "https://github.com/marin-community/marin/issues/10231#issuecomment-6231071121"
      ]
    },
    {
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "decided_at": "2026-11-15",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497",
      "source_id": "OpenGCUA:trajectories",
      "upstream_id": "https://tongyi-mai.github.io/OpenGCUA/trajectories.html",
      "upstream_revision": "no published trajectory pin",
      "reason": "No trajectory-data license; Apache-2.0 applies to the harness code.",
      "evidence_urls": [
        "https://tongyi-mai.github.io/OpenGCUA/trajectories.html",
        "https://github.com/Tongyi-MAI/OpenGCUA/tree/55c33aea792b48b9c1330a514bfcc8f6d3ecc704"
      ]
    },
    {
      "source_id": "kernelbot-data",
      "upstream_id": "GPUMODE/kernelbot-data",
      "upstream_revision": "4159cf6b2c6bab208be6dda885d6d87631cc16df",
      "classification": "unknown",
      "license_ids": [
        "custom"
      ],
      "reason": "User rejected the required model-license reciprocity condition for plain OpenMDW-1.1 releases.",
      "evidence_urls": [
        "https://huggingface.co/datasets/GPUMODE/kernelbot-data/blob/4159cf6b2c6bab208be6dda885d6d87631cc16df/LICENSE"
      ],
      "decided_at": "2026-11-15T18:11:47.896163+00:00",
      "decision_by": "user via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:exp_rpt_crosscodeeval-python",
      "upstream_id": "DCAgent/exp_rpt_crosscodeeval-python",
      "upstream_revision": "9cd539d223f101588e2604c9de7a58f16d2f26e7",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/exp_rpt_crosscodeeval-python/tree/9cd539d223f101588e2604c9de7a58f16d2f26e7?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:exp_rpt_crosscodeeval-typescript",
      "upstream_id": "DCAgent/exp_rpt_crosscodeeval-typescript",
      "upstream_revision": "0befa932636d239ceeeadeea84109c460d70cec9",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/exp_rpt_crosscodeeval-typescript/tree/0befa932636d239ceeeadeea84109c460d70cec9?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:exp_rpt_crosscodeeval-csharp",
      "upstream_id": "DCAgent/exp_rpt_crosscodeeval-csharp",
      "upstream_revision": "d1ef4c19a8b4c6cf0893cfc8b0d8f2821da64c62",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/exp_rpt_crosscodeeval-csharp/tree/d1ef4c19a8b4c6cf0893cfc8b0d8f2821da64c62?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:exp_rpt_crosscodeeval-python-v2",
      "upstream_id": "DCAgent/exp_rpt_crosscodeeval-python-v2",
      "upstream_revision": "ece476da1479f738677e5de691118c002eb9b610",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/exp_rpt_crosscodeeval-python-v2/tree/ece476da1479f738677e5de691118c002eb9b610?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:exp_rpt_crosscodeeval-java_10k",
      "upstream_id": "DCAgent/exp_rpt_crosscodeeval-java_10k",
      "upstream_revision": "d4305a34fe45ab97a1b18fdddba6755916ae1b3b",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/exp_rpt_crosscodeeval-java_10k/tree/d4305a34fe45ab97a1b18fdddba6755916ae1b3b?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:exp_rpt_crosscodeeval-csharp_10k",
      "upstream_id": "DCAgent/exp_rpt_crosscodeeval-csharp_10k",
      "upstream_revision": "3c247803bdf2c06e81d29a9d1149d2db48e9c85c",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/exp_rpt_crosscodeeval-csharp_10k/tree/3c247803bdf2c06e81d29a9d1149d2db48e9c85c?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:exp_rpt_crosscodeeval-python-v2_10k",
      "upstream_id": "DCAgent/exp_rpt_crosscodeeval-python-v2_10k",
      "upstream_revision": "2644406f15380e8805f52806861196047fa3de37",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/exp_rpt_crosscodeeval-python-v2_10k/tree/2644406f15380e8805f52806861196047fa3de37?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:exp_rpt_crosscodeeval-typescript_10k",
      "upstream_id": "DCAgent/exp_rpt_crosscodeeval-typescript_10k",
      "upstream_revision": "c93cf519ed37528f979183e5518e67ce80ae30ad",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/exp_rpt_crosscodeeval-typescript_10k/tree/c93cf519ed37528f979183e5518e67ce80ae30ad?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:a1_crosscodeeval_java",
      "upstream_id": "DCAgent/a1_crosscodeeval_java",
      "upstream_revision": "a32226762a9eb22eeb7ed132909af2fa5ac3c83e",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/datasets/DCAgent/a1_crosscodeeval_java/resolve/a32226762a9eb22eeb7ed132909af2fa5ac3c83e/README.md",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:a1_crosscodeeval_csharp",
      "upstream_id": "DCAgent/a1_crosscodeeval_csharp",
      "upstream_revision": "8267569628c25f373bd2e058821705881fa3545f",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/datasets/DCAgent/a1_crosscodeeval_csharp/resolve/8267569628c25f373bd2e058821705881fa3545f/README.md",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:a1_crosscodeeval_python",
      "upstream_id": "DCAgent/a1_crosscodeeval_python",
      "upstream_revision": "f3ba544b844fccc01adfa272bf4717b3fc691e7b",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/datasets/DCAgent/a1_crosscodeeval_python/resolve/f3ba544b844fccc01adfa272bf4717b3fc691e7b/README.md",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:a1_crosscodeeval_typescript",
      "upstream_id": "DCAgent/a1_crosscodeeval_typescript",
      "upstream_revision": "7663e75450af4ceae0c6684ff5ab183ebef5c409",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/datasets/DCAgent/a1_crosscodeeval_typescript/resolve/7663e75450af4ceae0c6684ff5ab183ebef5c409/README.md",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "DCAgent:a1_crosscodeeval",
      "upstream_id": "DCAgent/a1_crosscodeeval",
      "upstream_revision": "53074ddf9fdee51c8bca0fcc2ffb465c8539ceea",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "FAIL F2: no declared license; dataset banned for TRAIN/EVAL under the standing doubt rule.",
      "evidence_urls": [
        "https://huggingface.co/api/datasets/DCAgent/a1_crosscodeeval/tree/53074ddf9fdee51c8bca0fcc2ffb465c8539ceea?recursive=true",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "laion:tasktrove-bugsinpy-v4-oracle17-apptainer-v1",
      "upstream_id": "laion/tasktrove-bugsinpy-v4-oracle17-apptainer-v1",
      "upstream_revision": "a2f98ed57d4bf72604a79f78d0f95c3b847aa757",
      "classification": "unknown",
      "license_ids": [
        "Apache-2.0",
        "other"
      ],
      "reason": "E2: license:other/mixed terms leave dataset permission unresolved; doubt rule bans TRAIN/EVAL.",
      "evidence_urls": [
        "https://huggingface.co/datasets/laion/tasktrove-bugsinpy-v4-oracle17-apptainer-v1/resolve/a2f98ed57d4bf72604a79f78d0f95c3b847aa757/README.md",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-22T22:05:21.992865+00:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "TRL:kto_mix/argilla/distilabel-capybara-dpo-7k-binarized",
      "upstream_id": "trl-lib/kto-mix-14k",
      "upstream_revision": "4470f033f33364e7d064c9f920c3df54d0cce767",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "trl-lib/kto-mix-14k declares no license at the pin; standing doubt rule is binding.",
      "evidence_urls": [
        "https://huggingface.co/datasets/trl-lib/kto-mix-14k/resolve/4470f033f33364e7d064c9f920c3df54d0cce767/README.md",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-25T09:30:00-05:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "TRL:kto_mix/argilla/distilabel-intel-orca-dpo-pairs",
      "upstream_id": "trl-lib/kto-mix-14k",
      "upstream_revision": "4470f033f33364e7d064c9f920c3df54d0cce767",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "trl-lib/kto-mix-14k declares no license at the pin; standing doubt rule is binding.",
      "evidence_urls": [
        "https://huggingface.co/datasets/trl-lib/kto-mix-14k/resolve/4470f033f33364e7d064c9f920c3df54d0cce767/README.md",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-25T09:30:00-05:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    },
    {
      "source_id": "TRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned",
      "upstream_id": "trl-lib/kto-mix-14k",
      "upstream_revision": "4470f033f33364e7d064c9f920c3df54d0cce767",
      "classification": "missing",
      "license_ids": [
        "missing"
      ],
      "reason": "trl-lib/kto-mix-14k declares no license at the pin; standing doubt rule is binding.",
      "evidence_urls": [
        "https://huggingface.co/datasets/trl-lib/kto-mix-14k/resolve/4470f033f33364e7d064c9f920c3df54d0cce767/README.md",
        "https://github.com/marin-community/marin/issues/11497"
      ],
      "decided_at": "2026-11-25T09:30:00-05:00",
      "decision_by": "user standing rule via coordinator",
      "issue_url": "https://github.com/marin-community/marin/issues/11497"
    }
  ]
}
"""
