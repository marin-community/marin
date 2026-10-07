# Task curation tests

`test_sources.py` and `test_catalog.py` check source declarations and catalog coverage; `test_driver.py` and `test_campaign.py` cover campaign setup and reporting. The binding tests exercise source-specific normalization and private grader contracts. `test_executable.py` covers executable source behavior with local fixtures.

[`fixtures/`](fixtures/README.md) holds representative archived tasks and source rows. [`fixtures/nemotron_ultra/`](fixtures/nemotron_ultra/README.md) contains Ultra grader examples. These tests do not establish that every source has a runnable image or production rollout adapter; use each source's verification report for that evidence. See the [experiment overview](../README.md).
