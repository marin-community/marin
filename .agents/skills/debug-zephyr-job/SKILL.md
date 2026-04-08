---
name: debug-zephyr-job
description: Debug Zephyr pipeline execution issues (stuck stages, stragglers, idle workers). Use when zephyr workers are misbehaving or pipelines are slow.
---

Debug Zephyr pipeline execution. Follow the procedures in `lib/zephyr/OPS.md` — covers dashboard setup, observability (logs, profiling, coordinator queries), and diagnostic patterns (stragglers, data skew, worker failures).

After a fix, use the **babysit-zephyr** skill to stop, resubmit, and monitor the new run.
