---
name: debug-tpu
description: Identify and replace bad TPU nodes causing job failures. Use when logs show "No accelerator found", "FAILED_PRECONDITION", or "Device or resource busy".
---

Identify and replace bad TPU nodes. Follow the procedures in `lib/iris/OPS.md` — specifically the "TPU Bad-Node Recovery" section under "GCP (TPU) Operations".

After recovery, return to the active babysit loop (**babysit-job** or **babysit-zephyr**).
