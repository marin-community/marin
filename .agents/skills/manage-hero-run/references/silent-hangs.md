# Localizing a silent training hang

Use this reference after the cutover checklist identifies a stall. A blocked
collective may produce no error log. Preserve evidence without extending a failed
production trial through repeated retries.

- Compare NCCL RAS `collective_operations` and minimum/maximum `rank_statistic`
  values per communicator in `telemetry_v1.levanter`. A group of ranks lagging
  across communicators can identify the domain that stopped entering collectives.
- Compare per-task `cpu_millicores` in `iris.task` over the stall window. Ranks
  busy-polling in a collective may consume more CPU than ranks that never entered
  it. Map task indices to racks from the actual allocation, not a fixed topology.
- Compare `task_attempts.node_name` across attempts. Recurrence on the same nodes
  motivates hardware inspection; recurrence on different nodes motivates code
  and collective-order inspection. Neither alone proves the cause.
- Use `iris process profile threads` for host stacks. It cannot inspect GPU-side
  XLA execution; GPU diagnosis may require a CUDA core dump armed before launch.

Record the executable transition before the stall: instrumented to ordinary
training, evaluation to training, or checkpoint work. Shared NCCL symmetric-memory
arenas across executables were implicated in the historical incidents
[#8861](https://github.com/marin-community/marin/issues/8861) and
[#8870](https://github.com/marin-community/marin/issues/8870).

See the [RAS alert contract](../../../../docs/ops/training-stall-alert-contract.md)
and [Iris operations](../../../../lib/iris/OPS.md) for collection and query commands.
