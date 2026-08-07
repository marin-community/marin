# Wedge forensics: NCCL proxy-op pool inspection on a live #7344 wedge

These lldb scripts read NCCL's proxy-op pool state from a wedged rank and distinguish the
proxy-slot-leak deadlock (root cause of #7344 on NCCL 2.28.9 / aarch64) from any other stall
with the same surface signature. Total time on a live wedge: ~15 minutes.

The scripts were written against a specific wedge and carry hardcoded addresses. They are
kept as executable documentation of the method; edit the constants at the top for a new
wedge. The workflow that produced them, in order:

1. **Find the lagging rank** — NCCL RAS from any task:
   `echo verbose status | nc localhost 28028`, look for the `MISMATCH` warning naming the
   rank and node IP. Map node IP → pod with `kubectl -n iris get pods -o wide`
   (`KUBECONFIG=~/.kube/coreweave-iris`). The trainer is PID 1 in the `task` container;
   `/usr/bin/lldb` is in the image.
2. **Dump all thread stacks** —
   `lldb -p 1 --batch -o "thread backtrace all" -o detach -o quit > /tmp/btall.txt`.
   In a slot-leak wedge you find: one thread spinning in `ncclLocalOpAppend`
   (proxy.cc:499), its comm's own progress thread in `pthread_cond_wait` inside
   `ncclProxyGetPostedOps` (proxy.cc:797), and the other progress threads polling at
   proxy.cc:986. NCCL threads carry inherited names (`py_xla_execute` etc.) unless
   `NCCL_SET_THREAD_NAME=1`; identify them by stack, not name.
3. **Map the stuck comm** (`pool_map.py`) — from the `ncclProxySaveOp` frame's `comm`
   argument, print rank/opCount and the per-connection `proxyOps[]` table
   (`count/freeOp/nextOps`) plus the shared pool's `nextOps`/`freeOps[tpLocalRank]`.
   Deadlock state: everything `-1`/`0`.
4. **Prove client/service pool identity** (`pool_fingerprint_census.py`) — the same shm is
   mapped at different VAs in the client and service roles, so never compare pool pointers;
   compare the `freeOps[]` fingerprint. Also walks one partition's 2048 slots and
   reconstructs the orphaned chains (leaked return batches Y-merge into shared suffixes).
5. **Measure the leak** (`freeops_deficit.py`) — for every in-use partition, walk the free
   chain from `freeOps[r]` and report `2048 - length`. On the wedge this produced deficits
   2048 / 830 / 446 / 702 across the four local ranks: the leak is continuous on healthy
   ranks and the first partition to hit zero wedges the job.

Root cause (fixed upstream in NCCL v2.29.3-1, commit 25368a7f78ba): the slot-return path in
`ncclProxyGetPostedOps` used a weak `__atomic_compare_exchange_n` with a value-based retry
condition (`while (swap != oldFree)`), which treats a spurious CAS failure as success and
orphans the entire batch of freed slots. On aarch64 this compiles to ldaxr/stlxr with the
stlxr success flag never tested; the producer's `__atomic_exchange_n(&freeOps[i], -1)` spin
breaks the reservation while leaving the value unchanged, so leaks concentrate exactly at
backpressure episodes. x86 `lock cmpxchg` cannot fail spuriously, which is why only the
GB200 (Grace) fleet sees it.
