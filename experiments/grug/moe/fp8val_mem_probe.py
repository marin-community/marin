# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile-only probe attributing the fp8-vs-bf16 temp-arena gap (#7298, FP8VAL-004).

Compiles the exact row-13 validation train step twice in one process on one
8xH100 node — bf16 then fp8 — without running a single step, and prints:

- ``compiled.memory_analysis()`` per arm (temp arena = the 22.66 / 40.68 GiB
  contiguous allocation whose fragmentation-failure wedged the validation runs)
- the largest allocations from XLA's buffer-assignment dump, with the HLO
  values living in the temp arena ranked by size, so the extra ~18 GiB of fp8
  temporaries can be attributed to specific tensors.

Submit (XLA dump flags required for the buffer table)::

    iris job run --job-name fp8val-memprobe --cpu=2 --memory=3G \
      -e XLA_FLAGS "--xla_gpu_shard_autotuning=false --xla_dump_to=/tmp/xladump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_train_step" \
      -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.90 \
      -- python -m experiments.grug.moe.fp8val_mem_probe
"""

import dataclasses
import functools
import gc
import glob
import logging
import os
import re

from fray.cluster import ResourceConfig

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.llama import llama3_tokenizer_vocab_size

logger = logging.getLogger(__name__)

HIDDEN_DIM = 2560
SEQ_LEN = 4096
TOP_VALUES = 40


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    steps: int = 24000
    batch_size: int = 16
    expert_axis: int = 8


def _build_model(fp8: bool):
    from experiments.grug.moe.model import GrugFp8Config, GrugModelConfig  # noqa: PLC0415

    kwargs = {"fp8": GrugFp8Config(wire=True, dense=True)} if fp8 else {}
    return GrugModelConfig(
        vocab_size=llama3_tokenizer_vocab_size,
        hidden_dim=HIDDEN_DIM,
        num_layers=26,
        num_heads=20,
        num_kv_heads=20,
        head_dim=128,
        intermediate_dim=HIDDEN_DIM // 2,
        shared_expert_intermediate_dim=HIDDEN_DIM // 2,
        num_experts=64,
        num_experts_per_token=4,
        max_seq_len=SEQ_LEN,
        sliding_window=2048,
        initializer_std=0.5 / (HIDDEN_DIM**0.5),
        qk_mult=1.3,
        attention_implementation="gpu_fa4_cute",
        remat_mode="recompute_all",
        **kwargs,
    )


_VALUE_RE = re.compile(r"\(size=(\d+),offset=(\d+)\):\s*([A-Za-z0-9]+)\[")


def _report_buffer_assignment(arm: str, new_files: list[str]) -> None:
    # The authoritative arena table is `...-buffer-assignment.txt`; skip the
    # sibling `-buffer-assignment-values.txt` (different, uninformative format).
    ba_files = [
        f
        for f in new_files
        if os.path.basename(f).endswith("buffer-assignment.txt")
        and not os.path.basename(f).endswith("values.txt")
    ]
    if not ba_files:
        print(f"MEMPROBE {arm} no buffer-assignment.txt; dump files: {sorted(os.path.basename(f) for f in new_files)}", flush=True)
        return
    for path in ba_files:
        # Find the biggest allocation (the preallocated temp arena) and, within
        # it, every value line. Each arena offset is one physical slot reused
        # over time; attribute the slot to the dtype of the largest value that
        # ever lands there, then sum slot bytes per dtype.
        allocations: list[tuple[int, str, list[str]]] = []
        cur_size, cur_head, cur_lines = 0, "", []
        with open(path) as f:
            for line in f:
                m = re.match(r"allocation \d+: .*size (\d+)", line)
                if m:
                    if cur_head:
                        allocations.append((cur_size, cur_head, cur_lines))
                    cur_size, cur_head, cur_lines = int(m.group(1)), line.strip()[:160], []
                elif cur_head and "offset=" in line:
                    cur_lines.append(line)
        if cur_head:
            allocations.append((cur_size, cur_head, cur_lines))
        allocations.sort(key=lambda a: -a[0])
        print(f"MEMPROBE {arm} buffer file {os.path.basename(path)}: {len(allocations)} allocations", flush=True)
        for size, head, _ in allocations[:6]:
            print(f"MEMPROBE {arm} alloc {size / 2**30:8.2f}GiB  {head}", flush=True)
        if not allocations:
            continue
        _, _, lines = allocations[0]
        slot_size: dict[int, int] = {}
        slot_dtype: dict[int, str] = {}
        for line in lines:
            vm = _VALUE_RE.search(line)
            if not vm:
                continue
            size, offset, dtype = int(vm.group(1)), int(vm.group(2)), vm.group(3)
            if size > slot_size.get(offset, -1):
                slot_size[offset] = size
                slot_dtype[offset] = dtype
        dtypes: dict[str, tuple[int, int]] = {}
        for offset, size in slot_size.items():
            d = slot_dtype[offset]
            c, t = dtypes.get(d, (0, 0))
            dtypes[d] = (c + 1, t + size)
        covered = sum(slot_size.values())
        print(f"MEMPROBE {arm} arena slots={len(slot_size)} covered={covered / 2**30:.2f}GiB", flush=True)
        for d, (c, t) in sorted(dtypes.items(), key=lambda kv: -kv[1][1]):
            print(f"MEMPROBE {arm} dtype {t / 2**30:8.3f}GiB  n={c:5d}  {d}", flush=True)


def _probe_entry(cfg: ProbeConfig) -> None:
    import jax  # noqa: PLC0415
    import jmp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from haliax.partitioning import set_mesh  # noqa: PLC0415
    from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: PLC0415
    from levanter.data.text.examples import GrugLmExample  # noqa: PLC0415
    from levanter.grug.sharding import compact_grug_mesh  # noqa: PLC0415

    from experiments.grug.moe.optimizer import GrugMoeMuonHConfig  # noqa: PLC0415
    from experiments.grug.moe.train import _BATCH_AXES, _make_train_step, initial_state  # noqa: PLC0415

    print(f"MEMPROBE devices: {jax.device_count()} {jax.devices()[0].device_kind}", flush=True)
    dump_dir = None
    m = re.search(r"--xla_dump_to=(\S+)", os.environ.get("XLA_FLAGS", ""))
    if m:
        dump_dir = m.group(1)

    mesh = compact_grug_mesh(expert_axis_size=cfg.expert_axis, replica_axis_size=1)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    optimizer = GrugMoeMuonHConfig(learning_rate=1e-3, adam_lr=1e-4, min_lr_ratio=0.0, warmup=0.1).build(cfg.steps)

    for arm, fp8 in (("bf16", False), ("fp8", True)):
        model_cfg = _build_model(fp8)
        train_step = _make_train_step(optimizer, mp, z_loss_weight=1e-4, ema_beta=None, watch_config=None)
        with set_mesh(mesh):
            init = jax.jit(
                functools.partial(initial_state, model_cfg, optimizer=optimizer, mp=mp, ema_beta=None)
            )
            state = init(key=jax.random.PRNGKey(0))
            sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
            rng = np.random.default_rng(0)
            tokens = jax.device_put(
                rng.integers(0, model_cfg.vocab_size, size=(cfg.batch_size, SEQ_LEN), dtype=np.int32), sharding
            )
            loss_weight = jax.device_put(np.ones((cfg.batch_size, SEQ_LEN), np.float32), sharding)
            batch = GrugLmExample(tokens=tokens, loss_weight=loss_weight)

            before = set(glob.glob(dump_dir + "/*")) if dump_dir else set()
            print(f"MEMPROBE {arm} lowering + compiling...", flush=True)
            compiled = train_step.lower(state, batch).compile()
            ma = compiled.memory_analysis()
            fields = ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes")
            stats = {f: getattr(ma, f, None) for f in fields}
            pretty = {k: (f"{v / 2**30:.2f}GiB" if isinstance(v, int) else v) for k, v in stats.items()}
            print(f"MEMPROBE {arm} memory_analysis: {pretty}", flush=True)
            new_files = list(set(glob.glob(dump_dir + "/*")) - before) if dump_dir else []
            _report_buffer_assignment(arm, new_files)
        del state, batch, compiled, tokens, loss_weight
        gc.collect()
    print("MEMPROBE done", flush=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g", replicas=1)
    dispatch_grug_training_run(
        run_id="fp8val-memprobe",
        config=ProbeConfig(),
        local_entrypoint=_probe_entry,
        resources=resources,
        max_retries_failure=0,
    )


if __name__ == "__main__":
    main()
