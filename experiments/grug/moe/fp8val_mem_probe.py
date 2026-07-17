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


def _report_buffer_assignment(arm: str, new_files: list[str]) -> None:
    ba_files = [f for f in new_files if "buffer" in os.path.basename(f)]
    if not ba_files:
        print(f"MEMPROBE {arm} no buffer-assignment dump found; new dump files: {sorted(new_files)[:20]}", flush=True)
        return
    for path in ba_files:
        allocations: list[tuple[int, str, list[tuple[int, str]]]] = []
        cur_size, cur_head, cur_values = 0, "", []
        with open(path) as f:
            for line in f:
                m = re.match(r"allocation \d+: .*size (\d+)", line)
                if m:
                    if cur_head:
                        allocations.append((cur_size, cur_head, cur_values))
                    cur_size, cur_head, cur_values = int(m.group(1)), line.strip()[:160], []
                elif cur_head and "size" in line and ("value:" in line or "offset=" in line):
                    vm = re.search(r"size=?[ ]?(\d+)", line)
                    if vm:
                        cur_values.append((int(vm.group(1)), line.strip()[:200]))
        if cur_head:
            allocations.append((cur_size, cur_head, cur_values))
        allocations.sort(key=lambda a: -a[0])
        print(f"MEMPROBE {arm} buffer file {os.path.basename(path)}: {len(allocations)} allocations", flush=True)
        for size, head, _ in allocations[:8]:
            print(f"MEMPROBE {arm} alloc {size / 2**30:8.2f}GiB  {head}", flush=True)
        if allocations:
            _, _, values = allocations[0]
            values.sort(key=lambda v: -v[0])
            print(f"MEMPROBE {arm} top {TOP_VALUES} values in largest allocation:", flush=True)
            for size, text in values[:TOP_VALUES]:
                print(f"MEMPROBE {arm} val {size / 2**30:7.2f}GiB  {text}", flush=True)


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
