# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parsing guards for schedule_report against the shapes real GPU HLO dumps take."""

import pathlib

from experiments.grug.moe.schedule_report import computations, leg_of, pick_dump

DUMP = '''HloModule jit_train_step, entry_computation_layout={(f32[8]{0})->f32[8]{0}}, schedule={...}

%wrapped_all_to_all.112 (p: f32[8]) -> f32[8] {
  %p = f32[8]{0} parameter(0)
  ROOT %all_to_all.112.1 = f32[8]{0} all-to-all(f32[8]{0} %p), replica_groups={{0,1}}
}

%while_body.31 (arg: (f32[8], f32[8])) -> (f32[8], f32[8]) {
  %arg = (f32[8]{0}, f32[8]{0}) parameter(0)
  %gte.1 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}) %arg), index=0
  %all_to_all.112-start = ((f32[8]{0}), f32[8]{0}, u32[]) async-start(f32[8]{0} %gte.1), \
calls=%wrapped_all_to_all.112, backend_config={"collective_backend_config":{"is_sync":true}}, \
metadata={op_name="jit(train_step)/jvp(Transformer)/while/body/shard_map/dispatch/all_to_all"}
  %all_to_all.112-done = f32[8]{0} async-done(((f32[8]{0}), f32[8]{0}, u32[]) %all_to_all.112-start)
  %all-to-all-start.113 = ((f32[8]{0}), f32[8]{0}, u32[]) async-start(f32[8]{0} %gte.1), \
calls=%wrapped_all_to_all.112, backend_config={"collective_backend_config":{"is_sync":false}}, \
metadata={op_name="jit(train_step)/transpose(jvp(Transformer))/while/body/shard_map/combine/all_to_all"}
  %gemm.9 = f32[8]{0} fusion(f32[8]{0} %gte.1), kind=kLoop, calls=%wrapped_all_to_all.112
  %bitcast.3 = f32[8]{0} bitcast(f32[8]{0} %gemm.9)
  %all-to-all-done.113 = f32[8]{0} async-done(((f32[8]{0}), f32[8]{0}, u32[]) %all-to-all-start.113)
  ROOT %tup = (f32[8]{0}, f32[8]{0}) tuple(f32[8]{0} %all_to_all.112-done, f32[8]{0} %all-to-all-done.113)
}

ENTRY %main.99 (Arg_0.1: f32[8]) -> f32[8] {
  %Arg_0.1 = f32[8]{0} parameter(0)
  ROOT %call.1 = f32[8]{0} call(f32[8]{0} %Arg_0.1), to_apply=%while_body.31
}
'''


def test_computations_split_by_brace_depth_not_header_pattern():
    """backend_config={...} and tuple shapes must not be mistaken for computation headers."""
    names = [name for name, _ in computations(DUMP.splitlines())]
    assert names == ["wrapped_all_to_all.112", "while_body.31", "main.99"]


def test_while_body_instructions_are_collected_in_schedule_order():
    body = dict(computations(DUMP.splitlines()))["while_body.31"]
    assert [name for name, _ in body] == [
        "arg",
        "gte.1",
        "all_to_all.112-start",
        "all_to_all.112-done",
        "all-to-all-start.113",
        "gemm.9",
        "bitcast.3",
        "all-to-all-done.113",
        "tup",
    ]


def test_leg_of_reads_direction_and_scope_from_metadata():
    body = dict(computations(DUMP.splitlines()))["while_body.31"]
    lines = dict(body)
    assert leg_of(lines["all_to_all.112-start"]) == "fwd dispatch/all_to_all"
    assert leg_of(lines["all-to-all-start.113"]) == "bwd combine/all_to_all"


def test_pick_dump_prefers_the_largest_post_optimization_file(tmp_path: pathlib.Path):
    (tmp_path / "module_0000.jit_small.cpu_after_optimizations.txt").write_text("small")
    big = tmp_path / "module_0001.jit_train_step.sm_10.0a_gpu_after_optimizations.txt"
    big.write_text(DUMP)
    assert pick_dump(tmp_path) == big
