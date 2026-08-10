// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-opt --split-input-file --shuttle-verify-no-shuttle-ops %s 2>&1 | FileCheck %s

module {
  "shuttle.region"() ({
    "shuttle.yield"() : () -> ()
  }) {
    policy = #shuttle.policy<source_ordered>,
    source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
  } : () -> ()
}

// CHECK: Shuttle operation remains before HLO export

// -----

module {} loc(fused<#shuttle.source_ref<0, 0, 0, 0>>["nested"])

// CHECK: Shuttle attribute remains in a location before HLO export

// -----

module attributes {
  test.nested = {payload = [#shuttle.source_ref<0, 0, 0, 0>]}
} {}

// CHECK: Shuttle attribute remains before HLO export: "test.nested"
