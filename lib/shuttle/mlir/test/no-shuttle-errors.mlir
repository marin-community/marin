// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --shuttle-verify-no-shuttle-ops --verify-diagnostics %s

module {
  // expected-error @+1 {{Shuttle operation remains before HLO export}}
  "shuttle.region"() ({
    "shuttle.yield"() : () -> ()
  }) {
    policy = #shuttle.policy<source_ordered>,
    source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
  } : () -> ()
}

// -----

// expected-error @+1 {{Shuttle attribute remains in a location before HLO export}}
module {} loc(fused<#shuttle.source_ref<0, 0, 0, 0>>["nested"])

// -----

// expected-error @+1 {{Shuttle attribute remains before HLO export: test.nested}}
module attributes {
  test.nested = {payload = [#shuttle.source_ref<0, 0, 0, 0>]}
} {}
