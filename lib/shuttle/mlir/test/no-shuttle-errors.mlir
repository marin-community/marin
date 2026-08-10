// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --allow-unregistered-dialect --split-input-file --shuttle-verify-no-shuttle-ops --verify-diagnostics %s

module {
  // expected-error @+1 {{Shuttle operation remains before HLO export}}
  "shuttle.opaque"() : () -> ()
}

// -----

// expected-error @+1 {{Shuttle attribute remains in a location before HLO export}}
module {} loc(fused<#shuttle.source_ref<0, 0, 0, 0>>["nested"])

// -----

// expected-error @+1 {{Shuttle attribute remains before HLO export: test.nested}}
module attributes {
  test.nested = {payload = [#shuttle.source_ref<0, 0, 0, 0>]}
} {}
