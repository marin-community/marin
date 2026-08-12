// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

module {
  func.func @narrow_requires_rne(%arg: f32) -> bf16 {
    // expected-error @+1 {{f32 to bf16 requires round_nearest_even semantics}}
    %0 = "shuttle.scalar_convert"(%arg) {semantics = #shuttle.scalar_convert_semantics<exact>} : (f32) -> bf16
    return %0 : bf16
  }
}

// -----

module {
  func.func @widen_requires_exact(%arg: bf16) -> f32 {
    // expected-error @+1 {{bf16 to f32 requires exact semantics}}
    %0 = "shuttle.scalar_convert"(%arg) {semantics = #shuttle.scalar_convert_semantics<round_nearest_even>} : (bf16) -> f32
    return %0 : f32
  }
}

// -----

module {
  func.func @identity_requires_exact(%arg: f32) -> f32 {
    // expected-error @+1 {{same-type conversion supports only f32 exact semantics}}
    %0 = "shuttle.scalar_convert"(%arg) {semantics = #shuttle.scalar_convert_semantics<round_nearest_even>} : (f32) -> f32
    return %0 : f32
  }
}

// -----

module {
  func.func @bf16_identity_is_outside_slice(%arg: bf16) -> bf16 {
    // expected-error @+1 {{same-type conversion supports only f32 exact semantics}}
    %0 = "shuttle.scalar_convert"(%arg) {semantics = #shuttle.scalar_convert_semantics<exact>} : (bf16) -> bf16
    return %0 : bf16
  }
}

// -----

module {
  func.func @only_bf16_f32(%arg: f16) -> f32 {
    // expected-error @+1 {{supports only bf16 and f32 scalar types}}
    %0 = "shuttle.scalar_convert"(%arg) {semantics = #shuttle.scalar_convert_semantics<exact>} : (f16) -> f32
    return %0 : f32
  }
}

// -----

module {
  func.func @scalar_only(%arg: tensor<2xbf16>) -> tensor<2xf32> {
    // expected-error @+1 {{requires scalar input and result types}}
    %0 = "shuttle.scalar_convert"(%arg) {semantics = #shuttle.scalar_convert_semantics<exact>} : (tensor<2xbf16>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
