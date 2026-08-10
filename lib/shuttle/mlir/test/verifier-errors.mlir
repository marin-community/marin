// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

#identity = affine_map<(m, n) -> (m, n)>

module {
  func.func @map_scalar_contract(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      // expected-error @+1 {{scalar body argument types must equal input element types}}
      %mapped = "shuttle.map"(%region_arg) ({
      ^bb0(%not_scalar: tensor<2x4xf32>):
        "shuttle.yield"(%not_scalar) : (tensor<2x4xf32>) -> ()
      }) {
        indexing_maps = [#identity, #identity],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%mapped) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @contract_precision(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>)
      -> tensor<2x4xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{requires one precision entry per input}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @fold_dimensions(%arg: tensor<2x4xf32>) -> tensor<2xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      %zero = arith.constant 0.0 : f32
      // expected-error @+1 {{reduction dimensions must be non-negative and unique}}
      %folded = "shuttle.fold"(%region_arg, %zero) ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64: 1, 1>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>, f32) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// -----

module {
  func.func @fold_result_element(%arg: tensor<2x4xf32>) -> tensor<2xf16> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      %zero = arith.constant 0.0 : f32
      // expected-error @+1 {{result elements must use accumulator types}}
      %folded = "shuttle.fold"(%region_arg, %zero) ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>, f32) -> tensor<2xf16>
      "shuttle.yield"(%folded) : (tensor<2xf16>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<2xf16>
    return %result : tensor<2xf16>
  }
}

// -----

module {
  func.func @fold_shape(%arg: tensor<2x4xf32>) -> tensor<3xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      %zero = arith.constant 0.0 : f32
      // expected-error @+1 {{result shape must equal the input shape with reduced dimensions removed}}
      %folded = "shuttle.fold"(%region_arg, %zero) ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>, f32) -> tensor<3xf32>
      "shuttle.yield"(%folded) : (tensor<3xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }
}
