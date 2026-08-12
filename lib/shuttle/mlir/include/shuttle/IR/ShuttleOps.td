// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_IR_SHUTTLEOPS_TD_
#define SHUTTLE_IR_SHUTTLEOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "shuttle/IR/ShuttleAttrs.td"
include "shuttle/IR/ShuttleDialect.td"

class Shuttle_Op<string mnemonic, list<Trait> traits = []>
    : Op<Shuttle_Dialect, mnemonic, traits>;

def Shuttle_RegionOp : Shuttle_Op<"region", [RecursiveMemoryEffects,
                                             SingleBlockImplicitTerminator<
                                                 "::mlir::shuttle::YieldOp">]> {
  let summary = "Explicit SSA boundary for selected structural algebra";
  let arguments = (ins Variadic<AnyType>:$inputs, Shuttle_PolicyAttr:$policy,
      ArrayAttr:$source_refs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_MapOp
    : Shuttle_Op<
          "map", [ParentOneOf<["::mlir::shuttle::RegionOp"]>, Pure,
                  SingleBlockImplicitTerminator<"::mlir::shuttle::YieldOp">]> {
  let summary = "Pointwise scalar computation with explicit indexing maps";
  let arguments = (ins Variadic<AnyType>:$inputs,
      AffineMapArrayAttr:$indexing_maps, Shuttle_SourceRefAttr:$source);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_ContractOp
    : Shuttle_Op<"contract", [ParentOneOf<["::mlir::shuttle::RegionOp"]>,
                              Pure]> {
  let summary = "Generic contraction with normalized indexing metadata";
  let arguments = (ins Variadic<AnyType>:$inputs,
      AffineMapArrayAttr:$indexing_maps, ArrayAttr:$iterator_kinds,
      TypeArrayAttr:$accumulator_types, ArrayAttr:$precision,
      StrAttr:$algorithm, Shuttle_SourceRefAttr:$source);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
}

def Shuttle_ScalarConvertOp : Shuttle_Op<"scalar_convert", [Pure]> {
  let summary = "Scalar conversion with explicit rounding provenance";
  let arguments = (ins AnyType:$input,
      Shuttle_ScalarConvertSemanticsAttr:$semantics);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Shuttle_FoldOp
    : Shuttle_Op<
          "fold", [AttrSizedOperandSegments,
                   ParentOneOf<["::mlir::shuttle::RegionOp"]>, Pure,
                   SingleBlockImplicitTerminator<"::mlir::shuttle::YieldOp">]> {
  let summary = "Reduction with explicit accumulator and ordering contract";
  let arguments = (ins Variadic<AnyType>:$inputs,
      Variadic<AnyType>:$initializers, DenseI64ArrayAttr:$reduction_dimensions,
      TypeArrayAttr:$accumulator_types, BoolAttr:$order_free,
      Shuttle_SourceRefAttr:$source);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$combiner);
  let hasRegionVerifier = 1;
}

def Shuttle_YieldOp
    : Shuttle_Op<"yield", [ParentOneOf<["::mlir::shuttle::RegionOp",
                                        "::mlir::shuttle::MapOp",
                                        "::mlir::shuttle::FoldOp"]>,
                           Pure, Terminator]> {
  let summary = "Yield values from a Shuttle algebra region";
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

#endif // SHUTTLE_IR_SHUTTLEOPS_TD_
