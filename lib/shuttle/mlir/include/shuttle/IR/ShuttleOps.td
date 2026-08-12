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
      AffineMapArrayAttr:$indexing_maps,
      DefaultValuedAttr<Shuttle_MapSemanticsAttr,
                        "::mlir::shuttle::MapSemantics::Pointwise">:$semantics,
      Shuttle_SourceRefAttr:$source);
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

def Shuttle_MaterializationPlanOp
    : Shuttle_Op<
          "materialization_plan", [IsolatedFromAbove,
                            SingleBlockImplicitTerminator<
                                "::mlir::shuttle::MaterializationPlanYieldOp">]> {
  let summary = "Non-executable task, buffer, and lifetime plan";
  let arguments = (ins I64Attr:$schema_version, Shuttle_PolicyAttr:$policy,
      StrAttr:$fingerprint);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_MaterializationBufferOp
    : Shuttle_Op<"materialization_buffer",
                 [ParentOneOf<["::mlir::shuttle::MaterializationPlanOp"]>]> {
  let summary = "One explicitly materialized logical tensor buffer";
  let arguments = (ins I64Attr:$ordinal, TypeAttr:$tensor_type,
      Shuttle_MaterializationStorageAttr:$storage, BoolAttr:$live_in,
      BoolAttr:$live_out, OptionalAttr<I64Attr>:$producer,
      DenseI64ArrayAttr:$consumers, I64Attr:$lifetime_start,
      I64Attr:$lifetime_end);
}

def Shuttle_MaterializationTaskOp
    : Shuttle_Op<"materialization_task",
                 [ParentOneOf<["::mlir::shuttle::MaterializationPlanOp"]>]> {
  let summary = "One generic Map or Fold task family";
  let arguments = (ins I64Attr:$ordinal, Shuttle_MaterializationTaskKindAttr:$kind,
      DenseI64ArrayAttr:$domain_shape,
      DenseI64ArrayAttr:$reduction_dimensions,
      DenseI64ArrayAttr:$input_buffers, DenseI64ArrayAttr:$output_buffers,
      DenseI64ArrayAttr:$dependencies, OptionalAttr<BoolAttr>:$order_free,
      StrAttr:$semantic_fingerprint, Shuttle_SourceRefAttr:$source);
}

def Shuttle_MaterializationPlanYieldOp
    : Shuttle_Op<"materialization_plan_yield",
                 [ParentOneOf<["::mlir::shuttle::MaterializationPlanOp"]>,
                  Terminator]> {
  let summary = "Terminate a non-executable materialization plan";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_SchedulePlanOp
    : Shuttle_Op<
          "schedule_plan", [IsolatedFromAbove,
                            SingleBlockImplicitTerminator<
                                "::mlir::shuttle::SchedulePlanYieldOp">]> {
  let summary = "Non-executable target schedule and logical indexing contract";
  let arguments = (ins I64Attr:$schema_version, Shuttle_ScheduleTargetAttr:$target,
      Shuttle_PolicyAttr:$policy, StrAttr:$source_plan_fingerprint,
      StrAttr:$fingerprint);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_ScheduleBufferOp
    : Shuttle_Op<"schedule_buffer",
                 [ParentOneOf<["::mlir::shuttle::SchedulePlanOp"]>]> {
  let summary = "Logical iteration contract for one planned buffer";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$source_buffer,
      TypeAttr:$tensor_type, Shuttle_ScheduleBufferIndexingAttr:$indexing,
      DenseI64ArrayAttr:$iteration_order, I64Attr:$lifetime_start,
      I64Attr:$lifetime_end);
}

def Shuttle_ScheduleTaskOp
    : Shuttle_Op<"schedule_task",
                 [ParentOneOf<["::mlir::shuttle::SchedulePlanOp"]>]> {
  let summary = "One bounded logical SIMT32 schedule candidate";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$source_task,
      Shuttle_ScheduleTaskKindAttr:$kind, DenseI64ArrayAttr:$domain_shape,
      DenseI64ArrayAttr:$grid_shape, DenseI64ArrayAttr:$tile_shape,
      I64Attr:$serial_tiles, I64Attr:$workgroup_threads,
      I64Attr:$subgroup_size, I64Attr:$scratch_bytes,
      DenseI64ArrayAttr:$input_buffers, DenseI64ArrayAttr:$output_buffers,
      DenseI64ArrayAttr:$dependencies, OptionalAttr<I64Attr>:$reduction_axis,
      OptionalAttr<Shuttle_ScheduleReductionOrderAttr>:$reduction_order,
      StrAttr:$semantic_fingerprint);
}

def Shuttle_SchedulePlanYieldOp
    : Shuttle_Op<"schedule_plan_yield",
                 [ParentOneOf<["::mlir::shuttle::SchedulePlanOp"]>,
                  Terminator]> {
  let summary = "Terminate a non-executable schedule plan";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_DeviceModuleOp
    : Shuttle_Op<"device_module", [IsolatedFromAbove,
                         SingleBlockImplicitTerminator<
                             "::mlir::shuttle::DeviceModuleYieldOp">]> {
  let summary = "Closed generated CPU bytecode and entrypoint bindings";
  let arguments = (ins I64Attr:$schema_version,
      Shuttle_ExecutableCodeFormatAttr:$code_format, Shuttle_PolicyAttr:$policy,
      StrAttr:$source_schedule_fingerprint, DenseI8ArrayAttr:$code,
      StrAttr:$code_digest, StrAttr:$fingerprint);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_DeviceEntryOp
    : Shuttle_Op<"device_entry",
                 [ParentOneOf<["::mlir::shuttle::DeviceModuleOp"]>]> {
  let summary = "One bytecode entry bound to a scheduled task";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$source_task,
      I64Attr:$code_offset, I64Attr:$code_length,
      DenseI64ArrayAttr:$input_buffers, DenseI64ArrayAttr:$output_buffers,
      ArrayAttr:$input_accesses, ArrayAttr:$output_accesses,
      DenseI64ArrayAttr:$dependencies,
      Shuttle_ExecutablePredicationAttr:$predication,
      OptionalAttr<Shuttle_ScheduleReductionOrderAttr>:$reduction_order,
      StrAttr:$code_digest);
}

def Shuttle_DeviceModuleYieldOp
    : Shuttle_Op<"device_module_yield",
                 [ParentOneOf<["::mlir::shuttle::DeviceModuleOp"]>,
                  Terminator]> {
  let summary = "Terminate a generated device module";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_InvocationAbiOp
    : Shuttle_Op<"invocation_abi", [IsolatedFromAbove,
                         SingleBlockImplicitTerminator<
                             "::mlir::shuttle::InvocationAbiYieldOp">]> {
  let summary = "Closed static buffer ABI for a local executable consumer";
  let arguments = (ins I64Attr:$schema_version, StrAttr:$source_plan_fingerprint,
      StrAttr:$source_schedule_fingerprint, StrAttr:$fingerprint);
  let regions = (region SizedRegion<1>:$body);
  let hasRegionVerifier = 1;
}

def Shuttle_InvocationSlotOp
    : Shuttle_Op<"invocation_slot",
                 [ParentOneOf<["::mlir::shuttle::InvocationAbiOp"]>]> {
  let summary = "One statically bound runtime byte buffer";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$source_buffer,
      TypeAttr:$tensor_type, I64Attr:$required_bytes,
      DenseI64ArrayAttr:$strides, I64Attr:$offset, I64Attr:$alignment,
      Shuttle_ExecutableAddressSpaceAttr:$address_space,
      Shuttle_ExecutableAccessAttr:$access,
      Shuttle_MaterializationStorageAttr:$storage, I64Attr:$alias_group,
      I64Attr:$reuse_group);
}

def Shuttle_InvocationAbiYieldOp
    : Shuttle_Op<"invocation_abi_yield",
                 [ParentOneOf<["::mlir::shuttle::InvocationAbiOp"]>,
                  Terminator]> {
  let summary = "Terminate a static invocation ABI";
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
}

def Shuttle_ExecutableBundleOp : Shuttle_Op<"executable_bundle"> {
  let summary = "Closed root binding code, ABI, schedule, and completion";
  let arguments = (ins I64Attr:$schema_version,
      StrAttr:$source_schedule_fingerprint, StrAttr:$device_module_fingerprint,
      StrAttr:$invocation_abi_fingerprint,
      Shuttle_ExecutableCompletionAttr:$completion, StrAttr:$fingerprint);
  let hasVerifier = 1;
}

#endif // SHUTTLE_IR_SHUTTLEOPS_TD_
