// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_IR_SHUTTLEATTRS_TD_
#define SHUTTLE_IR_SHUTTLEATTRS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "shuttle/IR/ShuttleDialect.td"

class Shuttle_Attr<string name, list<Trait> traits = []>
    : AttrDef<Shuttle_Dialect, name, traits>;

def Shuttle_NumericalPolicy
    : I32EnumAttr<"NumericalPolicy", "Shuttle numerical policy",
                  [I32EnumAttrCase<"SourceOrdered", 0, "source_ordered">,
                   I32EnumAttrCase<"Fast", 1, "fast">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_PolicyAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_NumericalPolicy, "policy"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ScalarConvertSemantics
    : I32EnumAttr<"ScalarConvertSemantics", "Scalar conversion semantics",
                  [I32EnumAttrCase<"Exact", 0, "exact">,
                   I32EnumAttrCase<"RoundNearestEven", 1,
                                   "round_nearest_even">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ScalarConvertSemanticsAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ScalarConvertSemantics,
               "scalar_convert_semantics"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_MapSemantics
    : I32EnumAttr<"MapSemantics", "Map structural semantics",
                  [I32EnumAttrCase<"Pointwise", 0, "pointwise">,
                   I32EnumAttrCase<"BroadcastInDim", 1,
                                   "broadcast_in_dim">,
                   I32EnumAttrCase<"Reshape", 2, "reshape">,
                   I32EnumAttrCase<"Transpose", 3, "transpose">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_MapSemanticsAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_MapSemantics, "map_semantics"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_MaterializationTaskKind
    : I32EnumAttr<"MaterializationTaskKind", "Materialization task family kind",
                  [I32EnumAttrCase<"Map", 0, "map">,
                   I32EnumAttrCase<"Fold", 1, "fold">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_MaterializationTaskKindAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_MaterializationTaskKind,
               "materialization_task_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_MaterializationStorage
    : I32EnumAttr<"MaterializationStorage", "Materialized buffer storage class",
                  [I32EnumAttrCase<"External", 0, "external">,
                   I32EnumAttrCase<"Temporary", 1, "temporary">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_MaterializationStorageAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_MaterializationStorage,
               "materialization_storage"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ScheduleTarget
    : I32EnumAttr<"ScheduleTarget", "Abstract schedule capability profile",
                  [I32EnumAttrCase<"Simt32", 0, "simt32">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ScheduleTargetAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ScheduleTarget, "schedule_target"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ScheduleBufferIndexing
    : I32EnumAttr<"ScheduleBufferIndexing", "Logical tensor iteration contract",
                  [I32EnumAttrCase<"Scalar", 0, "scalar">,
                   I32EnumAttrCase<"Lexicographic", 1, "lexicographic">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ScheduleBufferIndexingAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ScheduleBufferIndexing,
               "schedule_buffer_indexing"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ScheduleReductionOrder
    : I32EnumAttr<"ScheduleReductionOrder", "Required Fold realization order",
                  [I32EnumAttrCase<"TreeAssociationFreeLeafOrderFixed", 0,
                      "tree_association_free_leaf_order_fixed">,
                   I32EnumAttrCase<"LeafOrderFree", 1, "leaf_order_free">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ScheduleReductionOrderAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ScheduleReductionOrder,
               "schedule_reduction_order"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ScheduleTaskKind
    : I32EnumAttr<"ScheduleTaskKind", "Logical SIMT32 task schedule kind",
                  [I32EnumAttrCase<"Scalar", 0, "scalar">,
                   I32EnumAttrCase<"Elementwise", 1, "elementwise">,
                   I32EnumAttrCase<"RowFold", 2, "row_fold">,
                   I32EnumAttrCase<"ColumnFold", 3, "column_fold">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ScheduleTaskKindAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ScheduleTaskKind,
               "schedule_task_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ExecutableCodeFormat
    : I32EnumAttr<"ExecutableCodeFormat", "Closed local executable format",
                  [I32EnumAttrCase<"CpuBytecodeV1", 0,
                                   "cpu_bytecode_v1">,
                   I32EnumAttrCase<"CpuBytecodeV2", 1,
                                   "cpu_bytecode_v2">,
                   I32EnumAttrCase<"CudaPtxSm90V1", 2,
                                   "cuda_ptx_sm90_v1">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ExecutableCodeFormatAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ExecutableCodeFormat,
               "executable_code_format"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ExecutableAccess
    : I32EnumAttr<"ExecutableAccess", "Executable buffer access mode",
                  [I32EnumAttrCase<"Read", 0, "read">,
                   I32EnumAttrCase<"Write", 1, "write">,
                   I32EnumAttrCase<"ReadWrite", 2, "read_write">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ExecutableAccessAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ExecutableAccess,
               "executable_access"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ExecutableAddressSpace
    : I32EnumAttr<"ExecutableAddressSpace", "Executable buffer address space",
                  [I32EnumAttrCase<"Host", 0, "host">,
                   I32EnumAttrCase<"Device", 1, "device">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ExecutableAddressSpaceAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ExecutableAddressSpace,
               "executable_address_space"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ExecutablePredication
    : I32EnumAttr<"ExecutablePredication", "Executable bounds policy",
                  [I32EnumAttrCase<"None", 0, "none">,
                   I32EnumAttrCase<"DomainBounds", 1, "domain_bounds">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ExecutablePredicationAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ExecutablePredication,
               "executable_predication"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ExecutableCompletion
    : I32EnumAttr<"ExecutableCompletion", "Local consumer completion policy",
                  [I32EnumAttrCase<"Synchronous", 0, "synchronous">,
                   I32EnumAttrCase<"StreamOrdered", 1,
                                   "stream_ordered">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ExecutableCompletionAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ExecutableCompletion,
               "executable_completion"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_ExecutableBindingKind
    : I32EnumAttr<"ExecutableBindingKind", "External invocation binding kind",
                  [I32EnumAttrCase<"None", 0, "none">,
                   I32EnumAttrCase<"Operand", 1, "operand">,
                   I32EnumAttrCase<"Result", 2, "result">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shuttle";
}

def Shuttle_ExecutableBindingKindAttr
    : EnumAttr<Shuttle_Dialect, Shuttle_ExecutableBindingKind,
               "executable_binding_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Shuttle_SourceRefAttr : Shuttle_Attr<"SourceRef"> {
  let mnemonic = "source_ref";
  let summary = "Rename-invariant structural reference to a source result";
  let description = [{
    A source reference identifies one result by its function, block, operation,
    and result ordinals in a deterministic preorder of the input MLIR module.
    It contains no workload or symbol names.
  }];
  let parameters = (ins "uint64_t":$function_ordinal, "uint64_t":$block_ordinal,
      "uint64_t":$operation_ordinal, "uint64_t":$result_ordinal);
  let assemblyFormat = [{
    `<` $function_ordinal `,` $block_ordinal `,` $operation_ordinal `,`
        $result_ordinal `>`
  }];
}

#endif // SHUTTLE_IR_SHUTTLEATTRS_TD_
