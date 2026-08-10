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
