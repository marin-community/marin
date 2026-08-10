#ifndef SHUTTLE_TRANSFORMS_PASSES_TD_
#define SHUTTLE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ShuttleAnnotateSourcePass
    : Pass<"shuttle-annotate-source", "::mlir::ModuleOp"> {
  let summary = "Attach rename-invariant structural source references";
  let dependentDialects = ["::mlir::shuttle::ShuttleDialect"];
  let constructor = "createAnnotateSourcePass()";
}

def ShuttleFormStructuralRegionsPass
    : Pass<"shuttle-form-structural-regions", "::mlir::ModuleOp"> {
  let summary = "Form maximal supported convex StableHLO regions";
  let constructor = "createFormStructuralRegionsPass()";
}

def ShuttleConvertStablehloToAlgebraPass
    : Pass<"shuttle-convert-stablehlo-to-algebra", "::mlir::ModuleOp"> {
  let summary = "Partially convert selected StableHLO to Shuttle algebra";
  let dependentDialects = ["::mlir::shuttle::ShuttleDialect"];
  let constructor = "createConvertStablehloToAlgebraPass()";
}

def ShuttleVerifySourceCoveragePass
    : Pass<"shuttle-verify-source-coverage", "::mlir::ModuleOp"> {
  let summary = "Verify exact source-reference coverage within Shuttle regions";
  let constructor = "createVerifySourceCoveragePass()";
}

def ShuttleVerifySemanticErasurePass
    : Pass<"shuttle-verify-semantic-erasure", "::mlir::ModuleOp"> {
  let summary = "Reject selected source operations that survived conversion";
  let constructor = "createVerifySemanticErasurePass()";
}

def ShuttleCanonicalizePass : Pass<"shuttle-canonicalize", "::mlir::ModuleOp"> {
  let summary = "Canonicalize generic Shuttle algebra";
  let constructor = "createShuttleCanonicalizePass()";
}

def ShuttleLowerAlgebraToStablehloPass
    : Pass<"shuttle-lower-algebra-to-stablehlo", "::mlir::ModuleOp"> {
  let summary = "Lower all Shuttle algebra back to typed StableHLO";
  let constructor = "createLowerAlgebraToStablehloPass()";
}

def ShuttleVerifyNoShuttleOpsPass
    : Pass<"shuttle-verify-no-shuttle-ops", "::mlir::ModuleOp"> {
  let summary = "Reject any Shuttle operation or attribute before HLO export";
  let constructor = "createVerifyNoShuttleOpsPass()";
}

#endif // SHUTTLE_TRANSFORMS_PASSES_TD_
