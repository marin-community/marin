// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

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
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::shuttle::ShuttleDialect"];
  let constructor = "createConvertStablehloToAlgebraPass()";
}

def ShuttleVerifySourceCoveragePass
    : Pass<"shuttle-verify-source-coverage", "::mlir::ModuleOp"> {
  let summary = "Verify pre-lowering source coverage within Shuttle regions";
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
  let dependentDialects = ["::mlir::stablehlo::StablehloDialect"];
  let constructor = "createLowerAlgebraToStablehloPass()";
}

def ShuttleStripSourceProvenancePass
    : Pass<"shuttle-strip-source-provenance", "::mlir::ModuleOp"> {
  let summary = "Strip verified transient Shuttle provenance";
  let constructor = "createStripSourceProvenancePass()";
}

def ShuttleVerifyNoShuttleOpsPass
    : Pass<"shuttle-verify-no-shuttle-ops", "::mlir::ModuleOp"> {
  let summary = "Reject any Shuttle operation or attribute before HLO export";
  let constructor = "createVerifyNoShuttleOpsPass()";
}

def ShuttlePlanRowFoldMaterializationPass
    : Pass<"shuttle-plan-row-fold-materialization", "::mlir::ModuleOp"> {
  let summary = "Derive one row Fold and Map materialization plan";
  let dependentDialects = ["::mlir::shuttle::ShuttleDialect"];
  let constructor = "createPlanRowFoldMaterializationPass()";
}

def ShuttleVerifyMaterializationPlanPass
    : Pass<"shuttle-verify-materialization-plan", "::mlir::ModuleOp"> {
  let summary = "Verify materialization plans remain bound to Shuttle algebra";
  let constructor = "createVerifyMaterializationPlanPass()";
}

#endif // SHUTTLE_TRANSFORMS_PASSES_TD_
