#ifndef SHUTTLE_IR_SHUTTLEDIALECT_TD_
#define SHUTTLE_IR_SHUTTLEDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Shuttle_Dialect : Dialect {
  let name = "shuttle";
  let cppNamespace = "::mlir::shuttle";
  let summary = "Generic tensor algebra selected from typed StableHLO";
  let description = [{
    Shuttle is a transient compiler dialect. Its operations describe structural
    tensor algebra after selection from StableHLO and before lowering back to
    legal XLA input. Workload names are deliberately absent from the dialect.
  }];
  let useDefaultAttributePrinterParser = 1;
}

#endif // SHUTTLE_IR_SHUTTLEDIALECT_TD_
