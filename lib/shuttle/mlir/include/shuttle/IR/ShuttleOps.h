// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_IR_SHUTTLEOPS_H_
#define SHUTTLE_IR_SHUTTLEOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleDialect.h"

#define GET_OP_CLASSES
#include "shuttle/IR/ShuttleOps.h.inc"

#endif // SHUTTLE_IR_SHUTTLEOPS_H_
