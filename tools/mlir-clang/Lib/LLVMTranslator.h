//===- LLVMTranslator.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_LLVM_TRANSLATOR
#define CLANG_MLIR_LLVM_TRANSLATOR

#include <clang/AST/Type.h>

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include <mlir/Dialect/SYCL/IR/SYCLOpsDialect.h>
#include <mlir/Dialect/SYCL/IR/SYCLOpsTypes.h>

#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

/// The LLVMToSYCLTranslator class is responsible to take LLVM IR as
/// input and will translate it to SYCL MLIR.
/// If the input can't be translated, it will fallback to the LLVM IR to LLVM
/// MLIR translator.
class LLVMToSYCLTranslator {
public:
  LLVMToSYCLTranslator(mlir::MLIRContext &Context,
                       mlir::LLVM::TypeFromLLVMIRTranslator &Translator)
      : Context(Context), Translator(Translator) {}
  ~LLVMToSYCLTranslator() = default;

  void setTemplateArguments(const llvm::StructType *ST,
                            const clang::RecordType *RT);
  mlir::Type translateType(llvm::Type *Type);

private:
  void translateTypes(llvm::ArrayRef<llvm::Type *> Types,
                      llvm::SmallVectorImpl<mlir::Type> &Result);

private:
  mlir::MLIRContext &Context;
  mlir::LLVM::TypeFromLLVMIRTranslator &Translator;

  llvm::Optional<mlir::Type> AttrType;
  llvm::Optional<int> AttrDimension;
  llvm::Optional<mlir::sycl::MemoryAccessMode> AttrMemAccessMode;
  llvm::Optional<mlir::sycl::MemoryTargetMode> AttrMemTargetMode;
  llvm::Optional<bool> AttrWithOffset;
};

#endif
