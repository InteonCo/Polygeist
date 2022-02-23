//===- LLVMTranslator.cc -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLVMTranslator.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclTemplate.h>

// JLE_QUEL::TODO
// Accessor's template argument type is still very specific to
// Integer and set as signless, should we generalize it?

/// The setTemplateArguments function is responsible to fetch and set the
/// template arguments that will be used to create the SYCL types
void LLVMToSYCLTranslator::setTemplateArguments(const llvm::StructType *ST,
                                                const clang::RecordType *RT) {
  if (ST->getName() == "class.cl::sycl::range") {
    if (const auto classTemplateSpecialization =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                RT->getAsRecordDecl())) {
      const auto &args = classTemplateSpecialization->getTemplateArgs();
      AttrDimension = args.get(0).getAsIntegral().getExtValue();
    }
  } else if (ST->getName() == "class.cl::sycl::detail::array") {
    if (const auto classTemplateSpecialization =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                RT->getAsRecordDecl())) {
      const auto &args = classTemplateSpecialization->getTemplateArgs();
      AttrDimension = args.get(0).getAsIntegral().getExtValue();
    }
  } else if (ST->getName() == "class.cl::sycl::id") {
    if (const auto classTemplateSpecialization =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                RT->getAsRecordDecl())) {
      const auto &args = classTemplateSpecialization->getTemplateArgs();
      AttrDimension = args.get(0).getAsIntegral().getExtValue();
    }
  } else if (ST->getName() == "class.cl::sycl::accessor") {
    if (const auto classTemplateSpecialization =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                RT->getAsRecordDecl())) {
      const auto &args = classTemplateSpecialization->getTemplateArgs();
      const auto TypeInfo =
          RT->getDecl()->getASTContext().getTypeInfo(args.get(0).getAsType());
      AttrType = mlir::IntegerType::get(&Context, TypeInfo.Width);
      AttrDimension = args.get(1).getAsIntegral().getExtValue();
      AttrMemAccessMode = static_cast<mlir::sycl::MemoryAccessMode>(
          args.get(2).getAsIntegral().getExtValue());
      AttrMemTargetMode = static_cast<mlir::sycl::MemoryTargetMode>(
          args.get(3).getAsIntegral().getExtValue());
    }
  } else if (ST->getName() == "class.cl::sycl::detail::AccessorImplDevice") {
    if (const auto classTemplateSpecialization =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                RT->getAsRecordDecl())) {
      const auto &args = classTemplateSpecialization->getTemplateArgs();
      AttrDimension = args.get(0).getAsIntegral().getExtValue();
    }
  } else if (ST->getName() == "class.cl::sycl::item") {
    if (const auto classTemplateSpecialization =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                RT->getAsRecordDecl())) {
      const auto &args = classTemplateSpecialization->getTemplateArgs();
      AttrDimension = args.get(0).getAsIntegral().getExtValue();
      AttrWithOffset = args.get(1).getAsIntegral().getExtValue();
    }
  } else if (ST->getName() == "struct.cl::sycl::detail::ItemBase") {
    if (const auto classTemplateSpecialization =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                RT->getAsRecordDecl())) {
      const auto &args = classTemplateSpecialization->getTemplateArgs();
      AttrDimension = args.get(0).getAsIntegral().getExtValue();
      AttrWithOffset = args.get(1).getAsIntegral().getExtValue();
    }
  } else {
    llvm_unreachable("SYCL type not handle (yet)");
  }
}

/// The translateType function is responsible to do the translation from
/// LLVM IR to SYCL MLIR.
/// If the input can't be translated, it will fallback to the LLVM IR to LLVM
/// MLIR translator.
mlir::Type LLVMToSYCLTranslator::translateType(llvm::Type *Type) {
  if (auto ST = llvm::dyn_cast<llvm::StructType>(Type)) {
    llvm::SmallVector<mlir::Type, 4> Body;

    if (ST->getName() == "class.cl::sycl::range") {
      translateTypes(ST->subtypes(), Body);
      assert(AttrDimension.hasValue() &&
             "sycl::range template argument (dimension) should not be None");
      return mlir::sycl::RangeType::get(&Context, AttrDimension.getValue(),
                                        Body);
    }
    if (ST->getName() == "class.cl::sycl::detail::array") {
      translateTypes(ST->subtypes(), Body);
      assert(AttrDimension.hasValue() &&
             "sycl::array template argument (dimension) should not be None");
      return mlir::sycl::ArrayType::get(&Context, AttrDimension.getValue(),
                                        Body);
    }
    if (ST->getName() == "class.cl::sycl::id") {
      translateTypes(ST->subtypes(), Body);
      assert(AttrDimension.hasValue() &&
             "sycl::id template argument (dimension) should not be None");
      return mlir::sycl::IDType::get(&Context, AttrDimension.getValue(), Body);
    }
    if (ST->getName() == "class.cl::sycl::accessor") {
      translateTypes(ST->subtypes(), Body);
      assert(AttrType.hasValue() == true &&
             "sycl::accessor template argument (type) should not be None");
      assert(AttrDimension.hasValue() == true &&
             "sycl::accessor template argument (dimension) should not be None");
      assert(AttrMemAccessMode.hasValue() == true &&
             "sycl::accessor template argument (memory access mode) should not "
             "be None");
      assert(AttrMemTargetMode.hasValue() == true &&
             "sycl::accessor template argument (memory target mode) should not "
             "be None");
      return mlir::sycl::AccessorType::get(
          &Context, AttrType.getValue(), AttrDimension.getValue(),
          AttrMemAccessMode.getValue(), AttrMemTargetMode.getValue(), Body);
    }
    if (ST->getName() == "class.cl::sycl::detail::AccessorImplDevice") {
      translateTypes(ST->subtypes(), Body);
      assert(AttrDimension.hasValue() &&
             "sycl::AccessorImplDevice template argument (dimension) should "
             "not be None");
      return mlir::sycl::AccessorImplDeviceType::get(
          &Context, AttrDimension.getValue(), Body);
    }
    if (ST->getName() == "class.cl::sycl::item") {
      translateTypes(ST->subtypes(), Body);
      assert(AttrDimension.hasValue() &&
             "sycl::item template argument (dimension) should not be None");
      assert(AttrWithOffset.hasValue() &&
             "sycl::item template argument (withoffset) should not be None");
      return mlir::sycl::ItemType::get(&Context, AttrDimension.getValue(),
                                       AttrWithOffset.getValue(), Body);
    }
    if (ST->getName() == "struct.cl::sycl::detail::ItemBase") {
      translateTypes(ST->subtypes(), Body);
      assert(AttrDimension.hasValue() &&
             "sycl::item template argument (dimension) should not be None");
      assert(AttrWithOffset.hasValue() &&
             "sycl::item template argument (withoffset) should not be None");
      return mlir::sycl::ItemBaseType::get(&Context, AttrDimension.getValue(),
                                           AttrWithOffset.getValue(), Body);
    }

    if (ST->getName().contains(".cl::sycl")) {
      llvm_unreachable("SYCL type not handle (yet)");
    }
  }

  return Translator.translateType(Type);
}

/// The translateTypes function is responsible to iterate over the members of
/// the aggregate type and translate thoses members
void LLVMToSYCLTranslator::translateTypes(
    llvm::ArrayRef<llvm::Type *> Types,
    llvm::SmallVectorImpl<mlir::Type> &Result) {
  Result.reserve(Result.size() + Types.size());
  for (llvm::Type *Type : Types) {
    Result.push_back(translateType(Type));
  }
}
