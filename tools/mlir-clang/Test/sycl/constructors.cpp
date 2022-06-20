// Copyright (C) Codeplay Software Limited

//===--- constructors.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: sycl-clang.py %s 2> /dev/null | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: func @_Z6cons_1v() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %false = arith.constant false
// CHECK-NEXT: %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl.id<2>>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl.id<2>> to memref<?x!sycl.id<2>>
// CHECK-NEXT: %2 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl.id<2>>) -> !llvm.ptr<i8>
// CHECK-NEXT: %3 = "polygeist.typeSize"() {source = !sycl.id<2>} : () -> index
// CHECK-NEXT: %4 = arith.index_cast %3 : index to i64
// CHECK-NEXT: "llvm.intr.memset"(%2, %c0_i8, %4, %false) : (!llvm.ptr<i8>, i8, i64, i1) -> ()
// CHECK-NEXT: sycl.constructor(%1) {Type = @id} : (memref<?x!sycl.id<2>>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

[[intel::halide]] void cons_1() {
  auto id = sycl::id<2>{};
}

// CHECK: func @_Z6cons_2mm(%arg0: i64, %arg1: i64) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl.id<2>>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl.id<2>> to memref<?x!sycl.id<2>>
// CHECK-NEXT: sycl.constructor(%1, %arg0, %arg1) {Type = @id} : (memref<?x!sycl.id<2>>, i64, i64) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

[[intel::halide]] void cons_2(size_t a, size_t b) {
  auto id = sycl::id<2>{a, b};
}

// CHECK: func @_Z6cons_3N2cl4sycl4itemILi2ELb1EEE(%arg0: !sycl.item<2, 1>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl.id<2>>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl.id<2>> to memref<?x!sycl.id<2>>
// CHECK-NEXT: %2 = memref.alloca() : memref<1x!sycl.item<2, 1>>
// CHECK-NEXT: %3 = memref.cast %2 : memref<1x!sycl.item<2, 1>> to memref<?x!sycl.item<2, 1>>
// CHECK-NEXT: affine.store %arg0, %2[0] : memref<1x!sycl.item<2, 1>>
// CHECK-NEXT: sycl.constructor(%1, %3) {Type = @id} : (memref<?x!sycl.id<2>>, memref<?x!sycl.item<2, 1>>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

[[intel::halide]] void cons_3(sycl::item<2, true> val) {
  auto id = sycl::id<2>{val};
}

// CHECK: func @_Z6cons_4N2cl4sycl2idILi2EEE(%arg0: !sycl.id<2>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl.id<2>>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl.id<2>> to memref<?x!sycl.id<2>>
// CHECK-NEXT: %2 = memref.alloca() : memref<1x!sycl.id<2>>
// CHECK-NEXT: %3 = memref.cast %2 : memref<1x!sycl.id<2>> to memref<?x!sycl.id<2>>
// CHECK-NEXT: affine.store %arg0, %2[0] : memref<1x!sycl.id<2>>
// CHECK-NEXT: sycl.constructor(%1, %3) {Type = @id} : (memref<?x!sycl.id<2>>, memref<?x!sycl.id<2>>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

[[intel::halide]] void cons_4(sycl::id<2> val) {
  auto id = sycl::id<2>{val};
}
