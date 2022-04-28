// RUN: sycl-clang.py %s | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: func @_Z8method_1N2cl4sycl4itemILi2ELb1EEE(%arg0: !sycl.item<2, 1>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl.item<2, 1>>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl.item<2, 1>> to memref<?x!sycl.item<2, 1>>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl.item<2, 1>>
// CHECK-NEXT: %2 = sycl.call(%1, %c0_i32) {Function = @get_id, Type = @item} : (memref<?x!sycl.item<2, 1>>, i32) -> i64
// CHECK-NEXT: return
// CHECK-NEXT: }

[[intel::halide]] void method_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}

// CHECK: func @_Z8method_2N2cl4sycl4itemILi2ELb1EEE(%arg0: !sycl.item<2, 1>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl.item<2, 1>>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl.item<2, 1>> to memref<?x!sycl.item<2, 1>>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl.item<2, 1>>
// CHECK-NEXT: %2 = sycl.call(%1, %1) {Function = @"operator==", Type = @item} : (memref<?x!sycl.item<2, 1>>, memref<?x!sycl.item<2, 1>>) -> i8
// CHECK-NEXT: return
// CHECK-NEXT: }

[[intel::halide]] void method_2(sycl::item<2, true> item) {
  auto id = item.operator==(item);
}

// CHECK: func @_Z4op_1N2cl4sycl2idILi2EEES2_(%arg0: !sycl.id<2>, %arg1: !sycl.id<2>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl.id<2>>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl.id<2>> to memref<?x!sycl.id<2>>
// CHECK-NEXT: %2 = memref.alloca() : memref<1x!sycl.id<2>>
// CHECK-NEXT: %3 = memref.cast %2 : memref<1x!sycl.id<2>> to memref<?x!sycl.id<2>>
// CHECK-NEXT: affine.store %arg0, %2[0] : memref<1x!sycl.id<2>>
// CHECK-NEXT: affine.store %arg1, %0[0] : memref<1x!sycl.id<2>>
// CHECK-NEXT: %4 = sycl.cast(%3) : (memref<?x!sycl.id<2>>) -> memref<?x!sycl.array<2>>
// CHECK-NEXT: %5 = sycl.cast(%1) : (memref<?x!sycl.id<2>>) -> memref<?x!sycl.array<2>>
// CHECK-NEXT: %6 = sycl.call(%4, %5) {Function = @"operator==", Type = @array} : (memref<?x!sycl.array<2>>, memref<?x!sycl.array<2>>) -> i8
// CHECK-NEXT: return
// CHECK-NEXT: }

[[intel::halide]] void op_1(sycl::id<2> a, sycl::id<2> b) {
  auto id = a == b;
}

// CHECK: func @_Z8static_1N2cl4sycl2idILi2EEES2_(%arg0: !sycl.id<2>, %arg1: !sycl.id<2>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl.id<2>>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl.id<2>> to memref<?x!sycl.id<2>>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl.id<2>>
// CHECK-NEXT: %2 = sycl.cast(%1) : (memref<?x!sycl.id<2>>) -> memref<?x!sycl.array<2>>
// CHECK-NEXT: %3 = sycl.call(%2, %c0_i32) {Function = @get, Type = @array} : (memref<?x!sycl.array<2>>, i32) -> i64
// CHECK-NEXT: %4 = sycl.call(%2, %c1_i32) {Function = @get, Type = @array} : (memref<?x!sycl.array<2>>, i32) -> i64
// CHECK-NEXT: %5 = arith.addi %3, %4 : i64
// CHECK-NEXT: %6 = sycl.call(%5) {Function = @abs} : (i64) -> i64
// CHECK-NEXT: return
// CHECK-NEXT: }

[[intel::halide]] void static_1(sycl::id<2> a, sycl::id<2> b) {
  auto abs = sycl::abs(a.get(0) + a.get(1));
}
