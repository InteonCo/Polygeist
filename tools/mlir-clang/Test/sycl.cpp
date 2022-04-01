// RUN: sycl-clang.py %s 2> /dev/null | FileCheck %s

#include <sycl/sycl.hpp>

// sycl range

// clang-format off
// CHECK: func @_Z8functionN2cl4sycl5rangeILi1EEE(%arg0: !sycl.range<1>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:         %0 = memref.alloca() : memref<1x!sycl.range<1>>
// CHECK-NEXT:         %1 = memref.cast %0 : memref<1x!sycl.range<1>> to memref<?x!sycl.range<1>>
// CHECK-NEXT:         %c0 = arith.constant 0 : index
// CHECK-NEXT:         memref.store %arg0, %1[%c0] : memref<?x!sycl.range<1>>
// CHECK-NEXT:         %true = arith.constant true
// CHECK-NEXT:         %2 = memref.alloca() : memref<i1>
// CHECK-NEXT:         %3 = memref.alloca() : memref<i1>
// CHECK-NEXT:         memref.store %true, %3[] : memref<i1>
// CHECK-NEXT:         memref.store %true, %2[] : memref<i1>
// CHECK-NEXT:        return
// CHECK-NEXT:      }
// clang-format on
[[intel::halide]] void function(sycl::range<1> range) {}

// clang-format off
// CHECK: func @_Z8functionN2cl4sycl2idILi2EEES2_(%arg0: !sycl.id<2>, %arg1: !sycl.id<2>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK: %[[VAL_0:.*]] = memref.cast %[[VAL_1:.*]] : memref<?x!sycl.id<2>> to memref<?x!sycl.array<2>>
// CHECK-NEXT: %[[VAL_2:.*]] = memref.cast %[[VAL_3:.*]] : memref<?x!sycl.id<2>> to memref<?x!sycl.array<2>>
// CHECK-NEXT: %[[VAL_4:.*]] = call @_ZNK2cl4sycl6detail5arrayILi2EEeqERKS3_(%[[VAL_0]], %[[VAL_2]]) : (memref<?x!sycl.array<2>>, memref<?x!sycl.array<2>>) -> i8
// CHECK-NEXT: %[[VAL_C:.*]] = arith.constant 0 : index
// CHECK-NEXT: memref.store %[[VAL_4]], %[[VAL_5:.*]][%[[VAL_C]]] : memref<?xi8>
// clang-format on
[[intel::halide]] void function(sycl::id<2> a, sycl::id<2> b) {
  auto id = a == b;
}

