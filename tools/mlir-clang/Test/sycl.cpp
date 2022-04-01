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

