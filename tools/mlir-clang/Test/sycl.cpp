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

// clang-format off
// CHECK:   func @_Z9function1N2cl4sycl2idILi2EEES2_(%arg0: !sycl.id<2>, %arg1: !sycl.id<2>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:      %10 = call @_ZN2cl4syclplERKNS0_2idILi2EEES4_(%5, %3) : (memref<?x!sycl.id<2>>, memref<?x!sycl.id<2>>) -> !sycl.id<2>
// CHECK-NEXT: %c0_1 = arith.constant 0 : index
// CHECK-NEXT: memref.store %10, %1[%c0_1] : memref<?x!sycl.id<2>>
// clang-format on
[[intel::halide]] void function1(sycl::id<2> a, sycl::id<2> b) {
  auto id = a + b;
}

// clang-format off
// CHECK:   func @_Z8functionN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEENS0_2idILi1EEE(%arg0: !sycl.accessor<1>, %arg1: !sycl.id<1>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:      call @_ZN2cl4sycl2idILi1EEC1ERKS2_(%[[VAL_0:.*]], %[[VAL_1:.*]]) : (memref<?x!sycl.id<1>>, memref<?x!sycl.id<1>>) -> ()
// CHECK-NEXT: %[[VAL_C:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[VAL_2:.*]] = memref.load %[[VAL_0]][%[[VAL_C]]] : memref<?x!sycl.id<1>>
// CHECK-NEXT: %[[VAL_3:.*]] = call @_ZNK2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE(%[[VAL_4:.*]], %[[VAL_2]]) : (memref<?x!sycl.accessor<1>>, !sycl.id<1>) -> memref<?xi32>
[[intel::halide]] void function(sycl::accessor<cl::sycl::cl_int, 1,
sycl::access::mode::read_write> A, sycl::id<1> id) {
  A[id];
}

// clang-format off
// CHECK:    func @_Z8functionN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEENS0_4itemILi1ELb1EEE(%arg0: !sycl.accessor<1>, %arg1: !sycl.item<1, 1>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:      call @_ZN2cl4sycl2idILi1EEC1ILi1ELb1EEERNSt9enable_ifIXeqT_Li1EEKNS0_4itemILi1EXT0_EEEE4typeE(%1, %3) : (memref<?x!sycl.id<1>>, memref<?x!sycl.item<1, 1>>) -> ()
// CHECK-NEXT: %c0_1 = arith.constant 0 : index
// CHECK-NEXT: %9 = memref.load %1[%c0_1] : memref<?x!sycl.id<1>>
// CHECK-NEXT: %10 = call @_ZNK2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE(%5, %9) : (memref<?x!sycl.accessor<1>>, !sycl.id<1>) -> memref<?xi32>
// clang-format on
[[intel::halide]] void
function(sycl::accessor<cl::sycl::cl_int, 1, sycl::access::mode::read_write> A,
         sycl::item<1, true> item) {
  A[item];
}
