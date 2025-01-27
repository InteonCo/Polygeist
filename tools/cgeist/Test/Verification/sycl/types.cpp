// Copyright (C) Codeplay Software Limited

//===--- types.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: sycl-clang.py %s | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: !sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK: !sycl_accessor_2_i32_read_write_global_buffer = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK: !sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
// CHECK: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
// CHECK: !sycl_group_1_ = !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>
// CHECK: !sycl_group_2_ = !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>
// CHECK: !sycl_id_1_ = !sycl.id<1>
// CHECK: !sycl_id_2_ = !sycl.id<2>
// CHECK: !sycl_item_1_1_ = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>
// CHECK: !sycl_item_2_0_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>
// CHECK: !sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>, !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl.range<1>, !sycl.id<1>)>)>, !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>)>
// CHECK: !sycl_nd_item_2_ = !sycl.nd_item<[2], (!sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>, !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>, !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>)>
// CHECK: !sycl_range_1_ = !sycl.range<1>
// CHECK: !sycl_range_2_ = !sycl.range<2>

// CHECK: func.func @_Z4id_1N2cl4sycl2idILi1EEE(%arg0: !sycl_id_1_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void id_1(sycl::id<1> id) {}

// CHECK: func.func @_Z4id_2N2cl4sycl2idILi2EEE(%arg0: !sycl_id_2_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void id_2(sycl::id<2> id) {}

// CHECK: func.func @_Z5acc_1N2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl_accessor_1_i32_read_write_global_buffer) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void acc_1(sycl::accessor<cl::sycl::cl_int, 1, sycl::access::mode::read_write>) {}

// CHECK: func.func @_Z5acc_2N2cl4sycl8accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl_accessor_2_i32_read_write_global_buffer) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void acc_2(sycl::accessor<cl::sycl::cl_int, 2, sycl::access::mode::read_write>) {}

// CHECK: func.func @_Z7range_1N2cl4sycl5rangeILi1EEE(%arg0: !sycl_range_1_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void range_1(sycl::range<1> range) {}

// CHECK: func.func @_Z7range_2N2cl4sycl5rangeILi2EEE(%arg0: !sycl_range_2_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void range_2(sycl::range<2> range) {}

// CHECK: func.func @_Z5arr_1N2cl4sycl6detail5arrayILi1EEE(%arg0: !sycl_array_1_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void arr_1(sycl::detail::array<1> arr) {}

// CHECK: func.func @_Z5arr_2N2cl4sycl6detail5arrayILi2EEE(%arg0: !sycl_array_2_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void arr_2(sycl::detail::array<2> arr) {}

// CHECK: func.func @_Z11item_1_trueN2cl4sycl4itemILi1ELb1EEE(%arg0: !sycl_item_1_1_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void item_1_true(sycl::item<1, true> item) {}

// CHECK: func.func @_Z12item_2_falseN2cl4sycl4itemILi2ELb0EEE(%arg0: !sycl_item_2_0_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void item_2_false(sycl::item<2, false> item) {}

// CHECK: func.func @_Z9nd_item_1N2cl4sycl7nd_itemILi1EEE(%arg0: !sycl_nd_item_1_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void nd_item_1(sycl::nd_item<1> nd_item) {}

// CHECK: func.func @_Z9nd_item_2N2cl4sycl7nd_itemILi2EEE(%arg0: !sycl_nd_item_2_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void nd_item_2(sycl::nd_item<2> nd_item) {}

// CHECK: func.func @_Z7group_1N2cl4sycl5groupILi1EEE(%arg0: !sycl_group_1_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void group_1(sycl::group<1> group) {}

// CHECK: func.func @_Z7group_2N2cl4sycl5groupILi2EEE(%arg0: !sycl_group_2_) attributes {llvm.linkage = #llvm.linkage<external>}

[[intel::halide]] void group_2(sycl::group<2> group) {}
