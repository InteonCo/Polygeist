// RUN: sycl-clang.py %s | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: func @_ZTS8kernel_1(%arg0: memref<?xi32>, %arg1: !sycl.range<1>, %arg2: !sycl.range<1>, %arg3: !sycl.id<1>) attributes {SYCLKernel = "_ZTS8kernel_1", llvm.linkage = #llvm.linkage<weak_odr>}
// CHECK-NOT: SYCLKernel =

class kernel_1 {
 sycl::accessor<cl::sycl::cl_int, 1, sycl::access::mode::read_write> A;

public:
	kernel_1(sycl::accessor<cl::sycl::cl_int, 1, sycl::access::mode::read_write> A) : A(A) {}

 [[intel::halide]] void operator()(sycl::id<1> id) const {
   A[id] = 42;
 }
};

void host_1() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
      auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
      auto ker =  kernel_1{A};
      cgh.parallel_for<kernel_1>(range, ker);
    });
  }
}

// CHECK: func @_ZTSZZ6host_2vENKUlRN2cl4sycl7handlerEE_clES2_E8kernel_2(%arg0: memref<?xi32>, %arg1: !sycl.range<1>, %arg2: !sycl.range<1>, %arg3: !sycl.id<1>) attributes {SYCLKernel = "_ZTSZZ6host_2vENKUlRN2cl4sycl7handlerEE_clES2_E8kernel_2", llvm.linkage = #llvm.linkage<weak_odr>}
// CHECK-NOT: SYCLKernel =

void host_2() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
      auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class kernel_2>(range, [=](sycl::id<1> id) [[intel::halide]] {
        A[id] = 42;
      });
    });
  }
}

// CHECK-NOT: SYCLKernel =
[[intel::halide]] void function_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}