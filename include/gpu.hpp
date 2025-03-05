#pragma once

#include <CL/sycl.hpp>

using namespace hipsycl;

template <typename Float>
struct GPUAlloc {
    GPUAlloc(sycl::queue &q, size_t gpu_allocN) :
        q(q) {
        mem = sycl::malloc_device<Float>(gpu_allocN, q);
    }

    ~GPUAlloc() {
        sycl::free(mem, q);
    }

    sycl::queue &q;
    Float *mem;
};
