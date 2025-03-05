#pragma once

#include <CL/sycl.hpp>
#include <vector>

using namespace hipsycl;

template <typename real_type>
auto forwardsub_CPU(const std::vector<real_type> &L, size_t n, std::vector<real_type> &b) {
}

template <typename real_type>
auto forwardsub_GPU_0(sycl::queue &q, real_type *const gpuA, size_t N, real_type *gpub) {
    const auto index = [](size_t i, size_t j) -> size_t { return i * (i + 1) / 2 + j; };

    sycl::event e;

    for (size_t _j = 1; _j < N; ++_j) {
        const auto j = _j - 1;

        e = q.parallel_for(sycl::range<1>(N - _j), e, [=](sycl::id<1> idx) {
            const auto i = idx[0] + _j;
            gpub[i] -= gpuA[index(i, j)] * gpub[j] / gpuA[index(j, j)];
        });
    }

    e = q.parallel_for(sycl::range<1>(N), e, [=](sycl::id<1> idx) {
        const auto j = idx[0];
        gpub[j] /= gpuA[index(j, j)];
    });
}

template <typename real_type>
auto backwardsub_GPU_0(sycl::queue &q, real_type *const gpuA, size_t N, real_type *gpub) {
    const auto index = [](size_t i, size_t j) -> size_t { return i * (i + 1) / 2 + j; };

    sycl::event e;

    for (size_t j = N - 1; j > 0; --j) {
        e = q.parallel_for(sycl::range<1>(j), e, [=](sycl::id<1> idx) {
            const auto i = idx[0];
            gpub[i] -= gpuA[index(j, i)] * gpub[j] / gpuA[index(j, j)];
        });
    }

    e = q.parallel_for(sycl::range<1>(N), e, [=](sycl::id<1> idx) {
        const auto j = idx[0];
        gpub[j] /= gpuA[index(j, j)];
    });
}
