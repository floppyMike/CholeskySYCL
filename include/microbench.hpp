#pragma once

#include "gpu.hpp"
#include <chrono>

template <typename Float, typename F, typename U>
void microbench(sycl::queue &q, size_t N, F f, U u) {
    GPUAlloc<Float> gpuA(q, N * N);
    q.wait_and_throw();

    {
        const auto start = std::chrono::high_resolution_clock::now();
        f(N, gpuA.mem);
        q.wait_and_throw();
        const auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;

        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();
        u(N, gpuA.mem);
        q.wait_and_throw();
        const auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;

        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    }
}
