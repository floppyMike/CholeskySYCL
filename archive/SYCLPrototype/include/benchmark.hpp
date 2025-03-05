#pragma once

#include <chrono>
#include <CL/sycl.hpp>
#include <iostream>

using namespace hipsycl;

template <typename real_type, typename IDX>
auto cholesky_CPU(std::vector<real_type> &A, size_t n, IDX index) -> void {
    for (size_t k = 0; k < n; ++k) {
        for (size_t j = k + 1; j < n; ++j) {
            for (size_t i = j; i < n; ++i) {
                A[index(i, j)] -= A[index(i, k)] * A[index(j, k)] / A[index(k, k)];
            }
        }
    }

    for (size_t k = 0; k < n; ++k) {
        A[index(k, k)] = std::sqrt(A[index(k, k)]);
    }

    for (size_t k = 0; k < n; ++k) {
        for (size_t i = k + 1; i < n; ++i) {
            A[index(i, k)] /= A[index(k, k)];
        }
    }
}

template <size_t N, typename Float, typename IDX, typename F>
void testCholesky(bool showall, std::vector<Float> &A, IDX index, F f) {
    sycl::queue q([](const sycl::device &dev) {
        if (dev.is_gpu()) {
            return 1;
        } else {
            return -1;
        }
    });

    // Init IO
    auto gpuA = sycl::malloc_device<Float>(A.size(), q);
    q.memcpy(gpuA, A.data(), sizeof(Float) * A.size());

    // Setup Benchmark
    q.wait_and_throw();
    std::cout << "Started Calculation..." << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();

    // Do Cholesky
    f(q, gpuA);

    // Report benchmark
    q.wait_and_throw();
    const auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time: " << elapsed << " (" << std::chrono::duration<double>(elapsed) << ')' << std::endl;

    // Compare with CPU
    // Copy over data for testing
    std::vector<Float> Asol(N * (N + 1) / 2);
    const auto indexCPU = [](size_t i, size_t j) -> size_t { return i * (i + 1) / 2 + j; };
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            Asol[indexCPU(i, j)] = A[index(i, j)];
        }
    }

    // Copy over result
    q.memcpy(A.data(), gpuA, sizeof(Float) * A.size());
    q.wait_and_throw();

    // Solve correctly
    std::cout << "Solving on CPU..." << std::endl;
    cholesky_CPU(Asol, N, indexCPU);

    // Compare
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            // if (std::abs(A[index(i, j)] - Asol[indexCPU(i, j)]) / A[index(i, j)] >= 10e-5) {
            if (showall || A[index(i, j)] != Asol[indexCPU(i, j)]) {
                std::cout << "Failed! With is: " << A[index(i, j)] << ", should: " << Asol[indexCPU(i, j)] << " at i: " << i << " j: " << j << std::endl;
            }
        }
    }

    // Clean up
    sycl::free(gpuA, q);
}

template <typename Float, typename F>
void benchmarkCholesky(const std::vector<Float> &A, F f) {
    sycl::queue q([](const sycl::device &dev) {
        if (dev.is_gpu()) {
            return 1;
        } else {
            return -1;
        }
    });

    // Init IO
    auto gpuA = sycl::malloc_device<Float>(A.size(), q);
    q.memcpy(gpuA, A.data(), sizeof(Float) * A.size());

    // Setup Benchmark
    q.wait_and_throw();
    std::cout << "Started Calculation..." << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();

    // Do Cholesky
    f(q, gpuA);

    // Report benchmark
    q.wait_and_throw();
    const auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time: " << elapsed << " (" << std::chrono::duration<double>(elapsed) << ')' << std::endl;

    // Clean up
    sycl::free(gpuA, q);
}

template <typename Float, typename F>
void get_matrix(sycl::queue &q, Float *gpuA, size_t size, F f) {
    std::vector<Float> A(size);

    q.wait_and_throw();
    q.memcpy(A.data(), gpuA, sizeof(Float) * A.size());
    q.wait_and_throw();

    f(A);
}

template <typename Float, typename F>
void simpleTestCholesky(std::vector<Float> &A, F f) {
    sycl::queue q([](const sycl::device &dev) {
        if (dev.is_gpu()) {
            return 1;
        } else {
            return -1;
        }
    });

    // Init IO
    auto gpuA = sycl::malloc_device<Float>(A.size(), q);
    q.memcpy(gpuA, A.data(), sizeof(Float) * A.size());

    // Test Cholesky
    q.wait_and_throw();
    f(q, gpuA);
    q.wait_and_throw();

    // Copy back
    q.memcpy(A.data(), gpuA, sizeof(Float) * A.size());
    q.wait_and_throw();

    // Clean up
    sycl::free(gpuA, q);
}
