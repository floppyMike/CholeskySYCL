#pragma once

#include "gpu.hpp"
#include "math.hpp"
#include "matrix.hpp"
#include <iostream>

inline auto sum_vector(const std::vector<uint64_t> &v) -> uint64_t {
    uint64_t sum = 0;
    for (const auto i : v) {
        sum += i;
    }
    return sum;
}

inline auto to_nano(uint64_t v) -> double {
    return v / 1000000000.;
}

// Test 1

template <typename Float, typename IDX>
void generate_ones(size_t N, Float *mem, IDX idx) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mem[idx(N, i, j)] = j + 1;
        }
    }
}

template <typename Float, typename IDX>
auto norm_ones(size_t N, const Float *mem, IDX idx) -> Float {
    Float sum = 0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            sum += pow2(mem[idx(N, i, j)] - 1);
        }
    }

    return std::sqrt(sum) / std::sqrt(N * (N + 1) / 2);
}

// Test 2

template <typename Float, typename IDX>
void generate_rowascend(size_t N, Float *mem, IDX idx) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mem[idx(N, i, j)] = std::min(i + 1, j + 1) * (i + 1) * (j + 1);
        }
    }
}

template <typename Float, typename VECIDX>
void generate_rowascend_forward_vecs(size_t N, size_t Nb, Float *mem, VECIDX idx) {
    for (size_t i = 0; i < Nb; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mem[idx(N, Nb, i, j)] = ((j + 1) * (j + 1) * (j + 2) / 2) * (i + 1);
        }
    }
}

template <typename Float, typename VECIDX>
void generate_rowascend_backward_vecs(size_t N, size_t Nb, Float *mem, VECIDX idx) {
    for (size_t i = 0; i < Nb; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mem[idx(N, Nb, i, j)] = (N * (N + 1) * (2 * N + 1) / 6 - j * (j + 1) * (2 * j + 1) / 6) * (i + 1);
        }
    }
}

template <typename Float, typename MATIDX>
void generate_rowascend_vecs_mat(size_t N, Float *mem, MATIDX idx) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mem[idx(N, i, j)] = i + 1;
        }
    }
}

template <typename Float, typename IDX>
auto norm_rowascend(size_t N, const Float *mem, IDX idx) -> Float {
    Float sum = 0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            sum += pow2(mem[idx(N, i, j)] - (i + 1));
        }
    }

    return std::sqrt(sum) / (N * (N + 1) / 2);
}

template <typename Float, typename VECIDX>
auto norm_rowascend_vecs(size_t N, size_t Nb, const Float *mem, VECIDX idx) -> Float {
    Float sum = 0;

    for (size_t i = 0; i < Nb; ++i) {
        for (size_t j = 0; j < N; ++j) {
            sum += pow2(mem[idx(N, Nb, i, j)] - (i + 1) * (j + 1));
        }
    }

    return std::sqrt(sum) / std::sqrt(N * (N + 1) * (2 * N + 1) / 6 * Nb * (Nb + 1) * (2 * Nb + 1) / 6);
}

// Test 3

template <typename Float, typename IDX>
void generate_colascend(size_t N, Float *mem, IDX idx) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mem[idx(N, i, j)] = (j + 1) * (j + 2) * (2 * (j + 1) + 1) / 6;
        }
    }
}

template <typename Float, typename VECIDX>
void generate_colascend_forward_vecs(size_t N, size_t Nb, Float *mem, VECIDX idx) {
    for (size_t i = 0; i < Nb; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mem[idx(N, Nb, i, j)] = ((j + 1) * (j + 2) * (2 * (j + 1) + 1) / 6) * (i + 1);
        }
    }
}

template <typename Float, typename VECIDX>
void generate_colascend_backward_vecs(size_t N, size_t Nb, Float *mem, VECIDX idx) {
    for (size_t i = 0; i < Nb; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mem[idx(N, Nb, i, j)] = (j + 1) * (N * (N + 1) / 2 - j * (j + 1) / 2) * (i + 1);
        }
    }
}

template <typename Float, typename MATIDX>
void generate_colascend_vecs_mat(size_t N, Float *mem, MATIDX idx) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mem[idx(N, i, j)] = j + 1;
        }
    }
}

template <typename Float, typename IDX>
auto norm_colascend(size_t N, const Float *mem, IDX idx) -> Float {
    Float sum = 0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            sum += pow2(mem[idx(N, i, j)] - (j + 1));
        }
    }

    return std::sqrt(sum) / std::sqrt(N * pow2(N + 1) * (N + 2) / 12);
}

template <typename Float, typename VECIDX>
auto norm_colascend_vecs(size_t N, size_t Nb, const Float *mem, VECIDX idx) -> Float {
    Float sum = 0;

    for (size_t i = 0; i < Nb; ++i) {
        for (size_t j = 0; j < N; ++j) {
            sum += pow2(mem[idx(N, Nb, i, j)] - (i + 1) * (j + 1));
        }
    }

    return std::sqrt(sum) / std::sqrt(N * (N + 1) * (2 * N + 1) / 6 * Nb * (Nb + 1) * (2 * Nb + 1) / 6);
}

template <typename Float, typename IDX, typename GEN, typename F>
void validateCholesky(bool printmatrix, sycl::queue &q, size_t N, IDX idx, GEN gen, F f) {
    const auto gpu_allocN = gen(N);

    std::vector<Float> gpu_io(gpu_allocN);
    GPUAlloc<Float> gpuA(q, gpu_allocN);

    // Test 1

    generate_ones(N, gpu_io.data(), idx);

    q.memcpy(gpuA.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto onestime = sum_vector(f(N, gpuA.mem, 0.));
    q.memcpy(gpu_io.data(), gpuA.mem, sizeof(Float) * gpu_io.size()).wait();
    const auto ones = norm_ones(N, gpu_io.data(), idx);

    if (printmatrix) {
        print_rowbased(gpu_io.data(), N, idx);
        std::cout << std::endl;
    }

    // Test 2

    generate_rowascend(N, gpu_io.data(), idx);

    q.memcpy(gpuA.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto rowascendtime = sum_vector(f(N, gpuA.mem, 0.));
    q.memcpy(gpu_io.data(), gpuA.mem, sizeof(Float) * gpu_io.size()).wait();
    const auto rowascend = norm_rowascend(N, gpu_io.data(), idx);

    if (printmatrix) {
        print_rowbased(gpu_io.data(), N, idx);
        std::cout << std::endl;
    }

    // Test 3

    generate_colascend(N, gpu_io.data(), idx);

    q.memcpy(gpuA.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto colascendtime = sum_vector(f(N, gpuA.mem, 0.));
    q.memcpy(gpu_io.data(), gpuA.mem, sizeof(Float) * gpu_io.size()).wait();
    const auto colascend = norm_colascend(N, gpu_io.data(), idx);

    if (printmatrix) {
        print_rowbased(gpu_io.data(), N, idx);
        std::cout << std::endl;
    }

    std::cout << "N: " << N
              << ", ones: " << ones << " (" << to_nano(onestime)
              << "), rowascend: " << rowascend << " (" << to_nano(rowascendtime)
              << "), colascend: " << colascend << " (" << to_nano(colascendtime) << ')' << std::endl;
}

template <typename Float, typename MATIDX, typename MATGEN, typename VECIDX, typename VECGEN, typename F>
void validateForwardSubsitution(bool printvector, sycl::queue &q, size_t N, size_t Nb, MATIDX matidx, MATGEN matgen, VECIDX vecidx, VECGEN vecgen, F f) {
    const auto gpuA_allocN = matgen(N);
    const auto gpuB_allocN = vecgen(N, Nb);

    GPUAlloc<Float> gpuA(q, gpuA_allocN);
    GPUAlloc<Float> gpuB(q, gpuB_allocN);

    std::vector<Float> gpu_io;

    // Test 2

    gpu_io.resize(gpuA_allocN);
    generate_rowascend_vecs_mat(N, gpu_io.data(), matidx);
    q.memcpy(gpuA.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    gpu_io.resize(gpuB_allocN);

    generate_rowascend_forward_vecs(N, Nb, gpu_io.data(), vecidx);

    q.memcpy(gpuB.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto rowascendtime = sum_vector(f(N, Nb, gpuA.mem, gpuB.mem));
    q.memcpy(gpu_io.data(), gpuB.mem, sizeof(Float) * gpu_io.size()).wait();
    const auto rowascend = norm_rowascend_vecs(N, Nb, gpu_io.data(), vecidx);

    if (printvector) {
        print_vectors(gpu_io.data(), N, Nb, vecidx);
        std::cout << std::endl;
    }

    // Test 3

    gpu_io.resize(gpuA_allocN);
    generate_colascend_vecs_mat(N, gpu_io.data(), matidx);
    q.memcpy(gpuA.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    gpu_io.resize(gpuB_allocN);

    generate_colascend_forward_vecs(N, Nb, gpu_io.data(), vecidx);

    q.memcpy(gpuB.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto colascendtime = sum_vector(f(N, Nb, gpuA.mem, gpuB.mem));
    q.memcpy(gpu_io.data(), gpuB.mem, sizeof(Float) * gpu_io.size()).wait();
    const auto colascend = norm_colascend_vecs(N, Nb, gpu_io.data(), vecidx);

    if (printvector) {
        print_vectors(gpu_io.data(), N, Nb, vecidx);
        std::cout << std::endl;
    }

    std::cout << "N: " << N
              << ", rowascend: " << rowascend << " (" << to_nano(rowascendtime)
              << "), colascend: " << colascend << " (" << to_nano(colascendtime) << ')'
              << std::endl;
}

template <typename Float, typename MATIDX, typename MATGEN, typename VECIDX, typename VECGEN, typename F>
void validateBackwardSubsitution(bool printvector, sycl::queue &q, size_t N, size_t Nb, MATIDX matidx, MATGEN matgen, VECIDX vecidx, VECGEN vecgen, F f) {
    const auto gpuA_allocN = matgen(N);
    const auto gpuB_allocN = vecgen(N, Nb);

    GPUAlloc<Float> gpuA(q, gpuA_allocN);
    GPUAlloc<Float> gpuB(q, gpuB_allocN);

    std::vector<Float> gpu_io;

    // Test 2

    gpu_io.resize(gpuA_allocN);
    generate_rowascend_vecs_mat(N, gpu_io.data(), matidx);
    q.memcpy(gpuA.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    gpu_io.resize(gpuB_allocN);

    generate_rowascend_backward_vecs(N, Nb, gpu_io.data(), vecidx);

    q.memcpy(gpuB.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto rowascendtime = sum_vector(f(N, Nb, gpuA.mem, gpuB.mem));
    q.memcpy(gpu_io.data(), gpuB.mem, sizeof(Float) * gpu_io.size()).wait();
    const auto rowascend = norm_rowascend_vecs(N, Nb, gpu_io.data(), vecidx);

    if (printvector) {
        print_vectors(gpu_io.data(), N, Nb, vecidx);
        std::cout << std::endl;
    }

    // Test 3

    gpu_io.resize(gpuA_allocN);
    generate_colascend_vecs_mat(N, gpu_io.data(), matidx);
    q.memcpy(gpuA.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    gpu_io.resize(gpuB_allocN);

    generate_colascend_backward_vecs(N, Nb, gpu_io.data(), vecidx);

    q.memcpy(gpuB.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto colascendtime = sum_vector(f(N, Nb, gpuA.mem, gpuB.mem));
    q.memcpy(gpu_io.data(), gpuB.mem, sizeof(Float) * gpu_io.size()).wait();
    const auto colascend = norm_colascend_vecs(N, Nb, gpu_io.data(), vecidx);

    if (printvector) {
        print_vectors(gpu_io.data(), N, Nb, vecidx);
        std::cout << std::endl;
    }

    std::cout << "N: " << N
              << ", rowascend: " << rowascend << " (" << to_nano(rowascendtime)
              << "), colascend: " << colascend << " (" << to_nano(colascendtime) << ')'
              << std::endl;
}

// 1
// 1 2
// 1 2 3
template <typename Float, typename IDX>
void generate_ones_all(size_t N, Float *mem, IDX idx) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mem[idx(N, i, j)] = j + 1;
        }
    }
}

// 03 05 06
// 06 10 12
// 12 20 24
template <typename Float, typename VECIDX>
void generate_ones_vec_all(size_t N, size_t Nb, Float *mem, VECIDX idx) {
    for (size_t i = 0; i < Nb; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mem[idx(N, Nb, i, j)] = ((j + 1) * (j + 2) / 2 + (N - j - 1) * (j + 1)) * (i + 1);
        }
    }
}

// 1 1 1
// 2 2 2
// 3 3 3
template <typename Float, typename VECIDX>
auto norm_ones_vec_all(size_t N, size_t Nb, const Float *mem, VECIDX idx) -> Float {
    Float sum = 0;

    for (size_t i = 0; i < Nb; ++i) {
        for (size_t j = 0; j < N; ++j) {
            sum += pow2(mem[idx(N, Nb, i, j)] - (i + 1));
        }
    }

    return std::sqrt(sum) / std::sqrt(Nb * (Nb + 1) / 2 * N);
}

template <typename Float, typename MATIDX, typename MATGEN, typename VECIDX, typename VECGEN, typename CHOL, typename FSUB, typename BSUB>
void validateAll(bool print, sycl::queue &q, size_t N, size_t Nb, MATIDX matidx, MATGEN matgen, VECIDX vecidx, VECGEN vecgen, CHOL cholkernel, FSUB fsubkernel, BSUB bsubkernel) {
    const auto gpuA_allocN = matgen(N);
    const auto gpuB_allocN = vecgen(N, Nb);

    GPUAlloc<Float> gpuA(q, gpuA_allocN);
    GPUAlloc<Float> gpuB(q, gpuB_allocN);

    std::vector<Float> gpu_io;

    // Test 1

    gpu_io.resize(gpuA_allocN);

    generate_ones_all(N, gpu_io.data(), matidx);
    if (print) {
        print_rowbased(gpu_io.data(), N, matidx);
        std::cout << std::endl;
    }
    q.memcpy(gpuA.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto choltime = sum_vector(cholkernel(N, gpuA.mem, 0.));

    gpu_io.resize(gpuB_allocN);

    generate_ones_vec_all(N, Nb, gpu_io.data(), vecidx);
    if (print) {
        print_vectors(gpu_io.data(), N, Nb, vecidx);
        std::cout << std::endl;
    }
    q.memcpy(gpuB.mem, gpu_io.data(), sizeof(Float) * gpu_io.size()).wait();
    const auto fsubtime = sum_vector(fsubkernel(N, Nb, gpuA.mem, gpuB.mem));
    const auto bsubtime = sum_vector(bsubkernel(N, Nb, gpuA.mem, gpuB.mem));
    q.memcpy(gpu_io.data(), gpuB.mem, sizeof(Float) * gpu_io.size()).wait();
    if (print) {
        print_vectors(gpu_io.data(), N, Nb, vecidx);
        std::cout << std::endl;
    }

    const auto error = norm_ones_vec_all(N, Nb, gpu_io.data(), vecidx);

    std::cout << "N: " << N
              << ", cholesky: " << to_nano(choltime)
              << ", fsub: " << to_nano(fsubtime)
              << ", bsub: " << to_nano(bsubtime)
              << ", total: " << to_nano(choltime + fsubtime + bsubtime)
              << ", error: " << error
              << std::endl;
}
