#pragma once

#include "gpu.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#ifndef USE_DOUBLE
constexpr static size_t divs[5] = { 0, 100, 1000, 10'000, 1'000'000 };
constexpr static const char *front = "float";
#else
constexpr static size_t divs[5] = { 0, 1'000'000'000, 100'000'000'000, 10'000'000'000'000, 1'000'000'000'000'000 };
constexpr static const char *front = "double";
#endif

inline auto append2filename(const std::string &filename, const std::string &append) -> std::string {
    if (const auto dotpos = filename.find_last_of('.'); dotpos != std::string::npos) {
        return filename.substr(0, dotpos) + append + filename.substr(dotpos);
    }

    return filename + append;
}

template <typename Float, typename IDX, typename GEN, typename F>
void benchmarkCholesky(const std::string &prefix, bool dotime, bool dovalue, sycl::queue &q, IDX idx, GEN gen, F f) {
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    std::cout << "Allocating Benchmark..."
              << std::endl;

    for (const auto &entry : std::filesystem::directory_iterator(front)) {
        const auto file = entry.path().string();

        if (!file.ends_with("input.bin") || !file.contains(front)) {
            continue;
        }

        std::cout << "Reading file: "
                  << file
                  << std::endl;

        size_t N;
        size_t gpu_allocN;
        std::vector<Float> gpu_in;
        std::vector<Float> gpu_out;

        // Read values
        if (std::ifstream file_in(file, std::ios::binary); file_in) {
            file_in.read(reinterpret_cast<char *>(&N), sizeof(N));
            gpu_allocN = gen(N);
            gpu_in.resize(gpu_allocN);
            gpu_out.resize(gpu_allocN);

            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    if (Float v; file_in.read(reinterpret_cast<char *>(&v), sizeof(v))) {
                        gpu_in[idx(N, i, j)] = v;
                    } else {
                        throw std::runtime_error("File matrix doesn't have enough bytes.");
                    }
                }
            }
        } else {
            throw std::runtime_error("Input file couldn't be opened.");
        }

        GPUAlloc<Float> gpuA(q, gpu_allocN);

        if (dovalue) {
            for (size_t dividx = 0; dividx < sizeof(divs) / sizeof(divs[0]); ++dividx) {
                const auto div = divs[dividx];

                std::cout << "Starting value benchmark: "
                          << div
                          << std::endl;

                const Float jitter = div != 0 ? 1. / div : 0;

                q.memcpy(gpuA.mem, gpu_in.data(), sizeof(Float) * gpu_in.size()).wait();
                f(N, gpuA.mem, jitter);
                q.wait_and_throw();
                q.memcpy(gpu_out.data(), gpuA.mem, sizeof(Float) * gpu_out.size()).wait();

                // Write values
                std::cout << "Writing results..."
                          << std::endl;

                if (std::ofstream file_out(append2filename(file, "-" + prefix + "-rand-" + std::to_string(dividx) + "-result"), std::ios::binary); file_out) {
                    file_out.write(reinterpret_cast<const char *>(&N), sizeof(N));
                    file_out.write(reinterpret_cast<const char *>(&div), sizeof(div));
                    for (size_t i = 0; i < N; ++i) {
                        for (size_t j = 0; j <= i; ++j) {
                            const auto v = gpu_out[idx(N, i, j)];
                            file_out.write(reinterpret_cast<const char *>(&v), sizeof(v));
                        }
                    }
                } else {
                    std::cerr << "Value file couldn't be created.\n";
                    return;
                }
            }
        }

        if (dotime) {
            const auto runkernel = [&](size_t N) -> std::vector<uint64_t> {
                q.memcpy(gpuA.mem, gpu_in.data(), sizeof(Float) * gpu_in.size()).wait();
                return f(N, gpuA.mem, 0.1);
            };

            const auto storeresult = [&](size_t N, const std::vector<uint64_t> times, const std::string &suffix) {
                if (std::ofstream file_out(append2filename(file, "-" + prefix + "-" + suffix), std::ios::binary); file_out) {
                    file_out.write(reinterpret_cast<const char *>(&N), sizeof(N));
                    file_out.write(reinterpret_cast<const char *>(times.data()), times.size() * sizeof(uint64_t));
                } else {
                    throw std::runtime_error("Time file couldn't be created.");
                }
            };

            const auto dotimebench = [&](size_t N, const std::string &suffix) {
                for (size_t i = 0; i < 5; ++i) {
                    std::cout << "Starting warmup: "
                              << i
                              << std::endl;

                    runkernel(8096);
                }

                std::cout << "Starting time benchmark: "
                          << N
                          << std::endl;

                const auto times = runkernel(N);

                std::cout << "Writing results..."
                          << std::endl;

                storeresult(N, times, suffix);
            };

            dotimebench(N, "wholetime");
            for (size_t Nidx = 0; (1 << Nidx) < N; ++Nidx) {
                dotimebench(1 << Nidx, std::to_string(Nidx) + "-parttime");
            }
        }
    }
}

template <typename Float, typename MATIDX, typename MATGEN, typename VECIDX, typename VECGEN, typename F>
void benchmarkSubstitution(const std::string &prefix, sycl::queue &q, MATIDX matidx, MATGEN matgen, VECIDX vecidx, VECGEN vecgen, F f) {
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    std::cout << "Allocating Benchmark..."
              << std::endl;

    constexpr size_t N = 60000;
    constexpr size_t Nb = 32;

    const size_t matgpu_allocN = matgen(N);
    const size_t vecgpu_allocN = vecgen(N, Nb);

    constexpr unsigned seed = 123'456'789;
    std::mt19937 generator(seed);

    std::uniform_real_distribution<Float> dest(.0001, 1.);

    for (size_t benchidx = 0; benchidx < 5; ++benchidx) {
        std::cout << "Forward Substitution benchmark: "
                  << benchidx
                  << std::endl;

        std::vector<Float> matgpu_in(matgpu_allocN);
        std::vector<Float> vecgpu_in(vecgpu_allocN);

        // Generate Values

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                matgpu_in[matidx(N, i, j)] = dest(generator);
            }
        }

        for (size_t i = 0; i < Nb; ++i) {
            for (size_t j = 0; j < N; ++j) {
                matgpu_in[vecidx(N, Nb, i, j)] = dest(generator);
            }
        }

        GPUAlloc<Float> gpuA(q, matgpu_in.size());
        GPUAlloc<Float> gpuB(q, vecgpu_in.size());

        const auto runkernel = [&](size_t N) -> std::vector<uint64_t> {
            q.memcpy(gpuA.mem, matgpu_in.data(), sizeof(Float) * matgpu_in.size()).wait();
            q.memcpy(gpuB.mem, vecgpu_in.data(), sizeof(Float) * vecgpu_in.size()).wait();
            return f(N, Nb, gpuA.mem, gpuB.mem);
        };

        const auto storeresult = [&](size_t _N, const std::vector<uint64_t> times, const std::string suffix) {
            if (std::ofstream file_out(std::string(front) + "/" + front + "-" + std::to_string(N) + "-" + std::to_string(benchidx) + "-input-" + prefix + "-" + suffix + ".bin", std::ios::binary); file_out) {
                file_out.write(reinterpret_cast<const char *>(&_N), sizeof(_N));
                file_out.write(reinterpret_cast<const char *>(times.data()), times.size() * sizeof(uint64_t));
            } else {
                throw std::runtime_error("Time file couldn't be created.");
            }
        };

        const auto dotimebench = [&](size_t N, const std::string &suffix) {
            for (size_t i = 0; i < 5; ++i) {
                std::cout << "Starting warmup: "
                          << i
                          << std::endl;

                runkernel(8096);
            }

            std::cout << "Starting time benchmark: "
                      << N
                      << std::endl;

            const auto times = runkernel(N);

            std::cout << "Writing results..."
                      << std::endl;

            storeresult(N, times, suffix);
        };

        dotimebench(N, "wholetime");
        for (size_t Nidx = 0; (1 << Nidx) < N; ++Nidx) {
            dotimebench(1 << Nidx, std::to_string(Nidx) + "-parttime");
        }
    }
}
