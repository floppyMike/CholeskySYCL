#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

#include "matrix.hpp"

using namespace hipsycl;

using real_type = float;

int main(int argc, char **argv) {
  constexpr size_t N = 16384;

  auto A = std::vector<real_type>(N * N);
  auto B = std::vector<real_type>(N * N);
  auto C = std::vector<real_type>(N * N);

  sycl::queue q([](const sycl::device &dev) {
    if (dev.is_gpu()) {
      return 1;
    } else {
      return -1;
    }
  });

  // Init Inputs

  const auto gpuA = sycl::malloc_device<real_type>(A.size(), q);
  const auto gpuB = sycl::malloc_device<real_type>(B.size(), q);
  auto e1 = q.memcpy(gpuA, A.data(), sizeof(real_type) * A.size());
  auto e2 = q.memcpy(gpuB, B.data(), sizeof(real_type) * B.size());

  // Init Outputs

  const auto gpuC = sycl::malloc_device<real_type>(C.size(), q);
  auto e3 = q.memset(gpuC, 0, sizeof(real_type) * C.size());

  // Setup benchmark

  q.wait();
  const auto start = std::chrono::high_resolution_clock::now();

  // Matrix multiply

  constexpr size_t GROUPSIZE = 32;

  sycl::range global(N, N);
  sycl::range local(GROUPSIZE, GROUPSIZE);

  auto e4 = q.submit([&](sycl::handler &cgh) {
    auto localA = sycl::local_accessor<real_type, 2>(local, cgh);
    auto localB = sycl::local_accessor<real_type, 2>(local, cgh);

    cgh.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> idx) {
      const auto groupi = idx.get_group(0);
      const auto groupj = idx.get_group(1);

      const auto locali = idx.get_local_id(0);
      const auto localj = idx.get_local_id(1);

      auto gpuAoffset = gpuA + groupi * GROUPSIZE * N;
      auto gpuBoffset = gpuB + groupj * GROUPSIZE;
      auto gpuCoffset = gpuC + groupi * GROUPSIZE * N + groupj * GROUPSIZE;

      real_type sum = 0;

      for (size_t blkIdx = 0; blkIdx < N; blkIdx += GROUPSIZE) {
        localA[locali][localj] = gpuAoffset[locali * N + localj];
        localB[locali][localj] = gpuBoffset[locali * N + localj];

        idx.barrier(sycl::access::fence_space::local_space);

        gpuAoffset += GROUPSIZE;
        gpuBoffset += GROUPSIZE * N;

        for (size_t k = 0; k < GROUPSIZE; ++k) {
          sum += localA[locali][k] * localB[k][localj];
        }

        idx.barrier(sycl::access::fence_space::local_space);
      }

      gpuCoffset[locali * N + localj] = sum;
    });
  });

  // Report benchmark

  q.wait();
  const auto elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Time: " << elapsed << std::endl;

  // Copy Output to Host

  q.memcpy(C.data(), gpuC, sizeof(real_type) * C.size()).wait();

  // Clean up & Results

  sycl::free(gpuA, q);
  sycl::free(gpuB, q);
  sycl::free(gpuC, q);

  return 0;
}
